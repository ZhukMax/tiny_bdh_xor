use anyhow::Result;
use tch::{nn, Device, Kind, Reduction, Tensor};
use tch::nn::OptimizerConfig;

/// y = ReLU( (x E) D^T )
fn relu_lowrank_forward(x: &Tensor, e: &Tensor, d: &Tensor) -> Tensor {
    let h = x.matmul(e);                           // [B,n]·[n,d] = [B,d]
    h.matmul(&d.transpose(-1, -2)).relu()      // [B,d]·[d,n] = [B,n]
}

fn main() -> Result<()> {
    let dev = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };

    // XOR
    let x = Tensor::from_slice(&[
        0f32,0.,1.,  0.,1.,1.,  1.,0.,1.,  1.,1.,1.
    ]).reshape([4,3]).to_device(dev);
    let y = Tensor::from_slice(&[0f32,1.,1.,0.]).reshape([4,1]).to_device(dev);

    // --- hyperparameters ---
    let n: i64 = 64;
    let d: i64 = 16;
    let u: f64 = 0.20;
    let hebb_lr: f64 = 0.01;
    let smax: f64 = 1.0;
    let sparsity_thresh: f64 = 5e-3;
    let lr: f64 = 5e-3;
    let steps = 3000;

    let vs = nn::VarStore::new(dev);
    let root = &vs.root();

    let e  = root.var("E",  &[n,d], nn::Init::Randn { mean: 0.0, stdev: 0.05 });
    let dx = root.var("Dx", &[n,d], nn::Init::Randn { mean: 0.0, stdev: 0.05 });
    let dy = root.var("Dy", &[n,d], nn::Init::Randn { mean: 0.0, stdev: 0.05 });

    let r_in   = root.var("R_in",   &[3,n], nn::Init::Randn { mean: 0.0, stdev: 0.20 });
    let w_read = root.var("W_read", &[n,1], nn::Init::Randn { mean: 0.0, stdev: 0.20 });

    let mut opt = nn::Adam::default().build(&vs, lr)?;
    let mut sigma = Tensor::zeros(&[n, n], (Kind::Float, dev));

    for step in 0..steps {
        // --- forward
        let x_neu = x.matmul(&r_in);
        let y1 = relu_lowrank_forward(&x_neu, &e, &dx);
        let a  = x_neu.matmul(&sigma.transpose(-1, -2));
        let y2 = y1 + a;
        let z  = relu_lowrank_forward(&y2, &e, &dy);
        let logits = z.matmul(&w_read);

        // --- loss/backward/step (logits → BCEWithLogits)
        let loss = logits.binary_cross_entropy_with_logits::<Tensor>(&y, None, None, Reduction::Mean);
        opt.zero_grad();
        loss.backward();
        opt.step();

        tch::no_grad(|| {
            let bsz = x.size()[0] as f64;

            // outer = (y2^T @ x_neu) * (hebb_lr / B)
            let outer = y2
                .detach()
                .transpose(-1, -2)
                .matmul(&x_neu.detach())
                .to_kind(Kind::Float)
                * (hebb_lr / bsz);

            let zeros = Tensor::zeros_like(&sigma);
            let mut s = sigma.shallow_clone();

            s *= 1.0 - u;
            s += &outer;

            s = s.clamp(-smax, smax);

            let keep = s.abs().ge(sparsity_thresh);
            s = s.where_self(&keep, &zeros);

            let row_norm = s.square().sum_dim_intlist([1].as_ref(), true, Kind::Float).sqrt();
            s = &s / &row_norm.clamp_min(1.0);

            sigma.copy_(&s);
        });

        if step % 300 == 0 {
            let y_hat = logits.sigmoid();
            let acc = y_hat.gt(0.5)
                .eq_tensor(&y.gt(0.5))
                .to_kind(Kind::Float)
                .mean(Kind::Float)
                .double_value(&[]);
            println!("step {:4}  loss {:.4}  acc {:.2}", step, loss.double_value(&[]), acc);
        }
    }

    let x_neu = x.matmul(&r_in);
    let y1 = relu_lowrank_forward(&x_neu, &e, &dx);
    let a  = x_neu.matmul(&sigma.transpose(-1, -2));
    let y2 = y1 + a;
    let z  = relu_lowrank_forward(&y2, &e, &dy);
    let preds = z.matmul(&w_read).sigmoid().gt(0.5).to_kind(Kind::Int64);
    println!("\nPred:\n{:?}", preds);

    let probs = z.matmul(&w_read).sigmoid();
    println!("\nProbs (σ=on):");
    probs.print();
    println!("Preds (σ=on):");
    preds.print();

    let y1_nos = relu_lowrank_forward(&x_neu, &e, &dx);
    let y2_nos = y1_nos;
    let z_nos  = relu_lowrank_forward(&y2_nos, &e, &dy);
    let preds_nos = z_nos.matmul(&w_read).sigmoid().gt(0.5).to_kind(Kind::Int64);
    println!("\nPreds (σ=off):");
    preds_nos.print();

    Ok(())
}
