//! No tokenizer, model weights or Vela dependency: X @ W.T with packed FP32 B.
use candle_core::{Device, Tensor};
fn main() -> anyhow::Result<()> {
    let library = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: linear /absolute/path/libcandle_mlas.so"))?;
    // The host supplies the ABI/source-qualified library built by our script,
    // and verifies that its build target supports this CPU.
    unsafe { candle_mlas::initialize(&library, 1)? };
    let weights = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (3, 2), &Device::Cpu)?;
    let packed = candle_mlas::PackedWeight::new(&weights)?;
    drop(weights); // the packed allocation owns all projection data
    let input = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &Device::Cpu)?;
    let output = packed.forward(&input)?;
    assert_eq!(
        output.to_vec2::<f32>()?,
        vec![vec![5., 11., 17.], vec![11., 25., 39.]]
    );
    println!("{:?}", output.to_vec2::<f32>()?);
    Ok(())
}
