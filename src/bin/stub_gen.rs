use pyo3_stub_gen::Result;
use symbolica_community::physics::tensors::stub_info;
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "info")).init();
        let stub = stub_info()?;
        stub.generate()?;
        Ok(())
}
