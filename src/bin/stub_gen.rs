use pyo3_stub_gen::Result;
use symbolica_community::stub_info;

fn main() -> Result<()> {
    let only = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => None,
        [arg] if arg == "--hep-only" => Some("hep"),
        [arg] if arg == "--spenso-only" => Some("spenso"),
        _ => {
            return Err(
                std::io::Error::other("Usage: stub_gen [--hep-only | --spenso-only]").into(),
            );
        }
    };
    let mut stub = stub_info()?;
    if only == Some("spenso") {
        let spenso = stub
            .modules
            .get("symbolica.community.spenso")
            .ok_or_else(|| {
                std::io::Error::other("Spenso did not register its Python stub inventory")
            })?;
        let directory = stub.python_root.join("symbolica/community/spenso");
        std::fs::create_dir_all(&directory)?;
        let source = spenso
            .to_string()
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(
            directory.join("__init__.pyi"),
            source.trim_end().to_owned() + "\n",
        )?;
        return Ok(());
    }
    let feynkit = feynkit_py::stub_info()?
        .modules
        .remove("symbolica.community.feynkit")
        .ok_or_else(|| {
            std::io::Error::other("FeynKit did not register its Python stub inventory")
        })?;
    stub.modules.remove("symbolica.community.feynkit");
    let hep = feynkit
        .to_string()
        .replace("symbolica.community.feynkit", "symbolica.community.hep");
    if only != Some("hep") {
        if let Some(community) = stub.modules.get_mut("symbolica.community") {
            community.submodules.remove("feynkit");
            community.submodules.insert("hep".to_owned());
        }
        stub.generate()?;
    }
    let directory = stub.python_root.join("symbolica/community/hep");
    std::fs::create_dir_all(&directory)?;
    let source = (hep + "\nfrom . import oneloop as oneloop\n")
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(
        directory.join("__init__.pyi"),
        source.trim_end().to_owned() + "\n",
    )?;
    Ok(())
}
