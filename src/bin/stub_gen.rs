use pyo3_stub_gen::Result;
use symbolica_community::stub_info;

fn main() -> Result<()> {
    let only = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => None,
        [arg] if arg == "--hep-only" => Some("hep"),
        [arg] if arg == "--spenso-only" => Some("spenso"),
        [arg] if arg == "--oneloop-only" => Some("oneloop"),
        _ => {
            return Err(std::io::Error::other(
                "Usage: stub_gen [--hep-only | --spenso-only | --oneloop-only]",
            )
            .into());
        }
    };
    let mut stub = stub_info()?;
    if let Some(module) = stub.modules.get_mut("symbolica.community.spenso") {
        spynso3::SpensoModule::prepare_stub_module(module);
    }
    if only == Some("spenso") {
        let spenso = stub
            .modules
            .get("symbolica.community.spenso")
            .ok_or_else(|| {
                std::io::Error::other("Spenso did not register its Python stub inventory")
            })?;
        let directory = stub.python_root.join("symbolica/community/spenso");
        std::fs::create_dir_all(&directory)?;
        let source = spynso3::SpensoModule::stub_source(spenso)
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
    let reducer = stub
        .modules
        .remove("symbolica.community.hep.oneloop")
        .ok_or_else(|| {
            std::io::Error::other("One-loop reducer did not register its Python stub inventory")
        })?;
    let directory = stub.python_root.join("symbolica/community/hep");
    std::fs::create_dir_all(&directory)?;
    let oneloop = format!("{}\n{}", include_str!("../../stubs/oneloop.pyi"), reducer)
        .replace("symbolica.community.feynkit", "symbolica.community.hep");
    let oneloop = oneloop
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(
        directory.join("oneloop.pyi"),
        oneloop.trim_end().to_owned() + "\n",
    )?;
    if only == Some("oneloop") {
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
        if let Some(spenso) = stub.modules.get("symbolica.community.spenso") {
            std::fs::write(
                stub.python_root
                    .join("symbolica/community/spenso/__init__.pyi"),
                spynso3::SpensoModule::stub_source(spenso),
            )?;
        }
    }
    let directory = stub.python_root.join("symbolica/community/hep");
    std::fs::create_dir_all(&directory)?;
    let source = (hep
        + "\n"
        + include_str!("../../python/symbolica/community/hep/ibp.pyi")
        + "\nfrom . import oneloop as oneloop\n")
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
