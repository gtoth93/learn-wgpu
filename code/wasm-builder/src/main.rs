use anyhow::Result;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::{path::PathBuf, process::Command, time::Instant};
use wasm_bindgen_cli_support::Bindgen;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct WasmTarget {
    package: String,
    out: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let Some(input) = std::env::args().nth(1) else {
        tracing::info!("No input file supplied");
        std::process::exit(1);
    };

    let json = std::fs::read_to_string(input)?;

    let targets: Vec<WasmTarget> = serde_json::from_str(&json)?;

    // Tell cargo to build all targets as wasm32-unknown-unknown
    let mut command = Command::new("cargo");

    let start_time = Instant::now();

    command
        .arg("build")
        .arg("--release")
        .arg("--target")
        .arg("wasm32-unknown-unknown");
    for target in &targets {
        command.arg("-p").arg(&target.package);
    }
    let status = command.spawn()?.wait()?;

    if !status.success() {
        anyhow::bail!("Failed to compile WASM with code ({status})");
    }

    let errors = targets
        .par_iter()
        .map(|target| {
            let path = PathBuf::from(&target.out);

            // Ignore contents to prevent cargo-watch looping
            std::fs::create_dir_all(&path)?;
            std::fs::write(path.join(".gitignore"), "*\n")?;

            Bindgen::new()
                .bundler(true)?
                .input_path(format!(
                    "target/wasm32-unknown-unknown/release/{}.wasm",
                    target.package
                ))
                .out_name("demo")
                .generate(path)
        })
        .filter_map(Result::err)
        .collect::<Vec<_>>();

    if !errors.is_empty() {
        for error in &errors {
            tracing::error!("{}", error);
        }
        tracing::error!("Encountered {} error(s)", errors.len());
        std::process::exit(1);
    }

    let time_taken = start_time.elapsed();
    tracing::info!(
        "Successfully processed {} target(s) in {:?}",
        targets.len(),
        time_taken
    );

    Ok(())
}
