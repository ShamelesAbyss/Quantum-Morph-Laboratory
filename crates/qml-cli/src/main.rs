use qml_core::DenseMatrix;
use qml_format::QmfRankOne;
use qml_model::inspect_safetensors as parse_safetensors_metadata;
use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::ExitCode;

const HELP: &str = r#"Quantum Morph Laboratory (qml)

USAGE:
    qml <COMMAND>

COMMANDS:
    doctor                         Show basic host platform diagnostics
    demo-rank1 [--output PATH]     Run a synthetic rank-one factorization demo
    inspect-qmf PATH               Validate an experimental QMF v0 file
    inspect-safetensors [--json] PATH  Read metadata only (16 MiB header cap)
    help                           Show this help text
    version                        Show the CLI version

The demo uses a generated synthetic matrix, not a pretrained model. The rank-one
pivot approximation is a simple baseline and is not guaranteed optimal for general
matrices. QMF v0 is an unstable, single-matrix experimental format.

Safetensors inspection reads only the bounded header, never payload bytes or tensor
values. GGUF inspection, transformations, benchmarking, and inference are unavailable."#;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("doctor") if args.next().is_none() => doctor(),
        Some("demo-rank1") => demo_rank_one(args.collect()),
        Some("inspect-qmf") => inspect_qmf(args.collect()),
        Some("inspect-safetensors") => inspect_safetensors(args.collect()),
        Some("help" | "--help" | "-h") if args.next().is_none() => {
            println!("{HELP}");
            ExitCode::SUCCESS
        }
        Some("version" | "--version" | "-V") if args.next().is_none() => {
            println!("qml {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some(command) => {
            eprintln!("unknown command or unexpected argument: {command}\n\n{HELP}");
            ExitCode::from(2)
        }
        None => {
            println!("{HELP}");
            ExitCode::SUCCESS
        }
    }
}

fn doctor() -> ExitCode {
    println!("QML host diagnostics (basic)");
    println!("  OS: {}", env::consts::OS);
    println!("  Architecture: {}", env::consts::ARCH);
    match std::thread::available_parallelism() {
        Ok(count) => println!("  Available logical parallelism: {}", count.get()),
        Err(error) => println!("  Available logical parallelism: unavailable ({error})"),
    }
    println!("  GPU, memory, disk, and backend discovery: not implemented");
    ExitCode::SUCCESS
}

fn demo_rank_one(args: Vec<String>) -> ExitCode {
    let output = match parse_output_path(&args) {
        Ok(path) => path.unwrap_or_else(|| PathBuf::from("experiments/rank1-demo.qmf")),
        Err(message) => {
            eprintln!("{message}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };

    match run_rank_one_demo(&output) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rank-one demo failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_output_path(args: &[String]) -> Result<Option<PathBuf>, String> {
    if args.is_empty() {
        return Ok(None);
    }
    if args.len() == 2 && args[0] == "--output" {
        return Ok(Some(PathBuf::from(&args[1])));
    }
    Err("expected either no options or exactly `--output PATH`".to_owned())
}

fn run_rank_one_demo(output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let rows = 64;
    let cols = 48;
    let left: Vec<f64> = (0..rows).map(|row| (row as f64 + 1.0) / 64.0).collect();
    let right: Vec<f64> = (0..cols)
        .map(|col| ((col % 11) as f64 - 5.0) / 5.0)
        .collect();
    let data: Vec<f64> = left
        .iter()
        .flat_map(|left_value| {
            right
                .iter()
                .map(move |right_value| left_value * right_value)
        })
        .collect();
    let original = DenseMatrix::from_flat(rows, cols, data)?;

    let factors = original.rank_one_pivot_approximation();
    let reconstruction_error = original.approximation_error(&factors)?;
    let vector: Vec<f64> = (0..cols)
        .map(|index| ((index % 7) as f64 - 3.0) / 7.0)
        .collect();
    let dense_output = original.matvec(&vector)?;
    let factorized_output = factors.matvec(&vector)?;
    let max_matvec_error = dense_output
        .iter()
        .zip(&factorized_output)
        .map(|(dense, compressed)| (dense - compressed).abs())
        .fold(0.0_f64, f64::max);

    let archive = QmfRankOne::new(factors);
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(output)?;
    archive.write_to(&mut BufWriter::new(file))?;

    let reloaded = QmfRankOne::read_from(File::open(output)?)?;
    let reloaded_output = reloaded.factors().matvec(&vector)?;
    let reload_matvec_error = factorized_output
        .iter()
        .zip(reloaded_output)
        .map(|(before, after)| (before - after).abs())
        .fold(0.0_f64, f64::max);

    let dense_bytes = rows * cols * std::mem::size_of::<f64>();
    let factor_bytes = reloaded.factors().parameter_bytes_f64();
    let archive_bytes = fs::metadata(output)?.len();

    println!("Synthetic rank-one matrix proof of concept");
    println!("  Matrix dimensions: {rows} x {cols}");
    println!("  Approximation: deterministic largest-pivot cross approximation (rank 1)");
    println!("  Dense f64 tensor bytes: {dense_bytes}");
    println!("  Factor f64 bytes: {factor_bytes}");
    println!("  QMF v0 container bytes: {archive_bytes}");
    println!(
        "  Dense-to-container size ratio: {:.2}x",
        dense_bytes as f64 / archive_bytes as f64
    );
    println!("  Frobenius reconstruction error: {reconstruction_error:.3e}");
    println!("  Dense-vs-factorized matvec max absolute error: {max_matvec_error:.3e}");
    println!("  Pre-save-vs-reload matvec max absolute error: {reload_matvec_error:.3e}");
    println!("  Output: {}", output.display());
    println!(
        "  Scope: generated synthetic fixture only; not a pretrained model or general benchmark"
    );
    println!("  Format status: QMF v0 is experimental and not compatibility-stable");
    Ok(())
}

fn inspect_qmf(args: Vec<String>) -> ExitCode {
    if args.len() != 1 {
        eprintln!("usage: qml inspect-qmf PATH\n\n{HELP}");
        return ExitCode::from(2);
    }
    let path = PathBuf::from(&args[0]);
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("could not open QMF file {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match QmfRankOne::read_from(file) {
        Ok(archive) => {
            println!("Experimental QMF file (not compatibility-stable)");
            println!("  Version: {}", archive.version());
            println!("  Representation: rank-one f64 matrix");
            println!(
                "  Dimensions: {} x {}",
                archive.factors().rows(),
                archive.factors().cols()
            );
            println!(
                "  Factor parameter bytes: {}",
                archive.factors().parameter_bytes_f64()
            );
            println!(
                "  Container bytes: {}",
                fs::metadata(&path)
                    .map(|metadata| metadata.len())
                    .unwrap_or(0)
            );
            println!("  Integrity: CRC-32 valid (error detection only; not authentication)");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("could not read QMF file {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn inspect_safetensors(args: Vec<String>) -> ExitCode {
    let (json, path) = match args.as_slice() {
        [path] => (false, PathBuf::from(path)),
        [flag, path] if flag == "--json" => (true, PathBuf::from(path)),
        _ => {
            eprintln!("usage: qml inspect-safetensors [--json] PATH\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!(
                "could not open Safetensors file {}: {error}",
                path.display()
            );
            return ExitCode::FAILURE;
        }
    };
    match parse_safetensors_metadata(file) {
        Ok(metadata) if json => match serde_json::to_string_pretty(&metadata) {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not encode Safetensors metadata as JSON: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(metadata) => {
            println!("Safetensors metadata (payload not read)");
            println!("  File bytes: {}", metadata.file_bytes);
            println!("  Header bytes: {}", metadata.header_bytes);
            println!("  Tensor payload bytes: {}", metadata.data_bytes);
            println!("  Tensor count: {}", metadata.tensors.len());
            for tensor in &metadata.tensors {
                println!(
                    "  {}: {} {:?}, bytes {}..{} ({} bytes)",
                    tensor.name,
                    tensor.dtype,
                    tensor.shape,
                    tensor.data_offsets[0],
                    tensor.data_offsets[1],
                    tensor.data_bytes
                );
            }
            if !metadata.metadata.is_empty() {
                println!("  File metadata: {:?}", metadata.metadata);
            }
            println!("  Scope: header validation only; tensor values were not inspected");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!(
                "could not inspect Safetensors file {}: {error}",
                path.display()
            );
            ExitCode::FAILURE
        }
    }
}
