use qml_core::DenseMatrix;
use qml_format::QmfRankOne;
use qml_model::{
    PayloadBudget, inspect_gguf as parse_gguf_metadata,
    inspect_safetensors as parse_safetensors_metadata,
};
use qml_scope::{ScopeConfig, analyze_safetensors};
use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::ExitCode;

const HELP: &str = r#"Quantum Morph Laboratory (qml)

USAGE:
    qml <COMMAND>

COMMANDS:
    doctor                               Show host platform & CPU SIMD diagnostics
    demo-rank1 [--output PATH]           Run a synthetic rank-one factorization demo
    inspect-qmf PATH                     Validate an experimental QMF v0 file
    inspect-safetensors [--json] PATH    Read Safetensors metadata only (16 MiB header cap)
    inspect-gguf [--json] PATH           Read GGUF v2/v3 metadata & tensor directory only
    scope [OPTIONS] PATH                 Run Q-Scope neural tensor & spectral analysis
    help                                 Show this help text
    version                              Show the CLI version

SCOPE OPTIONS:
    --json                               Emit machine-readable JSON report
    --max-tensor-mb N                    Per-tensor memory budget in MiB (default: 256)
    --svd-rank K                         Maximum singular values per 2D matrix (default: 16)
    --no-svd                             Skip SVD & cross-layer subspace analysis

The demo uses a generated synthetic matrix, not a pretrained model. QMF v0 is an
unstable, single-matrix experimental format. Imported models are opened read-only."#;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("doctor") if args.next().is_none() => doctor(),
        Some("demo-rank1") => demo_rank_one(args.collect()),
        Some("inspect-qmf") => inspect_qmf(args.collect()),
        Some("inspect-safetensors") => inspect_safetensors(args.collect()),
        Some("inspect-gguf") => inspect_gguf(args.collect()),
        Some("scope") => run_scope(args.collect()),
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
    println!("QML host diagnostics");
    println!("  OS: {}", env::consts::OS);
    println!("  Architecture: {}", env::consts::ARCH);
    match std::thread::available_parallelism() {
        Ok(count) => println!("  Available logical parallelism: {}", count.get()),
        Err(error) => println!("  Available logical parallelism: unavailable ({error})"),
    }
    println!("  CPU SIMD features: {}", detect_cpu_simd());
    println!("  Compute backend: CPU reference (safe Rust)");
    println!("  GPU/VRAM discovery: not yet implemented");
    ExitCode::SUCCESS
}

fn detect_cpu_simd() -> String {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let mut features = Vec::new();
        if std::arch::is_x86_feature_detected!("sse4.2") {
            features.push("SSE4.2");
        }
        if std::arch::is_x86_feature_detected!("avx") {
            features.push("AVX");
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            features.push("FMA");
        }
        if std::arch::is_x86_feature_detected!("f16c") {
            features.push("F16C");
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            features.push("AVX-512F");
        }
        if features.is_empty() {
            "baseline x86".to_owned()
        } else {
            features.join(", ")
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        "non-x86 platform baseline".to_owned()
    }
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

fn inspect_gguf(args: Vec<String>) -> ExitCode {
    let (json, path) = match args.as_slice() {
        [path] => (false, PathBuf::from(path)),
        [flag, path] if flag == "--json" => (true, PathBuf::from(path)),
        _ => {
            eprintln!("usage: qml inspect-gguf [--json] PATH\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("could not open GGUF file {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match parse_gguf_metadata(file) {
        Ok(metadata) if json => match serde_json::to_string_pretty(&metadata) {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not encode GGUF metadata as JSON: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(metadata) => {
            println!("GGUF v{} metadata (payload not read)", metadata.version);
            println!("  File bytes: {}", metadata.file_bytes);
            println!("  KV pairs: {}", metadata.kv_count);
            println!("  Tensor count: {}", metadata.tensor_count);
            for (k, v) in metadata.metadata.iter().take(20) {
                println!("  meta {k}: {v:?}");
            }
            if metadata.metadata.len() > 20 {
                println!(
                    "  ... ({} additional KV pairs)",
                    metadata.metadata.len() - 20
                );
            }
            for tensor in metadata.tensors.iter().take(25) {
                println!(
                    "  tensor {}: {} {:?}, offset {}",
                    tensor.name, tensor.ggml_type, tensor.dimensions, tensor.offset
                );
            }
            if metadata.tensors.len() > 25 {
                println!("  ... ({} additional tensors)", metadata.tensors.len() - 25);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("could not inspect GGUF file {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn run_scope(args: Vec<String>) -> ExitCode {
    let mut json = false;
    let mut cfg = ScopeConfig::default();
    let mut path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json = true;
                i += 1;
            }
            "--no-svd" => {
                cfg.compute_svd = false;
                cfg.compute_cross_layer = false;
                i += 1;
            }
            "--max-tensor-mb" => {
                let Some(val) = args.get(i + 1).and_then(|s| s.parse::<u64>().ok()) else {
                    eprintln!("--max-tensor-mb expects a positive integer\n\n{HELP}");
                    return ExitCode::from(2);
                };
                cfg.payload_budget = PayloadBudget {
                    max_tensor_bytes: val.saturating_mul(1024 * 1024),
                    ..cfg.payload_budget
                };
                i += 2;
            }
            "--svd-rank" => {
                let Some(val) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) else {
                    eprintln!("--svd-rank expects a positive integer\n\n{HELP}");
                    return ExitCode::from(2);
                };
                if val == 0 {
                    eprintln!("--svd-rank must be >= 1\n\n{HELP}");
                    return ExitCode::from(2);
                }
                cfg.svd_max_rank = val;
                i += 2;
            }
            arg if arg.starts_with('-') => {
                eprintln!("unknown option for scope: {arg}\n\n{HELP}");
                return ExitCode::from(2);
            }
            arg => {
                if path.is_some() {
                    eprintln!("unexpected extra positional argument: {arg}\n\n{HELP}");
                    return ExitCode::from(2);
                }
                path = Some(PathBuf::from(arg));
                i += 1;
            }
        }
    }

    let Some(path) = path else {
        eprintln!(
            "usage: qml scope [--json] [--max-tensor-mb N] [--svd-rank K] [--no-svd] PATH\n\n{HELP}"
        );
        return ExitCode::from(2);
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

    match analyze_safetensors(file, &cfg) {
        Ok(report) if json => match serde_json::to_string_pretty(&report) {
            Ok(text) => {
                println!("{text}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not serialize Q-Scope report as JSON: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(report) => {
            println!("Q-Scope Neural Model Observatory Report");
            println!("  File: {}", path.display());
            println!(
                "  Size: {} bytes (header {}, payload {})",
                report.file_bytes, report.header_bytes, report.data_bytes
            );
            println!(
                "  Tensors: {} total, {} statistically analyzed, {} SVD-analyzed",
                report.summary.total_tensors,
                report.summary.analyzed_tensors,
                report.summary.svd_analyzed_matrices
            );
            println!("  Total parameters: {}", report.summary.total_parameters);
            if let Some(ref name) = report.summary.max_quantization_susceptibility_tensor {
                println!(
                    "  Peak quantization susceptibility: {name} (score {:.2})",
                    report
                        .summary
                        .max_quantization_susceptibility_score
                        .unwrap_or(0.0)
                );
            }
            if let Some(mean_aff) = report.summary.mean_cross_layer_shared_score {
                println!(
                    "  Mean cross-layer subspace affinity ({} pairs): {:.4}",
                    report.cross_layer_relations.len(),
                    mean_aff
                );
            }
            println!("\nTensors:");
            for t in &report.tensors {
                if let Some(ref s) = t.stats {
                    let svd_note = match &t.matrix_structure {
                        Some(m) => format!(
                            " | srank={:.2} erank={:.2} r90={} r99={}",
                            m.stable_rank, m.effective_rank, m.rank_90_energy, m.rank_99_energy
                        ),
                        None => String::new(),
                    };
                    println!(
                        "  {} [{} {:?}] rms={:.4} absmax={:.4} crest={:.2} out6σ={} ent={:.2}b susc={:.2}{}",
                        t.name,
                        t.dtype,
                        t.shape,
                        s.rms,
                        s.absmax,
                        s.crest_factor,
                        s.outlier_6sigma_count,
                        s.shannon_entropy_bits,
                        s.quantization_susceptibility_score,
                        svd_note
                    );
                } else {
                    println!(
                        "  {} [{} {:?}] skipped: {}",
                        t.name,
                        t.dtype,
                        t.shape,
                        t.stats_skipped_reason.as_deref().unwrap_or("unavailable")
                    );
                }
            }
            if !report.cross_layer_relations.is_empty() {
                println!("\nCross-Layer Subspace Relationships (top 10 by shared score):");
                let mut sorted = report.cross_layer_relations.clone();
                sorted.sort_by(|a, b| {
                    b.shared_structure_score
                        .total_cmp(&a.shared_structure_score)
                });
                for rel in sorted.iter().take(10) {
                    println!(
                        "  {:?} {} <-> {}: shared={:.4} (left={:.4}, right={:.4}, cos={:.4})",
                        rel.role,
                        rel.tensor_a,
                        rel.tensor_b,
                        rel.shared_structure_score,
                        rel.left_subspace_affinity,
                        rel.right_subspace_affinity,
                        rel.weight_cosine_similarity
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Q-Scope analysis failed for {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}
