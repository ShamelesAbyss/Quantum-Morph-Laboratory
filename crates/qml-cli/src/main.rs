use qml_core::{DenseMatrix, DeterministicRng};
use qml_forge::{TransformStrategy, compare_safetensors_and_qmf, forge_safetensors_to_qmf};
use qml_format::{QmfArchiveV1, QmfRankOne, peek_qmf_version};
use qml_genesis::{GenesisConfig, run_genesis};
use qml_model::{
    PayloadBudget, SafetensorsFile, inspect_gguf as parse_gguf_metadata,
    inspect_safetensors as parse_safetensors_metadata,
};
use qml_quant::{FactorPrecision, benchmark_conventional_baselines, compare_rmq_vs_baselines};
use qml_runtime::{
    QCoreModel, TransformerConfig, evaluate_inference_fidelity,
    generate_synthetic_llama_safetensors,
};
use qml_scope::{ScopeConfig, analyze_safetensors};
use qml_tokenizer::BpeTokenizer;
use serde::Serialize;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Cursor};
use std::path::PathBuf;
use std::process::ExitCode;

const HELP: &str = r#"Quantum Morph Laboratory (qml)
Don't just compress intelligence. Transform its representation.

USAGE:
    qml <COMMAND> [OPTIONS]

COMMANDS:
    doctor                               Show host platform & CPU SIMD diagnostics
    inspect-safetensors [--json] PATH    Read Safetensors metadata only (16 MiB header cap)
    inspect-gguf [--json] PATH           Read GGUF v2/v3 metadata & tensor directory only
    inspect-qmf [--json] PATH            Validate an experimental QMF v0 or v1 file
    scope [OPTIONS] PATH                 Run Q-Scope neural tensor, spectral & cross-layer analysis
    baseline [--json] [--synthetic|PATH] Run conventional quantization baselines with full byte accounting
    rmq-ablate [--json]                  Run RMQ vs baseline ablation (correlated + Gaussian control)
    morph PATH --method M --output OUT   Transform a Safetensors file into an experimental QMF v1 artifact
    compare SOURCE.safetensors OUT.qmf   Verify a QMF v1 artifact directly against its source Safetensors
    infer-demo [--json]                  Run native Q-Core CPU inference & KL-divergence comparison
    generate QMF --config C --tokenizer T --prompt P [OPTS]
                                         Run real text generation with Q-Core + BPE tokenizer
    genesis [--json] [--synthetic|PATH]  Run Q-Genesis multi-objective evolutionary representation search
    demo-rank1 [--output PATH]           Run the original synthetic rank-one QMF v0 demo
    help                                 Show this help text
    version                              Show the CLI version

MORPH METHODS (--method):
    fp16 | bf16 | int8 | int4 | int4-asym | svd | rmq | map

Imported model files are always opened read-only and never modified. QMF v0/v1,
RMQ, MAP, Q-Core, and Q-Genesis are active experimental research prototypes."#;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("doctor") if args.next().is_none() => doctor(),
        Some("demo-rank1") => demo_rank_one(args.collect()),
        Some("inspect-qmf") => inspect_qmf(args.collect()),
        Some("inspect-safetensors") => inspect_safetensors(args.collect()),
        Some("inspect-gguf") => inspect_gguf(args.collect()),
        Some("scope") => run_scope(args.collect()),
        Some("baseline") => run_baseline(args.collect()),
        Some("rmq-ablate") => run_rmq_ablate(args.collect()),
        Some("morph") => run_morph(args.collect()),
        Some("compare") => run_compare(args.collect()),
        Some("infer-demo") => run_infer_demo(args.collect()),
        Some("generate") => run_generate(args.collect()),
        Some("genesis") => run_genesis_cmd(args.collect()),
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
    let hw = qml_hardware::probe_hardware();
    println!("QML host diagnostics");
    println!("  OS: {}", hw.os);
    println!("  Architecture: {}", hw.arch);
    println!("  CPU: {}", hw.cpu_brand);
    println!("  Available logical parallelism: {}", hw.logical_cores);
    let simd = if hw.simd_features.is_empty() {
        "platform baseline".to_owned()
    } else {
        hw.simd_features.join(", ")
    };
    println!("  CPU SIMD features: {simd}");
    println!(
        "  System RAM: {:.2} GiB available / {:.2} GiB total",
        hw.available_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        hw.total_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    println!("  Recommended backend: {}", hw.recommended_backend);
    if hw.gpus.is_empty() {
        println!("  GPU/WGPU adapters: none detected (CPU fallback active)");
    } else {
        println!("  GPU/WGPU adapters ({}):", hw.gpus.len());
        for gpu in &hw.gpus {
            println!(
                "    - {} [{} / {}] max_buffer={:.0} MiB",
                gpu.name,
                gpu.device_type,
                gpu.backend,
                gpu.max_buffer_size as f64 / (1024.0 * 1024.0)
            );
        }
    }
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
    let (json, path) = match args.as_slice() {
        [path] => (false, PathBuf::from(path)),
        [flag, path] if flag == "--json" => (true, PathBuf::from(path)),
        _ => {
            eprintln!("usage: qml inspect-qmf [--json] PATH\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(error) => {
            eprintln!("could not read QMF file {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let version = match peek_qmf_version(&bytes) {
        Ok(v) => v,
        Err(error) => {
            eprintln!("could not parse QMF header {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match version {
        0 => match QmfRankOne::from_bytes(&bytes) {
            Ok(archive) if json => {
                let summary = serde_json::json!({
                    "version": 0,
                    "representation": "rank_one_f64",
                    "rows": archive.factors().rows(),
                    "cols": archive.factors().cols(),
                    "container_bytes": bytes.len(),
                });
                println!("{}", serde_json::to_string_pretty(&summary).unwrap());
                ExitCode::SUCCESS
            }
            Ok(archive) => {
                println!("Experimental QMF v0 file (not compatibility-stable)");
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
                println!("  Container bytes: {}", bytes.len());
                println!("  Integrity: CRC-32 valid (error detection only; not authentication)");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not read QMF v0 file {}: {error}", path.display());
                ExitCode::FAILURE
            }
        },
        1 => match QmfArchiveV1::from_bytes(&bytes) {
            Ok(archive) if json => match serde_json::to_string_pretty(archive.manifest()) {
                Ok(text) => {
                    println!("{text}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("failed to serialize QMF v1 manifest: {error}");
                    ExitCode::FAILURE
                }
            },
            Ok(archive) => {
                let m = archive.manifest();
                println!("Experimental QMF v1 container ({})", m.format_status);
                println!("  Model ID: {}", m.model_id);
                println!("  Architecture: {}", m.architecture);
                println!(
                    "  Source: {} (license: {})",
                    m.source_format, m.source_license
                );
                println!(
                    "  Transformation: {} (seed {})",
                    m.transformation, m.random_seed
                );
                println!(
                    "  Container bytes: {} (payload {} bytes)",
                    bytes.len(),
                    archive.payload_bytes()
                );
                println!("  Shared latent tables: {}", m.shared_latents.len());
                println!("  Tensors: {}", m.tensors.len());
                for t in m.tensors.iter().take(20) {
                    println!("    {} {:?}", t.name, t.shape);
                }
                if m.tensors.len() > 20 {
                    println!("    ... ({} additional tensors)", m.tensors.len() - 20);
                }
                println!("  Integrity: per-blob & whole-file CRC-32 valid");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not read QMF v1 file {}: {error}", path.display());
                ExitCode::FAILURE
            }
        },
        other => {
            eprintln!("unsupported QMF version {other}");
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
            for tensor in metadata.tensors.iter().take(25) {
                println!(
                    "  tensor {}: {} {:?}, offset {}",
                    tensor.name, tensor.ggml_type, tensor.dimensions, tensor.offset
                );
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

fn run_baseline(args: Vec<String>) -> ExitCode {
    let mut json = false;
    let mut synthetic = false;
    let mut svd_rank = 16_usize;
    let mut tensor_name: Option<String> = None;
    let mut path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json = true;
                i += 1;
            }
            "--synthetic" => {
                synthetic = true;
                i += 1;
            }
            "--svd-rank" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) else {
                    eprintln!("--svd-rank requires a positive integer");
                    return ExitCode::from(2);
                };
                svd_rank = v.max(1);
                i += 2;
            }
            "--tensor" => {
                let Some(t) = args.get(i + 1) else {
                    eprintln!("--tensor requires a tensor name");
                    return ExitCode::from(2);
                };
                tensor_name = Some(t.clone());
                i += 2;
            }
            arg if arg.starts_with('-') => {
                eprintln!("unknown baseline option: {arg}");
                return ExitCode::from(2);
            }
            arg => {
                path = Some(PathBuf::from(arg));
                i += 1;
            }
        }
    }

    let (label, matrix) = if synthetic {
        let mut rng = DeterministicRng::new(2026);
        let rows = 64;
        let cols = 64;
        let mut data: Vec<f64> = (0..rows * cols).map(|_| rng.next_normal() * 0.25).collect();
        data[0] = 3.5;
        data[17] = -3.2;
        (
            "synthetic_64x64_with_outliers".to_owned(),
            DenseMatrix::from_flat(rows, cols, data).unwrap(),
        )
    } else if let Some(path) = path {
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("could not open {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        let budget = PayloadBudget::default();
        let mut st = match SafetensorsFile::open(file, &budget) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("could not parse {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        let target = if let Some(name) = tensor_name {
            name
        } else if let Some(t) = st
            .metadata()
            .tensors
            .iter()
            .find(|t| t.shape.len() == 2 && t.shape[0] > 0 && t.shape[1] > 0)
        {
            t.name.clone()
        } else {
            eprintln!("no non-empty 2D tensor found in {}", path.display());
            return ExitCode::FAILURE;
        };
        match st.read_tensor_matrix(&target, &budget) {
            Ok(m) => (target, m),
            Err(e) => {
                eprintln!("could not load tensor {target:?}: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        eprintln!(
            "usage: qml baseline [--json] [--svd-rank K] [--tensor NAME] [--synthetic | PATH]"
        );
        return ExitCode::from(2);
    };

    let results = match benchmark_conventional_baselines(&matrix, svd_rank, 42) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("baseline benchmark failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
    } else {
        println!(
            "Conventional Quantization Baselines ({label}, shape {}x{}, {} params)",
            matrix.rows(),
            matrix.cols(),
            matrix.rows() * matrix.cols()
        );
        println!(
            "{:<22} {:>9} {:>8} {:>10} {:>10} {:>8}",
            "Method", "Bytes", "Eff BPP", "Rel Frob", "MV Err", "SNR dB"
        );
        for r in &results {
            println!(
                "{:<22} {:>9} {:>8.3} {:>10.5} {:>10.5} {:>8.2}",
                r.method,
                r.total_bytes,
                r.effective_bpp,
                r.relative_frobenius_error,
                r.mean_matvec_rel_error,
                r.snr_db
            );
        }
    }
    ExitCode::SUCCESS
}

fn run_rmq_ablate(args: Vec<String>) -> ExitCode {
    let mut json = false;
    let mut rank = 6_usize;
    let mut num_layers = 8_usize;
    let mut dim = 64_usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json = true;
                i += 1;
            }
            "--rank" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                rank = v;
                i += 2;
            }
            "--layers" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                num_layers = v;
                i += 2;
            }
            "--dim" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                dim = v;
                i += 2;
            }
            other => {
                eprintln!("unknown rmq-ablate option: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let mut rng = DeterministicRng::new(2026);
    let r = rank.max(1).min(dim);
    let u = DenseMatrix::from_flat(
        dim,
        r,
        (0..dim * r)
            .map(|_| rng.next_normal() / (dim as f64).sqrt())
            .collect(),
    )
    .unwrap();
    let vt = DenseMatrix::from_flat(
        r,
        dim,
        (0..r * dim)
            .map(|_| rng.next_normal() / (dim as f64).sqrt())
            .collect(),
    )
    .unwrap();

    let mut correlated = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let core = DenseMatrix::from_flat(
            r,
            r,
            (0..r * r)
                .map(|_| rng.next_normal() * (1.0 + 0.1 * l as f64))
                .collect(),
        )
        .unwrap();
        let mut data = u
            .matmul(&core)
            .unwrap()
            .matmul(&vt)
            .unwrap()
            .as_slice()
            .to_vec();
        for v in &mut data {
            *v += rng.next_normal() * 0.008;
        }
        data[0] += 3.0;
        correlated.push(DenseMatrix::from_flat(dim, dim, data).unwrap());
    }

    let uncorrelated: Vec<DenseMatrix> = (0..num_layers)
        .map(|_| {
            DenseMatrix::from_flat(
                dim,
                dim,
                (0..dim * dim).map(|_| rng.next_normal() * 0.2).collect(),
            )
            .unwrap()
        })
        .collect();

    let corr_refs: Vec<&DenseMatrix> = correlated.iter().collect();
    let uncorr_refs: Vec<&DenseMatrix> = uncorrelated.iter().collect();
    let rep_corr = compare_rmq_vs_baselines("correlated_shared_subspace", &corr_refs, r, 42)
        .expect("correlated ablation");
    let rep_ctrl = compare_rmq_vs_baselines("uncorrelated_gaussian_control", &uncorr_refs, r, 42)
        .expect("control ablation");

    if json {
        let out = serde_json::json!({
            "correlated": rep_corr,
            "uncorrelated_control": rep_ctrl,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else {
        for rep in [&rep_corr, &rep_ctrl] {
            println!(
                "\nRMQ Ablation: {} ({} layers x {}x{}, {} params)",
                rep.cluster_label, rep.num_layers, rep.shape[0], rep.shape[1], rep.total_parameters
            );
            println!(
                "{:<28} {:>9} {:>8} {:>10} {:>10} {:>9}",
                "Method", "Bytes", "Eff BPP", "Rel Frob", "MV Err", "Enc (us)"
            );
            for row in &rep.rows {
                println!(
                    "{:<28} {:>9} {:>8.3} {:>10.5} {:>10.5} {:>9}",
                    row.method,
                    row.total_bytes_all_layers,
                    row.effective_bpp,
                    row.mean_rel_frobenius_error,
                    row.mean_matvec_rel_error,
                    row.total_encode_micros
                );
            }
        }
    }
    ExitCode::SUCCESS
}

fn run_morph(args: Vec<String>) -> ExitCode {
    let mut source: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut method = "int4".to_owned();
    let mut rank = 8_usize;
    let mut target_bpp = 5.5_f64;
    let mut sparse_pct = 1.0_f64;
    let mut seed = 42_u64;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--method" => {
                let Some(m) = args.get(i + 1) else {
                    return ExitCode::from(2);
                };
                method = m.clone();
                i += 2;
            }
            "--output" => {
                let Some(o) = args.get(i + 1) else {
                    return ExitCode::from(2);
                };
                output = Some(PathBuf::from(o));
                i += 2;
            }
            "--rank" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                rank = v;
                i += 2;
            }
            "--target-bpp" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                target_bpp = v;
                i += 2;
            }
            "--sparse-pct" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                sparse_pct = v;
                i += 2;
            }
            "--seed" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                seed = v;
                i += 2;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            arg if arg.starts_with('-') => {
                eprintln!("unknown morph option: {arg}");
                return ExitCode::from(2);
            }
            arg => {
                source = Some(PathBuf::from(arg));
                i += 1;
            }
        }
    }

    let (Some(source), Some(output)) = (source, output) else {
        eprintln!(
            "usage: qml morph SOURCE.safetensors --method <fp16|bf16|int8|int4|int4-asym|svd|rmq|map> --output OUT.qmf"
        );
        return ExitCode::from(2);
    };

    let strategy = match method.as_str() {
        "fp16" => TransformStrategy::Fp16,
        "bf16" => TransformStrategy::Bf16,
        "int8" => TransformStrategy::Int8Sym { group_size: 64 },
        "int4" => TransformStrategy::Int4Sym { group_size: 32 },
        "int4-asym" => TransformStrategy::Int4Asym { group_size: 32 },
        "svd" => TransformStrategy::LowRankSvd {
            rank,
            precision: FactorPrecision::Int8Sym,
        },
        "rmq" => TransformStrategy::RmqSharedBasis {
            rank,
            sparse_fraction: (sparse_pct / 100.0).clamp(0.0, 0.25),
            basis_precision: FactorPrecision::Int8Sym,
            core_precision: FactorPrecision::Fp16,
        },
        "rmq-core" => TransformStrategy::RmqSharedCore {
            rank,
            sparse_fraction: (sparse_pct / 100.0).clamp(0.0, 0.25),
            factor_precision: FactorPrecision::Int8Sym,
        },
        "map" => TransformStrategy::MapAdaptive {
            target_avg_bpp: target_bpp,
            propagation_weight: 0.35,
            include_svd_rank: None,
        },
        other => {
            eprintln!("unknown morph method {other:?}");
            return ExitCode::from(2);
        }
    };

    let file = match File::open(&source) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not open {}: {e}", source.display());
            return ExitCode::FAILURE;
        }
    };
    let model_id = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let budget = PayloadBudget::default();
    let (archive, report) = match forge_safetensors_to_qmf(
        file,
        model_id,
        "see-source-model-license",
        "unspecified",
        &strategy,
        &budget,
        seed,
    ) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Q-Forge transformation failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty())
        && let Err(e) = fs::create_dir_all(parent)
    {
        eprintln!("could not create output directory: {e}");
        return ExitCode::FAILURE;
    }
    let out_file = match File::create(&output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not create {}: {e}", output.display());
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = archive.write_to(&mut BufWriter::new(out_file)) {
        eprintln!("failed writing QMF v1 artifact: {e}");
        return ExitCode::FAILURE;
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("Q-Forge Transformation Complete");
        println!(
            "  Strategy: {} ({})",
            report.descriptor.name, report.descriptor.maturity
        );
        println!("  Source bytes: {}", report.source_file_bytes);
        println!("  QMF v1 bytes: {}", report.qmf_serialized_bytes);
        println!(
            "  Complete QMF effective BPP: {:.3}",
            report.effective_qmf_bpp
        );
        println!(
            "  Compression vs source: {:.2}x",
            report.compression_ratio_vs_source
        );
        println!("  Shared latent structures: {}", report.shared_latent_count);
        println!(
            "  Mean relative Frobenius error: {:.5}",
            report.mean_rel_frobenius_error
        );
        println!(
            "  Mean direct matvec relative error: {:.5}",
            report.mean_matvec_rel_error
        );
        println!("  Output: {}", output.display());
    }
    ExitCode::SUCCESS
}

fn run_compare(args: Vec<String>) -> ExitCode {
    let (json, source_path, qmf_path) = match args.as_slice() {
        [s, q] => (false, PathBuf::from(s), PathBuf::from(q)),
        [flag, s, q] if flag == "--json" => (true, PathBuf::from(s), PathBuf::from(q)),
        _ => {
            eprintln!("usage: qml compare [--json] SOURCE.safetensors TARGET.qmf");
            return ExitCode::from(2);
        }
    };
    let source_file = match File::open(&source_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not open {}: {e}", source_path.display());
            return ExitCode::FAILURE;
        }
    };
    let qmf_bytes = match fs::read(&qmf_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("could not read {}: {e}", qmf_path.display());
            return ExitCode::FAILURE;
        }
    };
    match compare_safetensors_and_qmf(source_file, &qmf_bytes, &PayloadBudget::default(), 42) {
        Ok(rep) if json => {
            println!("{}", serde_json::to_string_pretty(&rep).unwrap());
            ExitCode::SUCCESS
        }
        Ok(rep) => {
            println!("QMF v1 vs Source Safetensors Verification");
            println!("  Model: {} ({})", rep.qmf_model_id, rep.transformation);
            println!("  Tensors verified: {}", rep.tensors_compared);
            println!(
                "  Size: {} -> {} bytes (complete QMF {:.3} bpp)",
                rep.source_file_bytes, rep.qmf_file_bytes, rep.effective_qmf_bpp
            );
            println!(
                "  Relative Frobenius error: mean {:.5}, max {:.5}",
                rep.mean_rel_frobenius_error, rep.max_rel_frobenius_error
            );
            println!(
                "  Direct matvec relative error: mean {:.5}, max {:.5}",
                rep.mean_matvec_rel_error, rep.max_matvec_rel_error
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("comparison failed: {e}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Serialize)]
struct InferDemoRow {
    variant: String,
    qmf_bytes: usize,
    effective_bpp: f64,
    generated_tokens: Vec<usize>,
    tokens_per_second: f64,
    mean_kl_divergence: f64,
    mean_logit_rmse: f64,
    top1_token_agreement: f64,
    perplexity: f64,
}

fn run_infer_demo(args: Vec<String>) -> ExitCode {
    let json = match args.as_slice() {
        [] => false,
        [flag] if flag == "--json" => true,
        _ => {
            eprintln!("usage: qml infer-demo [--json]");
            return ExitCode::from(2);
        }
    };

    let cfg = TransformerConfig {
        vocab_size: 48,
        hidden_size: 32,
        intermediate_size: 64,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_seq_len: 32,
        tie_word_embeddings: false,
    };
    let model_bytes = generate_synthetic_llama_safetensors(&cfg, 4, 2026).unwrap();
    let budget = PayloadBudget::default();
    let prompt = [1, 7, 13, 21];
    let eval_seq = [1, 7, 13, 21, 5, 19, 32, 9, 41, 14];

    let (ref_archive, ref_rep) = forge_safetensors_to_qmf(
        Cursor::new(&model_bytes),
        "synthetic-llama-4L",
        "generated-test",
        "llama",
        &TransformStrategy::Fp16,
        &budget,
        42,
    )
    .unwrap();
    let ref_model = QCoreModel::from_qmf(&ref_archive, cfg.clone()).unwrap();

    let variants: Vec<(&str, TransformStrategy)> = vec![
        ("FP16 (Reference)", TransformStrategy::Fp16),
        (
            "INT8_SYM (g64)",
            TransformStrategy::Int8Sym { group_size: 64 },
        ),
        (
            "INT4_SYM (Q4_0 g32)",
            TransformStrategy::Int4Sym { group_size: 32 },
        ),
        (
            "RMQ Shared-Basis + 1% Sparse",
            TransformStrategy::RmqSharedBasis {
                rank: 4,
                sparse_fraction: 0.01,
                basis_precision: FactorPrecision::Int8Sym,
                core_precision: FactorPrecision::Fp16,
            },
        ),
        (
            "MAP Adaptive (5.5 bpp)",
            TransformStrategy::MapAdaptive {
                target_avg_bpp: 5.5,
                propagation_weight: 0.35,
                include_svd_rank: Some(4),
            },
        ),
    ];

    let mut rows = Vec::new();
    for (label, strat) in variants {
        let (arch, rep) = forge_safetensors_to_qmf(
            Cursor::new(&model_bytes),
            "synthetic-llama-4L",
            "generated-test",
            "llama",
            &strat,
            &budget,
            42,
        )
        .unwrap();
        let model = QCoreModel::from_qmf(&arch, cfg.clone()).unwrap();
        let gen_out = model.generate(&prompt, 8, 0.0, 99).unwrap();
        let fid = evaluate_inference_fidelity(&ref_model, &model, &eval_seq).unwrap();
        rows.push(InferDemoRow {
            variant: label.to_owned(),
            qmf_bytes: rep.qmf_serialized_bytes,
            effective_bpp: rep.effective_qmf_bpp,
            generated_tokens: gen_out.generated_tokens,
            tokens_per_second: gen_out.tokens_per_second,
            mean_kl_divergence: fid.mean_kl_divergence,
            mean_logit_rmse: fid.mean_logit_rmse,
            top1_token_agreement: fid.top1_token_agreement,
            perplexity: fid.transformed_perplexity,
        });
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&rows).unwrap());
    } else {
        println!(
            "Q-Core Native Direct-Compressed Inference Benchmark (Synthetic 4L Llama, {} params, source {} B)",
            ref_rep.total_parameters, ref_rep.source_file_bytes
        );
        println!(
            "{:<30} {:>8} {:>7} {:>9} {:>9} {:>7} {:>8}",
            "Variant", "QMF B", "BPP", "Mean KL", "RMSE", "Top1%", "Tok/s"
        );
        for r in &rows {
            println!(
                "{:<30} {:>8} {:>7.2} {:>9.5} {:>9.5} {:>6.1}% {:>8.0}",
                r.variant,
                r.qmf_bytes,
                r.effective_bpp,
                r.mean_kl_divergence,
                r.mean_logit_rmse,
                r.top1_token_agreement * 100.0,
                r.tokens_per_second
            );
        }
        println!(
            "  Scope: synthetic mini-Llama CPU reference; direct compressed matvec (zero dense rebuild)"
        );
    }
    ExitCode::SUCCESS
}

fn run_genesis_cmd(args: Vec<String>) -> ExitCode {
    let mut json = false;
    let mut synthetic = false;
    let mut path: Option<PathBuf> = None;
    let mut archive_out: Option<PathBuf> = None;
    let mut cfg = GenesisConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json = true;
                i += 1;
            }
            "--synthetic" => {
                synthetic = true;
                i += 1;
            }
            "--population" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                cfg.population_size = v;
                i += 2;
            }
            "--generations" => {
                let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    return ExitCode::from(2);
                };
                cfg.generations = v;
                i += 2;
            }
            "--archive" => {
                let Some(p) = args.get(i + 1) else {
                    return ExitCode::from(2);
                };
                archive_out = Some(PathBuf::from(p));
                i += 2;
            }
            arg if arg.starts_with('-') => {
                eprintln!("unknown genesis option: {arg}");
                return ExitCode::from(2);
            }
            arg => {
                path = Some(PathBuf::from(arg));
                i += 1;
            }
        }
    }

    let owned_layers: Vec<(String, DenseMatrix)> = if synthetic {
        let mut rng = DeterministicRng::new(777);
        let u = DenseMatrix::from_flat(
            32,
            4,
            (0..128)
                .map(|_| rng.next_normal() / 32.0_f64.sqrt())
                .collect(),
        )
        .unwrap();
        let vt = DenseMatrix::from_flat(
            4,
            32,
            (0..128)
                .map(|_| rng.next_normal() / 32.0_f64.sqrt())
                .collect(),
        )
        .unwrap();
        (0..4)
            .map(|l| {
                let core =
                    DenseMatrix::from_flat(4, 4, (0..16).map(|_| rng.next_normal()).collect())
                        .unwrap();
                let mut data = u
                    .matmul(&core)
                    .unwrap()
                    .matmul(&vt)
                    .unwrap()
                    .as_slice()
                    .to_vec();
                for v in &mut data {
                    *v += rng.next_normal() * 0.02;
                }
                if l == 0 {
                    data[0] += 5.0;
                }
                (
                    format!("layer.{l}.q_proj"),
                    DenseMatrix::from_flat(32, 32, data).unwrap(),
                )
            })
            .collect()
    } else if let Some(path) = path {
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("could not open {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        let budget = PayloadBudget::default();
        let mut st = match SafetensorsFile::open(file, &budget) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("could not parse {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        let names: Vec<String> = st
            .metadata()
            .tensors
            .iter()
            .filter(|t| t.shape.len() == 2 && t.shape[0] > 0 && t.shape[1] > 0)
            .take(8)
            .map(|t| t.name.clone())
            .collect();
        let mut loaded = Vec::new();
        for name in names {
            match st.read_tensor_matrix(&name, &budget) {
                Ok(m) => loaded.push((name, m)),
                Err(e) => {
                    eprintln!("failed loading {name:?}: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        loaded
    } else {
        eprintln!(
            "usage: qml genesis [--json] [--population P] [--generations G] [--archive OUT.json] [--synthetic | PATH]"
        );
        return ExitCode::from(2);
    };

    let refs: Vec<(&str, &DenseMatrix)> =
        owned_layers.iter().map(|(n, m)| (n.as_str(), m)).collect();

    let archive = match run_genesis(&refs, &cfg) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Q-Genesis failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(out_path) = archive_out {
        if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            let _ = fs::create_dir_all(parent);
        }
        if let Err(e) = fs::write(&out_path, serde_json::to_string_pretty(&archive).unwrap()) {
            eprintln!("could not write archive {}: {e}", out_path.display());
            return ExitCode::FAILURE;
        }
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&archive).unwrap());
    } else {
        println!(
            "Q-Genesis Evolutionary Discovery ({} candidates evaluated across {} generations)",
            archive.total_candidates_evaluated, archive.config.generations
        );
        println!(
            "Discovered Pareto Frontier ({} non-dominated configurations):",
            archive.pareto_frontier.len()
        );
        println!(
            "{:<5} {:<4} {:>8} {:>10} {:>10}  Genome",
            "ID", "Gen", "Eff BPP", "MV Err", "FLOPs"
        );
        for c in &archive.pareto_frontier {
            let summary: Vec<String> = c.genes.iter().map(|g| format!("{g:?}")).collect();
            println!(
                "{:<5} {:<4} {:>8.3} {:>10.5} {:>10}  {}",
                c.id,
                c.generation,
                c.fitness.effective_bpp,
                c.fitness.mean_matvec_rel_error,
                c.fitness.estimated_matvec_flops,
                summary.join(", ")
            );
        }
    }
    ExitCode::SUCCESS
}

#[derive(Serialize)]
struct GenerateOutput {
    qmf_path: String,
    prompt: String,
    formatted_prompt: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    generated_text: String,
    tokens_per_second: f64,
    elapsed_ms: f64,
    fidelity: Option<qml_runtime::InferenceComparison>,
}

fn run_generate(args: Vec<String>) -> ExitCode {
    use std::io::Write;
    let mut json = false;
    let mut qmf_path: Option<PathBuf> = None;
    let mut config_path: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut compare_qmf: Option<PathBuf> = None;
    let mut prompt = String::from("The mathematical beauty of neural networks is");
    let mut max_tokens: usize = 32;
    let mut temperature: f64 = 0.0;
    let mut use_chat = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json = true,
            "--chat" => use_chat = true,
            "--config" => {
                i += 1;
                config_path = args.get(i).map(PathBuf::from);
            }
            "--tokenizer" => {
                i += 1;
                tokenizer_path = args.get(i).map(PathBuf::from);
            }
            "--prompt" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    prompt = p.clone();
                }
            }
            "--max-tokens" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    max_tokens = v;
                }
            }
            "--temperature" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    temperature = v;
                }
            }
            "--compare-qmf" => {
                i += 1;
                compare_qmf = args.get(i).map(PathBuf::from);
            }
            s if !s.starts_with("--") && qmf_path.is_none() => qmf_path = Some(PathBuf::from(s)),
            other => {
                eprintln!("unknown option for generate: {other}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let (Some(qmf_p), Some(cfg_p), Some(tok_p)) = (qmf_path, config_path, tokenizer_path) else {
        eprintln!(
            "usage: qml generate MODEL.qmf --config config.json --tokenizer tokenizer.json [--prompt TEXT] [--max-tokens N] [--temperature T] [--chat] [--compare-qmf REF.qmf] [--json]"
        );
        return ExitCode::from(2);
    };

    let cfg_bytes = match fs::read(&cfg_p) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("could not read {}: {e}", cfg_p.display());
            return ExitCode::FAILURE;
        }
    };
    let cfg = match TransformerConfig::from_hf_config_bytes(&cfg_bytes) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("invalid config.json: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tokenizer = match BpeTokenizer::from_file(&tok_p) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("could not load tokenizer {}: {e}", tok_p.display());
            return ExitCode::FAILURE;
        }
    };
    let qmf_file = match File::open(&qmf_p) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not open {}: {e}", qmf_p.display());
            return ExitCode::FAILURE;
        }
    };
    let archive = match QmfArchiveV1::read_from(qmf_file) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("could not read QMF v1 {}: {e}", qmf_p.display());
            return ExitCode::FAILURE;
        }
    };
    let model = match QCoreModel::from_qmf(&archive, cfg.clone()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("could not initialize QCoreModel: {e}");
            return ExitCode::FAILURE;
        }
    };

    let formatted = if use_chat {
        tokenizer.format_chat_prompt(None, &prompt)
    } else {
        prompt.clone()
    };
    let prompt_u32 = tokenizer.encode(&formatted);
    let prompt_ids: Vec<usize> = prompt_u32.iter().map(|&id| id as usize).collect();
    let eos = tokenizer.eos_token_id().map(|id| id as usize);

    if !json {
        println!("Q-Core Native Compressed Inference");
        println!(
            "  Model: {} ({} layers, hidden={}, vocab={})",
            archive.manifest().model_id,
            cfg.num_layers,
            cfg.hidden_size,
            cfg.vocab_size
        );
        println!(
            "  Prompt tokens: {} | Max new tokens: {} | Temp: {:.2}",
            prompt_ids.len(),
            max_tokens,
            temperature
        );
        print!("  Output: ");
        let _ = std::io::stdout().flush();
    }

    let generation = match model.generate_with_callback(
        &prompt_ids,
        max_tokens,
        temperature,
        eos,
        42,
        |tok_id| {
            if !json {
                let piece = tokenizer.decode_token(tok_id as u32, true);
                print!("{piece}");
                let _ = std::io::stdout().flush();
            }
            true
        },
    ) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("\ngeneration failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if !json {
        println!();
    }

    let gen_u32: Vec<u32> = generation
        .generated_tokens
        .iter()
        .map(|&t| t as u32)
        .collect();
    let generated_text = tokenizer.decode(&gen_u32, true);

    let fidelity = if let Some(ref_path) = compare_qmf {
        let ref_file = match File::open(&ref_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("could not open reference QMF {}: {e}", ref_path.display());
                return ExitCode::FAILURE;
            }
        };
        let ref_archive = match QmfArchiveV1::read_from(ref_file) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("could not read reference QMF: {e}");
                return ExitCode::FAILURE;
            }
        };
        let ref_model = match QCoreModel::from_qmf(&ref_archive, cfg) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("could not init reference QCoreModel: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut eval_seq = prompt_ids.clone();
        eval_seq.extend(generation.generated_tokens.iter().copied().take(16));
        match evaluate_inference_fidelity(&ref_model, &model, &eval_seq) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("fidelity evaluation failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    let output = GenerateOutput {
        qmf_path: qmf_p.display().to_string(),
        prompt,
        formatted_prompt: formatted,
        prompt_tokens: prompt_ids.len(),
        generated_tokens: generation.generated_tokens.len(),
        generated_text,
        tokens_per_second: generation.tokens_per_second,
        elapsed_ms: generation.elapsed_micros as f64 / 1000.0,
        fidelity,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!(
            "  Speed: {:.2} tok/s ({} generated in {:.1} ms)",
            output.tokens_per_second, output.generated_tokens, output.elapsed_ms
        );
        if let Some(f) = &output.fidelity {
            println!(
                "  Fidelity vs reference: Mean KL={:.5} | Logit RMSE={:.5} | Top-1={:.1}% | PPL ref={:.3} vs tr={:.3}",
                f.mean_kl_divergence,
                f.mean_logit_rmse,
                f.top1_token_agreement * 100.0,
                f.reference_perplexity,
                f.transformed_perplexity
            );
        }
    }
    ExitCode::SUCCESS
}
