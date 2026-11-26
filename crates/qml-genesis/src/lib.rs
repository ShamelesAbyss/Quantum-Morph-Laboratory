//! Q-Genesis — Multi-Objective Evolutionary Representation Discovery Engine.
//!
//! Searches over heterogeneous per-layer and cross-layer neural representation
//! genomes (FP16, BF16, INT8, INT4-Q4_0, INT4-Q4_1, Truncated SVD, and RMQ
//! Shared-Basis clusters) using real measured fitness across competing objectives:
//! - Complete effective bits per parameter (every byte counted)
//! - Weighted matrix-vector multiplication relative error
//! - Estimated compressed matvec FLOPs
//!
//! Maintains a persistent archive preserving BOTH Pareto-optimal frontiers and
//! dominated/unsuccessful candidates for full scientific reproducibility.

use qml_core::{DenseMatrix, DeterministicRng, MatrixError};
use qml_format::{QmfArchiveV1, QmfV1Builder};
use qml_model::{PayloadBudget, SafetensorsFile, SafetensorsMetadata};
use qml_quant::{
    Bf16Matrix, EchoBaseOperator, EchoCorrectedMatrix, FactorPrecision, Fp16Matrix, HadamardSpec,
    Int2AsymGroupMatrix, Int3AsymGroupMatrix, Int4AsymGroupMatrix, Int4SymGroupMatrix,
    Int8SymMatrix, LowRankSvdMatrix, RmqResidualKind, RmqSharedBasisCluster, RmqSharedBasisConfig,
    TernaryPackedMatrix, deterministic_probe_vectors, evaluate_operator,
};
use qml_scope::{LayerRole, classify_tensor_name};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek};

/// Evolvable representation choice for one layer inside a Q-Genesis genome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum LayerGene {
    Fp16,
    Bf16,
    Int8Sym {
        group_size: usize,
    },
    Int4Sym {
        group_size: usize,
    },
    Int4Asym {
        group_size: usize,
    },
    /// Sub-3-bit: 3-bit asymmetric + Hadamard incoherence (~3.25-3.50 bpp).
    Int3Hadamard {
        group_size: usize,
    },
    /// Sub-3-bit: 2-bit clipped asymmetric + Hadamard incoherence (~2.25-2.50 bpp).
    Int2Hadamard {
        group_size: usize,
    },
    /// Sub-3-bit: Packed base-3 ternary (5 trits/byte) + Hadamard (~1.725-1.85 bpp).
    TernaryHadamard {
        group_size: usize,
    },
    /// PROJECT ECHO: 2-bit Hadamard base + compact INT8 low-rank recovery adapter.
    Int2Echo {
        group_size: usize,
        echo_rank: usize,
    },
    LowRankSvd {
        rank: usize,
    },
    /// Layer opts into a cross-layer RMQ shared-basis cluster with `(rank, sparse_permille)`.
    RmqSharedBasis {
        rank: usize,
        sparse_permille: u16,
    },
}

impl LayerGene {
    fn random(rng: &mut DeterministicRng, max_rank: usize) -> Self {
        let r_choices = [
            2.min(max_rank).max(1),
            4.min(max_rank).max(1),
            6.min(max_rank).max(1),
            8.min(max_rank).max(1),
        ];
        match rng.next_usize(11) {
            0 => Self::Fp16,
            1 => Self::Bf16,
            2 => Self::Int8Sym {
                group_size: if rng.next_usize(2) == 0 { 32 } else { 64 },
            },
            3 => Self::Int4Sym {
                group_size: match rng.next_usize(3) {
                    0 => 32,
                    1 => 64,
                    _ => 128,
                },
            },
            4 => Self::Int4Asym {
                group_size: if rng.next_usize(2) == 0 { 32 } else { 64 },
            },
            5 => Self::Int3Hadamard {
                group_size: if rng.next_usize(2) == 0 { 64 } else { 128 },
            },
            6 => Self::Int2Hadamard {
                group_size: if rng.next_usize(2) == 0 { 64 } else { 128 },
            },
            7 => Self::TernaryHadamard {
                group_size: if rng.next_usize(2) == 0 { 64 } else { 128 },
            },
            8 => Self::Int2Echo {
                group_size: 128,
                echo_rank: r_choices[rng.next_usize(r_choices.len())],
            },
            9 => Self::LowRankSvd {
                rank: r_choices[rng.next_usize(r_choices.len())],
            },
            _ => Self::RmqSharedBasis {
                rank: r_choices[rng.next_usize(r_choices.len())],
                sparse_permille: match rng.next_usize(3) {
                    0 => 0,
                    1 => 5,
                    _ => 10,
                },
            },
        }
    }
}

/// Measured multi-objective fitness for an evaluated genome.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitnessVector {
    /// Complete effective bits per original parameter across all layers.
    pub effective_bpp: f64,
    /// Mean matrix-vector multiplication relative error across layers and probes.
    pub mean_matvec_rel_error: f64,
    /// Estimated total multiply-add FLOPs per forward matvec pass across all layers.
    pub estimated_matvec_flops: u64,
    pub total_bytes: usize,
}

impl FitnessVector {
    /// Returns `true` if `self` Pareto-dominates `other` (better or equal on all objectives,
    /// strictly better on at least one).
    pub fn dominates(&self, other: &Self) -> bool {
        let no_worse = self.effective_bpp <= other.effective_bpp
            && self.mean_matvec_rel_error <= other.mean_matvec_rel_error
            && self.estimated_matvec_flops <= other.estimated_matvec_flops;
        let strictly_better = self.effective_bpp < other.effective_bpp - 1e-9
            || self.mean_matvec_rel_error < other.mean_matvec_rel_error - 1e-9
            || self.estimated_matvec_flops < other.estimated_matvec_flops;
        no_worse && strictly_better
    }
}

/// Complete record of an evaluated candidate genome in the persistent archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub id: usize,
    pub generation: usize,
    pub parent_ids: Vec<usize>,
    pub genes: Vec<LayerGene>,
    pub fitness: FitnessVector,
    pub on_pareto_frontier: bool,
}

/// Configuration for a deterministic Q-Genesis evolutionary search campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenesisConfig {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub max_rank: usize,
    pub seed: u64,
}

impl Default for GenesisConfig {
    fn default() -> Self {
        Self {
            population_size: 12,
            generations: 6,
            mutation_rate: 0.25,
            max_rank: 8,
            seed: 0x4745_4e45_5349_5301,
        }
    }
}

/// User-selectable preference profile for choosing a winner from the Q-Genesis Pareto frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenesisProfile {
    #[default]
    Balanced,
    Smallest,
    BestQuality,
    Fastest,
}

impl GenesisProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Balanced => "Balanced (Normalized Knee)",
            Self::Smallest => "Smallest (Minimum BPP)",
            Self::BestQuality => "Best Quality (Minimum Matvec Error)",
            Self::Fastest => "Fastest (Minimum FLOPs)",
        }
    }
}

/// Persistent experimental archive returned by Q-Genesis (includes Pareto winners & dominated trials).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenesisArchive {
    pub config: GenesisConfig,
    pub layer_names: Vec<String>,
    pub total_parameters: usize,
    pub total_candidates_evaluated: usize,
    pub pareto_frontier: Vec<CandidateRecord>,
    pub all_candidates: Vec<CandidateRecord>,
}

impl GenesisArchive {
    /// Select the best Pareto-frontier candidate according to a [`GenesisProfile`].
    pub fn select_best(&self, profile: GenesisProfile) -> Option<&CandidateRecord> {
        let pool = if self.pareto_frontier.is_empty() {
            &self.all_candidates
        } else {
            &self.pareto_frontier
        };
        if pool.is_empty() {
            return None;
        }
        match profile {
            GenesisProfile::Smallest => pool
                .iter()
                .min_by(|a, b| a.fitness.effective_bpp.total_cmp(&b.fitness.effective_bpp)),
            GenesisProfile::BestQuality => pool.iter().min_by(|a, b| {
                a.fitness
                    .mean_matvec_rel_error
                    .total_cmp(&b.fitness.mean_matvec_rel_error)
            }),
            GenesisProfile::Fastest => pool.iter().min_by_key(|c| c.fitness.estimated_matvec_flops),
            GenesisProfile::Balanced => {
                let min_bpp = pool
                    .iter()
                    .map(|c| c.fitness.effective_bpp)
                    .fold(f64::INFINITY, f64::min);
                let max_bpp = pool
                    .iter()
                    .map(|c| c.fitness.effective_bpp)
                    .fold(f64::NEG_INFINITY, f64::max);
                let min_err = pool
                    .iter()
                    .map(|c| c.fitness.mean_matvec_rel_error)
                    .fold(f64::INFINITY, f64::min);
                let max_err = pool
                    .iter()
                    .map(|c| c.fitness.mean_matvec_rel_error)
                    .fold(f64::NEG_INFINITY, f64::max);
                let bpp_span = (max_bpp - min_bpp).max(1e-9);
                let err_span = (max_err - min_err).max(1e-9);
                pool.iter().min_by(|a, b| {
                    let da = ((a.fitness.effective_bpp - min_bpp) / bpp_span).powi(2)
                        + ((a.fitness.mean_matvec_rel_error - min_err) / err_span).powi(2);
                    let db = ((b.fitness.effective_bpp - min_bpp) / bpp_span).powi(2)
                        + ((b.fitness.mean_matvec_rel_error - min_err) / err_span).powi(2);
                    da.total_cmp(&db)
                })
            }
        }
    }
}

/// Select up to `max_matrices` 2D projection tensors stratified evenly across transformer
/// layer roles (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
/// and network depth (early / middle / late layers), excluding oversized embedding/head tables.
pub fn select_stratified_tensor_names(
    metadata: &SafetensorsMetadata,
    max_matrices: usize,
) -> Vec<String> {
    if max_matrices == 0 {
        return Vec::new();
    }
    let mut by_role: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
    for t in &metadata.tensors {
        if t.shape.len() != 2 || t.shape[0] == 0 || t.shape[1] == 0 {
            continue;
        }
        let (layer_idx, role) = classify_tensor_name(&t.name);
        if matches!(role, LayerRole::Embedding | LayerRole::LmHead) {
            continue;
        }
        by_role
            .entry(role.as_str().to_owned())
            .or_default()
            .push((layer_idx.unwrap_or(usize::MAX), t.name.clone()));
    }
    for list in by_role.values_mut() {
        list.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    }

    let roles: Vec<String> = by_role.keys().cloned().collect();
    if roles.is_empty() {
        return metadata
            .tensors
            .iter()
            .filter(|t| t.shape.len() == 2 && t.shape[0] > 0 && t.shape[1] > 0)
            .take(max_matrices)
            .map(|t| t.name.clone())
            .collect();
    }

    // Round-robin across roles, picking evenly spaced depth indices (early -> late -> mid).
    let mut selected = Vec::new();
    let per_role_quota = max_matrices.div_ceil(roles.len());
    for role in &roles {
        let list = &by_role[role];
        let take = per_role_quota.min(list.len());
        if take == 1 {
            selected.push(list[list.len() / 2].1.clone());
        } else {
            for k in 0..take {
                let idx = (k * (list.len() - 1)) / (take - 1);
                let name = &list[idx].1;
                if !selected.contains(name) {
                    selected.push(name.clone());
                }
            }
        }
        if selected.len() >= max_matrices {
            break;
        }
    }
    selected.truncate(max_matrices);
    selected
}

/// Materialize a discovered Q-Genesis genome over an immutable Safetensors model into a
/// complete, verifiable **QMF v1** archive.
///
/// Tensors explicitly present in `evolved_layer_names` receive their exact evolved [`LayerGene`].
/// Other projection tensors inherit the evolved gene of their matching [`LayerRole`] (or `Int4Sym`),
/// embeddings/`lm_head` are protected in `Int8Sym`, and 1D normalization vectors remain `DenseF32`.
#[allow(clippy::too_many_arguments)]
pub fn export_genome_to_qmf_v1<R: Read + Seek>(
    reader: R,
    model_id: &str,
    license: &str,
    source_format: &str,
    evolved_layer_names: &[String],
    genes: &[LayerGene],
    budget: &PayloadBudget,
    seed: u64,
) -> Result<QmfArchiveV1, String> {
    if evolved_layer_names.len() != genes.len() {
        return Err("evolved layer names and gene count mismatch".to_owned());
    }
    let mut file = SafetensorsFile::open(reader, budget).map_err(|e| e.to_string())?;
    let tensor_infos = file.metadata().tensors.clone();

    let mut exact_genes: HashMap<String, LayerGene> = HashMap::new();
    let mut role_genes: HashMap<LayerRole, LayerGene> = HashMap::new();
    for (name, &gene) in evolved_layer_names.iter().zip(genes) {
        exact_genes.insert(name.clone(), gene);
        let (_, role) = classify_tensor_name(name);
        role_genes.entry(role).or_insert(gene);
    }

    let mut builder = QmfV1Builder::new(
        model_id,
        license,
        source_format,
        "q-genesis",
        "q_genesis_genome",
        seed,
    );

    // Group RMQ-selected tensors by (role, rows, cols, rank, sparse_permille).
    type RmqKey = (LayerRole, usize, usize, usize, u16);
    let mut rmq_groups: BTreeMap<RmqKey, Vec<String>> = BTreeMap::new();
    let mut single_tensors: Vec<(String, LayerGene)> = Vec::new();

    for info in &tensor_infos {
        if info.shape.len() == 1 {
            let vals = file
                .read_tensor_f64(&info.name, budget)
                .map_err(|e| e.to_string())?;
            let f32_vals: Vec<f32> = vals.into_iter().map(|v| v as f32).collect();
            builder
                .add_dense_f32(&info.name, vec![info.shape[0] as usize], &f32_vals)
                .map_err(|e| e.to_string())?;
            continue;
        }
        if info.shape.len() != 2 {
            continue;
        }
        let (_, role) = classify_tensor_name(&info.name);
        let gene = if matches!(role, LayerRole::Embedding | LayerRole::LmHead) {
            LayerGene::Int8Sym { group_size: 64 }
        } else {
            exact_genes
                .get(&info.name)
                .copied()
                .or_else(|| role_genes.get(&role).copied())
                .unwrap_or(LayerGene::Int4Sym { group_size: 32 })
        };
        match gene {
            LayerGene::RmqSharedBasis {
                rank,
                sparse_permille,
            } => {
                rmq_groups
                    .entry((
                        role,
                        info.shape[0] as usize,
                        info.shape[1] as usize,
                        rank,
                        sparse_permille,
                    ))
                    .or_default()
                    .push(info.name.clone());
            }
            other => single_tensors.push((info.name.clone(), other)),
        }
    }

    for (idx, (name, gene)) in single_tensors.iter().enumerate() {
        let mat = file
            .read_tensor_matrix(name, budget)
            .map_err(|e| e.to_string())?;
        match *gene {
            LayerGene::Fp16 => {
                let op = Fp16Matrix::quantize(&mat);
                builder.add_fp16(name, &op).map_err(|e| e.to_string())?;
            }
            LayerGene::Bf16 => {
                let op = Bf16Matrix::quantize(&mat);
                builder.add_bf16(name, &op).map_err(|e| e.to_string())?;
            }
            LayerGene::Int8Sym { group_size } => {
                let op = Int8SymMatrix::quantize(&mat, group_size).map_err(|e| e.to_string())?;
                builder.add_int8_sym(name, &op).map_err(|e| e.to_string())?;
            }
            LayerGene::Int4Sym { group_size } => {
                let op =
                    Int4SymGroupMatrix::quantize(&mat, group_size).map_err(|e| e.to_string())?;
                builder.add_int4_sym(name, &op).map_err(|e| e.to_string())?;
            }
            LayerGene::Int4Asym { group_size } => {
                let op =
                    Int4AsymGroupMatrix::quantize(&mat, group_size).map_err(|e| e.to_string())?;
                builder
                    .add_int4_asym(name, &op)
                    .map_err(|e| e.to_string())?;
            }
            LayerGene::Int3Hadamard { group_size } => {
                let had = HadamardSpec::new(64, seed ^ (idx as u64).wrapping_mul(0x9e37_79b9));
                let op = Int3AsymGroupMatrix::quantize(&mat, group_size, had)
                    .map_err(|e| e.to_string())?;
                builder
                    .add_int3_asym(name, &op)
                    .map_err(|e| e.to_string())?;
            }
            LayerGene::Int2Hadamard { group_size } => {
                let had = HadamardSpec::new(64, seed ^ (idx as u64).wrapping_mul(0x9e37_79b9));
                let op = Int2AsymGroupMatrix::quantize(&mat, group_size, had)
                    .map_err(|e| e.to_string())?;
                builder
                    .add_int2_asym(name, &op)
                    .map_err(|e| e.to_string())?;
            }
            LayerGene::TernaryHadamard { group_size } => {
                let had = HadamardSpec::new(64, seed ^ (idx as u64).wrapping_mul(0x9e37_79b9));
                let op = TernaryPackedMatrix::quantize(&mat, group_size, had)
                    .map_err(|e| e.to_string())?;
                builder
                    .add_ternary_packed(name, &op)
                    .map_err(|e| e.to_string())?;
            }
            LayerGene::Int2Echo {
                group_size,
                echo_rank,
            } => {
                let had = HadamardSpec::new(64, seed ^ (idx as u64).wrapping_mul(0x9e37_79b9));
                let base = Int2AsymGroupMatrix::quantize(&mat, group_size, had)
                    .map_err(|e| e.to_string())?;
                let op = EchoCorrectedMatrix::fit(
                    &mat,
                    EchoBaseOperator::Int2Asym(base),
                    echo_rank.min(mat.rows().min(mat.cols())),
                    0.0,
                    seed + idx as u64,
                )
                .map_err(|e| e.to_string())?;
                builder
                    .add_echo_corrected(name, &op)
                    .map_err(|e| e.to_string())?;
            }
            LayerGene::LowRankSvd { rank } => {
                let r = rank.max(1).min(mat.rows().min(mat.cols()));
                let op = LowRankSvdMatrix::from_matrix(
                    &mat,
                    r,
                    FactorPrecision::Int8Sym,
                    seed + idx as u64,
                )
                .map_err(|e| e.to_string())?;
                builder
                    .add_low_rank_svd(name, &op)
                    .map_err(|e| e.to_string())?;
            }
            LayerGene::RmqSharedBasis { .. } => unreachable!("handled in rmq_groups"),
        }
    }

    for ((role, rows, cols, rank, sparse_permille), names) in rmq_groups {
        let cluster_mats: Vec<DenseMatrix> = names
            .iter()
            .map(|n| file.read_tensor_matrix(n, budget))
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        if cluster_mats.len() >= 2 {
            let refs: Vec<&DenseMatrix> = cluster_mats.iter().collect();
            let r = rank.max(1).min(rows.min(cols));
            let frac = f64::from(sparse_permille) / 1000.0;
            let residual_kind = if frac > 0.0 {
                RmqResidualKind::SparseTopFraction { fraction: frac }
            } else {
                RmqResidualKind::None
            };
            let cluster = RmqSharedBasisCluster::fit(
                &refs,
                RmqSharedBasisConfig {
                    left_rank: r,
                    right_rank: r,
                    hooi_iters: 4,
                    basis_precision: FactorPrecision::Int8Sym,
                    core_precision: FactorPrecision::Fp16,
                    residual_kind,
                    seed,
                },
            )
            .map_err(|e| e.to_string())?;
            let cid = format!("genesis_{}_{rows}x{cols}_r{r}", role.as_str());
            builder
                .add_rmq_shared_basis_cluster(&cid, &names, &cluster)
                .map_err(|e| e.to_string())?;
        } else {
            let op =
                Int4SymGroupMatrix::quantize(&cluster_mats[0], 32).map_err(|e| e.to_string())?;
            builder
                .add_int4_sym(&names[0], &op)
                .map_err(|e| e.to_string())?;
        }
    }

    builder.build().map_err(|e| e.to_string())
}

/// Run deterministic multi-objective evolutionary search over a sequence of layer matrices.
pub fn run_genesis(
    layers: &[(&str, &DenseMatrix)],
    config: &GenesisConfig,
) -> Result<GenesisArchive, MatrixError> {
    if layers.is_empty() {
        return Err(MatrixError::Empty);
    }
    let n_layers = layers.len();
    let total_params: usize = layers.iter().map(|(_, m)| m.rows() * m.cols()).sum();
    let min_dim = layers
        .iter()
        .map(|(_, m)| m.rows().min(m.cols()))
        .min()
        .unwrap_or(1);
    let max_rank = config.max_rank.max(1).min(min_dim);

    // Precompute deterministic probes per column dimension.
    let probes_per_layer: Vec<Vec<Vec<f64>>> = layers
        .iter()
        .enumerate()
        .map(|(idx, (_, m))| deterministic_probe_vectors(m.cols(), 4, config.seed + idx as u64))
        .collect();

    // Cache single-layer gene evaluations (layer_idx, gene) -> (bytes, mv_err, flops)
    // so evolutionary search evaluates hundreds of combinations rapidly!
    let mut single_cache: BTreeMap<(usize, String), (usize, f64, u64)> = BTreeMap::new();

    let mut rng = DeterministicRng::new(config.seed);
    let mut next_id = 0_usize;
    let mut all_candidates: Vec<CandidateRecord> = Vec::new();

    let pop_size = config.population_size.max(4);

    // Generation 0: seed with interpretable uniform baselines (including sub-3-bit Resonance tiers) + random genomes.
    let mut initial_genomes: Vec<Vec<LayerGene>> = vec![
        vec![LayerGene::Int8Sym { group_size: 64 }; n_layers],
        vec![LayerGene::Int4Sym { group_size: 32 }; n_layers],
        vec![LayerGene::Int3Hadamard { group_size: 128 }; n_layers],
        vec![LayerGene::Int2Hadamard { group_size: 128 }; n_layers],
        vec![LayerGene::TernaryHadamard { group_size: 128 }; n_layers],
        vec![
            LayerGene::RmqSharedBasis {
                rank: 4.min(max_rank),
                sparse_permille: 10,
            };
            n_layers
        ],
    ];
    while initial_genomes.len() < pop_size {
        initial_genomes.push(
            (0..n_layers)
                .map(|_| LayerGene::random(&mut rng, max_rank))
                .collect(),
        );
    }

    let mut population: Vec<CandidateRecord> = Vec::new();
    for genes in initial_genomes.into_iter().take(pop_size) {
        let fitness = evaluate_genome(
            layers,
            &genes,
            &probes_per_layer,
            &mut single_cache,
            total_params,
            config.seed,
        )?;
        let rec = CandidateRecord {
            id: next_id,
            generation: 0,
            parent_ids: Vec::new(),
            genes,
            fitness,
            on_pareto_frontier: false,
        };
        next_id += 1;
        all_candidates.push(rec.clone());
        population.push(rec);
    }

    for generation_idx in 1..=config.generations {
        mark_pareto(&mut population);
        let mut elites: Vec<CandidateRecord> = population
            .iter()
            .filter(|c| c.on_pareto_frontier)
            .cloned()
            .collect();
        // Keep at most half the population as elites (evenly spaced across BPP) so every
        // generation actively breeds new crossover and mutated offspring.
        let max_elites = (pop_size / 2).max(2);
        if elites.len() > max_elites {
            elites.sort_by(|a, b| a.fitness.effective_bpp.total_cmp(&b.fitness.effective_bpp));
            let mut selected = Vec::with_capacity(max_elites);
            let step = (elites.len() - 1) as f64 / (max_elites - 1) as f64;
            for k in 0..max_elites {
                let idx = ((k as f64) * step).round() as usize;
                selected.push(elites[idx.min(elites.len() - 1)].clone());
            }
            elites = selected;
        }

        let mut next_pop = elites;
        while next_pop.len() < pop_size {
            let p1 = tournament_select(&population, &mut rng);
            let p2 = tournament_select(&population, &mut rng);
            let mut child_genes = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                let mut gene = if rng.next_usize(2) == 0 {
                    p1.genes[l]
                } else {
                    p2.genes[l]
                };
                if rng.next_f64() < config.mutation_rate {
                    gene = LayerGene::random(&mut rng, max_rank);
                }
                child_genes.push(gene);
            }
            let fitness = evaluate_genome(
                layers,
                &child_genes,
                &probes_per_layer,
                &mut single_cache,
                total_params,
                config.seed,
            )?;
            let rec = CandidateRecord {
                id: next_id,
                generation: generation_idx,
                parent_ids: vec![p1.id, p2.id],
                genes: child_genes,
                fitness,
                on_pareto_frontier: false,
            };
            next_id += 1;
            all_candidates.push(rec.clone());
            next_pop.push(rec);
        }
        population = next_pop;
    }

    // Mark global Pareto frontier across ALL evaluated candidates.
    mark_pareto(&mut all_candidates);
    let mut pareto_frontier: Vec<CandidateRecord> = all_candidates
        .iter()
        .filter(|c| c.on_pareto_frontier)
        .cloned()
        .collect();
    pareto_frontier.sort_by(|a, b| a.fitness.effective_bpp.total_cmp(&b.fitness.effective_bpp));

    Ok(GenesisArchive {
        config: config.clone(),
        layer_names: layers.iter().map(|(n, _)| (*n).to_owned()).collect(),
        total_parameters: total_params,
        total_candidates_evaluated: all_candidates.len(),
        pareto_frontier,
        all_candidates,
    })
}

fn evaluate_genome(
    layers: &[(&str, &DenseMatrix)],
    genes: &[LayerGene],
    probes_per_layer: &[Vec<Vec<f64>>],
    single_cache: &mut BTreeMap<(usize, String), (usize, f64, u64)>,
    total_params: usize,
    seed: u64,
) -> Result<FitnessVector, MatrixError> {
    let mut total_bytes = 0_usize;
    let mut sum_mv_err = 0.0_f64;
    let mut total_flops = 0_u64;

    // Group any compatible RMQ genes by (rows, cols, rank, sparse_permille).
    type RmqKey = (usize, usize, usize, u16);
    let mut rmq_groups: BTreeMap<RmqKey, Vec<usize>> = BTreeMap::new();

    for (idx, gene) in genes.iter().enumerate() {
        let mat = layers[idx].1;
        if let LayerGene::RmqSharedBasis {
            rank,
            sparse_permille,
        } = *gene
        {
            rmq_groups
                .entry((mat.rows(), mat.cols(), rank, sparse_permille))
                .or_default()
                .push(idx);
            continue;
        }

        let cache_key = (idx, format!("{gene:?}"));
        let (bytes, mv_err, flops) = if let Some(&cached) = single_cache.get(&cache_key) {
            cached
        } else {
            let res = eval_single_layer(mat, *gene, &probes_per_layer[idx], seed + idx as u64)?;
            single_cache.insert(cache_key, res);
            res
        };
        total_bytes += bytes;
        sum_mv_err += mv_err;
        total_flops += flops;
    }

    for ((rows, cols, rank, sparse_permille), indices) in rmq_groups {
        let r = rank.max(1).min(rows.min(cols));
        let refs: Vec<&DenseMatrix> = indices.iter().map(|&i| layers[i].1).collect();
        let residual_kind = if sparse_permille > 0 {
            RmqResidualKind::SparseTopFraction {
                fraction: f64::from(sparse_permille) / 1000.0,
            }
        } else {
            RmqResidualKind::None
        };
        let cluster = RmqSharedBasisCluster::fit(
            &refs,
            RmqSharedBasisConfig {
                left_rank: r,
                right_rank: r,
                hooi_iters: 1,
                basis_precision: FactorPrecision::Int8Sym,
                core_precision: FactorPrecision::Fp16,
                residual_kind,
                seed,
            },
        )?;
        total_bytes += cluster.total_cluster_bytes();
        let nnz_per_layer = ((rows * cols) as f64 * (f64::from(sparse_permille) / 1000.0)) as u64;
        for (local_i, &layer_idx) in indices.iter().enumerate() {
            let op = cluster.layer(local_i).unwrap();
            let ev = evaluate_operator(layers[layer_idx].1, &op, &probes_per_layer[layer_idx], 0)?;
            sum_mv_err += ev.mean_matvec_rel_error;
            // FLOPs for U(C(Vᵀ x)) + sparse residual: 2*(r*cols + r*r + rows*r + nnz)
            total_flops += 2 * ((r * cols + r * r + rows * r) as u64 + nnz_per_layer);
        }
    }

    Ok(FitnessVector {
        effective_bpp: (total_bytes as f64 * 8.0) / total_params.max(1) as f64,
        mean_matvec_rel_error: sum_mv_err / layers.len() as f64,
        estimated_matvec_flops: total_flops,
        total_bytes,
    })
}

fn eval_single_layer(
    mat: &DenseMatrix,
    gene: LayerGene,
    probes: &[Vec<f64>],
    seed: u64,
) -> Result<(usize, f64, u64), MatrixError> {
    let dense_flops = 2 * (mat.rows() * mat.cols()) as u64;
    match gene {
        LayerGene::Fp16 => {
            let op = Fp16Matrix::quantize(mat);
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, dense_flops))
        }
        LayerGene::Bf16 => {
            let op = Bf16Matrix::quantize(mat);
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, dense_flops))
        }
        LayerGene::Int8Sym { group_size } => {
            let op = Int8SymMatrix::quantize(mat, group_size)?;
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, dense_flops))
        }
        LayerGene::Int4Sym { group_size } => {
            let op = Int4SymGroupMatrix::quantize(mat, group_size)?;
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, dense_flops))
        }
        LayerGene::Int4Asym { group_size } => {
            let op = Int4AsymGroupMatrix::quantize(mat, group_size)?;
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, dense_flops))
        }
        LayerGene::Int3Hadamard { group_size } => {
            let had = HadamardSpec::new(64, seed);
            let op = Int3AsymGroupMatrix::quantize(mat, group_size, had)?;
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, dense_flops))
        }
        LayerGene::Int2Hadamard { group_size } => {
            let had = HadamardSpec::new(64, seed);
            let op = Int2AsymGroupMatrix::quantize(mat, group_size, had)?;
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, dense_flops))
        }
        LayerGene::TernaryHadamard { group_size } => {
            let had = HadamardSpec::new(64, seed);
            let op = TernaryPackedMatrix::quantize(mat, group_size, had)?;
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, dense_flops))
        }
        LayerGene::Int2Echo {
            group_size,
            echo_rank,
        } => {
            let had = HadamardSpec::new(64, seed);
            let base = Int2AsymGroupMatrix::quantize(mat, group_size, had)?;
            let r = echo_rank.max(1).min(mat.rows().min(mat.cols()));
            let op = EchoCorrectedMatrix::fit(mat, EchoBaseOperator::Int2Asym(base), r, 0.0, seed)?;
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            let flops = dense_flops + 2 * (r * (mat.rows() + mat.cols())) as u64;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, flops))
        }
        LayerGene::LowRankSvd { rank } => {
            let r = rank.max(1).min(mat.rows().min(mat.cols()));
            let op = LowRankSvdMatrix::from_matrix(mat, r, FactorPrecision::Int8Sym, seed)?;
            let ev = evaluate_operator(mat, &op, probes, 0)?;
            let flops = 2 * (r * (mat.rows() + mat.cols())) as u64;
            Ok((ev.total_bytes, ev.mean_matvec_rel_error, flops))
        }
        LayerGene::RmqSharedBasis { .. } => unreachable!("handled in RMQ cluster pass"),
    }
}

fn mark_pareto(candidates: &mut [CandidateRecord]) {
    let n = candidates.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        for j in 0..n {
            if i != j && candidates[j].fitness.dominates(&candidates[i].fitness) {
                dominated[i] = true;
                break;
            }
        }
    }
    for (cand, is_dom) in candidates.iter_mut().zip(dominated) {
        cand.on_pareto_frontier = !is_dom;
    }
}

fn tournament_select<'a>(
    pop: &'a [CandidateRecord],
    rng: &mut DeterministicRng,
) -> &'a CandidateRecord {
    let a = &pop[rng.next_usize(pop.len())];
    let b = &pop[rng.next_usize(pop.len())];
    if a.on_pareto_frontier && !b.on_pareto_frontier {
        return a;
    }
    if b.on_pareto_frontier && !a.on_pareto_frontier {
        return b;
    }
    if a.fitness.dominates(&b.fitness) {
        a
    } else if b.fitness.dominates(&a.fitness) {
        b
    } else if rng.next_usize(2) == 0 {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_evolves_deterministic_pareto_frontier_and_preserves_archive() {
        let mut rng = DeterministicRng::new(555);
        // Build 4 correlated low-rank matrices + 1 outlier matrix so hybrid genomes shine!
        let u = DenseMatrix::from_flat(24, 3, (0..72).map(|_| rng.next_normal() * 0.4).collect())
            .unwrap();
        let vt = DenseMatrix::from_flat(3, 24, (0..72).map(|_| rng.next_normal() * 0.4).collect())
            .unwrap();
        let mut mats = Vec::new();
        for l in 0..4 {
            let core =
                DenseMatrix::from_flat(3, 3, (0..9).map(|_| rng.next_normal()).collect()).unwrap();
            let mut data = u
                .matmul(&core)
                .unwrap()
                .matmul(&vt)
                .unwrap()
                .as_slice()
                .to_vec();
            if l == 0 {
                data[0] += 8.0;
            }
            mats.push(DenseMatrix::from_flat(24, 24, data).unwrap());
        }
        let layers = [
            ("l0", &mats[0]),
            ("l1", &mats[1]),
            ("l2", &mats[2]),
            ("l3", &mats[3]),
        ];

        let cfg = GenesisConfig {
            population_size: 8,
            generations: 3,
            mutation_rate: 0.3,
            max_rank: 6,
            seed: 42,
        };
        let run1 = run_genesis(&layers, &cfg).unwrap();
        let run2 = run_genesis(&layers, &cfg).unwrap();

        assert_eq!(run1, run2, "Q-Genesis must be strictly deterministic");
        assert!(!run1.pareto_frontier.is_empty());
        assert!(run1.total_candidates_evaluated > run1.pareto_frontier.len());
        // Verify no Pareto frontier member dominates another Pareto frontier member.
        for i in 0..run1.pareto_frontier.len() {
            for j in 0..run1.pareto_frontier.len() {
                if i != j {
                    assert!(
                        !run1.pareto_frontier[i]
                            .fitness
                            .dominates(&run1.pareto_frontier[j].fitness)
                    );
                }
            }
        }

        // Verify all 4 GenesisProfile selectors return valid Pareto winners.
        for profile in [
            GenesisProfile::Balanced,
            GenesisProfile::Smallest,
            GenesisProfile::BestQuality,
            GenesisProfile::Fastest,
        ] {
            let best = run1.select_best(profile).unwrap();
            assert!(best.on_pareto_frontier);
        }
    }
}
