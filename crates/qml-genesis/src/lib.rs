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
use qml_quant::{
    Bf16Matrix, FactorPrecision, Fp16Matrix, Int4AsymGroupMatrix, Int4SymGroupMatrix,
    Int8SymMatrix, LowRankSvdMatrix, RmqResidualKind, RmqSharedBasisCluster, RmqSharedBasisConfig,
    deterministic_probe_vectors, evaluate_operator,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
        match rng.next_usize(7) {
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
            5 => Self::LowRankSvd {
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

    // Generation 0: seed with interpretable uniform baselines + diverse random genomes.
    let mut initial_genomes: Vec<Vec<LayerGene>> = vec![
        vec![LayerGene::Fp16; n_layers],
        vec![LayerGene::Int8Sym { group_size: 64 }; n_layers],
        vec![LayerGene::Int4Sym { group_size: 32 }; n_layers],
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
        let elites: Vec<CandidateRecord> = population
            .iter()
            .filter(|c| c.on_pareto_frontier)
            .cloned()
            .collect();

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
    }
}
