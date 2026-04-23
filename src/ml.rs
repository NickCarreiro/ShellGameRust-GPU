use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::tree::NodeSnapshot;

// ──────────────────────────────────────────────────────────────────────────────
// Global candle device — initialised once, shared across all Rayon threads.
// ──────────────────────────────────────────────────────────────────────────────

static CANDLE_DEVICE: OnceLock<Device> = OnceLock::new();

fn candle_device() -> &'static Device {
    CANDLE_DEVICE.get_or_init(|| {
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(0) {
                Ok(d) => {
                    eprintln!("[skoll/hati] CUDA device 0 ready — GPU batch inference active.");
                    d
                }
                Err(e) => {
                    panic!("[skoll/hati] CUDA init failed ({e}). GPU required by default — use --no-default-features only for an intentional CPU comparison.");
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("[skoll/hati] CUDA feature disabled — using Candle CPU backend. Default builds enable CUDA; CPU runs require --no-default-features.");
            Device::Cpu
        }
    })
}

fn accelerator_description() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "CUDA enabled: batched MLP inference runs on GPU; tree simulation/evolution still runs on CPU"
    }

    #[cfg(not(feature = "cuda"))]
    {
        "CUDA disabled: batched MLP inference is running on CPU"
    }
}

/// Returns true only when the training/inference backend is actually CUDA.
///
/// This intentionally goes through the same global Candle device used by the
/// batched training paths, so tests catch accidental CPU fallbacks.
pub fn training_accelerator_is_cuda() -> bool {
    candle_device().is_cuda()
}

// Feature counts
/// Searcher features: 10 original + 4 strategic features (fraction_guessed,
/// subtree_unguessed_norm, is_pivot_norm, unguessed_sibling_norm).
pub const SEARCHER_FEATURE_COUNT: usize = 14;
/// Evader features: 10 original + subtree_size_norm + cold_subtree_flag + recent_same_depth_norm.
pub const EVADER_FEATURE_COUNT: usize = 14;

// MLP architecture — two hidden layers for both roles.
pub const EVADER_MLP_HIDDEN1: usize = 1024;
pub const EVADER_MLP_HIDDEN2: usize = 512;
pub const SEARCHER_MLP_HIDDEN1: usize = 1024;
pub const SEARCHER_MLP_HIDDEN2: usize = 512;

// Legacy aliases kept so any remaining references compile.
#[allow(dead_code)]
pub const EVADER_MLP_HIDDEN: usize = EVADER_MLP_HIDDEN1;
#[allow(dead_code)]
pub const SEARCHER_MLP_HIDDEN: usize = SEARCHER_MLP_HIDDEN1;

// Episode-memory tuning.
const RECENT_MEMORY: usize = 6;

// ──────────────────────────────────────────────────────────────────────────────
// Searcher: LinearPolicyModel (unchanged from original)
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearPolicyModel {
    pub role: String,
    pub feature_names: Vec<String>,
    pub bias: f64,
    pub weights: Vec<f64>,
}

impl LinearPolicyModel {
    pub fn new_random(role: &str, rng: &mut StdRng) -> Self {
        let mut weights = Vec::with_capacity(SEARCHER_FEATURE_COUNT);
        for _ in 0..SEARCHER_FEATURE_COUNT {
            weights.push(rng.gen_range(-0.5..0.5));
        }

        Self {
            role: role.to_string(),
            feature_names: searcher_feature_names(),
            bias: rng.gen_range(-0.25..0.25),
            weights,
        }
    }

    pub fn score(&self, features: &[f64]) -> f64 {
        self.bias
            + self
                .weights
                .iter()
                .zip(features.iter())
                .map(|(weight, feature)| weight * feature)
                .sum::<f64>()
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(self).map_err(|err| err.to_string())?;
        fs::write(path, serialized).map_err(|err| err.to_string())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
        serde_json::from_str(&text).map_err(|err| err.to_string())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Evader: MlpPolicyModel  (10+2 inputs → 16 hidden ReLU → 1 output)
// ──────────────────────────────────────────────────────────────────────────────

/// Three-layer MLP: input_dim → hidden1 ReLU → hidden2 ReLU → 1 output.
/// Weight layout (row-major):
///   w1: [hidden1 × input_dim]
///   b1: [hidden1]
///   w2: [hidden2 × hidden1]
///   b2: [hidden2]
///   w3: [hidden2]   (output weights)
///   b3: scalar      (output bias)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpPolicyModel {
    pub role: String,
    pub feature_names: Vec<String>,
    pub input_dim: usize,
    pub hidden1_dim: usize,
    pub hidden2_dim: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub w3: Vec<f64>,
    pub b3: f64,
}

impl MlpPolicyModel {
    pub fn new_random(role: &str, rng: &mut StdRng) -> Self {
        let input_dim = EVADER_FEATURE_COUNT;
        let hidden1_dim = EVADER_MLP_HIDDEN1;
        let hidden2_dim = EVADER_MLP_HIDDEN2;
        // He initialisation scale for ReLU: sqrt(2 / fan_in)
        let scale_w1 = (2.0_f64 / input_dim as f64).sqrt();
        let scale_w2 = (2.0_f64 / hidden1_dim as f64).sqrt();
        let scale_w3 = (2.0_f64 / hidden2_dim as f64).sqrt();

        let w1: Vec<f64> = (0..hidden1_dim * input_dim)
            .map(|_| rng.gen_range(-scale_w1..scale_w1))
            .collect();
        let b1: Vec<f64> = vec![0.0; hidden1_dim];
        let w2: Vec<f64> = (0..hidden2_dim * hidden1_dim)
            .map(|_| rng.gen_range(-scale_w2..scale_w2))
            .collect();
        let b2: Vec<f64> = vec![0.0; hidden2_dim];
        let w3: Vec<f64> = (0..hidden2_dim)
            .map(|_| rng.gen_range(-scale_w3..scale_w3))
            .collect();
        let b3 = 0.0;

        Self {
            role: role.to_string(),
            feature_names: evader_feature_names(),
            input_dim,
            hidden1_dim,
            hidden2_dim,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        }
    }

    pub fn score(&self, features: &[f64]) -> f64 {
        let input_dim = self.input_dim;
        let h1_dim = self.hidden1_dim;
        let h2_dim = self.hidden2_dim;

        // Layer 1: h1 = ReLU(W1 * x + b1)
        let mut h1 = vec![0.0f64; h1_dim];
        for h in 0..h1_dim {
            let mut val = self.b1[h];
            let row = h * input_dim;
            for i in 0..input_dim.min(features.len()) {
                val += self.w1[row + i] * features[i];
            }
            h1[h] = val.max(0.0);
        }

        // Layer 2: h2 = ReLU(W2 * h1 + b2)
        let mut h2 = vec![0.0f64; h2_dim];
        for h in 0..h2_dim {
            let mut val = self.b2[h];
            let row = h * h1_dim;
            for i in 0..h1_dim {
                val += self.w2[row + i] * h1[i];
            }
            h2[h] = val.max(0.0);
        }

        // Output: y = W3 · h2 + b3
        let mut out = self.b3;
        for h in 0..h2_dim {
            out += self.w3[h] * h2[h];
        }
        out
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(self).map_err(|err| err.to_string())?;
        fs::write(path, serialized).map_err(|err| err.to_string())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
        serde_json::from_str(&text).map_err(|err| err.to_string())
    }

    /// Score all nodes in one GPU batch.
    /// `feature_rows` is a flat row-major [n_nodes × input_dim] slice.
    pub fn batch_score_nodes(&self, feature_rows: &[f64]) -> Vec<f64> {
        let input_dim = self.input_dim;
        let n_nodes = feature_rows.len() / input_dim.max(1);
        if n_nodes == 0 {
            return Vec::new();
        }

        let device = candle_device();

        let xf: Vec<f32> = feature_rows.iter().map(|&v| v as f32).collect();
        let x  = Tensor::from_slice(&xf,                                          (n_nodes, input_dim),         device).expect("GPU tensor alloc");
        let w1 = Tensor::from_slice(&self.w1.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden1_dim, input_dim),         device).expect("GPU tensor alloc");
        let b1 = Tensor::from_slice(&self.b1.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden1_dim,),                   device).expect("GPU tensor alloc");
        let h1 = x.matmul(&w1.t().expect("t")).expect("matmul").broadcast_add(&b1).expect("add").relu().expect("relu");

        let w2 = Tensor::from_slice(&self.w2.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden2_dim, self.hidden1_dim), device).expect("GPU tensor alloc");
        let b2 = Tensor::from_slice(&self.b2.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden2_dim,),                  device).expect("GPU tensor alloc");
        let h2 = h1.matmul(&w2.t().expect("t")).expect("matmul").broadcast_add(&b2).expect("add").relu().expect("relu");

        let w3 = Tensor::from_slice(&self.w3.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden2_dim, 1),               device).expect("GPU tensor alloc");
        let b3 = Tensor::from_slice(&[self.b3 as f32],                            (1,),                         device).expect("GPU tensor alloc");
        let out = h2.matmul(&w3).expect("matmul").broadcast_add(&b3).expect("add").squeeze(1).expect("squeeze");

        out.to_vec1::<f32>().expect("GPU readback").into_iter().map(|v| v as f64).collect()
    }

    /// Mutate: add uniform noise to every parameter.
    pub fn mutate(&self, scale: f64, rng: &mut StdRng) -> Self {
        let mut out = self.clone();
        for w in &mut out.w1 { *w += rng.gen_range(-scale..scale); }
        for b in &mut out.b1 { *b += rng.gen_range(-scale..scale); }
        for w in &mut out.w2 { *w += rng.gen_range(-scale..scale); }
        for b in &mut out.b2 { *b += rng.gen_range(-scale..scale); }
        for w in &mut out.w3 { *w += rng.gen_range(-scale..scale); }
        out.b3 += rng.gen_range(-scale..scale);
        out
    }

    /// Average a slice of MLP models (elite aggregation).
    pub fn aggregate(role: &str, elites: &[MlpPolicyModel]) -> MlpPolicyModel {
        let n = elites.len().max(1) as f64;
        let first = &elites[0];
        let mut result = first.clone();
        result.role = role.to_string();

        for elite in elites.iter().skip(1) {
            result.b3 += elite.b3;
            for (a, b) in result.w1.iter_mut().zip(elite.w1.iter()) { *a += b; }
            for (a, b) in result.b1.iter_mut().zip(elite.b1.iter()) { *a += b; }
            for (a, b) in result.w2.iter_mut().zip(elite.w2.iter()) { *a += b; }
            for (a, b) in result.b2.iter_mut().zip(elite.b2.iter()) { *a += b; }
            for (a, b) in result.w3.iter_mut().zip(elite.w3.iter()) { *a += b; }
        }

        result.b3 /= n;
        for w in &mut result.w1 { *w /= n; }
        for b in &mut result.b1 { *b /= n; }
        for w in &mut result.w2 { *w /= n; }
        for b in &mut result.b2 { *b /= n; }
        for w in &mut result.w3 { *w /= n; }
        result
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SearcherMlpModel  (SEARCHER_FEATURE_COUNT inputs → 32 hidden ReLU → 1 output)
// ──────────────────────────────────────────────────────────────────────────────

/// Three-layer MLP for the searcher role: input_dim → hidden1 ReLU → hidden2 ReLU → 1 output.
/// Layout mirrors MlpPolicyModel but uses SEARCHER_* constants and searcher feature names.
///   w1: [hidden1 × input_dim]
///   b1: [hidden1]
///   w2: [hidden2 × hidden1]
///   b2: [hidden2]
///   w3: [hidden2]   (output weights)
///   b3: scalar      (output bias)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearcherMlpModel {
    pub role: String,
    pub feature_names: Vec<String>,
    pub input_dim: usize,
    pub hidden1_dim: usize,
    pub hidden2_dim: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub w3: Vec<f64>,
    pub b3: f64,
}

impl SearcherMlpModel {
    pub fn new_random(role: &str, rng: &mut StdRng) -> Self {
        let input_dim = SEARCHER_FEATURE_COUNT;
        let hidden1_dim = SEARCHER_MLP_HIDDEN1;
        let hidden2_dim = SEARCHER_MLP_HIDDEN2;
        let scale_w1 = (2.0_f64 / input_dim as f64).sqrt();
        let scale_w2 = (2.0_f64 / hidden1_dim as f64).sqrt();
        let scale_w3 = (2.0_f64 / hidden2_dim as f64).sqrt();

        let w1: Vec<f64> = (0..hidden1_dim * input_dim)
            .map(|_| rng.gen_range(-scale_w1..scale_w1))
            .collect();
        let b1: Vec<f64> = vec![0.0; hidden1_dim];
        let w2: Vec<f64> = (0..hidden2_dim * hidden1_dim)
            .map(|_| rng.gen_range(-scale_w2..scale_w2))
            .collect();
        let b2: Vec<f64> = vec![0.0; hidden2_dim];
        let w3: Vec<f64> = (0..hidden2_dim)
            .map(|_| rng.gen_range(-scale_w3..scale_w3))
            .collect();
        let b3 = 0.0;

        Self {
            role: role.to_string(),
            feature_names: searcher_feature_names(),
            input_dim,
            hidden1_dim,
            hidden2_dim,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        }
    }

    pub fn score(&self, features: &[f64]) -> f64 {
        let input_dim = self.input_dim;
        let h1_dim = self.hidden1_dim;
        let h2_dim = self.hidden2_dim;

        // Layer 1: h1 = ReLU(W1 * x + b1)
        let mut h1 = vec![0.0f64; h1_dim];
        for h in 0..h1_dim {
            let mut val = self.b1[h];
            let row = h * input_dim;
            for i in 0..input_dim.min(features.len()) {
                val += self.w1[row + i] * features[i];
            }
            h1[h] = val.max(0.0);
        }

        // Layer 2: h2 = ReLU(W2 * h1 + b2)
        let mut h2 = vec![0.0f64; h2_dim];
        for h in 0..h2_dim {
            let mut val = self.b2[h];
            let row = h * h1_dim;
            for i in 0..h1_dim {
                val += self.w2[row + i] * h1[i];
            }
            h2[h] = val.max(0.0);
        }

        // Output: y = W3 · h2 + b3
        let mut out = self.b3;
        for h in 0..h2_dim {
            out += self.w3[h] * h2[h];
        }
        out
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(self).map_err(|err| err.to_string())?;
        fs::write(path, serialized).map_err(|err| err.to_string())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
        serde_json::from_str(&text).map_err(|err| err.to_string())
    }

    /// Score all nodes in one GPU batch.
    /// `feature_rows` is a flat row-major [n_nodes × input_dim] slice.
    pub fn batch_score_nodes(&self, feature_rows: &[f64]) -> Vec<f64> {
        let input_dim = self.input_dim;
        let n_nodes = feature_rows.len() / input_dim.max(1);
        if n_nodes == 0 {
            return Vec::new();
        }

        let device = candle_device();

        let xf: Vec<f32> = feature_rows.iter().map(|&v| v as f32).collect();
        let x  = Tensor::from_slice(&xf,                                          (n_nodes, input_dim),          device).expect("GPU tensor alloc");
        let w1 = Tensor::from_slice(&self.w1.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden1_dim, input_dim),          device).expect("GPU tensor alloc");
        let b1 = Tensor::from_slice(&self.b1.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden1_dim,),                    device).expect("GPU tensor alloc");
        let h1 = x.matmul(&w1.t().expect("t")).expect("matmul").broadcast_add(&b1).expect("add").relu().expect("relu");

        let w2 = Tensor::from_slice(&self.w2.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden2_dim, self.hidden1_dim),  device).expect("GPU tensor alloc");
        let b2 = Tensor::from_slice(&self.b2.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden2_dim,),                   device).expect("GPU tensor alloc");
        let h2 = h1.matmul(&w2.t().expect("t")).expect("matmul").broadcast_add(&b2).expect("add").relu().expect("relu");

        let w3 = Tensor::from_slice(&self.w3.iter().map(|&v| v as f32).collect::<Vec<_>>(), (self.hidden2_dim, 1),                device).expect("GPU tensor alloc");
        let b3 = Tensor::from_slice(&[self.b3 as f32],                            (1,),                          device).expect("GPU tensor alloc");
        let out = h2.matmul(&w3).expect("matmul").broadcast_add(&b3).expect("add").squeeze(1).expect("squeeze");

        out.to_vec1::<f32>().expect("GPU readback").into_iter().map(|v| v as f64).collect()
    }

    /// Mutate: add uniform noise to every parameter.
    pub fn mutate(&self, scale: f64, rng: &mut StdRng) -> Self {
        let mut out = self.clone();
        for w in &mut out.w1 { *w += rng.gen_range(-scale..scale); }
        for b in &mut out.b1 { *b += rng.gen_range(-scale..scale); }
        for w in &mut out.w2 { *w += rng.gen_range(-scale..scale); }
        for b in &mut out.b2 { *b += rng.gen_range(-scale..scale); }
        for w in &mut out.w3 { *w += rng.gen_range(-scale..scale); }
        out.b3 += rng.gen_range(-scale..scale);
        out
    }

    /// Average a slice of SearcherMlpModel models (elite aggregation).
    pub fn aggregate(role: &str, elites: &[SearcherMlpModel]) -> SearcherMlpModel {
        let n = elites.len().max(1) as f64;
        let first = &elites[0];
        let mut result = first.clone();
        result.role = role.to_string();

        for elite in elites.iter().skip(1) {
            result.b3 += elite.b3;
            for (a, b) in result.w1.iter_mut().zip(elite.w1.iter()) { *a += b; }
            for (a, b) in result.b1.iter_mut().zip(elite.b1.iter()) { *a += b; }
            for (a, b) in result.w2.iter_mut().zip(elite.w2.iter()) { *a += b; }
            for (a, b) in result.b2.iter_mut().zip(elite.b2.iter()) { *a += b; }
            for (a, b) in result.w3.iter_mut().zip(elite.w3.iter()) { *a += b; }
        }

        result.b3 /= n;
        for w in &mut result.w1 { *w /= n; }
        for b in &mut result.b1 { *b /= n; }
        for w in &mut result.w2 { *w /= n; }
        for b in &mut result.b2 { *b /= n; }
        for w in &mut result.w3 { *w /= n; }
        result
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public model bundle
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfPlayModels {
    pub evader: MlpPolicyModel,
    pub searcher: SearcherMlpModel,
}

// ──────────────────────────────────────────────────────────────────────────────
// Training configuration and summaries
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub generations: usize,
    pub population_size: usize,
    pub episodes_per_eval: usize,
    pub min_nodes: i32,
    pub max_nodes: i32,
    pub max_attempts_factor: usize,
    pub max_attempts_ratio: Option<f64>,
    pub max_attempts_cap: Option<usize>,
    pub mutation_scale: f64,
    pub seed: u64,
    pub output_dir: PathBuf,
    pub resume_from: Option<PathBuf>,
    pub hall_of_fame_size: usize,
    /// How many hall-of-fame opponents to sample each generation (keeps cost bounded).
    pub hall_sample_count: usize,
    /// In static mode, how many fixed algorithm opponents to sample per generation.
    pub static_opponent_sample_count: usize,
    pub training_mode: TrainingMode,
    /// Adam learning rate for the OpenAI-ES gradient update (applied after each generation).
    pub es_lr: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingMode {
    CoAgent,
    Static,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub generations: usize,
    pub final_searcher_score: f64,
    pub final_evader_score: f64,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub episodes: usize,
    pub found_rate: f64,
    pub average_attempts: f64,
    pub average_searcher_reward: f64,
    pub average_evader_reward: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerationRecord {
    generation: usize,
    searcher_score: f64,
    evader_score: f64,
    found_rate: f64,
    avg_attempts: f64,
    /// CoAgent mode only: evader score when evaluated against the fixed static searchers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    static_evader_score: Option<f64>,
    /// CoAgent mode only: found rate when evaluated against the fixed static searchers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    static_found_rate: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GenerationMode {
    Balanced,
    Uneven,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EpisodeSpec {
    pub node_count: i32,
    pub generation_mode: GenerationMode,
    pub seed: u64,
}

#[derive(Debug, Clone)]
struct NodeMeta {
    key: i32,
    depth: usize,
    path_bits: u64,
    path_len: u8,
    child_count: usize,
    encounter_index: usize,
    /// Number of nodes in this node's subtree (including itself).
    subtree_size: usize,
}

#[derive(Debug, Clone)]
struct TrainingNode {
    key: i32,
    left: Option<usize>,
    right: Option<usize>,
}

#[derive(Debug, Clone)]
struct FastTrainingTree {
    nodes: Vec<TrainingNode>,
    root: Option<usize>,
    rng: StdRng,
    cached_meta: Option<Vec<NodeMeta>>,
}

impl FastTrainingTree {
    fn new(spec: EpisodeSpec) -> Self {
        let mut tree = Self {
            nodes: Vec::new(),
            root: None,
            rng: StdRng::seed_from_u64(spec.seed),
            cached_meta: None,
        };

        match spec.generation_mode {
            GenerationMode::Balanced => {
                for key in 1..=spec.node_count.max(1) {
                    tree.insert_level_order(key);
                }
            }
            GenerationMode::Uneven => tree.build_uneven(spec.node_count.max(1)),
        }

        tree
    }

    fn make_node(&mut self, key: i32) -> usize {
        let index = self.nodes.len();
        self.nodes.push(TrainingNode {
            key,
            left: None,
            right: None,
        });
        index
    }

    fn mark_dirty(&mut self) {
        self.cached_meta = None;
    }

    fn insert_level_order(&mut self, key: i32) {
        self.mark_dirty();
        if self.root.is_none() {
            self.root = Some(self.make_node(key));
            return;
        }

        let mut queue = vec![self.root.expect("root exists")];
        let mut cursor = 0usize;
        while cursor < queue.len() {
            let node_idx = queue[cursor];
            cursor += 1;

            if self.nodes[node_idx].left.is_none() {
                let child = self.make_node(key);
                self.nodes[node_idx].left = Some(child);
                return;
            }
            if self.nodes[node_idx].right.is_none() {
                let child = self.make_node(key);
                self.nodes[node_idx].right = Some(child);
                return;
            }
            queue.push(self.nodes[node_idx].left.expect("left exists"));
            queue.push(self.nodes[node_idx].right.expect("right exists"));
        }
    }

    fn build_uneven(&mut self, node_count: i32) {
        let mut next_key = 1;

        fn opposite(side: &str) -> &'static str {
            if side == "left" { "right" } else { "left" }
        }

        fn attach_subtree(
            tree: &mut FastTrainingTree,
            parent: Option<usize>,
            side: Option<&str>,
            size: i32,
            bias: &str,
            depth: i32,
            next_key: &mut i32,
        ) -> Option<usize> {
            if size <= 0 {
                return None;
            }

            let node_idx = tree.make_node(*next_key);
            *next_key += 1;

            if let Some(parent_idx) = parent {
                if side == Some("left") {
                    tree.nodes[parent_idx].left = Some(node_idx);
                } else {
                    tree.nodes[parent_idx].right = Some(node_idx);
                }
            }

            let remaining = size - 1;
            if remaining <= 0 {
                return Some(node_idx);
            }

            if remaining == 1 {
                let _ = attach_subtree(tree, Some(node_idx), Some(bias), 1, opposite(bias), depth + 1, next_key);
                return Some(node_idx);
            }

            let heavy_ratio = if depth % 2 != 0 { 0.68 } else { 0.58 };
            let mut heavy_size = (remaining as f32 * heavy_ratio).round() as i32;
            heavy_size = heavy_size.max(1).min(remaining - 1);
            let light_size = remaining - heavy_size;

            let mut heavy_side = bias;
            if depth % 3 == 2 {
                heavy_side = opposite(heavy_side);
            }
            let light_side = opposite(heavy_side);
            let next_bias = if depth % 2 != 0 { opposite(bias) } else { bias };

            let _ = attach_subtree(tree, Some(node_idx), Some(heavy_side), heavy_size, next_bias, depth + 1, next_key);
            if light_size > 0 {
                let _ = attach_subtree(
                    tree,
                    Some(node_idx),
                    Some(light_side),
                    light_size,
                    opposite(next_bias),
                    depth + 1,
                    next_key,
                );
            }

            Some(node_idx)
        }

        self.mark_dirty();
        self.root = attach_subtree(self, None, None, node_count, "left", 0, &mut next_key);
    }

    #[allow(dead_code)]
    fn node_keys(&self) -> Vec<i32> {
        self.node_meta().iter().map(|node| node.key).collect()
    }

    fn swap_children(&mut self, idx: usize) {
        let left = self.nodes[idx].left;
        let right = self.nodes[idx].right;
        self.nodes[idx].left = right;
        self.nodes[idx].right = left;
    }

    fn shuffle_subtree(&mut self, idx: Option<usize>) {
        let Some(node_idx) = idx else {
            return;
        };

        if self.rng.gen::<f64>() >= 0.5 {
            self.swap_children(node_idx);
        }

        let left = self.nodes[node_idx].left;
        let right = self.nodes[node_idx].right;
        self.shuffle_subtree(left);
        self.shuffle_subtree(right);
    }

    fn node_meta(&self) -> &Vec<NodeMeta> {
        self.cached_meta.as_ref().expect("node metadata cache should be present")
    }

    fn ensure_meta(&mut self) {
        if self.cached_meta.is_some() {
            return;
        }

        let mut ordered = Vec::new();

        // First pass: collect keys and tree structure (DFS order).
        fn walk_collect(
            tree: &FastTrainingTree,
            idx: Option<usize>,
            depth: usize,
            path_bits: u64,
            path_len: u8,
            ordered: &mut Vec<NodeMeta>,
        ) -> usize {
            let Some(node_idx) = idx else {
                return 0;
            };

            let node = &tree.nodes[node_idx];
            let encounter_index = ordered.len();
            let child_count = node.left.is_some() as usize + node.right.is_some() as usize;
            ordered.push(NodeMeta {
                key: node.key,
                depth,
                path_bits,
                path_len,
                child_count,
                encounter_index,
                subtree_size: 0, // filled in second pass
            });

            let left_size = walk_collect(tree, node.left, depth + 1, path_bits << 1, path_len + 1, ordered);
            let right_size = walk_collect(tree, node.right, depth + 1, (path_bits << 1) | 1, path_len + 1, ordered);
            let total = 1 + left_size + right_size;

            // Back-fill subtree_size for the node we just pushed.
            ordered[encounter_index].subtree_size = total;
            total
        }

        walk_collect(self, self.root, 0, 1, 1, &mut ordered);
        self.cached_meta = Some(ordered);
    }

    /// Advance one episode step: shuffle the tree in-place once for the vectorized runner.
    fn shuffle_step(&mut self) {
        let root = self.root;
        self.shuffle_subtree(root);
        self.mark_dirty();
    }

    fn meta_snapshot(&mut self) -> Vec<NodeMeta> {
        self.ensure_meta();
        self.node_meta().clone()
    }

}

// ──────────────────────────────────────────────────────────────────────────────
// Feature construction
// ──────────────────────────────────────────────────────────────────────────────

pub fn searcher_feature_names() -> Vec<String> {
    vec![
        // Original 10 node-centric features
        "depth_norm".to_string(),
        "leaf_flag".to_string(),
        "child_count_norm".to_string(),
        "key_norm".to_string(),
        "encounter_index_norm".to_string(),
        "path_len_norm".to_string(),
        "min_recent_distance_norm".to_string(),
        "avg_recent_distance_norm".to_string(),
        "recent_guess_flag".to_string(),
        "last_guess_distance_norm".to_string(),
        // 4 search-strategy features
        "fraction_guessed".to_string(),
        "subtree_unguessed_norm".to_string(),
        "is_pivot_norm".to_string(),
        "unguessed_sibling_norm".to_string(),
    ]
}

fn evader_feature_names() -> Vec<String> {
    vec![
        "depth_norm".to_string(),
        "leaf_flag".to_string(),
        "child_count_norm".to_string(),
        "key_norm".to_string(),
        "encounter_index_norm".to_string(),
        "path_len_norm".to_string(),
        "min_recent_distance_norm".to_string(),
        "avg_recent_distance_norm".to_string(),
        "recent_guess_flag".to_string(),
        "last_guess_distance_norm".to_string(),
        // Evader-only features:
        "subtree_size_norm".to_string(),
        "cold_subtree_flag".to_string(),
        "recent_same_depth_norm".to_string(),
        // Anti-EvasionAware: 1 if this node is ≤1 hop from a recent guess (warm zone).
        "warm_zone_flag".to_string(),
    ]
}

fn path_distance(a_bits: u64, a_len: u8, b_bits: u64, b_len: u8) -> usize {
    let min_len = a_len.min(b_len) as usize;
    let mut common = 0usize;
    for shift in 0..min_len {
        let a_bit = (a_bits >> ((a_len as usize - 1) - shift)) & 1;
        let b_bit = (b_bits >> ((b_len as usize - 1) - shift)) & 1;
        if a_bit == b_bit {
            common += 1;
        } else {
            break;
        }
    }
    (a_len as usize - common) + (b_len as usize - common)
}

/// Returns true if any recently-guessed key is an ancestor of or inside the subtree rooted
/// at `node`. We detect "inside the subtree" via the path-bit prefix: a node B is in A's
/// subtree iff A's path is a prefix of B's path.
fn any_recent_in_subtree(node: &NodeMeta, recent_guesses: &[i32], all_nodes: &[NodeMeta]) -> bool {
    // Build the prefix mask for node's path.
    // node.path_len includes the sentinel leading 1 bit; the actual depth is path_len - 1.
    // A node R is in the subtree of N iff:
    //   R.path_len >= N.path_len  AND
    //   the top N.path_len bits of R.path_bits == N.path_bits (after aligning).
    for &guess_key in recent_guesses {
        if let Some(r) = all_nodes.iter().find(|m| m.key == guess_key) {
            if r.path_len >= node.path_len {
                // Shift r's bits right by (r.path_len - node.path_len) to align.
                let shift = (r.path_len - node.path_len) as usize;
                let r_prefix = r.path_bits >> shift;
                if r_prefix == node.path_bits {
                    return true;
                }
            }
        }
    }
    false
}

fn build_searcher_features(node: &NodeMeta, all_nodes: &[NodeMeta], recent_guesses: &[i32]) -> [f64; SEARCHER_FEATURE_COUNT] {
    let node_count = all_nodes.len().max(1) as f64;
    let max_key = all_nodes.iter().map(|item| item.key).max().unwrap_or(1).max(1) as f64;
    let recent_nodes: Vec<&NodeMeta> = recent_guesses
        .iter()
        .filter_map(|guess| all_nodes.iter().find(|item| item.key == *guess))
        .collect();

    let min_recent_distance = if recent_nodes.is_empty() {
        node.depth + 1
    } else {
        recent_nodes
            .iter()
            .map(|recent| path_distance(node.path_bits, node.path_len, recent.path_bits, recent.path_len))
            .min()
            .unwrap_or(0)
    };
    let avg_recent_distance = if recent_nodes.is_empty() {
        node.depth as f64 + 1.0
    } else {
        recent_nodes
            .iter()
            .map(|recent| path_distance(node.path_bits, node.path_len, recent.path_bits, recent.path_len) as f64)
            .sum::<f64>()
            / recent_nodes.len() as f64
    };
    let last_guess_distance = recent_nodes
        .last()
        .map(|recent| path_distance(node.path_bits, node.path_len, recent.path_bits, recent.path_len) as f64)
        .unwrap_or(node.depth as f64 + 1.0);

    // ── 4 search-strategy features ──────────────────────────────────────
    //
    // Feature 10: fraction_guessed
    //   What proportion of all tree nodes have been guessed so far?
    //   Range [0, 1]. Gives the searcher a "how far along?" signal so it can
    //   modulate urgency (e.g. be more aggressive at covering leaves when
    //   budget is running out).
    let fraction_guessed = (recent_guesses.len() as f64 / node_count).min(1.0);

    // Feature 11: subtree_unguessed_norm
    //   Number of not-yet-guessed nodes inside this node's subtree,
    //   normalised by tree size.  A high value means "lots of unexplored
    //   territory under this node" — guessing here covers maximum ground.
    //   This is the key strategic signal for systematic search.
    let subtree_start = node.encounter_index;
    let subtree_end   = subtree_start + node.subtree_size;
    let subtree_unguessed = all_nodes
        .iter()
        .filter(|n| n.encounter_index >= subtree_start && n.encounter_index < subtree_end && !recent_guesses.contains(&n.key))
        .count();
    let subtree_unguessed_norm = subtree_unguessed as f64 / node_count;

    // Feature 12: is_pivot_norm
    //   A "pivot" node divides the remaining unguessed tree into approximately
    //   equal halves — the binary-search ideal.  We measure this as
    //   1 - |0.5 - (unguessed_in_subtree / total_unguessed)| * 2,
    //   so 1.0 means perfect half-split and 0.0 means this node is a leaf
    //   with respect to unguessed nodes.
    let total_unguessed = all_nodes.iter().filter(|n| !recent_guesses.contains(&n.key)).count();
    let is_pivot_norm = if total_unguessed == 0 {
        0.0
    } else {
        let ratio = subtree_unguessed as f64 / total_unguessed as f64;
        (1.0 - (0.5 - ratio).abs() * 2.0).max(0.0)
    };

    // Feature 13: unguessed_sibling_norm
    //   Fraction of unguessed nodes *outside* this subtree. Complements
    //   `subtree_unguessed_norm` to tell the model whether this node balances the
    //   remaining search space. 0 when this subtree contains all remaining options.
    let unguessed_sibling = total_unguessed.saturating_sub(subtree_unguessed);
    let unguessed_sibling_norm = unguessed_sibling as f64 / node_count;

    [
        node.depth as f64 / node_count.max(1.0),
        if node.child_count == 0 { 1.0 } else { 0.0 },
        node.child_count as f64 / 2.0,
        node.key as f64 / max_key,
        node.encounter_index as f64 / node_count.max(1.0),
        node.path_len as f64 / (node_count + 1.0),
        min_recent_distance as f64 / (node_count + 1.0),
        avg_recent_distance / (node_count + 1.0),
        if recent_guesses.contains(&node.key) { 1.0 } else { 0.0 },
        last_guess_distance / (node_count + 1.0),
        fraction_guessed,
        subtree_unguessed_norm,
        is_pivot_norm,
        unguessed_sibling_norm,
    ]
}

fn build_evader_features(node: &NodeMeta, all_nodes: &[NodeMeta], recent_guesses: &[i32]) -> [f64; EVADER_FEATURE_COUNT] {
    let node_count = all_nodes.len().max(1) as f64;
    let max_key = all_nodes.iter().map(|item| item.key).max().unwrap_or(1).max(1) as f64;
    let recent_nodes: Vec<&NodeMeta> = recent_guesses
        .iter()
        .filter_map(|guess| all_nodes.iter().find(|item| item.key == *guess))
        .collect();

    let min_recent_distance = if recent_nodes.is_empty() {
        node.depth + 1
    } else {
        recent_nodes
            .iter()
            .map(|recent| path_distance(node.path_bits, node.path_len, recent.path_bits, recent.path_len))
            .min()
            .unwrap_or(0)
    };
    let avg_recent_distance = if recent_nodes.is_empty() {
        node.depth as f64 + 1.0
    } else {
        recent_nodes
            .iter()
            .map(|recent| path_distance(node.path_bits, node.path_len, recent.path_bits, recent.path_len) as f64)
            .sum::<f64>()
            / recent_nodes.len() as f64
    };
    let last_guess_distance = recent_nodes
        .last()
        .map(|recent| path_distance(node.path_bits, node.path_len, recent.path_bits, recent.path_len) as f64)
        .unwrap_or(node.depth as f64 + 1.0);

    // New feature 1: subtree_size_norm — what fraction of the tree lives under this node?
    let subtree_size_norm = node.subtree_size as f64 / node_count.max(1.0);

    // New feature 2: cold_subtree_flag — 1 if NO recent guess falls inside this subtree.
    let cold_subtree_flag = if recent_guesses.is_empty() || !any_recent_in_subtree(node, recent_guesses, all_nodes) {
        1.0
    } else {
        0.0
    };

    // New feature 3: recent_same_depth_norm — fraction of recent guesses at the same depth as this node.
    let same_depth = recent_guesses
        .iter()
        .filter_map(|guess| all_nodes.iter().find(|item| item.key == *guess))
        .filter(|m| m.depth == node.depth)
        .count();
    let recent_same_depth_norm = if recent_guesses.is_empty() {
        0.0
    } else {
        same_depth as f64 / recent_guesses.len().max(1) as f64
    };

    // warm_zone_flag: 1 if this node is ≤1 tree-hop from any recent guess.
    // EvasionAware maximises depth + distance-from-recent, so it never looks here —
    // the evader can exploit this by hiding in the "warm zone" it ignores.
    let warm_zone_flag = if min_recent_distance <= 1 && !recent_guesses.is_empty() { 1.0 } else { 0.0 };

    [
        node.depth as f64 / node_count.max(1.0),
        if node.child_count == 0 { 1.0 } else { 0.0 },
        node.child_count as f64 / 2.0,
        node.key as f64 / max_key,
        node.encounter_index as f64 / node_count.max(1.0),
        node.path_len as f64 / (node_count + 1.0),
        min_recent_distance as f64 / (node_count + 1.0),
        avg_recent_distance / (node_count + 1.0),
        if recent_guesses.contains(&node.key) { 1.0 } else { 0.0 },
        last_guess_distance / (node_count + 1.0),
        subtree_size_norm,
        cold_subtree_flag,
        recent_same_depth_norm,
        warm_zone_flag,
    ]
}

// ──────────────────────────────────────────────────────────────────────────────
// Node ranking helpers
// ──────────────────────────────────────────────────────────────────────────────

fn rank_nodes_evader(model: &MlpPolicyModel, all_nodes: &[NodeMeta], recent_guesses: &[i32]) -> Vec<(i32, f64)> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * EVADER_FEATURE_COUNT);
    for node in all_nodes {
        feature_rows.extend_from_slice(&build_evader_features(node, all_nodes, recent_guesses));
    }
    let scores = model.batch_score_nodes(&feature_rows);
    let mut ranked: Vec<(i32, f64)> = all_nodes.iter().zip(scores.iter())
        .filter(|(node, _)| node.depth != 0)   // root is never a valid hiding spot
        .map(|(node, &score)| (node.key, score))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    ranked
}

fn choose_key_evader(
    model: &MlpPolicyModel,
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    excluded: &[i32],
) -> Option<i32> {
    rank_nodes_evader(model, all_nodes, recent_guesses)
        .into_iter()
        .map(|(key, _)| key)
        .find(|key| !excluded.contains(key))
}

fn collect_node_meta_from_snapshot(
    node: &NodeSnapshot,
    depth: usize,
    path_bits: u64,
    path_len: u8,
    ordered: &mut Vec<NodeMeta>,
) -> usize {
    let encounter_index = ordered.len();
    let child_count = node.left.is_some() as usize + node.right.is_some() as usize;
    ordered.push(NodeMeta {
        key: node.key,
        depth,
        path_bits,
        path_len,
        child_count,
        encounter_index,
        subtree_size: 0,
    });
    let left_size = node.left.as_deref().map(|n| {
        collect_node_meta_from_snapshot(n, depth + 1, path_bits << 1, path_len + 1, ordered)
    }).unwrap_or(0);
    let right_size = node.right.as_deref().map(|n| {
        collect_node_meta_from_snapshot(n, depth + 1, (path_bits << 1) | 1, path_len + 1, ordered)
    }).unwrap_or(0);
    let total = 1 + left_size + right_size;
    ordered[encounter_index].subtree_size = total;
    total
}

/// Returns the ML evader's preferred relocation key given a live tree snapshot.
pub fn choose_evader_relocation_from_snapshot(
    model: &MlpPolicyModel,
    snapshot: &NodeSnapshot,
    recent_guesses: &[i32],
    excluded: &[i32],
) -> Option<i32> {
    let mut all_nodes = Vec::new();
    collect_node_meta_from_snapshot(snapshot, 0, 1u64, 1u8, &mut all_nodes);
    choose_key_evader(model, &all_nodes, recent_guesses, excluded)
}

/// Returns the ML searcher's preferred guess given a live tree snapshot.
pub fn choose_searcher_guess_from_snapshot(
    model: &SearcherMlpModel,
    snapshot: &NodeSnapshot,
    recent_guesses: &[i32],
    excluded: &[i32],
) -> Option<i32> {
    let mut all_nodes = Vec::new();
    collect_node_meta_from_snapshot(snapshot, 0, 1u64, 1u8, &mut all_nodes);
    choose_key_searcher_mlp(model, &all_nodes, recent_guesses, excluded)
}

fn rank_nodes_searcher_mlp(model: &SearcherMlpModel, all_nodes: &[NodeMeta], recent_guesses: &[i32]) -> Vec<(i32, f64)> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * SEARCHER_FEATURE_COUNT);
    for node in all_nodes {
        feature_rows.extend_from_slice(&build_searcher_features(node, all_nodes, recent_guesses));
    }
    let scores = model.batch_score_nodes(&feature_rows);
    let mut ranked: Vec<(i32, f64)> = all_nodes.iter().zip(scores.iter())
        .map(|(node, &score)| (node.key, score))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    ranked
}

fn choose_key_searcher_mlp(
    model: &SearcherMlpModel,
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    excluded: &[i32],
) -> Option<i32> {
    rank_nodes_searcher_mlp(model, all_nodes, recent_guesses)
        .into_iter()
        .map(|(key, _)| key)
        .find(|key| !excluded.contains(key))
}

fn push_recent_guess(recent_guesses: &mut Vec<i32>, guess: i32) {
    recent_guesses.push(guess);
    if recent_guesses.len() > RECENT_MEMORY {
        let overflow = recent_guesses.len() - RECENT_MEMORY;
        recent_guesses.drain(0..overflow);
    }
}

fn compute_max_attempts(
    node_count: usize,
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
) -> usize {
    let mut max_attempts = node_count.max(1) * max_attempts_factor.max(1);

    if let Some(ratio) = max_attempts_ratio {
        let ratio_budget = ((node_count.max(1) as f64) * ratio.max(0.05)).ceil() as usize;
        max_attempts = max_attempts.min(ratio_budget.max(1));
    }

    if let Some(cap) = max_attempts_cap {
        max_attempts = max_attempts.min(cap.max(1));
    }

    max_attempts.max(1)
}

fn frontier_exposure_from_order(target_key: i32, ordered_keys: &[i32]) -> f64 {
    const IMMEDIATE_WINDOW: usize = 6;

    let Some(index) = ordered_keys.iter().position(|key| *key == target_key) else {
        return 0.0;
    };
    if index >= IMMEDIATE_WINDOW {
        return 0.0;
    }

    (IMMEDIATE_WINDOW.saturating_sub(index) as f64) / IMMEDIATE_WINDOW as f64
}

fn frontier_exposure_penalty(
    relocate_key: i32,
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    guessed_keys: &[i32],
) -> f64 {
    let remaining_nodes: Vec<&NodeMeta> = all_nodes
        .iter()
        .filter(|node| !guessed_keys.contains(&node.key))
        .collect();
    if remaining_nodes.is_empty() {
        return 0.0;
    }

    let mut exposures = Vec::new();

    let mut ascending: Vec<i32> = remaining_nodes.iter().map(|node| node.key).collect();
    ascending.sort();
    exposures.push(frontier_exposure_from_order(relocate_key, &ascending));

    let mut breadth_first: Vec<i32> = remaining_nodes.iter().map(|node| node.key).collect();
    breadth_first.sort_by_key(|key| {
        all_nodes
            .iter()
            .find(|node| node.key == *key)
            .map(|node| (node.depth, node.path_bits))
            .unwrap_or((usize::MAX, u64::MAX))
    });
    exposures.push(frontier_exposure_from_order(relocate_key, &breadth_first));

    let mut preorder: Vec<i32> = remaining_nodes.iter().map(|node| node.key).collect();
    preorder.sort_by_key(|key| {
        all_nodes
            .iter()
            .find(|node| node.key == *key)
            .map(|node| node.encounter_index)
            .unwrap_or(usize::MAX)
    });
    exposures.push(frontier_exposure_from_order(relocate_key, &preorder));

    let mut deepest_first: Vec<i32> = remaining_nodes.iter().map(|node| node.key).collect();
    deepest_first.sort_by_key(|key| {
        all_nodes
            .iter()
            .find(|node| node.key == *key)
            .map(|node| (usize::MAX - node.depth, node.path_bits))
            .unwrap_or((usize::MAX, u64::MAX))
    });
    exposures.push(frontier_exposure_from_order(relocate_key, &deepest_first));

    let mut evasion_aware: Vec<i32> = remaining_nodes.iter().map(|node| node.key).collect();
    evasion_aware.sort_by_key(|key| {
        let Some(node) = all_nodes.iter().find(|node| node.key == *key) else {
            return (i64::MAX, i64::MAX);
        };
        let min_recent_distance = if recent_guesses.is_empty() {
            node.depth + 1
        } else {
            recent_guesses
                .iter()
                .filter_map(|guess| all_nodes.iter().find(|candidate| candidate.key == *guess))
                .map(|recent| path_distance(node.path_bits, node.path_len, recent.path_bits, recent.path_len))
                .min()
                .unwrap_or(node.depth + 1)
        };
        let score = ((node.depth as i64) * 10) + ((min_recent_distance as i64) * 6);
        (-score, node.key as i64)
    });
    exposures.push(frontier_exposure_from_order(relocate_key, &evasion_aware));

    // Only fixed-algorithm orderings are used. Including the co-evolving searcher's
    // ranking here creates a feedback loop: the searcher can "claim" any node by
    // ranking it high, herding the evader into a predictable corner over generations.
    exposures.iter().sum::<f64>() / exposures.len().max(1) as f64
}

// ──────────────────────────────────────────────────────────────────────────────
// Reward functions
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the searcher's reward for one episode.
///
/// # Gradient-collapse problem
/// When found_rate → 0, every candidate scores the same constant
/// `-120 - max_attempts * 2`, giving zero selection variance.  The evolutionary
/// algorithm cannot improve because there is nothing to select on.
///
/// # Solution: partial credit on failure
/// Even when the shell is never found, candidates differ in:
///   - **Proximity**: how close did any single guess get to the shell?
///     `min_dist_to_shell == 1` means the searcher guessed the immediate
///     neighbour — almost found it.  That is much better than random walking.
///   - **Coverage**: how many *distinct* nodes were visited?
///     A searcher that revisits the same nodes wastes budget; one that
///     systematically covers distinct nodes should be rewarded.
///
/// Scaling rationale:
///   - On failure the base reward is reduced to -60 (from -120) to create
///     headroom for the bonuses.
///   - Proximity bonus: up to +40.  Scales linearly from 0 (far away) to 40
///     (distance == 1, i.e. one hop away from the shell).
///   - Coverage bonus: up to +20.  Scales linearly with fraction of tree
///     covered by unique guesses.
///   - Total possible on failure: -60 - attempts*2 + 40 + 20 = 0 - attempts*2.
///     This means a perfect-coverage, almost-found episode at max_attempts still
///     gives a negative reward — correctly worse than any found episode.
///
/// When found the reward is unchanged so the selection pressure to actually
/// find the shell is fully preserved.
fn searcher_reward(
    found: bool,
    attempts: usize,
    min_dist_to_shell: usize,
    unique_guesses: usize,
    node_count: usize,
) -> f64 {
    if found {
        // Early-find tempo bonus: up to +30 when the shell is found quickly relative
        // to tree size. Also small bonus for avoiding duplicate guesses.
        let tempo_bonus = ((node_count as f64) / (attempts as f64).max(1.0)).min(3.0) * 10.0;
        let diversity_bonus = 5.0 * (unique_guesses as f64 / (attempts as f64).max(1.0)).min(1.0);
        120.0 - (attempts as f64 * 8.0) + tempo_bonus + diversity_bonus
    } else {
        let nc = node_count.max(1) as f64;

        // Proximity bonus: 0..40. Best when dist == 1 (one hop from shell).
        let proximity_bonus = if min_dist_to_shell == 0 {
            // Guard: found == true should have been caught above.
            40.0
        } else {
            40.0 * (1.0 - (min_dist_to_shell as f64 - 1.0) / nc).max(0.0)
        };

        // Coverage bonus: 0..25. Rewards visiting diverse nodes more strongly.
        let coverage_bonus = 25.0 * (unique_guesses as f64 / nc).min(1.0);

        -60.0 - (attempts as f64 * 2.0) + proximity_bonus + coverage_bonus
    }
}

/// Evader reward now includes a relocation-distance penalty.
///
/// The penalty discourages unlimited teleportation: on each unsuccessful guess the evader
/// is penalised proportionally to how far it moved the shell (measured in tree hops).
/// This is scaled by `node_count` so it stays meaningful across tree sizes.
///
/// Design rationale:
///   - Without the penalty the evader's globally-optimal strategy is "always jump to the
///     farthest node", a pure linear policy that a linear searcher can never beat.
///   - With the penalty the evader must trade off survival vs. relocation cost, creating
///     a richer strategy space that requires the MLP's non-linearity to navigate.
fn evader_reward(
    found: bool,
    attempts: usize,
    max_attempts: usize,
    total_reloc_cost: f64,
    total_frontier_exposure: f64,
    root_relocations: usize,
    node_count: usize,
) -> f64 {
    // Per-hop penalty: 0.6 per relocation edge, scaled to tree size.
    let reloc_penalty = if node_count > 1 {
        total_reloc_cost * 0.6 / (node_count as f64).ln().max(1.0)
    } else {
        0.0
    };
    let frontier_penalty = total_frontier_exposure * 9.0;
    // Root node (min-key) is always guessed first by the ascending searcher.
    // Each time the evader hides there it is trivially found next turn.
    let root_penalty = root_relocations as f64 * 28.0;
    let early_survival_window = max_attempts.min(6);
    let early_survival_bonus = attempts.min(early_survival_window) as f64 * 7.5;
    let immediate_capture_penalty = if found && attempts <= 3 {
        (4usize.saturating_sub(attempts) as f64) * 18.0
    } else {
        0.0
    };

    if found {
        let survival_ratio = attempts as f64 / max_attempts.max(1) as f64;
        (survival_ratio * 70.0) - 35.0 + early_survival_bonus
            - reloc_penalty - frontier_penalty - root_penalty - immediate_capture_penalty
    } else {
        120.0 + (attempts as f64 * 2.5) + early_survival_bonus
            - reloc_penalty - frontier_penalty - root_penalty
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Episode and evaluation machinery
// ──────────────────────────────────────────────────────────────────────────────

fn sample_generation_mode(rng: &mut StdRng) -> GenerationMode {
    if rng.gen_bool(0.5) {
        GenerationMode::Balanced
    } else {
        GenerationMode::Uneven
    }
}

pub fn build_episode_specs(
    episodes: usize,
    seed: u64,
    min_nodes: i32,
    max_nodes: i32,
) -> Vec<EpisodeSpec> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..episodes.max(1))
        .map(|_| EpisodeSpec {
            node_count: rng.gen_range(min_nodes.max(1)..=max_nodes.max(min_nodes).max(1)),
            generation_mode: sample_generation_mode(&mut rng),
            seed: rng.gen(),
        })
        .collect()
}

pub fn evaluate_pair_on_specs_mlp_searcher(
    evader: &MlpPolicyModel,
    searcher: &SearcherMlpModel,
    episode_specs: &[EpisodeSpec],
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
) -> EvaluationSummary {
    let candidates = [evader.clone()];
    let gpu_batch = GpuEvaderBatch::new(&candidates, candle_device())
        .expect("GPU evaluation evader upload failed — GPU required");
    let gpu_searcher =
        GpuSearcherModel::new(searcher, candle_device()).expect("GPU evaluation searcher upload failed");
    let results: Vec<(bool, usize, f64, f64)> = episode_specs
        .par_iter()
        .copied()
        .filter_map(|spec| {
            run_vectorized_episode(
                &gpu_batch,
                &candidates,
                &gpu_searcher,
                spec,
                max_attempts_factor,
                max_attempts_ratio,
                max_attempts_cap,
            )
            .into_iter()
            .next()
        })
        .collect();

    let found_count = results.iter().filter(|(found, _, _, _)| *found).count();
    let total_attempts: usize = results.iter().map(|(_, attempts, _, _)| *attempts).sum();
    let total_searcher_reward: f64 = results.iter().map(|(_, _, reward, _)| *reward).sum();
    let total_evader_reward: f64 = results.iter().map(|(_, _, _, reward)| *reward).sum();

    let episode_count = episode_specs.len().max(1) as f64;
    EvaluationSummary {
        episodes: episode_specs.len().max(1),
        found_rate: found_count as f64 / episode_count,
        average_attempts: total_attempts as f64 / episode_count,
        average_searcher_reward: total_searcher_reward / episode_count,
        average_evader_reward: total_evader_reward / episode_count,
    }
}

fn evaluate_evader_against_fixed_searchers(
    evader: &MlpPolicyModel,
    fixed_searchers: &[SearcherMlpModel],
    episode_specs: &[EpisodeSpec],
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
) -> EvaluationSummary {
    let total: Vec<EvaluationSummary> = fixed_searchers
        .iter()
        .map(|searcher| {
            evaluate_pair_on_specs_mlp_searcher(
                evader,
                searcher,
                episode_specs,
                max_attempts_factor,
                max_attempts_ratio,
                max_attempts_cap,
            )
        })
        .collect();
    let denom = total.len().max(1) as f64;
    EvaluationSummary {
        episodes: episode_specs.len().max(1),
        found_rate: total.iter().map(|e| e.found_rate).sum::<f64>() / denom,
        average_attempts: total.iter().map(|e| e.average_attempts).sum::<f64>() / denom,
        average_searcher_reward: total.iter().map(|e| e.average_searcher_reward).sum::<f64>() / denom,
        average_evader_reward: total.iter().map(|e| e.average_evader_reward).sum::<f64>() / denom,
    }
}

fn fixed_searcher_role_names(searchers: &[SearcherMlpModel]) -> String {
    searchers
        .iter()
        .map(|searcher| searcher.role.clone())
        .collect::<Vec<_>>()
        .join(", ")
}

fn sample_fixed_searchers(
    fixed_searchers: &[SearcherMlpModel],
    sample_count: usize,
    rng: &mut StdRng,
) -> Vec<SearcherMlpModel> {
    if fixed_searchers.is_empty() {
        return Vec::new();
    }

    if sample_count == 0 || sample_count >= fixed_searchers.len() {
        return fixed_searchers.to_vec();
    }

    let mut indices: Vec<usize> = (0..fixed_searchers.len()).collect();
    indices.sort_by_key(|_| rng.gen::<u64>());
    indices
        .into_iter()
        .take(sample_count)
        .map(|index| fixed_searchers[index].clone())
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Evolutionary optimisation
// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// OpenAI-style Evolution Strategies with Adam
// ──────────────────────────────────────────────────────────────────────────────
//
// Each generation:
//   1. Generate N perturbed candidates: θᵢ = θ + σ·εᵢ  (εᵢ ~ N(0,I))
//   2. Evaluate fitness Fᵢ for every candidate via the existing GPU batch.
//   3. Rank-normalise fitness → rᵢ ∈ [-0.5, 0.5].
//   4. ES gradient:  g = (1/Nσ) Σᵢ rᵢ εᵢ
//   5. Apply Adam(g) to the base model θ.
//
// Memory trick: each εᵢ is never stored.  It is reproduced on demand from a
// per-candidate seed, so gradient accumulation costs O(n_params) extra RAM.

const ES_BETA1:   f64 = 0.9;
const ES_BETA2:   f64 = 0.999;
const ES_EPS:     f64 = 1e-8;

/// Box-Muller normal sample from a uniform RNG.
#[inline]
/// Convert raw fitness scores to rank-normalised values in [-0.5, 0.5].
fn rank_normalize(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    if n == 0 { return vec![]; }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(Ordering::Equal));
    let mut out = vec![0.0f64; n];
    for (rank, i) in idx.into_iter().enumerate() {
        out[i] = (rank as f64) / (n as f64) - 0.5;
    }
    out
}

/// Adam update for a parameter slice.  Updates moments in-place, returns new params.
fn adam_update_slice(
    params: &[f64],
    grad:   &[f64],
    m:      &mut Vec<f64>,
    v:      &mut Vec<f64>,
    t:      u64,
    lr:     f64,
) -> Vec<f64> {
    let bc1 = 1.0 - ES_BETA1.powi(t as i32);
    let bc2 = 1.0 - ES_BETA2.powi(t as i32);
    params.iter().zip(grad).zip(m.iter_mut()).zip(v.iter_mut())
        .map(|(((p, g), mi), vi)| {
            *mi = ES_BETA1 * *mi + (1.0 - ES_BETA1) * g;
            *vi = ES_BETA2 * *vi + (1.0 - ES_BETA2) * g * g;
            p + lr * (*mi / bc1) / ((*vi / bc2).sqrt() + ES_EPS)
        })
        .collect()
}

fn adam_update_scalar(param: f64, grad: f64, m: &mut f64, v: &mut f64, t: u64, lr: f64) -> f64 {
    let bc1 = 1.0 - ES_BETA1.powi(t as i32);
    let bc2 = 1.0 - ES_BETA2.powi(t as i32);
    *m = ES_BETA1 * *m + (1.0 - ES_BETA1) * grad;
    *v = ES_BETA2 * *v + (1.0 - ES_BETA2) * grad * grad;
    param + lr * (*m / bc1) / ((*v / bc2).sqrt() + ES_EPS)
}

/// Persistent Adam moment state for one MLP (evader or searcher).
/// Zeroed at the start of each training run, updated every generation.
#[derive(Debug, Clone)]
pub struct EsState {
    m_w1: Vec<f64>, m_b1: Vec<f64>,
    m_w2: Vec<f64>, m_b2: Vec<f64>,
    m_w3: Vec<f64>, m_b3: f64,
    v_w1: Vec<f64>, v_b1: Vec<f64>,
    v_w2: Vec<f64>, v_b2: Vec<f64>,
    v_w3: Vec<f64>, v_b3: f64,
    t: u64,
}

impl EsState {
    pub fn for_evader(m: &MlpPolicyModel) -> Self {
        Self {
            m_w1: vec![0.0; m.w1.len()], m_b1: vec![0.0; m.b1.len()],
            m_w2: vec![0.0; m.w2.len()], m_b2: vec![0.0; m.b2.len()],
            m_w3: vec![0.0; m.w3.len()], m_b3: 0.0,
            v_w1: vec![0.0; m.w1.len()], v_b1: vec![0.0; m.b1.len()],
            v_w2: vec![0.0; m.w2.len()], v_b2: vec![0.0; m.b2.len()],
            v_w3: vec![0.0; m.w3.len()], v_b3: 0.0,
            t: 0,
        }
    }

    pub fn for_searcher(m: &SearcherMlpModel) -> Self {
        Self {
            m_w1: vec![0.0; m.w1.len()], m_b1: vec![0.0; m.b1.len()],
            m_w2: vec![0.0; m.w2.len()], m_b2: vec![0.0; m.b2.len()],
            m_w3: vec![0.0; m.w3.len()], m_b3: 0.0,
            v_w1: vec![0.0; m.w1.len()], v_b1: vec![0.0; m.b1.len()],
            v_w2: vec![0.0; m.w2.len()], v_b2: vec![0.0; m.b2.len()],
            v_w3: vec![0.0; m.w3.len()], v_b3: 0.0,
            t: 0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GPU vectorized batch: all N candidates × all M nodes in one stacked matmul
// ──────────────────────────────────────────────────────────────────────────────

/// Weight tensors for the entire candidate population, pre-uploaded to the GPU.
/// Layout: first dim is candidate index.
struct GpuEvaderBatch {
    w1: Tensor, // [n, h1, id]
    b1: Tensor, // [n, h1]
    w2: Tensor, // [n, h2, h1]
    b2: Tensor, // [n, h2]
    w3: Tensor, // [n, h2, 1]
    b3: Tensor, // [n, 1]
    n: usize,
}

/// Weight tensors for a population of searcher candidates, pre-uploaded to GPU.
/// Layout mirrors `GpuEvaderBatch`: first dim is candidate index.
struct GpuSearcherBatch {
    w1: Tensor, // [n, h1, id]
    b1: Tensor, // [n, h1]
    w2: Tensor, // [n, h2, h1]
    b2: Tensor, // [n, h2]
    w3: Tensor, // [n, h2, 1]
    b3: Tensor, // [n, 1]
    n: usize,
}

impl GpuEvaderBatch {
    fn new(models: &[MlpPolicyModel], device: &Device) -> candle_core::Result<Self> {
        let n = models.len();
        let first = &models[0];
        let (h1, h2, id) = (first.hidden1_dim, first.hidden2_dim, first.input_dim);

        fn stack<F: Fn(&MlpPolicyModel) -> Vec<f32>>(
            models: &[MlpPolicyModel],
            shape: &[usize],
            device: &Device,
            f: F,
        ) -> candle_core::Result<Tensor> {
            let ts: Vec<Tensor> = models
                .iter()
                .map(|m| Tensor::from_slice(&f(m), shape, device))
                .collect::<candle_core::Result<_>>()?;
            Tensor::stack(&ts, 0)
        }

        Ok(Self {
            w1: stack(models, &[h1, id],   device, |m| m.w1.iter().map(|&v| v as f32).collect())?,
            b1: stack(models, &[h1],       device, |m| m.b1.iter().map(|&v| v as f32).collect())?,
            w2: stack(models, &[h2, h1],   device, |m| m.w2.iter().map(|&v| v as f32).collect())?,
            b2: stack(models, &[h2],       device, |m| m.b2.iter().map(|&v| v as f32).collect())?,
            w3: stack(models, &[h2, 1],    device, |m| m.w3.iter().map(|&v| v as f32).collect())?,
            b3: stack(models, &[1],        device, |m| vec![m.b3 as f32])?,
            n,
        })
    }

    /// Build a population of ES-perturbed candidates entirely on GPU.
    /// Returns the batch and the flat noise matrix [n, p_total] for gradient computation.
    /// p_total = h1*id + h1 + h2*h1 + h2 + h2 + 1  (w1,b1,w2,b2,w3,b3 in order).
    fn new_es(base: &MlpPolicyModel, n: usize, sigma: f64, device: &Device) -> candle_core::Result<(Self, Tensor)> {
        let h1 = base.hidden1_dim;
        let h2 = base.hidden2_dim;
        let id = base.input_dim;
        let p  = h1*id + h1 + h2*h1 + h2 + h2 + 1;

        // One GPU kernel for all noise.
        let noise = Tensor::randn(0.0f32, 1.0f32, (n, p), device)?;

        let f32v = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let bw1 = Tensor::from_slice(&f32v(&base.w1), (1, h1, id), device)?;
        let bb1 = Tensor::from_slice(&f32v(&base.b1), (1, h1),     device)?;
        let bw2 = Tensor::from_slice(&f32v(&base.w2), (1, h2, h1), device)?;
        let bb2 = Tensor::from_slice(&f32v(&base.b2), (1, h2),     device)?;
        let bw3 = Tensor::from_slice(&f32v(&base.w3), (1, h2, 1),  device)?;
        let bb3 = Tensor::from_slice(&[base.b3 as f32], (1, 1),    device)?;

        let mut off = 0;
        let nw1 = noise.narrow(1, off, h1*id)?.reshape((n, h1, id))?; off += h1*id;
        let nb1 = noise.narrow(1, off, h1)?;                           off += h1;
        let nw2 = noise.narrow(1, off, h2*h1)?.reshape((n, h2, h1))?; off += h2*h1;
        let nb2 = noise.narrow(1, off, h2)?;                           off += h2;
        let nw3 = noise.narrow(1, off, h2)?.reshape((n, h2, 1))?;     off += h2;
        let nb3 = noise.narrow(1, off, 1)?;

        let batch = Self {
            w1: bw1.broadcast_add(&nw1.affine(sigma, 0.0)?)?,
            b1: bb1.broadcast_add(&nb1.affine(sigma, 0.0)?)?,
            w2: bw2.broadcast_add(&nw2.affine(sigma, 0.0)?)?,
            b2: bb2.broadcast_add(&nb2.affine(sigma, 0.0)?)?,
            w3: bw3.broadcast_add(&nw3.affine(sigma, 0.0)?)?,
            b3: bb3.broadcast_add(&nb3.affine(sigma, 0.0)?)?,
            n,
        };
        Ok((batch, noise))
    }

    /// One GPU call: scores[candidate][node].
    /// `feature_rows` is flat [n_nodes × input_dim] row-major f64.
    fn score_all(&self, feature_rows: &[f64], input_dim: usize) -> candle_core::Result<Vec<Vec<f64>>> {
        let n_nodes = feature_rows.len() / input_dim;
        let n = self.n;
        let device = candle_device();

        let xf: Vec<f32> = feature_rows.iter().map(|&v| v as f32).collect();
        // x: [n_nodes, id] → expand to [n, n_nodes, id]
        let x = Tensor::from_slice(&xf, (n_nodes, input_dim), device)?
            .unsqueeze(0)?
            .repeat(&[n, 1, 1])?;

        // Layer 1: [n, n_nodes, id] @ [n, id, h1] → relu → [n, n_nodes, h1]
        let h1 = x
            .matmul(&self.w1.transpose(1, 2)?)?
            .broadcast_add(&self.b1.unsqueeze(1)?)?
            .relu()?;

        // Layer 2: [n, n_nodes, h1] @ [n, h1, h2] → relu → [n, n_nodes, h2]
        let h2 = h1
            .matmul(&self.w2.transpose(1, 2)?)?
            .broadcast_add(&self.b2.unsqueeze(1)?)?
            .relu()?;

        // Output: [n, n_nodes, h2] @ [n, h2, 1] → squeeze → [n, n_nodes]
        let out = h2
            .matmul(&self.w3)?
            .broadcast_add(&self.b3.unsqueeze(1)?)?
            .squeeze(2)?;

        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;
        Ok((0..n)
            .map(|c| (0..n_nodes).map(|i| flat[c * n_nodes + i] as f64).collect())
            .collect())
    }
}

impl GpuSearcherBatch {
    /// Build a population of ES-perturbed searcher candidates entirely on GPU.
    fn new_es(base: &SearcherMlpModel, n: usize, sigma: f64, device: &Device) -> candle_core::Result<(Self, Tensor)> {
        let h1 = base.hidden1_dim;
        let h2 = base.hidden2_dim;
        let id = base.input_dim;
        let p  = h1*id + h1 + h2*h1 + h2 + h2 + 1;

        let noise = Tensor::randn(0.0f32, 1.0f32, (n, p), device)?;

        let f32v = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let bw1 = Tensor::from_slice(&f32v(&base.w1), (1, h1, id), device)?;
        let bb1 = Tensor::from_slice(&f32v(&base.b1), (1, h1),     device)?;
        let bw2 = Tensor::from_slice(&f32v(&base.w2), (1, h2, h1), device)?;
        let bb2 = Tensor::from_slice(&f32v(&base.b2), (1, h2),     device)?;
        let bw3 = Tensor::from_slice(&f32v(&base.w3), (1, h2, 1),  device)?;
        let bb3 = Tensor::from_slice(&[base.b3 as f32], (1, 1),    device)?;

        let mut off = 0;
        let nw1 = noise.narrow(1, off, h1*id)?.reshape((n, h1, id))?; off += h1*id;
        let nb1 = noise.narrow(1, off, h1)?;                           off += h1;
        let nw2 = noise.narrow(1, off, h2*h1)?.reshape((n, h2, h1))?; off += h2*h1;
        let nb2 = noise.narrow(1, off, h2)?;                           off += h2;
        let nw3 = noise.narrow(1, off, h2)?.reshape((n, h2, 1))?;     off += h2;
        let nb3 = noise.narrow(1, off, 1)?;

        let batch = Self {
            w1: bw1.broadcast_add(&nw1.affine(sigma, 0.0)?)?,
            b1: bb1.broadcast_add(&nb1.affine(sigma, 0.0)?)?,
            w2: bw2.broadcast_add(&nw2.affine(sigma, 0.0)?)?,
            b2: bb2.broadcast_add(&nb2.affine(sigma, 0.0)?)?,
            w3: bw3.broadcast_add(&nw3.affine(sigma, 0.0)?)?,
            b3: bb3.broadcast_add(&nb3.affine(sigma, 0.0)?)?,
            n,
        };
        Ok((batch, noise))
    }

    /// One GPU call: scores[candidate][node].
    /// `feature_rows` is flat [n_candidates x n_nodes x input_dim] row-major f64.
    fn score_all_feature_batches(
        &self,
        feature_rows: &[f64],
        n_nodes: usize,
        input_dim: usize,
    ) -> candle_core::Result<Vec<Vec<f64>>> {
        let n = self.n;
        if n == 0 || n_nodes == 0 {
            return Ok(Vec::new());
        }

        let device = candle_device();
        let xf: Vec<f32> = feature_rows.iter().map(|&v| v as f32).collect();
        let x = Tensor::from_slice(&xf, (n, n_nodes, input_dim), device)?;

        let h1 = x
            .matmul(&self.w1.transpose(1, 2)?)?
            .broadcast_add(&self.b1.unsqueeze(1)?)?
            .relu()?;
        let h2 = h1
            .matmul(&self.w2.transpose(1, 2)?)?
            .broadcast_add(&self.b2.unsqueeze(1)?)?
            .relu()?;
        let out = h2
            .matmul(&self.w3)?
            .broadcast_add(&self.b3.unsqueeze(1)?)?
            .squeeze(2)?;

        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;
        Ok((0..n)
            .map(|c| (0..n_nodes).map(|i| flat[c * n_nodes + i] as f64).collect())
            .collect())
    }
}

/// Compute the ES gradient from a flat noise matrix and rank-normalised fitness.
/// Works for both evader and searcher batches — the noise layout is the same.
/// Returns a flat Vec<f64> of length p_total; callers split it by param group.
fn es_gradient_gpu(noise: &Tensor, ranked: &[f64], n: usize, sigma: f64) -> candle_core::Result<Vec<f64>> {
    let device = candle_device();
    let r: Vec<f32> = ranked.iter().map(|&v| v as f32).collect();
    // r_row: [1, n] — one row-vector to left-multiply noise [n, p]
    let r_row = Tensor::from_slice(&r, (1, n), device)?;
    // grad = (1/Nσ) * r_row @ noise  →  [1, p]  →  squeeze  →  [p]
    let grad = r_row.matmul(noise)?.squeeze(0)?.affine(1.0 / (n as f64 * sigma), 0.0)?;
    Ok(grad.to_vec1::<f32>()?.into_iter().map(|v| v as f64).collect())
}

/// Single searcher model with weights resident on the selected Candle device.
/// This avoids re-uploading the same searcher weights on every guess attempt.
struct GpuSearcherModel {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
    input_dim: usize,
}

impl GpuSearcherModel {
    fn new(model: &SearcherMlpModel, device: &Device) -> candle_core::Result<Self> {
        Ok(Self {
            w1: Tensor::from_slice(
                &model.w1.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden1_dim, model.input_dim),
                device,
            )?,
            b1: Tensor::from_slice(
                &model.b1.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden1_dim,),
                device,
            )?,
            w2: Tensor::from_slice(
                &model.w2.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden2_dim, model.hidden1_dim),
                device,
            )?,
            b2: Tensor::from_slice(
                &model.b2.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden2_dim,),
                device,
            )?,
            w3: Tensor::from_slice(
                &model.w3.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden2_dim, 1),
                device,
            )?,
            b3: Tensor::from_slice(&[model.b3 as f32], (1,), device)?,
            input_dim: model.input_dim,
        })
    }

    fn score_nodes(&self, feature_rows: &[f64]) -> candle_core::Result<Vec<f64>> {
        let n_nodes = feature_rows.len() / self.input_dim.max(1);
        if n_nodes == 0 {
            return Ok(Vec::new());
        }

        let device = candle_device();
        let xf: Vec<f32> = feature_rows.iter().map(|&v| v as f32).collect();
        let x = Tensor::from_slice(&xf, (n_nodes, self.input_dim), device)?;
        let h1 = x
            .matmul(&self.w1.t()?)?
            .broadcast_add(&self.b1)?
            .relu()?;
        let h2 = h1
            .matmul(&self.w2.t()?)?
            .broadcast_add(&self.b2)?
            .relu()?;
        let out = h2
            .matmul(&self.w3)?
            .broadcast_add(&self.b3)?
            .squeeze(1)?;

        Ok(out
            .to_vec1::<f32>()?
            .into_iter()
            .map(|v| v as f64)
            .collect())
    }
}

fn choose_key_searcher_gpu(
    model: &GpuSearcherModel,
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    excluded: &[i32],
) -> Option<i32> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * SEARCHER_FEATURE_COUNT);
    for node in all_nodes {
        feature_rows.extend_from_slice(&build_searcher_features(node, all_nodes, recent_guesses));
    }

    let scores = model
        .score_nodes(&feature_rows)
        .expect("GPU searcher scoring failed");

    all_nodes
        .iter()
        .zip(scores.iter())
        .filter(|(node, _)| !excluded.contains(&node.key))
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
        .map(|(node, _)| node.key)
}

// ──────────────────────────────────────────────────────────────────────────────
// GpuEvaderModel — single evader model pre-uploaded to GPU
// ──────────────────────────────────────────────────────────────────────────────

struct GpuEvaderModel {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
    input_dim: usize,
}

impl GpuEvaderModel {
    fn new(model: &MlpPolicyModel, device: &Device) -> candle_core::Result<Self> {
        Ok(Self {
            w1: Tensor::from_slice(
                &model.w1.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden1_dim, model.input_dim),
                device,
            )?,
            b1: Tensor::from_slice(
                &model.b1.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden1_dim,),
                device,
            )?,
            w2: Tensor::from_slice(
                &model.w2.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden2_dim, model.hidden1_dim),
                device,
            )?,
            b2: Tensor::from_slice(
                &model.b2.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden2_dim,),
                device,
            )?,
            w3: Tensor::from_slice(
                &model.w3.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                (model.hidden2_dim, 1),
                device,
            )?,
            b3: Tensor::from_slice(&[model.b3 as f32], (1,), device)?,
            input_dim: model.input_dim,
        })
    }

    /// Score `n_nodes` rows of evader features in one GPU matmul chain.
    /// `feature_rows` is flat row-major [n_nodes × input_dim] f64.
    fn score_nodes(&self, feature_rows: &[f64]) -> candle_core::Result<Vec<f64>> {
        let n_nodes = feature_rows.len() / self.input_dim.max(1);
        if n_nodes == 0 {
            return Ok(Vec::new());
        }
        let device = candle_device();
        let xf: Vec<f32> = feature_rows.iter().map(|&v| v as f32).collect();
        let x = Tensor::from_slice(&xf, (n_nodes, self.input_dim), device)?;
        let h1 = x.matmul(&self.w1.t()?)?.broadcast_add(&self.b1)?.relu()?;
        let h2 = h1.matmul(&self.w2.t()?)?.broadcast_add(&self.b2)?.relu()?;
        let out = h2.matmul(&self.w3)?.broadcast_add(&self.b3)?.squeeze(1)?;
        Ok(out.to_vec1::<f32>()?.into_iter().map(|v| v as f64).collect())
    }
}

fn choose_key_evader_gpu(
    model: &GpuEvaderModel,
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    excluded: &[i32],
) -> Option<i32> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * EVADER_FEATURE_COUNT);
    for node in all_nodes {
        feature_rows.extend_from_slice(&build_evader_features(node, all_nodes, recent_guesses));
    }
    let scores = model
        .score_nodes(&feature_rows)
        .expect("GPU evader scoring failed");
    all_nodes
        .iter()
        .zip(scores.iter())
        .filter(|(node, _)| node.depth != 0 && !excluded.contains(&node.key))
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
        .map(|(node, _)| node.key)
}

/// GPU version of `sample_key_evader_training`: scores nodes on GPU, samples with temperature.
fn sample_key_evader_training_gpu(
    gpu_model: &GpuEvaderModel,
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    excluded: &[i32],
    rng: &mut StdRng,
) -> Option<i32> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * EVADER_FEATURE_COUNT);
    for node in all_nodes {
        feature_rows.extend_from_slice(&build_evader_features(node, all_nodes, recent_guesses));
    }
    let raw_scores = gpu_model
        .score_nodes(&feature_rows)
        .expect("GPU evader scoring failed");

    let mut ranked: Vec<(i32, f64)> = all_nodes
        .iter()
        .zip(raw_scores.iter())
        .filter(|(node, _)| node.depth != 0)    // root is never a valid hiding spot
        .map(|(node, &score)| (node.key, score))
        .filter(|(key, _)| !excluded.contains(key))
        .collect();
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let top_k = ranked.len().min(4);
    let temperature = 0.85_f64;
    let max_score = ranked.iter().take(top_k).map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
    let mut cumulative = Vec::with_capacity(top_k);
    let mut total = 0.0;
    for (key, score) in ranked.iter().take(top_k) {
        let weight = ((*score - max_score) / temperature).exp().max(1e-9);
        total += weight;
        cumulative.push((*key, total));
    }

    let draw = rng.gen_range(0.0..total.max(1e-9));
    cumulative
        .into_iter()
        .find(|(_, threshold)| draw <= *threshold)
        .map(|(key, _)| key)
        .or_else(|| ranked.first().map(|(key, _)| *key))
}

fn choose_initial_evader_keys_gpu(
    gpu_batch: &GpuEvaderBatch,
    candidate_count: usize,
    all_nodes: &[NodeMeta],
    fallback_key: i32,
) -> Vec<i32> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * EVADER_FEATURE_COUNT);
    for node in all_nodes {
        feature_rows.extend_from_slice(&build_evader_features(node, all_nodes, &[]));
    }

    let scores = gpu_batch
        .score_all(&feature_rows, EVADER_FEATURE_COUNT)
        .expect("GPU initial evader scoring failed");

    (0..candidate_count)
        .map(|candidate| {
            scores
                .get(candidate)
                .and_then(|candidate_scores| {
                    all_nodes
                        .iter()
                        .zip(candidate_scores.iter())
                        .filter(|(node, _)| node.depth != 0)   // root is never a valid hiding spot
                        .max_by(|(_, left), (_, right)| {
                            left.partial_cmp(right).unwrap_or(Ordering::Equal)
                        })
                        .map(|(node, _)| node.key)
                })
                .unwrap_or(fallback_key)
        })
        .collect()
}

/// Run one episode with all evader candidates simultaneously.
/// The tree and searcher guess are shared; only shell positions diverge per candidate.
/// Returns (found, attempts, searcher_reward, evader_reward) per candidate.
fn run_vectorized_episode(
    gpu_batch: &GpuEvaderBatch,
    candidates: &[MlpPolicyModel],
    gpu_searcher: &GpuSearcherModel,
    spec: EpisodeSpec,
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
) -> Vec<(bool, usize, f64, f64)> {
    let n = candidates.len();
    let mut tree = FastTrainingTree::new(spec);
    let initial_meta = tree.meta_snapshot();
    let all_keys: Vec<i32> = initial_meta.iter().map(|m| m.key).collect();
    let min_key = all_keys.iter().copied().min().unwrap_or(1);
    let node_count = spec.node_count as usize;
    let max_attempts = compute_max_attempts(node_count, max_attempts_factor, max_attempts_ratio, max_attempts_cap);

    // Each candidate independently picks an initial shell position in one GPU batch.
    let mut shell_keys =
        choose_initial_evader_keys_gpu(gpu_batch, n, &initial_meta, all_keys[1.min(all_keys.len() - 1)]);

    let mut alive       = vec![true; n];
    let mut found_at    = vec![0usize; n];
    let mut reloc_costs = vec![0.0f64; n];
    let mut frontiers   = vec![0.0f64; n];
    let mut root_relocs = vec![0usize; n];
    let mut min_dists   = vec![node_count; n];

    let mut recent_guesses: Vec<i32> = Vec::new();
    let mut unique_guesses: HashSet<i32> = HashSet::new();

    for attempt in 1..=max_attempts {
        let meta = tree.meta_snapshot();
        let guessed_keys: Vec<i32> = unique_guesses.iter().copied().collect();
        let excluded = if unique_guesses.len() < all_keys.len() { guessed_keys.clone() } else { vec![] };

        // Searcher guess is the same for all candidates (shared tree + shared history).
        let guess = choose_key_searcher_gpu(gpu_searcher, &meta, &recent_guesses, &excluded)
            .or_else(|| choose_key_searcher_gpu(gpu_searcher, &meta, &recent_guesses, &[]))
            .unwrap_or(min_key);

        // Check found / min-distance per candidate.
        let guess_node = meta.iter().find(|m| m.key == guess);
        for c in 0..n {
            if !alive[c] { continue; }
            if shell_keys[c] == guess {
                alive[c] = false;
                found_at[c] = attempt;
            } else if let (Some(gn), Some(sn)) = (guess_node, meta.iter().find(|m| m.key == shell_keys[c])) {
                let d = path_distance(gn.path_bits, gn.path_len, sn.path_bits, sn.path_len);
                if d < min_dists[c] { min_dists[c] = d; }
            }
        }

        // Advance shared tree state.
        tree.shuffle_step();
        unique_guesses.insert(guess);
        push_recent_guess(&mut recent_guesses, guess);

        if alive.iter().all(|&a| !a) || attempt == max_attempts { break; }

        // Build feature matrix for the updated tree (same for all candidates).
        let meta_new = tree.meta_snapshot();
        let mut feature_rows = Vec::with_capacity(meta_new.len() * EVADER_FEATURE_COUNT);
        for node in &meta_new {
            let feats = build_evader_features(node, &meta_new, &recent_guesses);
            feature_rows.extend_from_slice(&feats);
        }

        // One GPU call: all candidates × all nodes.
        let guessed_now: Vec<i32> = unique_guesses.iter().copied().collect();
        let all_scores = gpu_batch.score_all(&feature_rows, EVADER_FEATURE_COUNT)
            .expect("GPU score_all failed — GPU required");

        // Apply per-candidate relocation.
        for c in 0..n {
            if !alive[c] { continue; }
            let scores = &all_scores[c];
            let reloc = meta_new.iter().enumerate()
                .filter(|(_, nd)| nd.depth != 0 && !guessed_now.contains(&nd.key))
                .max_by(|(i, _), (j, _)| scores[*i].partial_cmp(&scores[*j]).unwrap_or(Ordering::Equal))
                .map(|(_, nd)| nd.key)
                .or_else(|| meta_new.iter().find(|nd| nd.depth != 0 && !guessed_now.contains(&nd.key)).map(|nd| nd.key));

            if let Some(rk) = reloc {
                frontiers[c] += frontier_exposure_penalty(rk, &meta_new, &recent_guesses, &guessed_now);
                if rk == min_key { root_relocs[c] += 1; }
                if let (Some(om), Some(nm)) = (
                    meta_new.iter().find(|m| m.key == shell_keys[c]),
                    meta_new.iter().find(|m| m.key == rk),
                ) {
                    reloc_costs[c] += path_distance(om.path_bits, om.path_len, nm.path_bits, nm.path_len) as f64;
                }
                shell_keys[c] = rk;
            }
        }
    }

    let unique_count = unique_guesses.len();
    (0..n).map(|c| {
        let found   = found_at[c] > 0;
        let attempts = if found { found_at[c] } else { max_attempts };
        (
            found,
            attempts,
            searcher_reward(found, attempts, if found { 0 } else { min_dists[c] }, unique_count, node_count),
            evader_reward(found, attempts, max_attempts, reloc_costs[c], frontiers[c], root_relocs[c], node_count),
        )
    }).collect()
}

fn optimize_evader(
    current_model: &MlpPolicyModel,
    es_state: &mut EsState,
    es_lr: f64,
    searcher_pool: &[SearcherMlpModel],
    population_size: usize,
    mutation_scale: f64,
    episode_specs: &[EpisodeSpec],
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
) -> (MlpPolicyModel, f64) {
    let n = population_size.max(2);
    let device = candle_device();

    // Build the perturbed batch AND generate all noise on GPU in one randn kernel.
    let (gpu_batch, noise) = GpuEvaderBatch::new_es(current_model, n, mutation_scale, device)
        .expect("GPU ES evader batch failed — GPU required");

    // MlpPolicyModel stubs needed only for run_vectorized_episode's type signature;
    // the actual weights are already resident in gpu_batch.
    let candidate_stubs: Vec<MlpPolicyModel> = (0..n)
        .map(|_| current_model.clone())
        .collect();

    let gpu_searchers: Vec<GpuSearcherModel> = searcher_pool
        .iter()
        .map(|s| GpuSearcherModel::new(s, device).expect("GPU searcher upload failed"))
        .collect();

    let combos: Vec<(usize, EpisodeSpec)> = gpu_searchers
        .iter()
        .enumerate()
        .flat_map(|(idx, _)| episode_specs.iter().map(move |&sp| (idx, sp)))
        .collect();
    let n_evals = combos.len().max(1) as f64;

    let per_episode: Vec<Vec<f64>> = combos
        .par_iter()
        .map(|(searcher_idx, spec)| {
            run_vectorized_episode(
                &gpu_batch, &candidate_stubs, &gpu_searchers[*searcher_idx], *spec,
                max_attempts_factor, max_attempts_ratio, max_attempts_cap,
            )
            .into_iter()
            .map(|(_, _, _, er)| er)
            .collect()
        })
        .collect();

    let mut total_scores = vec![0.0f64; n];
    for episode_scores in &per_episode {
        for (c, &s) in episode_scores.iter().enumerate() {
            total_scores[c] += s / n_evals;
        }
    }
    let best_score = total_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // GPU matmul: grad = (1/Nσ) * r @ noise  —  one kernel, no CPU accumulation loop.
    let ranked = rank_normalize(&total_scores);
    let grad_flat = es_gradient_gpu(&noise, &ranked, n, mutation_scale)
        .expect("GPU ES gradient failed");

    // Split flat gradient by param group and apply Adam (CPU, ~540K multiply-adds).
    let h1 = current_model.hidden1_dim;
    let h2 = current_model.hidden2_dim;
    let id = current_model.input_dim;
    let mut off = 0;
    let gw1 = &grad_flat[off..off + h1*id]; off += h1*id;
    let gb1 = &grad_flat[off..off + h1];    off += h1;
    let gw2 = &grad_flat[off..off + h2*h1]; off += h2*h1;
    let gb2 = &grad_flat[off..off + h2];    off += h2;
    let gw3 = &grad_flat[off..off + h2];    off += h2;
    let gb3 = grad_flat[off];

    es_state.t += 1;
    let t = es_state.t;
    let new_model = MlpPolicyModel {
        role:          current_model.role.clone(),
        feature_names: current_model.feature_names.clone(),
        input_dim:     id,
        hidden1_dim:   h1,
        hidden2_dim:   h2,
        w1: adam_update_slice(&current_model.w1, gw1, &mut es_state.m_w1, &mut es_state.v_w1, t, es_lr),
        b1: adam_update_slice(&current_model.b1, gb1, &mut es_state.m_b1, &mut es_state.v_b1, t, es_lr),
        w2: adam_update_slice(&current_model.w2, gw2, &mut es_state.m_w2, &mut es_state.v_w2, t, es_lr),
        b2: adam_update_slice(&current_model.b2, gb2, &mut es_state.m_b2, &mut es_state.v_b2, t, es_lr),
        w3: adam_update_slice(&current_model.w3, gw3, &mut es_state.m_w3, &mut es_state.v_w3, t, es_lr),
        b3: adam_update_scalar(current_model.b3, gb3, &mut es_state.m_b3, &mut es_state.v_b3, t, es_lr),
    };

    (new_model, best_score)
}



/// Run one episode with all searcher candidates simultaneously.
/// The shuffled tree is shared; each candidate keeps its own guesses and shell position.
fn run_vectorized_searcher_episode(
    gpu_batch: &GpuSearcherBatch,
    candidates: &[SearcherMlpModel],
    gpu_evader: &GpuEvaderModel,
    spec: EpisodeSpec,
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
) -> Vec<(bool, usize, f64, f64)> {
    let n = candidates.len();
    let mut episode_rng = StdRng::seed_from_u64(spec.seed ^ 0xE0AD_2026);
    let mut tree = FastTrainingTree::new(spec);
    let initial_meta = tree.meta_snapshot();
    // GPU-based initial shell placement with temperature sampling. Root is always forbidden.
    let initial_shell = sample_key_evader_training_gpu(gpu_evader, &initial_meta, &[], &[], &mut episode_rng)
        .or_else(|| choose_key_evader_gpu(gpu_evader, &initial_meta, &[], &[]))
        .unwrap_or_else(|| initial_meta.iter().find(|m| m.depth != 0).map(|m| m.key).unwrap_or(2));
    let all_keys: Vec<i32> = initial_meta.iter().map(|m| m.key).collect();
    let min_key = all_keys.iter().copied().min().unwrap_or(1);
    let node_count = spec.node_count as usize;
    let max_attempts = compute_max_attempts(
        node_count,
        max_attempts_factor,
        max_attempts_ratio,
        max_attempts_cap,
    );

    let mut shell_keys = vec![initial_shell; n];
    let mut alive = vec![true; n];
    let mut found_at = vec![0usize; n];
    let mut reloc_costs = vec![0.0f64; n];
    let mut frontiers = vec![0.0f64; n];
    let mut root_relocs = vec![0usize; n];
    let mut min_dists = vec![node_count.max(1); n];
    let mut recent_guesses = vec![Vec::<i32>::new(); n];
    let mut unique_guesses = vec![HashSet::<i32>::new(); n];

    for attempt in 1..=max_attempts {
        let meta = tree.meta_snapshot();
        let n_nodes = meta.len();
        let mut searcher_features = Vec::with_capacity(n * n_nodes * SEARCHER_FEATURE_COUNT);
        for c in 0..n {
            for node in &meta {
                searcher_features.extend_from_slice(&build_searcher_features(
                    node,
                    &meta,
                    &recent_guesses[c],
                ));
            }
        }

        let all_scores = gpu_batch
            .score_all_feature_batches(&searcher_features, n_nodes, SEARCHER_FEATURE_COUNT)
            .expect("GPU searcher score_all failed — GPU required");

        let mut guesses = vec![min_key; n];
        for c in 0..n {
            if !alive[c] {
                continue;
            }

            let excluded: Vec<i32> = if unique_guesses[c].len() < all_keys.len() {
                unique_guesses[c].iter().copied().collect()
            } else {
                Vec::new()
            };
            let scores = &all_scores[c];
            guesses[c] = meta
                .iter()
                .zip(scores.iter())
                .filter(|(node, _)| !excluded.contains(&node.key))
                .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
                .map(|(node, _)| node.key)
                .or_else(|| {
                    meta.iter()
                        .zip(scores.iter())
                        .max_by(|(_, left), (_, right)| {
                            left.partial_cmp(right).unwrap_or(Ordering::Equal)
                        })
                        .map(|(node, _)| node.key)
                })
                .unwrap_or(min_key);
        }

        for c in 0..n {
            if !alive[c] {
                continue;
            }
            let guess = guesses[c];
            if shell_keys[c] == guess {
                alive[c] = false;
                found_at[c] = attempt;
            } else if let (Some(guess_node), Some(shell_node)) = (
                meta.iter().find(|m| m.key == guess),
                meta.iter().find(|m| m.key == shell_keys[c]),
            ) {
                let d = path_distance(
                    guess_node.path_bits,
                    guess_node.path_len,
                    shell_node.path_bits,
                    shell_node.path_len,
                );
                if d < min_dists[c] {
                    min_dists[c] = d;
                }
            }

            unique_guesses[c].insert(guess);
        }

        tree.shuffle_step();
        if alive.iter().all(|&a| !a) || attempt == max_attempts {
            break;
        }

        let meta_new = tree.meta_snapshot();
        let n_new_nodes = meta_new.len();

        // Batch evader relocation for all alive candidates in one GPU call.
        let alive_indices: Vec<usize> = (0..n).filter(|&c| alive[c]).collect();
        if !alive_indices.is_empty() {
            let mut reloc_features =
                Vec::with_capacity(alive_indices.len() * n_new_nodes * EVADER_FEATURE_COUNT);
            for &c in &alive_indices {
                for node in &meta_new {
                    reloc_features.extend_from_slice(&build_evader_features(
                        node,
                        &meta_new,
                        &recent_guesses[c],
                    ));
                }
            }
            let all_reloc_scores = gpu_evader
                .score_nodes(&reloc_features)
                .expect("GPU evader relocation scoring failed — GPU required");

            for (batch_idx, &c) in alive_indices.iter().enumerate() {
                let guessed_keys: Vec<i32> = unique_guesses[c].iter().copied().collect();
                let offset = batch_idx * n_new_nodes;
                let scores = &all_reloc_scores[offset..offset + n_new_nodes];

                let relocate_to = meta_new
                    .iter()
                    .zip(scores.iter())
                    .filter(|(nd, _)| nd.depth != 0 && !guessed_keys.contains(&nd.key))
                    .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap_or(Ordering::Equal))
                    .map(|(nd, _)| nd.key)
                    .or_else(|| {
                        meta_new
                            .iter()
                            .zip(scores.iter())
                            .filter(|(nd, _)| nd.depth != 0)
                            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap_or(Ordering::Equal))
                            .map(|(nd, _)| nd.key)
                    });

                if let Some(relocate_key) = relocate_to {
                    frontiers[c] += frontier_exposure_penalty(
                        relocate_key,
                        &meta_new,
                        &recent_guesses[c],
                        &guessed_keys,
                    );
                    if relocate_key == min_key {
                        root_relocs[c] += 1;
                    }
                    if let (Some(old_node), Some(new_node)) = (
                        meta_new.iter().find(|m| m.key == shell_keys[c]),
                        meta_new.iter().find(|m| m.key == relocate_key),
                    ) {
                        reloc_costs[c] += path_distance(
                            old_node.path_bits,
                            old_node.path_len,
                            new_node.path_bits,
                            new_node.path_len,
                        ) as f64;
                    }
                    shell_keys[c] = relocate_key;
                }
                push_recent_guess(&mut recent_guesses[c], guesses[c]);
            }
        }
    }

    (0..n)
        .map(|c| {
            let found = found_at[c] > 0;
            let attempts = if found { found_at[c] } else { max_attempts };
            (
                found,
                attempts,
                searcher_reward(
                    found,
                    attempts,
                    if found { 0 } else { min_dists[c] },
                    unique_guesses[c].len(),
                    node_count,
                ),
                evader_reward(
                    found,
                    attempts,
                    max_attempts,
                    reloc_costs[c],
                    frontiers[c],
                    root_relocs[c],
                    node_count,
                ),
            )
        })
        .collect()
}

fn optimize_searcher_mlp(
    current_model: &SearcherMlpModel,
    es_state: &mut EsState,
    es_lr: f64,
    evader_pool: &[MlpPolicyModel],
    population_size: usize,
    mutation_scale: f64,
    episode_specs: &[EpisodeSpec],
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
) -> (SearcherMlpModel, f64) {
    let n = population_size.max(2);
    let device = candle_device();

    let (gpu_batch, noise) = GpuSearcherBatch::new_es(current_model, n, mutation_scale, device)
        .expect("GPU ES searcher batch failed — GPU required");

    let pool_stubs: Vec<SearcherMlpModel> = (0..n).map(|_| current_model.clone()).collect();

    let gpu_evader_pool: Vec<GpuEvaderModel> = evader_pool
        .iter()
        .map(|e| GpuEvaderModel::new(e, device).expect("GPU evader upload failed — GPU required"))
        .collect();

    let combos: Vec<(usize, EpisodeSpec)> = (0..gpu_evader_pool.len())
        .flat_map(|ei| episode_specs.iter().map(move |&sp| (ei, sp)))
        .collect();
    let n_evals = combos.len().max(1) as f64;

    let per_episode: Vec<Vec<f64>> = combos
        .par_iter()
        .map(|(ei, spec)| {
            run_vectorized_searcher_episode(
                &gpu_batch, &pool_stubs, &gpu_evader_pool[*ei], *spec,
                max_attempts_factor, max_attempts_ratio, max_attempts_cap,
            )
            .into_iter()
            .map(|(_, _, sr, _)| sr)
            .collect()
        })
        .collect();

    let mut total_scores = vec![0.0f64; n];
    for episode_scores in &per_episode {
        for (c, &s) in episode_scores.iter().enumerate() {
            total_scores[c] += s / n_evals;
        }
    }
    let best_score = total_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let ranked = rank_normalize(&total_scores);
    let grad_flat = es_gradient_gpu(&noise, &ranked, n, mutation_scale)
        .expect("GPU ES searcher gradient failed");

    let h1 = current_model.hidden1_dim;
    let h2 = current_model.hidden2_dim;
    let id = current_model.input_dim;
    let mut off = 0;
    let gw1 = &grad_flat[off..off + h1*id]; off += h1*id;
    let gb1 = &grad_flat[off..off + h1];    off += h1;
    let gw2 = &grad_flat[off..off + h2*h1]; off += h2*h1;
    let gb2 = &grad_flat[off..off + h2];    off += h2;
    let gw3 = &grad_flat[off..off + h2];    off += h2;
    let gb3 = grad_flat[off];

    es_state.t += 1;
    let t = es_state.t;
    let new_model = SearcherMlpModel {
        role:          current_model.role.clone(),
        feature_names: current_model.feature_names.clone(),
        input_dim:     id,
        hidden1_dim:   h1,
        hidden2_dim:   h2,
        w1: adam_update_slice(&current_model.w1, gw1, &mut es_state.m_w1, &mut es_state.v_w1, t, es_lr),
        b1: adam_update_slice(&current_model.b1, gb1, &mut es_state.m_b1, &mut es_state.v_b1, t, es_lr),
        w2: adam_update_slice(&current_model.w2, gw2, &mut es_state.m_w2, &mut es_state.v_w2, t, es_lr),
        b2: adam_update_slice(&current_model.b2, gb2, &mut es_state.m_b2, &mut es_state.v_b2, t, es_lr),
        w3: adam_update_slice(&current_model.w3, gw3, &mut es_state.m_w3, &mut es_state.v_w3, t, es_lr),
        b3: adam_update_scalar(current_model.b3, gb3, &mut es_state.m_b3, &mut es_state.v_b3, t, es_lr),
    };

    (new_model, best_score)
}

// ──────────────────────────────────────────────────────────────────────────────
// Public training entry point
// ──────────────────────────────────────────────────────────────────────────────

pub fn train_self_play_models(config: &TrainingConfig) -> Result<(SelfPlayModels, TrainingSummary), String> {
    fs::create_dir_all(&config.output_dir).map_err(|err| err.to_string())?;

    #[cfg(debug_assertions)]
    {
        println!("Debug build detected. For much faster training, prefer: cargo run --release --bin ml_self_play -- train ...");
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let fixed_evader_only_searchers = build_fixed_evader_training_searchers();
    let mut models = if let Some(resume_path) = &config.resume_from {
        load_model_bundle(resume_path)?
    } else {
        SelfPlayModels {
            evader: MlpPolicyModel::new_random("skoll", &mut rng),
            searcher: match config.training_mode {
                TrainingMode::CoAgent => SearcherMlpModel::new_random("hati", &mut rng),
                TrainingMode::Static => build_naive_walker_searcher(),
            },
        }
    };

    // Adam momentum states persist across generations — this is what makes ES accumulate
    // directed progress rather than random-walking around the fitness landscape.
    let mut evader_es  = EsState::for_evader(&models.evader);
    let mut searcher_es = EsState::for_searcher(&models.searcher);

    let initial_specs = build_episode_specs(
        config.episodes_per_eval,
        config.seed,
        config.min_nodes,
        config.max_nodes,
    );
    let mut current = match config.training_mode {
        TrainingMode::CoAgent => evaluate_pair_on_specs_mlp_searcher(
            &models.evader,
            &models.searcher,
            &initial_specs,
            config.max_attempts_factor,
            config.max_attempts_ratio,
            config.max_attempts_cap,
        ),
        TrainingMode::Static => evaluate_evader_against_fixed_searchers(
            &models.evader,
            &fixed_evader_only_searchers,
            &initial_specs,
            config.max_attempts_factor,
            config.max_attempts_ratio,
            config.max_attempts_cap,
        ),
    };

    println!(
        "Starting self-play training: generations={} population={} episodes_per_eval={} nodes={}..{} hall_of_fame={} seed={}",
        config.generations,
        config.population_size,
        config.episodes_per_eval,
        config.min_nodes,
        config.max_nodes,
        config.hall_of_fame_size,
        config.seed
    );
    println!("Accelerator: {}.", accelerator_description());
    println!(
        "Evader model: MLP ({}→{}→{}→1 ReLU), {} features. Searcher model: MLP ({}→{}→{}→1 ReLU), {} features.",
        EVADER_FEATURE_COUNT, EVADER_MLP_HIDDEN1, EVADER_MLP_HIDDEN2, EVADER_FEATURE_COUNT,
        SEARCHER_FEATURE_COUNT, SEARCHER_MLP_HIDDEN1, SEARCHER_MLP_HIDDEN2, SEARCHER_FEATURE_COUNT,
    );
    println!(
        "Training mode: {}.",
        match config.training_mode {
            TrainingMode::Static => "static (evader vs fixed search algorithms)",
            TrainingMode::CoAgent => "coagent (evader vs learned searcher)",
        }
    );
    println!(
        "Attempt budget: factor={} ratio={} cap={}",
        config.max_attempts_factor,
        config
            .max_attempts_ratio
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "none".to_string()),
        config
            .max_attempts_cap
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
    );
    if config.training_mode == TrainingMode::Static {
        println!(
            "Static opponents: {}.",
            fixed_searcher_role_names(&fixed_evader_only_searchers),
        );
        println!(
            "Static opponent sampling: {} per generation (from {} total fixed strategies).",
            config.static_opponent_sample_count.max(1).min(fixed_evader_only_searchers.len().max(1)),
            fixed_evader_only_searchers.len(),
        );
    }
    if let Some(resume_path) = &config.resume_from {
        println!("Resuming from model bundle: {}", resume_path.display());
    }
    println!(
        "Searcher: reward shaping (proximity + coverage bonuses) restores gradient signal on collapse."
    );
    println!(
        "Evader reward includes relocation-distance penalty (0.6 per hop / ln(n))."
    );
    match config.training_mode {
        TrainingMode::Static => println!(
            "Initial scores: opponent_searchers={:.2} evader={:.2} found_rate={:.3} avg_attempts={:.2}",
            current.average_searcher_reward,
            current.average_evader_reward,
            current.found_rate,
            current.average_attempts
        ),
        TrainingMode::CoAgent => println!(
            "Initial scores: searcher={:.2} evader={:.2} found_rate={:.3} avg_attempts={:.2}",
            current.average_searcher_reward,
            current.average_evader_reward,
            current.found_rate,
            current.average_attempts
        ),
    }
    let _ = io::stdout().flush();

    let history_path = config.output_dir.join("training_history.json");
    let mut history: Vec<GenerationRecord> = Vec::new();
    history.push(GenerationRecord {
        generation: 0,
        searcher_score: current.average_searcher_reward,
        evader_score: current.average_evader_reward,
        found_rate: current.found_rate,
        avg_attempts: current.average_attempts,
        static_evader_score: None,
        static_found_rate: None,
    });

    let hall_size = config.hall_of_fame_size;
    let snapshot_interval = if hall_size == 0 { usize::MAX } else { (config.generations / hall_size.max(1)).max(1) };
    let mut evader_hall: Vec<MlpPolicyModel> = Vec::new();
    let mut searcher_hall: Vec<SearcherMlpModel> = Vec::new();

    for generation in 0..config.generations {
        let generation_seed = config.seed ^ generation as u64 ^ 0xA11CE;
        let episode_specs = build_episode_specs(
            config.episodes_per_eval,
            generation_seed,
            config.min_nodes,
            config.max_nodes,
        );

        // Snapshot current models into the hall of fame at regular intervals.
        if generation > 0 && generation % snapshot_interval == 0 {
            evader_hall.push(models.evader.clone());
            if config.training_mode == TrainingMode::CoAgent {
                searcher_hall.push(models.searcher.clone());
            }
            if evader_hall.len() > hall_size { evader_hall.remove(0); }
            if searcher_hall.len() > hall_size { searcher_hall.remove(0); }
        }

        // Build opponent pools: current opponent + a random sample from the hall.
        let sample = config.hall_sample_count;
        let mut pool_rng = StdRng::seed_from_u64(generation_seed ^ 0xC0FFEE);

        let mut evader_pool: Vec<MlpPolicyModel> = vec![models.evader.clone()];
        if !evader_hall.is_empty() && sample > 0 {
            let mut indices: Vec<usize> = (0..evader_hall.len()).collect();
            indices.sort_by_key(|_| pool_rng.gen::<u64>());
            for i in indices.into_iter().take(sample) {
                evader_pool.push(evader_hall[i].clone());
            }
        }

        let searcher_pool: Vec<SearcherMlpModel> = match config.training_mode {
            TrainingMode::CoAgent => {
                let mut pool: Vec<SearcherMlpModel> = vec![models.searcher.clone()];
                if !searcher_hall.is_empty() && sample > 0 {
                    let mut indices: Vec<usize> = (0..searcher_hall.len()).collect();
                    indices.sort_by_key(|_| pool_rng.gen::<u64>());
                    for i in indices.into_iter().take(sample) {
                        pool.push(searcher_hall[i].clone());
                    }
                }
                // Include fixed-algorithm opponents so the evader maintains resistance to
                // simple strategies while co-evolving against the ML searcher.
                pool.extend(sample_fixed_searchers(&fixed_evader_only_searchers, config.static_opponent_sample_count, &mut pool_rng));
                pool
            }
            TrainingMode::Static => sample_fixed_searchers(
                &fixed_evader_only_searchers,
                config.static_opponent_sample_count,
                &mut pool_rng,
            ),
        };

        let (best_evader, _evader_score) = optimize_evader(
            &models.evader,
            &mut evader_es,
            config.es_lr,
            &searcher_pool,
            config.population_size,
            config.mutation_scale,
            &episode_specs,
            config.max_attempts_factor,
            config.max_attempts_ratio,
            config.max_attempts_cap,
        );
        models.evader = best_evader;

        if config.training_mode == TrainingMode::CoAgent {
            let (best_searcher, _searcher_score) = optimize_searcher_mlp(
                &models.searcher,
                &mut searcher_es,
                config.es_lr,
                &evader_pool,
                config.population_size,
                config.mutation_scale,
                &episode_specs,
                config.max_attempts_factor,
                config.max_attempts_ratio,
                config.max_attempts_cap,
            );
            models.searcher = best_searcher;
        }

        current = match config.training_mode {
            TrainingMode::CoAgent => evaluate_pair_on_specs_mlp_searcher(
                &models.evader,
                &models.searcher,
                &episode_specs,
                config.max_attempts_factor,
                config.max_attempts_ratio,
                config.max_attempts_cap,
            ),
            TrainingMode::Static => evaluate_evader_against_fixed_searchers(
                &models.evader,
                &fixed_evader_only_searchers,
                &episode_specs,
                config.max_attempts_factor,
                config.max_attempts_ratio,
                config.max_attempts_cap,
            ),
        };

        // In coagent mode, also evaluate against static searchers to track hybrid performance.
        let static_eval = if config.training_mode == TrainingMode::CoAgent {
            Some(evaluate_evader_against_fixed_searchers(
                &models.evader,
                &fixed_evader_only_searchers,
                &episode_specs,
                config.max_attempts_factor,
                config.max_attempts_ratio,
                config.max_attempts_cap,
            ))
        } else {
            None
        };

        match config.training_mode {
            TrainingMode::Static => println!(
                "Generation {}/{}: opponent_searchers={:.2} evader={:.2} found_rate={:.3} avg_attempts={:.2}",
                generation + 1,
                config.generations,
                current.average_searcher_reward,
                current.average_evader_reward,
                current.found_rate,
                current.average_attempts
            ),
            TrainingMode::CoAgent => println!(
                "Generation {}/{}: ml_searcher={:.2} evader={:.2} found_rate={:.3} static_evader={:.2} static_found={:.3}",
                generation + 1,
                config.generations,
                current.average_searcher_reward,
                current.average_evader_reward,
                current.found_rate,
                static_eval.as_ref().map(|e| e.average_evader_reward).unwrap_or(0.0),
                static_eval.as_ref().map(|e| e.found_rate).unwrap_or(0.0),
            ),
        }
        let _ = io::stdout().flush();

        history.push(GenerationRecord {
            generation: generation + 1,
            searcher_score: current.average_searcher_reward,
            evader_score: current.average_evader_reward,
            found_rate: current.found_rate,
            avg_attempts: current.average_attempts,
            static_evader_score: static_eval.as_ref().map(|e| e.average_evader_reward),
            static_found_rate: static_eval.as_ref().map(|e| e.found_rate),
        });
        if let Ok(json) = serde_json::to_string_pretty(&history) {
            let _ = fs::write(&history_path, json);
        }
    }

    let bundle_path = config.output_dir.join("self_play_models.json");
    let searcher_path = config.output_dir.join("searcher_model.json");
    let evader_path = config.output_dir.join("evader_model.json");

    let serialized_bundle = serde_json::to_string_pretty(&models).map_err(|err| err.to_string())?;
    fs::write(bundle_path, serialized_bundle).map_err(|err| err.to_string())?;
    models.searcher.save_json(searcher_path)?;
    models.evader.save_json(evader_path)?;

    Ok((
        models,
        TrainingSummary {
            generations: config.generations,
            final_searcher_score: current.average_searcher_reward,
            final_evader_score: current.average_evader_reward,
            seed: config.seed,
        },
    ))
}

// ──────────────────────────────────────────────────────────────────────────────
// Public evaluation and loading helpers
// ──────────────────────────────────────────────────────────────────────────────

pub fn load_model_bundle(path: impl AsRef<Path>) -> Result<SelfPlayModels, String> {
    let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
    serde_json::from_str(&text).map_err(|err| err.to_string())
}

pub fn evaluate_model_bundle(
    models: &SelfPlayModels,
    episodes: usize,
    seed: u64,
    min_nodes: i32,
    max_nodes: i32,
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
) -> EvaluationSummary {
    let episode_specs = build_episode_specs(episodes, seed, min_nodes, max_nodes);
    evaluate_pair_on_specs_mlp_searcher(
        &models.evader,
        &models.searcher,
        &episode_specs,
        max_attempts_factor,
        max_attempts_ratio,
        max_attempts_cap,
    )
}

/// Helper: build the passthrough second layer for hand-coded searchers.
/// Layer 1 (hidden1 neurons) all compute the same strategic score V.
/// Layer 2 averages hidden1 outputs so each h2 neuron ≈ V.
/// Output sums h2 neurons × (1/hidden2) to recover the original V magnitude.
fn fixed_searcher_shell(role: &str, h1: usize, h2: usize) -> SearcherMlpModel {
    let fwd = 1.0 / h1 as f64;
    SearcherMlpModel {
        role: role.to_string(),
        feature_names: searcher_feature_names(),
        input_dim: SEARCHER_FEATURE_COUNT,
        hidden1_dim: h1,
        hidden2_dim: h2,
        w1: vec![0.0; SEARCHER_FEATURE_COUNT * h1],
        b1: vec![0.0; h1],
        w2: vec![fwd; h2 * h1],   // each h2 neuron = mean(h1)
        b2: vec![0.0; h2],
        w3: vec![1.0 / h2 as f64; h2], // scale output back to ~V
        b3: 0.0,
    }
}

pub fn build_breadth_first_searcher() -> SearcherMlpModel {
    let h1 = SEARCHER_MLP_HIDDEN1;
    let h2 = SEARCHER_MLP_HIDDEN2;
    let mut model = fixed_searcher_shell("breadth_first", h1, h2);

    const DEPTH_IDX: usize = 0;
    const ENCOUNTER_IDX: usize = 4;
    const RECENT_FLAG_IDX: usize = 8;

    for h in 0..h1 {
        let row = h * SEARCHER_FEATURE_COUNT;
        model.w1[row + DEPTH_IDX] = -6.0;
        model.w1[row + ENCOUNTER_IDX] = -2.5;
        model.w1[row + RECENT_FLAG_IDX] = -4.0;
        model.b1[h] = 1.0;
    }
    model
}

pub fn build_depth_first_searcher() -> SearcherMlpModel {
    let h1 = SEARCHER_MLP_HIDDEN1;
    let h2 = SEARCHER_MLP_HIDDEN2;
    let mut model = fixed_searcher_shell("depth_first_preorder", h1, h2);

    const ENCOUNTER_IDX: usize = 4;
    const RECENT_FLAG_IDX: usize = 8;

    for h in 0..h1 {
        let row = h * SEARCHER_FEATURE_COUNT;
        model.w1[row + ENCOUNTER_IDX] = -6.0;
        model.w1[row + RECENT_FLAG_IDX] = -4.0;
        model.b1[h] = 1.0;
    }
    model
}

pub fn build_deepest_first_searcher() -> SearcherMlpModel {
    let h1 = SEARCHER_MLP_HIDDEN1;
    let h2 = SEARCHER_MLP_HIDDEN2;
    let mut model = fixed_searcher_shell("deepest_first", h1, h2);

    const DEPTH_IDX: usize = 0;
    const RECENT_FLAG_IDX: usize = 8;

    for h in 0..h1 {
        let row = h * SEARCHER_FEATURE_COUNT;
        model.w1[row + DEPTH_IDX] = 6.0;
        model.w1[row + RECENT_FLAG_IDX] = -4.0;
        model.b1[h] = 0.5;
    }
    model
}

/// Ascending key-order searcher: guesses keys 1, 2, 3 … in order.
/// Mirrors the visualizer's `SearchStrategy::Ascending`.
pub fn build_ascending_searcher() -> SearcherMlpModel {
    let h1 = SEARCHER_MLP_HIDDEN1;
    let h2 = SEARCHER_MLP_HIDDEN2;
    let mut model = fixed_searcher_shell("ascending", h1, h2);

    const KEY_IDX: usize = 3;
    const RECENT_FLAG_IDX: usize = 8;

    for h in 0..h1 {
        let row = h * SEARCHER_FEATURE_COUNT;
        model.w1[row + KEY_IDX] = -7.0;         // prefer lowest key
        model.w1[row + RECENT_FLAG_IDX] = -4.0; // avoid already-guessed
        model.b1[h] = 0.5;
    }
    model
}

/// Evasion-aware searcher: mirrors the visualizer's `SearchStrategy::EvasionAware`.
/// Scores nodes by depth*10 + min_path_distance_from_recent_guesses*6,
/// preferring deep nodes that are far from where it has already looked.
pub fn build_evasion_aware_searcher() -> SearcherMlpModel {
    let h1 = SEARCHER_MLP_HIDDEN1;
    let h2 = SEARCHER_MLP_HIDDEN2;
    let mut model = fixed_searcher_shell("evasion_aware", h1, h2);

    const DEPTH_IDX: usize = 0;
    const MIN_DIST_IDX: usize = 6;
    const RECENT_FLAG_IDX: usize = 8;

    for h in 0..h1 {
        let row = h * SEARCHER_FEATURE_COUNT;
        model.w1[row + DEPTH_IDX] = 10.0;
        model.w1[row + MIN_DIST_IDX] = 6.0;
        model.w1[row + RECENT_FLAG_IDX] = -5.0;
        model.b1[h] = 0.5;
    }
    model
}

/// Neighbor-hunter: targets nodes CLOSE to recent guesses.
/// Directly counters the evader strategy of hiding far from recent activity.
pub fn build_neighbor_hunter_searcher() -> SearcherMlpModel {
    let h1 = SEARCHER_MLP_HIDDEN1;
    let h2 = SEARCHER_MLP_HIDDEN2;
    let mut model = fixed_searcher_shell("neighbor_hunter", h1, h2);

    const MIN_DIST_IDX: usize = 6;
    const AVG_DIST_IDX: usize = 7;
    const RECENT_FLAG_IDX: usize = 8;

    for h in 0..h1 {
        let row = h * SEARCHER_FEATURE_COUNT;
        model.w1[row + MIN_DIST_IDX] = -8.0;  // prefer nodes close to recent guesses
        model.w1[row + AVG_DIST_IDX] = -4.0;
        model.w1[row + RECENT_FLAG_IDX] = -5.0;
        model.b1[h] = 1.0;
    }
    model
}

/// Subtree-sweeper: clears subtrees systematically by preferring nodes with the most
/// unguessed descendants. Harder to evade than single-feature heuristics.
pub fn build_subtree_sweeper_searcher() -> SearcherMlpModel {
    let h1 = SEARCHER_MLP_HIDDEN1;
    let h2 = SEARCHER_MLP_HIDDEN2;
    let mut model = fixed_searcher_shell("subtree_sweeper", h1, h2);

    const SUBTREE_UNGUESSED_IDX: usize = 11;
    const FRACTION_GUESSED_IDX: usize = 10;
    const RECENT_FLAG_IDX: usize = 8;

    for h in 0..h1 {
        let row = h * SEARCHER_FEATURE_COUNT;
        model.w1[row + SUBTREE_UNGUESSED_IDX] = 9.0;   // prefer large unguessed subtrees
        model.w1[row + FRACTION_GUESSED_IDX] = -5.0;   // avoid already-covered areas
        model.w1[row + RECENT_FLAG_IDX] = -4.0;
        model.b1[h] = 0.5;
    }
    model
}

pub fn build_fixed_evader_training_searchers() -> Vec<SearcherMlpModel> {
    vec![
        build_ascending_searcher(),
        build_breadth_first_searcher(),
        build_depth_first_searcher(),
        build_deepest_first_searcher(),
        build_evasion_aware_searcher(),
        build_neighbor_hunter_searcher(),
        build_subtree_sweeper_searcher(),
    ]
}

/// Build a naïve deterministic walking searcher (preorder by encounter index).
/// Scores favour low encounter_index_norm and unguessed nodes.
pub fn build_naive_walker_searcher() -> SearcherMlpModel {
    let h1 = SEARCHER_MLP_HIDDEN1;
    let h2 = SEARCHER_MLP_HIDDEN2;
    let mut model = fixed_searcher_shell("naive_walker", h1, h2);

    const ENCOUNTER_IDX: usize = 4;
    const RECENT_FLAG_IDX: usize = 8;

    for h in 0..h1 {
        let row = h * SEARCHER_FEATURE_COUNT;
        model.w1[row + ENCOUNTER_IDX] = -5.0;
        model.w1[row + RECENT_FLAG_IDX] = -3.0;
        model.b1[h] = 0.5;
    }
    model
}

/// Evaluate the trained evader against the naïve walking searcher.
pub fn evaluate_evader_vs_naive(
    models: &SelfPlayModels,
    episodes: usize,
    seed: u64,
    min_nodes: i32,
    max_nodes: i32,
    max_attempts_factor: usize,
) -> EvaluationSummary {
    let episode_specs = build_episode_specs(episodes, seed, min_nodes, max_nodes);
    let naive = build_naive_walker_searcher();
    evaluate_pair_on_specs_mlp_searcher(
        &models.evader,
        &naive,
        &episode_specs,
        max_attempts_factor,
        None,
        None,
    )
}
