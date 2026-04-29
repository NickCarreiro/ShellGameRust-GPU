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
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
    Mutex, MutexGuard, OnceLock,
};
use std::time::Instant;

use crate::tree::NodeSnapshot;

// ──────────────────────────────────────────────────────────────────────────────
// Global candle device — initialised once, shared across all Rayon threads.
// ──────────────────────────────────────────────────────────────────────────────

static CANDLE_DEVICE: OnceLock<Device> = OnceLock::new();
static CANDLE_CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn candle_device() -> &'static Device {
    CANDLE_DEVICE.get_or_init(|| {
        if force_cpu_backend() {
            eprintln!(
                "[skoll/hati] CPU backend forced by environment — CUDA will not be initialized."
            );
            return Device::Cpu;
        }

        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(0) {
                Ok(d) => {
                    eprintln!("[skoll/hati] CUDA device 0 ready — GPU batch inference active.");
                    d
                }
                Err(e) => {
                    let strict = std::env::var("REQUIRE_CUDA")
                        .map(|value| {
                            let norm = value.trim().to_ascii_lowercase();
                            matches!(norm.as_str(), "1" | "true" | "yes" | "on")
                        })
                        .unwrap_or(false);
                    if strict {
                        panic!(
                            "[skoll/hati] CUDA init failed ({e}) and REQUIRE_CUDA is set; refusing CPU fallback."
                        );
                    }
                    eprintln!(
                        "[skoll/hati] WARNING: CUDA init failed ({e}). Falling back to Candle CPU backend. \
Set REQUIRE_CUDA=1 to enforce hard failure."
                    );
                    Device::Cpu
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

fn force_cpu_backend() -> bool {
    parse_env_bool("SHELLGAME_FORCE_CPU")
        .or_else(|| parse_env_bool("FORCE_CPU"))
        .unwrap_or(false)
}

fn accelerator_description() -> &'static str {
    if force_cpu_backend() {
        return "CPU forced by SHELLGAME_FORCE_CPU/FORCE_CPU; CUDA will not be initialized";
    }
    if candle_device().is_cuda() {
        "CUDA enabled: batched MLP inference runs on GPU; tree simulation/evolution still runs on CPU"
    } else {
        #[cfg(feature = "cuda")]
        {
            "CUDA feature enabled, but runtime CUDA unavailable: using Candle CPU backend"
        }
        #[cfg(not(feature = "cuda"))]
        {
            "CUDA disabled: batched MLP inference is running on CPU"
        }
    }
}

/// Returns true only when the training/inference backend is actually CUDA.
///
/// This intentionally goes through the same global Candle device used by the
/// batched training paths, so tests catch accidental CPU fallbacks.
pub fn training_accelerator_is_cuda() -> bool {
    candle_device().is_cuda()
}

fn cuda_serialization_enabled() -> bool {
    parse_env_bool("SHELLGAME_SERIALIZE_CUDA")
        .or_else(|| parse_env_bool("SERIALIZE_CUDA"))
        .unwrap_or(true)
}

fn cuda_work_guard() -> Option<MutexGuard<'static, ()>> {
    if candle_device().is_cuda() && cuda_serialization_enabled() {
        Some(
            CANDLE_CUDA_LOCK
                .get_or_init(|| Mutex::new(()))
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    } else {
        None
    }
}

fn parse_env_bool(name: &str) -> Option<bool> {
    std::env::var(name).ok().and_then(|raw| {
        let value = raw.trim().to_ascii_lowercase();
        match value.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn training_heartbeat_enabled() -> bool {
    static HEARTBEAT: OnceLock<bool> = OnceLock::new();
    *HEARTBEAT.get_or_init(|| {
        if let Some(force) = parse_env_bool("TRAIN_HEARTBEAT") {
            return force;
        }
        !training_accelerator_is_cuda()
    })
}

fn should_emit_attempt_heartbeat(attempt: usize, max_attempts: usize, interval: usize) -> bool {
    attempt == 1 || attempt == max_attempts || attempt % interval.max(1) == 0
}

static TRAINING_STOP_REQUESTED: AtomicBool = AtomicBool::new(false);
static TRAINING_INTERRUPT_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRAINING_INTERRUPT_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();

fn install_training_interrupt_handler() -> Result<(), String> {
    TRAINING_STOP_REQUESTED.store(false, AtomicOrdering::SeqCst);
    TRAINING_INTERRUPT_COUNT.store(0, AtomicOrdering::SeqCst);
    TRAINING_INTERRUPT_HANDLER
        .get_or_init(|| {
            ctrlc::set_handler(|| {
                let previous = TRAINING_INTERRUPT_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
                TRAINING_STOP_REQUESTED.store(true, AtomicOrdering::SeqCst);
                if previous == 0 {
                    eprintln!(
                        "\nGraceful stop requested. Training will finish the current generation, save the best model, and exit cleanly."
                    );
                } else {
                    eprintln!(
                        "\nGraceful stop is already pending. Waiting for the current generation checkpoint boundary."
                    );
                }
            })
            .map_err(|err| err.to_string())
        })
        .clone()
}

fn training_stop_requested() -> bool {
    TRAINING_STOP_REQUESTED.load(AtomicOrdering::SeqCst)
}

// Feature counts
/// Searcher features: 10 original + 4 strategic features (fraction_guessed,
/// subtree_unguessed_norm, is_pivot_norm, unguessed_sibling_norm).
pub const SEARCHER_FEATURE_COUNT: usize = 14;
/// Evader features: 10 original + subtree_size_norm + cold_subtree_flag + recent_same_depth_norm.
pub const EVADER_FEATURE_COUNT: usize = 14;

/// Default cap for candidate-node cells fed to one batched GPU MLP call.
/// The crash log showed an NVIDIA Xid 31 MMU fault after adaptive growth reached
/// population=100.  Keeping `candidates * rows` bounded avoids very large 3-D
/// batched matmuls that can poison the CUDA context on this laptop GPU.
const DEFAULT_GPU_SCORE_BATCH_CELLS: usize = 4_096;
/// Default cap for rows fed to one single-model GPU MLP call.
/// Saved-opponent relocation can create hundreds of thousands of rows at high
/// adaptive populations; chunking prevents transient hidden activations from
/// exceeding VRAM while keeping CUDA enabled.
const DEFAULT_GPU_SINGLE_SCORE_ROWS: usize = 2_048;

fn gpu_score_batch_rows(candidate_count: usize) -> usize {
    if let Some(rows) = parse_env_usize("GPU_SCORE_BATCH_ROWS") {
        return rows;
    }
    let cells = parse_env_usize("GPU_SCORE_BATCH_CELLS")
        .unwrap_or(DEFAULT_GPU_SCORE_BATCH_CELLS);
    (cells / candidate_count.max(1)).max(1)
}

fn gpu_single_score_rows() -> usize {
    parse_env_usize("GPU_SINGLE_SCORE_ROWS").unwrap_or(DEFAULT_GPU_SINGLE_SCORE_ROWS)
}

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
        let _cuda_guard = cuda_work_guard();
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
        let _cuda_guard = cuda_work_guard();
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

fn write_json_pretty_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(value).map_err(|err| err.to_string())?;
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("invalid checkpoint path: {}", path.display()))?
        .to_string_lossy();
    let tmp_path = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));

    fs::write(&tmp_path, serialized).map_err(|err| err.to_string())?;
    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        err.to_string()
    })
}

fn write_recovery_checkpoint(
    output_dir: &Path,
    models: &SelfPlayModels,
    best_evader_model: &MlpPolicyModel,
    best_searcher_model: Option<&SearcherMlpModel>,
) -> Result<(), String> {
    write_json_pretty_atomic(&output_dir.join("self_play_models.json"), models)?;
    write_json_pretty_atomic(&output_dir.join("evader_model.json"), &models.evader)?;
    write_json_pretty_atomic(&output_dir.join("searcher_model.json"), &models.searcher)?;
    write_json_pretty_atomic(&output_dir.join("best_evader_model.json"), best_evader_model)?;
    if let Some(best_searcher_model) = best_searcher_model {
        write_json_pretty_atomic(
            &output_dir.join("best_searcher_model.json"),
            best_searcher_model,
        )?;
    }
    Ok(())
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
    /// Multiplier applied to `es_lr` for the co-evolving searcher.  Keeping the
    /// searcher slower prevents it from sprinting past the evader and flattening
    /// the useful learning signal.
    pub searcher_lr_scale: f64,
    /// Update the co-evolving searcher only every N generations.  Values <= 1
    /// update every generation.
    pub searcher_update_interval: usize,
    /// Roll back a searcher update if the learned matchup found-rate rises above
    /// this cap while the evader is still adapting.
    pub searcher_max_found_rate: f64,
    /// Roll back a searcher update if found-rate jumps by more than this amount
    /// over the pre-searcher-update baseline for the same generation.
    pub searcher_max_found_rate_jump: f64,
    /// Stop training early when evader score does not improve for this many consecutive
    /// generations. `None` disables early stopping (train for the full `generations` count).
    pub patience: Option<usize>,
    /// Grow the curriculum after this many consecutive generations without evader improvement.
    /// `None` disables adaptive growth; `Some(0)` is treated as disabled.
    pub stagnation_grow_after: Option<usize>,
    /// Node-count increase applied to both min_nodes and max_nodes on each growth event.
    pub stagnation_node_step: i32,
    /// Population-size increase applied on each growth event.
    pub stagnation_population_step: usize,
    /// Optional upper bound for adaptively grown node counts.
    pub stagnation_max_nodes_cap: Option<i32>,
    /// Optional upper bound for adaptively grown population size.
    pub stagnation_population_cap: Option<usize>,
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
    pub final_escape_score: f64,
    /// Best evader score seen across all generations (may differ from final if training overshot).
    pub best_evader_score: f64,
    /// Promotion score used to select the best evader. This prioritizes low found-rate,
    /// then high budget usage, then raw evader reward as a small tie-breaker.
    pub best_evader_selection_score: f64,
    /// Generation at which best_evader_score was achieved (0 = before training).
    pub best_generation: usize,
    /// True when training ended due to patience exhaustion rather than reaching `generations`.
    pub stopped_early: bool,
    /// True when the user requested a graceful stop with Ctrl+C.
    pub interrupted: bool,
    /// Best searcher score seen across all generations (CoAgent mode only; 0 otherwise).
    pub best_searcher_score: f64,
    pub best_searcher_generation: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub episodes: usize,
    pub found_rate: f64,
    pub average_attempts: f64,
    pub average_max_attempts: f64,
    pub survival_budget_ratio: f64,
    pub escape_quality_score: f64,
    pub average_searcher_reward: f64,
    pub average_evader_reward: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerationRecord {
    generation: usize,
    searcher_score: f64,
    evader_score: f64,
    escape_score: f64,
    found_rate: f64,
    avg_attempts: f64,
    avg_max_attempts: f64,
    survival_budget_ratio: f64,
    /// CoAgent mode only: evader score when evaluated against the fixed static searchers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    static_evader_score: Option<f64>,
    /// CoAgent mode only: escape-quality score against the fixed static searchers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    static_escape_score: Option<f64>,
    /// CoAgent mode only: found rate when evaluated against the fixed static searchers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    static_found_rate: Option<f64>,
    /// CoAgent mode only: survival-budget ratio against the fixed static searchers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    static_survival_budget_ratio: Option<f64>,
    /// Std dev of evader fitness across the ES population; near-zero signals gradient collapse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    evader_fitness_std: Option<f64>,
    /// Std dev of searcher fitness across the ES population (CoAgent mode only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    searcher_fitness_std: Option<f64>,
    /// Active ES population size for this generation.
    population_size: usize,
    /// Active minimum node count for this generation's episode specs.
    min_nodes: i32,
    /// Active maximum node count for this generation's episode specs.
    max_nodes: i32,
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

const MISSING_NODE_INDEX: usize = usize::MAX;

fn build_key_to_index(all_nodes: &[NodeMeta]) -> (Vec<usize>, usize) {
    let max_key = all_nodes
        .iter()
        .map(|item| item.key)
        .max()
        .unwrap_or(1)
        .max(0) as usize;
    let mut key_to_index = vec![MISSING_NODE_INDEX; max_key + 1];
    for (idx, node) in all_nodes.iter().enumerate() {
        if node.key >= 0 {
            let key = node.key as usize;
            if key >= key_to_index.len() {
                key_to_index.resize(key + 1, MISSING_NODE_INDEX);
            }
            key_to_index[key] = idx;
        }
    }
    (key_to_index, max_key)
}

fn lookup_key_index(key_to_index: &[usize], key: i32) -> Option<usize> {
    if key < 0 {
        return None;
    }
    key_to_index
        .get(key as usize)
        .copied()
        .filter(|&index| index != MISSING_NODE_INDEX)
}

fn recent_indices_from_keys(key_to_index: &[usize], recent_guesses: &[i32]) -> Vec<usize> {
    recent_guesses
        .iter()
        .filter_map(|&guess| lookup_key_index(key_to_index, guess))
        .collect()
}

fn recent_distance_stats_from_indices(
    all_nodes: &[NodeMeta],
    recent_indices: &[usize],
    node: &NodeMeta,
) -> (usize, f64, f64) {
    if recent_indices.is_empty() {
        let fallback = node.depth + 1;
        return (fallback, fallback as f64, fallback as f64);
    }

    let mut min_distance = usize::MAX;
    let mut total_distance = 0.0;
    for &recent_idx in recent_indices {
        let recent = &all_nodes[recent_idx];
        let distance = path_distance(
            node.path_bits,
            node.path_len,
            recent.path_bits,
            recent.path_len,
        );
        min_distance = min_distance.min(distance);
        total_distance += distance as f64;
    }

    let last_distance = recent_indices
        .last()
        .map(|&recent_idx| {
            let recent = &all_nodes[recent_idx];
            path_distance(
                node.path_bits,
                node.path_len,
                recent.path_bits,
                recent.path_len,
            ) as f64
        })
        .unwrap_or(node.depth as f64 + 1.0);

    (
        min_distance,
        total_distance / recent_indices.len().max(1) as f64,
        last_distance,
    )
}

struct FeatureContext<'a> {
    all_nodes: &'a [NodeMeta],
    node_count_f: f64,
    max_key_f: f64,
    recent_guess_count: usize,
    recent_indices: Vec<usize>,
    recent_key_present: Vec<bool>,
    recent_in_subtree: Vec<bool>,
    subtree_unguessed: Vec<usize>,
    total_unguessed: usize,
}

impl<'a> FeatureContext<'a> {
    fn new(all_nodes: &'a [NodeMeta], recent_guesses: &[i32]) -> Self {
        let node_count = all_nodes.len();
        let (key_to_index, max_key) = build_key_to_index(all_nodes);
        let recent_indices = recent_indices_from_keys(&key_to_index, recent_guesses);
        let mut recent_key_present = vec![false; node_count];
        for &index in &recent_indices {
            recent_key_present[index] = true;
        }

        let mut recent_in_subtree = vec![false; node_count];
        let mut subtree_unguessed = vec![0usize; node_count];
        for (idx, node) in all_nodes.iter().enumerate() {
            let start = node.encounter_index.min(node_count);
            let end = start.saturating_add(node.subtree_size).min(node_count);
            let mut unguessed = 0usize;
            let mut has_recent = false;
            for child_idx in start..end {
                if recent_key_present[child_idx] {
                    has_recent = true;
                } else {
                    unguessed += 1;
                }
            }
            recent_in_subtree[idx] = has_recent;
            subtree_unguessed[idx] = unguessed;
        }

        let total_unguessed = recent_key_present.iter().filter(|&&seen| !seen).count();

        Self {
            all_nodes,
            node_count_f: node_count.max(1) as f64,
            max_key_f: max_key.max(1) as f64,
            recent_guess_count: recent_guesses.len(),
            recent_indices,
            recent_key_present,
            recent_in_subtree,
            subtree_unguessed,
            total_unguessed,
        }
    }

    fn recent_distance_stats(&self, node: &NodeMeta) -> (usize, f64, f64) {
        recent_distance_stats_from_indices(self.all_nodes, &self.recent_indices, node)
    }

    fn node_index(&self, node: &NodeMeta) -> usize {
        node.encounter_index.min(self.all_nodes.len().saturating_sub(1))
    }
}

fn node_index_by_key(all_nodes: &[NodeMeta], key: i32) -> Option<usize> {
    all_nodes.iter().position(|node| node.key == key)
}

fn build_searcher_features_with_context(node: &NodeMeta, context: &FeatureContext<'_>) -> [f64; SEARCHER_FEATURE_COUNT] {
    let node_count = context.node_count_f;
    let max_key = context.max_key_f;
    let (min_recent_distance, avg_recent_distance, last_guess_distance) =
        context.recent_distance_stats(node);

    // ── 4 search-strategy features ──────────────────────────────────────
    //
    // Feature 10: fraction_guessed
    //   What proportion of all tree nodes have been guessed so far?
    //   Range [0, 1]. Gives the searcher a "how far along?" signal so it can
    //   modulate urgency (e.g. be more aggressive at covering leaves when
    //   budget is running out).
    let fraction_guessed = (context.recent_guess_count as f64 / node_count).min(1.0);

    // Feature 11: subtree_unguessed_norm
    //   Number of not-yet-guessed nodes inside this node's subtree,
    //   normalised by tree size.  A high value means "lots of unexplored
    //   territory under this node" — guessing here covers maximum ground.
    //   This is the key strategic signal for systematic search.
    let node_index = context.node_index(node);
    let subtree_unguessed = context.subtree_unguessed[node_index];
    let subtree_unguessed_norm = subtree_unguessed as f64 / node_count;

    // Feature 12: is_pivot_norm
    //   A "pivot" node divides the remaining unguessed tree into approximately
    //   equal halves — the binary-search ideal.  We measure this as
    //   1 - |0.5 - (unguessed_in_subtree / total_unguessed)| * 2,
    //   so 1.0 means perfect half-split and 0.0 means this node is a leaf
    //   with respect to unguessed nodes.
    let total_unguessed = context.total_unguessed;
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
        if context.recent_key_present[node_index] { 1.0 } else { 0.0 },
        last_guess_distance / (node_count + 1.0),
        fraction_guessed,
        subtree_unguessed_norm,
        is_pivot_norm,
        unguessed_sibling_norm,
    ]
}

fn build_evader_features_with_context(node: &NodeMeta, context: &FeatureContext<'_>) -> [f64; EVADER_FEATURE_COUNT] {
    let node_count = context.node_count_f;
    let max_key = context.max_key_f;
    let (min_recent_distance, avg_recent_distance, last_guess_distance) =
        context.recent_distance_stats(node);
    let node_index = context.node_index(node);

    // New feature 1: subtree_size_norm — what fraction of the tree lives under this node?
    let subtree_size_norm = node.subtree_size as f64 / node_count.max(1.0);

    // New feature 2: cold_subtree_flag — 1 if NO recent guess falls inside this subtree.
    let cold_subtree_flag = if context.recent_guess_count == 0 || !context.recent_in_subtree[node_index] {
        1.0
    } else {
        0.0
    };

    // New feature 3: recent_same_depth_norm — fraction of recent guesses at the same depth as this node.
    let same_depth = context
        .recent_indices
        .iter()
        .map(|&idx| &context.all_nodes[idx])
        .filter(|m| m.depth == node.depth)
        .count();
    let recent_same_depth_norm = if context.recent_guess_count == 0 {
        0.0
    } else {
        same_depth as f64 / context.recent_guess_count.max(1) as f64
    };

    // warm_zone_flag: 1 if this node is ≤1 tree-hop from any recent guess.
    // EvasionAware maximises depth + distance-from-recent, so it never looks here —
    // the evader can exploit this by hiding in the "warm zone" it ignores.
    let warm_zone_flag = if min_recent_distance <= 1 && context.recent_guess_count > 0 { 1.0 } else { 0.0 };

    [
        node.depth as f64 / node_count.max(1.0),
        if node.child_count == 0 { 1.0 } else { 0.0 },
        node.child_count as f64 / 2.0,
        node.key as f64 / max_key,
        node.encounter_index as f64 / node_count.max(1.0),
        node.path_len as f64 / (node_count + 1.0),
        min_recent_distance as f64 / (node_count + 1.0),
        avg_recent_distance / (node_count + 1.0),
        if context.recent_key_present[node_index] { 1.0 } else { 0.0 },
        last_guess_distance / (node_count + 1.0),
        subtree_size_norm,
        cold_subtree_flag,
        recent_same_depth_norm,
        warm_zone_flag,
    ]
}

fn append_searcher_feature_rows(out: &mut Vec<f64>, all_nodes: &[NodeMeta], recent_guesses: &[i32]) {
    let context = FeatureContext::new(all_nodes, recent_guesses);
    for node in all_nodes {
        out.extend_from_slice(&build_searcher_features_with_context(node, &context));
    }
}

fn append_evader_feature_rows(out: &mut Vec<f64>, all_nodes: &[NodeMeta], recent_guesses: &[i32]) {
    let context = FeatureContext::new(all_nodes, recent_guesses);
    for node in all_nodes {
        out.extend_from_slice(&build_evader_features_with_context(node, &context));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Node ranking helpers
// ──────────────────────────────────────────────────────────────────────────────

fn rank_nodes_evader(model: &MlpPolicyModel, all_nodes: &[NodeMeta], recent_guesses: &[i32]) -> Vec<(i32, f64)> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * EVADER_FEATURE_COUNT);
    append_evader_feature_rows(&mut feature_rows, all_nodes, recent_guesses);
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
    append_searcher_feature_rows(&mut feature_rows, all_nodes, recent_guesses);
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

fn survival_budget_ratio(average_attempts: f64, average_max_attempts: f64) -> f64 {
    if average_max_attempts <= 0.0 {
        0.0
    } else {
        (average_attempts / average_max_attempts).clamp(0.0, 1.0)
    }
}

fn escape_quality_score_from_parts(
    found_rate: f64,
    average_attempts: f64,
    average_max_attempts: f64,
    average_evader_reward: f64,
) -> f64 {
    let escape_rate = (1.0 - found_rate).clamp(0.0, 1.0);
    let budget_ratio = survival_budget_ratio(average_attempts, average_max_attempts);

    // Promotion score, not episode reward:
    //   1. escape_rate dominates, because the graph showed long but frequently caught runs;
    //   2. budget_ratio preserves the "make them spend the whole budget" behavior;
    //   3. raw reward is only a small tie-breaker for relocation/frontier quality.
    (escape_rate * 1_000.0) + (budget_ratio * 120.0) + (average_evader_reward * 0.05)
}

fn robust_evader_selection_score(current: &EvaluationSummary, static_eval: Option<&EvaluationSummary>) -> f64 {
    const STATIC_ROBUSTNESS_WEIGHT: f64 = 0.15;
    const STATIC_CREDIT_MARGIN: f64 = 200.0;

    if let Some(static_eval) = static_eval {
        // Static-script robustness is useful, but it must not rescue an evader
        // that the learned searcher immediately crushes. Cap static credit near
        // the learned-match score and penalize "found immediately" collapses.
        let static_credit = static_eval
            .escape_quality_score
            .min(current.escape_quality_score + STATIC_CREDIT_MARGIN);
        let learned_collapse_penalty =
            ((current.found_rate - 0.90).max(0.0) * 400.0)
                + ((0.35 - current.survival_budget_ratio).max(0.0) * 300.0);

        current.escape_quality_score * (1.0 - STATIC_ROBUSTNESS_WEIGHT)
            + static_credit * STATIC_ROBUSTNESS_WEIGHT
            - learned_collapse_penalty
    } else {
        current.escape_quality_score
    }
}

fn searcher_opponent_weight(searcher: &SearcherMlpModel) -> f64 {
    if searcher.role == "hati" {
        6.0
    } else {
        1.0
    }
}

fn frontier_exposure_from_rank(rank: usize) -> f64 {
    const IMMEDIATE_WINDOW: usize = 6;
    if rank >= IMMEDIATE_WINDOW {
        0.0
    } else {
        (IMMEDIATE_WINDOW - rank) as f64 / IMMEDIATE_WINDOW as f64
    }
}

fn add_frontier_order_exposure(scores: &mut [f64], ordered_indices: &[usize]) {
    for (rank, &node_index) in ordered_indices.iter().take(6).enumerate() {
        if let Some(score) = scores.get_mut(node_index) {
            *score += frontier_exposure_from_rank(rank);
        }
    }
}

fn frontier_exposure_scores(
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    guessed_keys: &HashSet<i32>,
) -> Vec<f64> {
    const ORDERING_COUNT: f64 = 5.0;

    let mut scores = vec![0.0; all_nodes.len()];
    let remaining: Vec<usize> = all_nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| !guessed_keys.contains(&node.key))
        .map(|(index, _)| index)
        .collect();
    if remaining.is_empty() {
        return scores;
    }

    let mut ordered = remaining.clone();
    ordered.sort_unstable_by_key(|&index| all_nodes[index].key);
    add_frontier_order_exposure(&mut scores, &ordered);

    ordered.clone_from(&remaining);
    ordered.sort_unstable_by_key(|&index| (all_nodes[index].depth, all_nodes[index].path_bits));
    add_frontier_order_exposure(&mut scores, &ordered);

    ordered.clone_from(&remaining);
    ordered.sort_unstable_by_key(|&index| all_nodes[index].encounter_index);
    add_frontier_order_exposure(&mut scores, &ordered);

    ordered.clone_from(&remaining);
    ordered.sort_unstable_by(|&left, &right| {
        all_nodes[right]
            .depth
            .cmp(&all_nodes[left].depth)
            .then_with(|| all_nodes[left].path_bits.cmp(&all_nodes[right].path_bits))
    });
    add_frontier_order_exposure(&mut scores, &ordered);

    let (key_to_index, _) = build_key_to_index(all_nodes);
    let recent_indices = recent_indices_from_keys(&key_to_index, recent_guesses);
    ordered.clone_from(&remaining);
    ordered.sort_unstable_by_key(|&index| {
        let node = &all_nodes[index];
        let min_recent_distance =
            recent_distance_stats_from_indices(all_nodes, &recent_indices, node).0;
        let score = ((node.depth as i64) * 10) + ((min_recent_distance as i64) * 6);
        (-score, node.key as i64)
    });
    add_frontier_order_exposure(&mut scores, &ordered);

    for score in &mut scores {
        *score /= ORDERING_COUNT;
    }
    scores
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
    max_attempts: usize,
    min_dist_to_shell: usize,
    unique_guesses: usize,
    node_count: usize,
) -> f64 {
    if found {
        // Any successful capture must beat any failure. Tempo is still rewarded,
        // but the searcher should never prefer "almost found" over "found late".
        let attempt_ratio = attempts as f64 / max_attempts.max(1) as f64;
        let tempo_bonus = (1.0 - attempt_ratio).clamp(0.0, 1.0) * 70.0;
        let diversity_bonus = 5.0 * (unique_guesses as f64 / (attempts as f64).max(1.0)).min(1.0);
        140.0 + tempo_bonus + diversity_bonus
    } else {
        let nc = node_count.max(1) as f64;
        // Use an approximate tree diameter so proximity retains signal as trees grow.
        // Normalizing by node_count made most "failed" episodes look similarly good/bad
        // on larger trees, reducing selection pressure.
        let approx_diameter = ((node_count.max(2) as f64).log2().ceil() * 2.0).max(2.0);

        // Proximity bonus: 0..40. Best when dist == 1 (one hop from shell).
        let proximity_bonus = if min_dist_to_shell == 0 {
            // Guard: found == true should have been caught above.
            40.0
        } else {
            let dist_from_near = (min_dist_to_shell as f64 - 1.0).max(0.0);
            let closeness = (1.0 - dist_from_near / approx_diameter).clamp(0.0, 1.0);
            40.0 * closeness
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
        // The graph showed too many "long but caught" runs. Preserve the gradient
        // for surviving longer, but make any capture much worse than a full escape.
        let capture_penalty = 95.0 + ((1.0 - survival_ratio).clamp(0.0, 1.0) * 75.0);
        (survival_ratio * 70.0) + early_survival_bonus - capture_penalty
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
    let run_one = |spec| {
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
    };
    let results: Vec<(bool, usize, f64, f64)> = if candle_device().is_cuda() {
        // Candle/cudarc CUDA objects are process-global enough that sharing them
        // across Rayon workers can poison the driver after a launch fault.  Keep
        // CUDA evaluation single-lane; CPU-only builds can still parallelize.
        episode_specs.iter().copied().filter_map(run_one).collect()
    } else {
        episode_specs.par_iter().copied().filter_map(run_one).collect()
    };

    let found_count = results.iter().filter(|(found, _, _, _)| *found).count();
    let total_attempts: usize = results.iter().map(|(_, attempts, _, _)| *attempts).sum();
    let total_max_attempts: usize = episode_specs
        .iter()
        .map(|spec| {
            compute_max_attempts(
                spec.node_count as usize,
                max_attempts_factor,
                max_attempts_ratio,
                max_attempts_cap,
            )
        })
        .sum();
    let total_searcher_reward: f64 = results.iter().map(|(_, _, reward, _)| *reward).sum();
    let total_evader_reward: f64 = results.iter().map(|(_, _, _, reward)| *reward).sum();

    let episode_count = episode_specs.len().max(1) as f64;
    let found_rate = found_count as f64 / episode_count;
    let average_attempts = total_attempts as f64 / episode_count;
    let average_max_attempts = total_max_attempts as f64 / episode_count;
    let average_evader_reward = total_evader_reward / episode_count;
    EvaluationSummary {
        episodes: episode_specs.len().max(1),
        found_rate,
        average_attempts,
        average_max_attempts,
        survival_budget_ratio: survival_budget_ratio(average_attempts, average_max_attempts),
        escape_quality_score: escape_quality_score_from_parts(
            found_rate,
            average_attempts,
            average_max_attempts,
            average_evader_reward,
        ),
        average_searcher_reward: total_searcher_reward / episode_count,
        average_evader_reward,
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
    let found_rate = total.iter().map(|e| e.found_rate).sum::<f64>() / denom;
    let average_attempts = total.iter().map(|e| e.average_attempts).sum::<f64>() / denom;
    let average_max_attempts = total.iter().map(|e| e.average_max_attempts).sum::<f64>() / denom;
    let average_evader_reward = total.iter().map(|e| e.average_evader_reward).sum::<f64>() / denom;
    EvaluationSummary {
        episodes: episode_specs.len().max(1),
        found_rate,
        average_attempts,
        average_max_attempts,
        survival_budget_ratio: survival_budget_ratio(average_attempts, average_max_attempts),
        escape_quality_score: escape_quality_score_from_parts(
            found_rate,
            average_attempts,
            average_max_attempts,
            average_evader_reward,
        ),
        average_searcher_reward: total.iter().map(|e| e.average_searcher_reward).sum::<f64>() / denom,
        average_evader_reward,
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
        let _cuda_guard = cuda_work_guard();
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
        let _cuda_guard = cuda_work_guard();
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
        let _cuda_guard = cuda_work_guard();
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
        let _cuda_guard = cuda_work_guard();
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
        let _cuda_guard = cuda_work_guard();
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
    let _cuda_guard = cuda_work_guard();
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
        let _cuda_guard = cuda_work_guard();
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
        let _cuda_guard = cuda_work_guard();
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

    fn score_nodes_chunked(&self, feature_rows: &[f64]) -> candle_core::Result<Vec<f64>> {
        let input_dim = self.input_dim.max(1);
        let total_rows = feature_rows.len() / input_dim;
        let max_rows = gpu_single_score_rows();
        if total_rows <= max_rows {
            return self.score_nodes(feature_rows);
        }

        let mut out = Vec::with_capacity(total_rows);
        let mut row_off = 0usize;
        while row_off < total_rows {
            let row_end = (row_off + max_rows).min(total_rows);
            let chunk = &feature_rows[row_off * input_dim..row_end * input_dim];
            out.extend(self.score_nodes(chunk)?);
            row_off = row_end;
        }
        Ok(out)
    }
}

fn choose_key_searcher_gpu(
    model: &GpuSearcherModel,
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    excluded: &[i32],
) -> Option<i32> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * SEARCHER_FEATURE_COUNT);
    append_searcher_feature_rows(&mut feature_rows, all_nodes, recent_guesses);

    let scores = model
        .score_nodes_chunked(&feature_rows)
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
        let _cuda_guard = cuda_work_guard();
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
        let _cuda_guard = cuda_work_guard();
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

    fn score_nodes_chunked(&self, feature_rows: &[f64]) -> candle_core::Result<Vec<f64>> {
        let input_dim = self.input_dim.max(1);
        let total_rows = feature_rows.len() / input_dim;
        let max_rows = gpu_single_score_rows();
        if total_rows <= max_rows {
            return self.score_nodes(feature_rows);
        }

        let mut out = Vec::with_capacity(total_rows);
        let mut row_off = 0usize;
        while row_off < total_rows {
            let row_end = (row_off + max_rows).min(total_rows);
            let chunk = &feature_rows[row_off * input_dim..row_end * input_dim];
            out.extend(self.score_nodes(chunk)?);
            row_off = row_end;
        }
        Ok(out)
    }
}

fn choose_key_evader_gpu(
    model: &GpuEvaderModel,
    all_nodes: &[NodeMeta],
    recent_guesses: &[i32],
    excluded: &[i32],
) -> Option<i32> {
    let mut feature_rows = Vec::with_capacity(all_nodes.len() * EVADER_FEATURE_COUNT);
    append_evader_feature_rows(&mut feature_rows, all_nodes, recent_guesses);
    let scores = model
        .score_nodes_chunked(&feature_rows)
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
    append_evader_feature_rows(&mut feature_rows, all_nodes, recent_guesses);
    let raw_scores = gpu_model
        .score_nodes_chunked(&feature_rows)
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
    append_evader_feature_rows(&mut feature_rows, all_nodes, &[]);

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
        append_evader_feature_rows(&mut feature_rows, &meta_new, &recent_guesses);

        // One GPU call: all candidates × all nodes.
        let guessed_now: Vec<i32> = unique_guesses.iter().copied().collect();
        let all_scores = gpu_batch.score_all(&feature_rows, EVADER_FEATURE_COUNT)
            .expect("GPU score_all failed — GPU required");

        // Apply per-candidate relocation.
        let guessed_set: HashSet<i32> = guessed_now.iter().copied().collect();
        let exposure_scores = frontier_exposure_scores(&meta_new, &recent_guesses, &guessed_set);
        for c in 0..n {
            if !alive[c] { continue; }
            let scores = &all_scores[c];
            let reloc = meta_new.iter().enumerate()
                .filter(|(_, nd)| nd.depth != 0 && !guessed_set.contains(&nd.key))
                .max_by(|(i, _), (j, _)| scores[*i].partial_cmp(&scores[*j]).unwrap_or(Ordering::Equal))
                .map(|(idx, nd)| (idx, nd.key))
                .or_else(|| meta_new.iter().enumerate()
                    .find(|(_, nd)| nd.depth != 0 && !guessed_set.contains(&nd.key))
                    .map(|(idx, nd)| (idx, nd.key)));

            if let Some((reloc_idx, rk)) = reloc {
                frontiers[c] += exposure_scores.get(reloc_idx).copied().unwrap_or(0.0);
                if rk == min_key { root_relocs[c] += 1; }
                if let Some(old_idx) = node_index_by_key(&meta_new, shell_keys[c]) {
                    let om = &meta_new[old_idx];
                    let nm = &meta_new[reloc_idx];
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
            searcher_reward(
                found,
                attempts,
                max_attempts,
                if found { 0 } else { min_dists[c] },
                unique_count,
                node_count,
            ),
            evader_reward(found, attempts, max_attempts, reloc_costs[c], frontiers[c], root_relocs[c], node_count),
        )
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory-bounded GPU scoring helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Like `GpuEvaderBatch::score_all` but splits rows into population-aware chunks
/// to avoid huge batched CUDA launches on large node counts or combo batches.
/// Returns `[n_candidates][total_rows]`.
fn score_all_chunked(
    gpu_batch: &GpuEvaderBatch,
    feature_rows: &[f64],
    input_dim: usize,
) -> candle_core::Result<Vec<Vec<f64>>> {
    let total_rows = feature_rows.len() / input_dim.max(1);
    if total_rows == 0 { return Ok(vec![Vec::new(); gpu_batch.n]); }
    let max_rows = gpu_score_batch_rows(gpu_batch.n);
    if total_rows <= max_rows {
        return gpu_batch.score_all(feature_rows, input_dim);
    }
    let n = gpu_batch.n;
    let mut result: Vec<Vec<f64>> = vec![Vec::with_capacity(total_rows); n];
    let mut off = 0;
    while off < total_rows {
        let end = (off + max_rows).min(total_rows);
        let chunk = &feature_rows[off * input_dim..end * input_dim];
        for (c, cs) in gpu_batch.score_all(chunk, input_dim)?.into_iter().enumerate() {
            result[c].extend(cs);
        }
        off = end;
    }
    Ok(result)
}

/// Like `GpuSearcherBatch::score_all_feature_batches` but chunks across the
/// node dimension to bound peak GPU memory.  Returns `[n_candidates][total_nodes]`.
fn score_feature_batches_chunked(
    gpu_batch: &GpuSearcherBatch,
    feature_rows: &[f64],
    total_nodes: usize,
    input_dim: usize,
) -> candle_core::Result<Vec<Vec<f64>>> {
    if total_nodes == 0 { return Ok(vec![Vec::new(); gpu_batch.n]); }
    let max_rows = gpu_score_batch_rows(gpu_batch.n);
    if total_nodes <= max_rows {
        return gpu_batch.score_all_feature_batches(feature_rows, total_nodes, input_dim);
    }
    let n = gpu_batch.n;
    let mut result: Vec<Vec<f64>> = vec![Vec::with_capacity(total_nodes); n];
    let mut node_off = 0;
    while node_off < total_nodes {
        let node_end = (node_off + max_rows).min(total_nodes);
        let chunk_nodes = node_end - node_off;
        // Interleaved layout: candidate c's chunk is at [c*total_nodes + node_off .. chunk_nodes].
        let mut chunk_feats = vec![0.0f64; n * chunk_nodes * input_dim];
        for c in 0..n {
            let src = c * total_nodes * input_dim + node_off * input_dim;
            let dst = c * chunk_nodes * input_dim;
            chunk_feats[dst..dst + chunk_nodes * input_dim]
                .copy_from_slice(&feature_rows[src..src + chunk_nodes * input_dim]);
        }
        for (c, cs) in gpu_batch.score_all_feature_batches(&chunk_feats, chunk_nodes, input_dim)?
            .into_iter().enumerate()
        {
            result[c].extend(cs);
        }
        node_off = node_end;
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Step-synchronous batched episode runners
//
// These replace the par_iter loops in optimize_evader / optimize_searcher_mlp.
//
// Problem with par_iter: N CPU threads each issue tiny GPU calls (M=25 nodes).
// Those calls serialize on a single CUDA stream, so the GPU sees 1200 sequential
// tiny matmuls per generation.  Kernel-launch overhead dominates; SM utilisation
// stays at 30–40%.
//
// Fix: process all combos in lockstep.  At each attempt step, concatenate every
// active combo's node features into one feature matrix and issue ONE GPU call.
// M grows from 25 → n_combos×25 ≈ 3000, driving SM utilisation to ~80%.
// ─────────────────────────────────────────────────────────────────────────────

struct ComboEvState {
    tree: FastTrainingTree,
    s_idx: usize,
    node_count: usize,
    max_attempts: usize,
    min_key: i32,
    // Per-candidate (n entries):
    shell_keys: Vec<i32>,
    alive: Vec<bool>,
    found_at: Vec<usize>,
    reloc_costs: Vec<f64>,
    frontiers: Vec<f64>,
    root_relocs: Vec<usize>,
    min_dists: Vec<usize>,
    // Shared within a combo (one searcher, same guess history for all candidates):
    recent_guesses: Vec<i32>,
    unique_guesses: HashSet<i32>,
    done: bool,
}

/// Batched evader episode runner.  Returns [combo][candidate] evader rewards.
fn run_episodes_batched_evader(
    gpu_batch: &GpuEvaderBatch,
    gpu_searchers: &[GpuSearcherModel],
    combos: &[(usize, EpisodeSpec)],
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
    phase_label: &str,
) -> Vec<Vec<f64>> {
    let n = gpu_batch.n;
    let n_combos = combos.len();
    if n_combos == 0 { return Vec::new(); }

    let mut states: Vec<ComboEvState> = combos.iter().map(|&(s_idx, spec)| {
        let node_count = spec.node_count as usize;
        let max_attempts = compute_max_attempts(
            node_count, max_attempts_factor, max_attempts_ratio, max_attempts_cap,
        );
        let mut tree = FastTrainingTree::new(spec);
        let initial_meta = tree.meta_snapshot();
        let all_keys: Vec<i32> = initial_meta.iter().map(|m| m.key).collect();
        let min_key = all_keys.iter().copied().min().unwrap_or(1);
        let fallback = all_keys.get(1).copied().unwrap_or(min_key);
        let shell_keys = choose_initial_evader_keys_gpu(gpu_batch, n, &initial_meta, fallback);
        ComboEvState {
            tree, s_idx, node_count, max_attempts, min_key, shell_keys,
            alive: vec![true; n],
            found_at: vec![0usize; n],
            reloc_costs: vec![0.0; n],
            frontiers: vec![0.0; n],
            root_relocs: vec![0usize; n],
            min_dists: vec![node_count; n],
            recent_guesses: Vec::new(),
            unique_guesses: HashSet::new(),
            done: false,
        }
    }).collect();

    let global_max = states.iter().map(|s| s.max_attempts).max().unwrap_or(1);
    let heartbeat = training_heartbeat_enabled();
    let heartbeat_interval = (global_max / 4).max(1);

    for attempt in 1..=global_max {
        let active: Vec<usize> = (0..n_combos).filter(|&i| !states[i].done).collect();
        if active.is_empty() { break; }
        if heartbeat && should_emit_attempt_heartbeat(attempt, global_max, heartbeat_interval) {
            let alive_candidates: usize = active
                .iter()
                .map(|&ci| states[ci].alive.iter().filter(|&&alive| alive).count())
                .sum();
            println!(
                "      {:>12} attempt {:>2}/{:<2} | active episodes {:>3}/{:<3} | alive candidates {:>6}",
                phase_label,
                attempt,
                global_max,
                active.len(),
                n_combos,
                alive_candidates
            );
            let _ = io::stdout().flush();
        }

        // ── Phase A: searcher picks a guess for each active combo ──────────
        // Group by searcher model → one batched GPU call per model.
        let mut combo_guesses = vec![0i32; n_combos];

        for s_idx in 0..gpu_searchers.len() {
            let s_combos: Vec<usize> = active.iter().copied()
                .filter(|&ci| states[ci].s_idx == s_idx)
                .collect();
            if s_combos.is_empty() { continue; }

            let mut all_s_feats: Vec<f64> = Vec::new();
            let mut offsets: Vec<usize> = Vec::new();
            let mut metas: Vec<Vec<NodeMeta>> = Vec::new();
            let mut cumul = 0usize;

            for &ci in &s_combos {
                let st = &mut states[ci];
                let meta = st.tree.meta_snapshot();
                append_searcher_feature_rows(&mut all_s_feats, &meta, &st.recent_guesses);
                offsets.push(cumul);
                cumul += meta.len();
                metas.push(meta);
            }

            // Chunked because one saved searcher may score many active episode trees.
            let all_s_scores = gpu_searchers[s_idx].score_nodes_chunked(&all_s_feats)
                .expect("GPU batched searcher scoring failed");

            for (gi, &ci) in s_combos.iter().enumerate() {
                let st = &states[ci];
                let meta = &metas[gi];
                let offset = offsets[gi];
                let scores = &all_s_scores[offset..offset + meta.len()];
                let exclude_guessed = st.unique_guesses.len() < meta.len();

                combo_guesses[ci] = meta.iter().zip(scores)
                    .filter(|(nd, _)| !exclude_guessed || !st.unique_guesses.contains(&nd.key))
                    .max_by(|(_, a): &(_, &f64), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .or_else(|| meta.iter().zip(scores)
                        .max_by(|(_, a): &(_, &f64), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)))
                    .map(|(nd, _)| nd.key)
                    .unwrap_or(st.min_key);
            }
        }

        // ── Phase B: check found, shuffle tree, update guess history ───────
        for &ci in &active {
            let st = &mut states[ci];
            let guess = combo_guesses[ci];
            let meta = st.tree.meta_snapshot();
            let guess_node_index = node_index_by_key(&meta, guess);

            for c in 0..n {
                if !st.alive[c] { continue; }
                if st.shell_keys[c] == guess {
                    st.alive[c] = false;
                    st.found_at[c] = attempt;
                } else if let (Some(guess_idx), Some(shell_idx)) = (
                    guess_node_index,
                    node_index_by_key(&meta, st.shell_keys[c]),
                ) {
                    let gn = &meta[guess_idx];
                    let sn = &meta[shell_idx];
                    let d = path_distance(gn.path_bits, gn.path_len, sn.path_bits, sn.path_len);
                    if d < st.min_dists[c] { st.min_dists[c] = d; }
                }
            }
            st.tree.shuffle_step();
            st.unique_guesses.insert(guess);
            push_recent_guess(&mut st.recent_guesses, guess);
            if st.alive.iter().all(|&a| !a) || attempt >= st.max_attempts {
                st.done = true;
            }
        }

        // ── Phase C: evader relocation — one GPU call for all still-active combos ──
        let still_active: Vec<usize> = (0..n_combos).filter(|&i| !states[i].done).collect();
        if still_active.is_empty() { break; }

        let mut all_e_feats: Vec<f64> = Vec::new();
        let mut e_offsets: Vec<usize> = Vec::new();
        let mut e_ncounts: Vec<usize> = Vec::new();
        let mut e_metas: Vec<Vec<NodeMeta>> = Vec::new();
        let mut cumul = 0usize;

        for &ci in &still_active {
            let st = &mut states[ci];
            let meta_new = st.tree.meta_snapshot();
            append_evader_feature_rows(&mut all_e_feats, &meta_new, &st.recent_guesses);
            e_offsets.push(cumul);
            let nn = meta_new.len();
            e_ncounts.push(nn);
            cumul += nn;
            e_metas.push(meta_new);
        }

        // THE KEY CALL: all combos × all candidates — chunked to bound GPU memory.
        // M grows from 25 to n_active_combos × avg_nodes; chunks keep activations ≤ ~307 MB.
        let all_e_scores = score_all_chunked(gpu_batch, &all_e_feats, EVADER_FEATURE_COUNT)
            .expect("GPU batched evader score_all failed");

        for (gi, &ci) in still_active.iter().enumerate() {
            let nn = e_ncounts[gi];
            let offset = e_offsets[gi];
            let meta_new = &e_metas[gi];
            let st = &mut states[ci];
            let exposure_scores =
                frontier_exposure_scores(meta_new, &st.recent_guesses, &st.unique_guesses);

            for c in 0..n {
                if !st.alive[c] { continue; }
                let scores = &all_e_scores[c][offset..offset + nn];
                let reloc = meta_new.iter().enumerate()
                    .filter(|(_, nd)| nd.depth != 0 && !st.unique_guesses.contains(&nd.key))
                    .max_by(|(i, _), (j, _)| scores[*i].partial_cmp(&scores[*j]).unwrap_or(Ordering::Equal))
                    .map(|(idx, nd)| (idx, nd.key))
                    .or_else(|| meta_new.iter().enumerate()
                        .find(|(_, nd)| nd.depth != 0 && !st.unique_guesses.contains(&nd.key))
                        .map(|(idx, nd)| (idx, nd.key)));
                if let Some((reloc_idx, rk)) = reloc {
                    st.frontiers[c] += exposure_scores.get(reloc_idx).copied().unwrap_or(0.0);
                    if rk == st.min_key { st.root_relocs[c] += 1; }
                    if let Some(old_idx) = node_index_by_key(meta_new, st.shell_keys[c]) {
                        let om = &meta_new[old_idx];
                        let nm = &meta_new[reloc_idx];
                        st.reloc_costs[c] += path_distance(
                            om.path_bits, om.path_len, nm.path_bits, nm.path_len,
                        ) as f64;
                    }
                    st.shell_keys[c] = rk;
                }
            }
        }
    }

    states.iter().map(|st| {
        (0..n).map(|c| {
            let found = st.found_at[c] > 0;
            let attempts = if found { st.found_at[c] } else { st.max_attempts };
            evader_reward(
                found, attempts, st.max_attempts,
                st.reloc_costs[c], st.frontiers[c], st.root_relocs[c], st.node_count,
            )
        }).collect()
    }).collect()
}

struct ComboSrState {
    tree: FastTrainingTree,
    e_idx: usize,
    node_count: usize,
    max_attempts: usize,
    min_key: i32,
    // Per-candidate (n entries) — candidates guess independently so histories diverge:
    shell_keys: Vec<i32>,
    alive: Vec<bool>,
    found_at: Vec<usize>,
    reloc_costs: Vec<f64>,
    frontiers: Vec<f64>,
    root_relocs: Vec<usize>,
    min_dists: Vec<usize>,
    recent_guesses: Vec<Vec<i32>>,
    unique_guesses: Vec<HashSet<i32>>,
    done: bool,
}

/// Batched searcher episode runner.  Returns [combo][candidate] searcher rewards.
fn run_episodes_batched_searcher(
    gpu_batch: &GpuSearcherBatch,
    gpu_evaders: &[GpuEvaderModel],
    combos: &[(usize, EpisodeSpec)],
    max_attempts_factor: usize,
    max_attempts_ratio: Option<f64>,
    max_attempts_cap: Option<usize>,
    phase_label: &str,
) -> Vec<Vec<f64>> {
    let n = gpu_batch.n;
    let n_combos = combos.len();
    if n_combos == 0 { return Vec::new(); }

    let mut states: Vec<ComboSrState> = combos.iter().map(|&(e_idx, spec)| {
        let node_count = spec.node_count as usize;
        let max_attempts = compute_max_attempts(
            node_count, max_attempts_factor, max_attempts_ratio, max_attempts_cap,
        );
        let mut episode_rng = StdRng::seed_from_u64(spec.seed ^ 0xE0AD_2026);
        let mut tree = FastTrainingTree::new(spec);
        let initial_meta = tree.meta_snapshot();
        let all_keys: Vec<i32> = initial_meta.iter().map(|m| m.key).collect();
        let min_key = all_keys.iter().copied().min().unwrap_or(1);
        let initial_shell = sample_key_evader_training_gpu(
                &gpu_evaders[e_idx], &initial_meta, &[], &[], &mut episode_rng)
            .or_else(|| choose_key_evader_gpu(&gpu_evaders[e_idx], &initial_meta, &[], &[]))
            .unwrap_or_else(|| initial_meta.iter().find(|m| m.depth != 0).map(|m| m.key).unwrap_or(2));
        ComboSrState {
            tree, e_idx, node_count, max_attempts, min_key,
            shell_keys: vec![initial_shell; n],
            alive: vec![true; n],
            found_at: vec![0usize; n],
            reloc_costs: vec![0.0; n],
            frontiers: vec![0.0; n],
            root_relocs: vec![0usize; n],
            min_dists: vec![node_count.max(1); n],
            recent_guesses: vec![Vec::new(); n],
            unique_guesses: vec![HashSet::new(); n],
            done: false,
        }
    }).collect();

    let global_max = states.iter().map(|s| s.max_attempts).max().unwrap_or(1);
    let heartbeat = training_heartbeat_enabled();
    let heartbeat_interval = (global_max / 4).max(1);

    for attempt in 1..=global_max {
        let active: Vec<usize> = (0..n_combos).filter(|&i| !states[i].done).collect();
        if active.is_empty() { break; }
        if heartbeat && should_emit_attempt_heartbeat(attempt, global_max, heartbeat_interval) {
            let alive_candidates: usize = active
                .iter()
                .map(|&ci| states[ci].alive.iter().filter(|&&alive| alive).count())
                .sum();
            println!(
                "      {:>12} attempt {:>2}/{:<2} | active episodes {:>3}/{:<3} | alive candidates {:>6}",
                phase_label,
                attempt,
                global_max,
                active.len(),
                n_combos,
                alive_candidates
            );
            let _ = io::stdout().flush();
        }

        // ── Phase A: all searcher candidates guess — one batched GPU call ──
        // For all active combos together:
        //   feature layout per candidate: [combo0_nodes || combo1_nodes || ...]
        //   total_nodes = Σ n_nodes_active_combos
        let mut total_nodes = 0usize;
        let mut combo_offsets: Vec<usize> = Vec::with_capacity(active.len());
        let mut combo_ncounts: Vec<usize> = Vec::with_capacity(active.len());
        let mut metas: Vec<Vec<NodeMeta>> = Vec::with_capacity(active.len());

        for &ci in &active {
            combo_offsets.push(total_nodes);
            let meta = states[ci].tree.meta_snapshot();
            combo_ncounts.push(meta.len());
            total_nodes += meta.len();
            metas.push(meta);
        }

        // Build [n × total_nodes × SEARCHER_FEAT] features.
        let mut all_sr_feats: Vec<f64> = vec![0.0; n * total_nodes * SEARCHER_FEATURE_COUNT];
        for (gi, &ci) in active.iter().enumerate() {
            let st = &states[ci];
            let meta = &metas[gi];
            let base_node_off = combo_offsets[gi];
            for c in 0..n {
                if !st.alive[c] { continue; }
                let feature_context = FeatureContext::new(meta, &st.recent_guesses[c]);
                let cand_base = c * total_nodes * SEARCHER_FEATURE_COUNT;
                for (ni, node) in meta.iter().enumerate() {
                    let feats = build_searcher_features_with_context(node, &feature_context);
                    let dst = cand_base + (base_node_off + ni) * SEARCHER_FEATURE_COUNT;
                    all_sr_feats[dst..dst + SEARCHER_FEATURE_COUNT].copy_from_slice(&feats);
                }
            }
        }

        // All candidates × all (batched) nodes — chunked to bound GPU memory.
        let all_sr_scores = score_feature_batches_chunked(gpu_batch, &all_sr_feats, total_nodes, SEARCHER_FEATURE_COUNT)
            .expect("GPU batched searcher score_all failed");
        // all_sr_scores: [n][total_nodes]

        // Pick guess per (combo, candidate) and check found.
        for (gi, &ci) in active.iter().enumerate() {
            let st = &mut states[ci];
            let meta = &metas[gi];
            let offset = combo_offsets[gi];
            let nn = combo_ncounts[gi];

            let mut guesses = vec![st.min_key; n];
            for c in 0..n {
                if !st.alive[c] { continue; }
                let scores = &all_sr_scores[c][offset..offset + nn];
                let exclude_guessed = st.unique_guesses[c].len() < meta.len();
                guesses[c] = meta.iter().zip(scores)
                    .filter(|(nd, _)| !exclude_guessed || !st.unique_guesses[c].contains(&nd.key))
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .or_else(|| meta.iter().zip(scores)
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)))
                    .map(|(nd, _)| nd.key)
                    .unwrap_or(st.min_key);
            }

            for c in 0..n {
                if !st.alive[c] { continue; }
                let guess = guesses[c];
                if st.shell_keys[c] == guess {
                    st.alive[c] = false;
                    st.found_at[c] = attempt;
                } else if let (Some(guess_idx), Some(shell_idx)) = (
                    node_index_by_key(meta, guess),
                    node_index_by_key(meta, st.shell_keys[c]),
                ) {
                    let gn = &meta[guess_idx];
                    let sn = &meta[shell_idx];
                    let d = path_distance(gn.path_bits, gn.path_len, sn.path_bits, sn.path_len);
                    if d < st.min_dists[c] { st.min_dists[c] = d; }
                }
                st.unique_guesses[c].insert(guess);
                push_recent_guess(&mut st.recent_guesses[c], guess);
            }

            st.tree.shuffle_step();
            if st.alive.iter().all(|&a| !a) || attempt >= st.max_attempts {
                st.done = true;
            }
        }

        let still_active: Vec<usize> = (0..n_combos).filter(|&i| !states[i].done).collect();
        if still_active.is_empty() { break; }

        // ── Phase B: evader relocation — batch by evader model ─────────────
        for e_idx in 0..gpu_evaders.len() {
            let e_combos: Vec<usize> = still_active.iter().copied()
                .filter(|&ci| states[ci].e_idx == e_idx)
                .collect();
            if e_combos.is_empty() { continue; }

            // Build flat feature rows: [Σ alive_c × n_nodes, EVADER_FEAT].
            let mut reloc_feats: Vec<f64> = Vec::new();
            let mut meta_newvec: Vec<Vec<NodeMeta>> = Vec::new();

            for &ci in &e_combos {
                let st = &mut states[ci];
                let meta_new = st.tree.meta_snapshot();
                for c in 0..n {
                    if !st.alive[c] { continue; }
                    let feature_context = FeatureContext::new(&meta_new, &st.recent_guesses[c]);
                    for node in &meta_new {
                        reloc_feats.extend_from_slice(
                            &build_evader_features_with_context(node, &feature_context),
                        );
                    }
                }
                meta_newvec.push(meta_new);
            }

            let reloc_scores = gpu_evaders[e_idx]
                .score_nodes_chunked(&reloc_feats)
                .expect("GPU batched evader relocation failed");

            // Distribute: reconstruct node-level scores per (combo, candidate).
            let mut row_ptr = 0usize;
            for (gi, &ci) in e_combos.iter().enumerate() {
                let st = &mut states[ci];
                let meta_new = &meta_newvec[gi];
                let nn = meta_new.len();

                for c in 0..n {
                    if !st.alive[c] { continue; }
                    let scores = &reloc_scores[row_ptr..row_ptr + nn];
                    row_ptr += nn;
                    let exposure_scores =
                        frontier_exposure_scores(meta_new, &st.recent_guesses[c], &st.unique_guesses[c]);
                    let relocate_to = meta_new.iter().enumerate().zip(scores)
                        .filter(|((_, nd), _)| nd.depth != 0 && !st.unique_guesses[c].contains(&nd.key))
                        .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap_or(Ordering::Equal))
                        .map(|((idx, nd), _)| (idx, nd.key))
                        .or_else(|| meta_new.iter().enumerate().zip(scores)
                            .filter(|((_, nd), _)| nd.depth != 0)
                            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap_or(Ordering::Equal))
                            .map(|((idx, nd), _)| (idx, nd.key)));
                    if let Some((reloc_idx, rk)) = relocate_to {
                        st.frontiers[c] += exposure_scores.get(reloc_idx).copied().unwrap_or(0.0);
                        if rk == st.min_key { st.root_relocs[c] += 1; }
                        if let Some(old_idx) = node_index_by_key(meta_new, st.shell_keys[c]) {
                            let om = &meta_new[old_idx];
                            let nm = &meta_new[reloc_idx];
                            st.reloc_costs[c] += path_distance(
                                om.path_bits, om.path_len, nm.path_bits, nm.path_len,
                            ) as f64;
                        }
                        st.shell_keys[c] = rk;
                    }
                }
            }
        }
    }

    states.iter().map(|st| {
        let unique_per_c: Vec<usize> = (0..n).map(|c| st.unique_guesses[c].len()).collect();
        (0..n).map(|c| {
            let found = st.found_at[c] > 0;
            let attempts = if found { st.found_at[c] } else { st.max_attempts };
            searcher_reward(
                found,
                attempts,
                st.max_attempts,
                if found { 0 } else { st.min_dists[c] },
                unique_per_c[c], st.node_count,
            )
        }).collect()
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
    let phase_start = Instant::now();

    // Build the perturbed batch AND generate all noise on GPU in one randn kernel.
    let (gpu_batch, noise) = GpuEvaderBatch::new_es(current_model, n, mutation_scale, device)
        .expect("GPU ES evader batch failed — GPU required");


    let gpu_searchers: Vec<GpuSearcherModel> = searcher_pool
        .iter()
        .map(|s| GpuSearcherModel::new(s, device).expect("GPU searcher upload failed"))
        .collect();

    let combos: Vec<(usize, EpisodeSpec)> = gpu_searchers
        .iter()
        .enumerate()
        .flat_map(|(idx, _)| episode_specs.iter().map(move |&sp| (idx, sp)))
        .collect();
    if training_heartbeat_enabled() {
        println!(
            "      evader-opt setup\n        population     {:>6}\n        searcher_pool  {:>6}\n        episodes       {:>6}\n        matchups       {:>6}",
            n,
            gpu_searchers.len(),
            episode_specs.len(),
            combos.len(),
        );
        let _ = io::stdout().flush();
    }
    let combo_weights: Vec<f64> = combos
        .iter()
        .map(|(searcher_idx, _)| searcher_opponent_weight(&searcher_pool[*searcher_idx]))
        .collect();
    let total_combo_weight = combo_weights.iter().sum::<f64>().max(1e-9);

    let per_episode: Vec<Vec<f64>> = run_episodes_batched_evader(
        &gpu_batch, &gpu_searchers, &combos,
        max_attempts_factor, max_attempts_ratio, max_attempts_cap,
        "evader-opt",
    );

    let mut total_scores = vec![0.0f64; n];
    for (episode_idx, episode_scores) in per_episode.iter().enumerate() {
        let weight = combo_weights.get(episode_idx).copied().unwrap_or(1.0) / total_combo_weight;
        for (c, &s) in episode_scores.iter().enumerate() {
            total_scores[c] += s * weight;
        }
    }
    let mean_score = total_scores.iter().sum::<f64>() / n as f64;
    let fitness_std = (total_scores.iter().map(|&s| (s - mean_score).powi(2)).sum::<f64>() / n as f64).sqrt();

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
    if training_heartbeat_enabled() {
        println!(
            "      evader-opt done   | elapsed {:>7.1}s | fitness_std {:>7.3}",
            phase_start.elapsed().as_secs_f64(),
            fitness_std
        );
        let _ = io::stdout().flush();
    }

    (new_model, fitness_std)
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
    let phase_start = Instant::now();

    let (gpu_batch, noise) = GpuSearcherBatch::new_es(current_model, n, mutation_scale, device)
        .expect("GPU ES searcher batch failed — GPU required");

    let gpu_evader_pool: Vec<GpuEvaderModel> = evader_pool
        .iter()
        .map(|e| GpuEvaderModel::new(e, device).expect("GPU evader upload failed — GPU required"))
        .collect();

    let combos: Vec<(usize, EpisodeSpec)> = (0..gpu_evader_pool.len())
        .flat_map(|ei| episode_specs.iter().map(move |&sp| (ei, sp)))
        .collect();
    if training_heartbeat_enabled() {
        println!(
            "      searcher-opt setup\n        population     {:>6}\n        evader_pool    {:>6}\n        episodes       {:>6}\n        matchups       {:>6}",
            n,
            gpu_evader_pool.len(),
            episode_specs.len(),
            combos.len(),
        );
        let _ = io::stdout().flush();
    }
    let n_evals = combos.len().max(1) as f64;

    let per_episode: Vec<Vec<f64>> = run_episodes_batched_searcher(
        &gpu_batch, &gpu_evader_pool, &combos,
        max_attempts_factor, max_attempts_ratio, max_attempts_cap,
        "searcher-opt",
    );

    let mut total_scores = vec![0.0f64; n];
    for episode_scores in &per_episode {
        for (c, &s) in episode_scores.iter().enumerate() {
            total_scores[c] += s / n_evals;
        }
    }
    let mean_score = total_scores.iter().sum::<f64>() / n as f64;
    let fitness_std = (total_scores.iter().map(|&s| (s - mean_score).powi(2)).sum::<f64>() / n as f64).sqrt();

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
    if training_heartbeat_enabled() {
        println!(
            "      searcher-opt done | elapsed {:>7.1}s | fitness_std {:>7.3}",
            phase_start.elapsed().as_secs_f64(),
            fitness_std
        );
        let _ = io::stdout().flush();
    }

    (new_model, fitness_std)
}

// ──────────────────────────────────────────────────────────────────────────────
// Public training entry point
// ──────────────────────────────────────────────────────────────────────────────

pub fn train_self_play_models(config: &TrainingConfig) -> Result<(SelfPlayModels, TrainingSummary), String> {
    fs::create_dir_all(&config.output_dir).map_err(|err| err.to_string())?;
    if config.population_size == 0 {
        return Err("population_size must be at least 1".to_string());
    }
    if config.max_nodes < config.min_nodes {
        return Err(format!(
            "max_nodes ({}) must be >= min_nodes ({})",
            config.max_nodes, config.min_nodes
        ));
    }
    if config.stagnation_node_step < 0 {
        return Err("stagnation_node_step must be >= 0".to_string());
    }
    if let Err(err) = install_training_interrupt_handler() {
        eprintln!(
            "WARNING: could not install graceful Ctrl+C handler ({err}); interrupting may abort immediately."
        );
    }
    if let Some(cap) = config.stagnation_max_nodes_cap {
        if cap < config.max_nodes {
            return Err(format!(
                "stagnation_max_nodes_cap ({cap}) must be >= initial max_nodes ({})",
                config.max_nodes
            ));
        }
    }
    if let Some(cap) = config.stagnation_population_cap {
        if cap < config.population_size {
            return Err(format!(
                "stagnation_population_cap ({cap}) must be >= initial population_size ({})",
                config.population_size
            ));
        }
    }
    if !config.searcher_lr_scale.is_finite() || config.searcher_lr_scale < 0.0 {
        return Err("searcher_lr_scale must be a finite value >= 0".to_string());
    }
    if !config.searcher_max_found_rate.is_finite()
        || !(0.0..=1.0).contains(&config.searcher_max_found_rate)
    {
        return Err("searcher_max_found_rate must be in [0, 1]".to_string());
    }
    if !config.searcher_max_found_rate_jump.is_finite()
        || !(0.0..=1.0).contains(&config.searcher_max_found_rate_jump)
    {
        return Err("searcher_max_found_rate_jump must be in [0, 1]".to_string());
    }

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

    let mut active_population_size = config.population_size;
    let mut active_min_nodes = config.min_nodes;
    let mut active_max_nodes = config.max_nodes;
    let adaptive_growth_after = config.stagnation_grow_after.filter(|value| *value > 0);
    let mut adaptive_growth_events = 0usize;

    let initial_specs = build_episode_specs(
        config.episodes_per_eval,
        config.seed,
        active_min_nodes,
        active_max_nodes,
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
    let initial_static_eval = if config.training_mode == TrainingMode::CoAgent {
        Some(evaluate_evader_against_fixed_searchers(
            &models.evader,
            &fixed_evader_only_searchers,
            &initial_specs,
            config.max_attempts_factor,
            config.max_attempts_ratio,
            config.max_attempts_cap,
        ))
    } else {
        None
    };
    let initial_evader_selection_score =
        robust_evader_selection_score(&current, initial_static_eval.as_ref());

    println!("Training configuration");
    println!("  mode: {}", match config.training_mode {
        TrainingMode::Static => "static (evader vs fixed search algorithms)",
        TrainingMode::CoAgent => "coagent (evader vs learned searcher)",
    });
    println!("  generations:       {:>6}", config.generations);
    println!("  population:        {:>6}", active_population_size);
    println!("  episodes/eval:     {:>6}", config.episodes_per_eval);
    println!("  nodes:             {:>3}..{:<3}", active_min_nodes, active_max_nodes);
    println!("  hall of fame:      {:>6}", config.hall_of_fame_size);
    println!("  seed:              {:>6}", config.seed);
    println!("  accelerator: {}", accelerator_description());
    println!("  controls:          Ctrl+C = finish current generation, save best, exit");
    if training_accelerator_is_cuda() {
        if let Some(row_override) = parse_env_usize("GPU_SCORE_BATCH_ROWS") {
            println!(
                "  gpu batching:      rows={} single_rows={} (explicit GPU_SCORE_BATCH_ROWS; population={} serial={})",
                row_override,
                gpu_single_score_rows(),
                active_population_size,
                cuda_serialization_enabled(),
            );
        } else {
            println!(
                "  gpu batching:      cells={} rows={} single_rows={} (population={} serial={})",
                parse_env_usize("GPU_SCORE_BATCH_CELLS").unwrap_or(DEFAULT_GPU_SCORE_BATCH_CELLS),
                gpu_score_batch_rows(active_population_size),
                gpu_single_score_rows(),
                active_population_size,
                cuda_serialization_enabled(),
            );
        }
    }
    println!(
        "  evader model:      MLP ({}→{}→{}→1 ReLU), {} features",
        EVADER_FEATURE_COUNT, EVADER_MLP_HIDDEN1, EVADER_MLP_HIDDEN2, EVADER_FEATURE_COUNT,
    );
    println!(
        "  searcher model:    MLP ({}→{}→{}→1 ReLU), {} features",
        SEARCHER_FEATURE_COUNT, SEARCHER_MLP_HIDDEN1, SEARCHER_MLP_HIDDEN2, SEARCHER_FEATURE_COUNT,
    );
    println!(
        "  attempt budget:    factor={} ratio={} cap={}",
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
    if let Some(grow_after) = adaptive_growth_after {
        println!(
            "  adaptive growth:   after={} node_step={} population_step={}{}{}",
            grow_after,
            config.stagnation_node_step,
            config.stagnation_population_step,
            config
                .stagnation_max_nodes_cap
                .map(|cap| format!(" node_cap={cap}"))
                .unwrap_or_default(),
            config
                .stagnation_population_cap
                .map(|cap| format!(" population_cap={cap}"))
                .unwrap_or_default(),
        );
    }
    if config.training_mode == TrainingMode::Static {
        println!(
            "  static opponents:  {}",
            fixed_searcher_role_names(&fixed_evader_only_searchers),
        );
        println!(
            "  static sampling:   {} per generation ({} total fixed strategies)",
            config.static_opponent_sample_count.max(1).min(fixed_evader_only_searchers.len().max(1)),
            fixed_evader_only_searchers.len(),
        );
    }
    if let Some(resume_path) = &config.resume_from {
        println!("  resume bundle:     {}", resume_path.display());
    }
    println!(
        "  searcher reward:   proximity + coverage shaping"
    );
    println!(
        "  evader reward:     relocation penalty 0.6 per hop / ln(n)"
    );
    if config.training_mode == TrainingMode::CoAgent {
        println!(
            "  searcher throttle: lr_scale={:.3} update_every={} found_rate_cap={:.3} max_jump={:.3}",
            config.searcher_lr_scale,
            config.searcher_update_interval.max(1),
            config.searcher_max_found_rate,
            config.searcher_max_found_rate_jump,
        );
    }
    match config.training_mode {
        TrainingMode::Static => println!(
            "Initial scores\n  searcher:      {:>8.2}\n  evader:        {:>8.2}\n  escape score:  {:>8.2}\n  found rate:    {:>8.3}\n  budget used:   {:>8.1}%\n  avg attempts:  {:>8.2}/{:>8.2}",
            current.average_searcher_reward,
            current.average_evader_reward,
            current.escape_quality_score,
            current.found_rate,
            current.survival_budget_ratio * 100.0,
            current.average_attempts,
            current.average_max_attempts,
        ),
        TrainingMode::CoAgent => println!(
            "Initial scores\n  learned\n    searcher:    {:>8.2}\n    evader:      {:>8.2}\n    escape score:{:>8.2}\n    found rate:  {:>8.3}\n    budget used: {:>8.1}%\n    avg attempts:{:>8.2}/{:>8.2}\n  static check\n    evader:      {:>8.2}\n    escape score:{:>8.2}\n    found rate:  {:>8.3}\n    budget used: {:>8.1}%\n  promotion score:{:>7.2}",
            current.average_searcher_reward,
            current.average_evader_reward,
            current.escape_quality_score,
            current.found_rate,
            current.survival_budget_ratio * 100.0,
            current.average_attempts,
            current.average_max_attempts,
            initial_static_eval.as_ref().map(|e| e.average_evader_reward).unwrap_or(0.0),
            initial_static_eval.as_ref().map(|e| e.escape_quality_score).unwrap_or(0.0),
            initial_static_eval.as_ref().map(|e| e.found_rate).unwrap_or(0.0),
            initial_static_eval.as_ref().map(|e| e.survival_budget_ratio * 100.0).unwrap_or(0.0),
            initial_evader_selection_score,
        ),
    }
    let _ = io::stdout().flush();

    let history_path = config.output_dir.join("training_history.json");
    let bundle_path = config.output_dir.join("self_play_models.json");
    let best_evader_path = config.output_dir.join("best_evader_model.json");
    let best_searcher_path = config.output_dir.join("best_searcher_model.json");
    let mut history: Vec<GenerationRecord> = Vec::new();
    history.push(GenerationRecord {
        generation: 0,
        searcher_score: current.average_searcher_reward,
        evader_score: current.average_evader_reward,
        escape_score: current.escape_quality_score,
        found_rate: current.found_rate,
        avg_attempts: current.average_attempts,
        avg_max_attempts: current.average_max_attempts,
        survival_budget_ratio: current.survival_budget_ratio,
        static_evader_score: initial_static_eval.as_ref().map(|e| e.average_evader_reward),
        static_escape_score: initial_static_eval.as_ref().map(|e| e.escape_quality_score),
        static_found_rate: initial_static_eval.as_ref().map(|e| e.found_rate),
        static_survival_budget_ratio: initial_static_eval.as_ref().map(|e| e.survival_budget_ratio),
        evader_fitness_std: None,
        searcher_fitness_std: None,
        population_size: active_population_size,
        min_nodes: active_min_nodes,
        max_nodes: active_max_nodes,
    });

    let hall_size = config.hall_of_fame_size;
    let snapshot_interval = if hall_size == 0 { usize::MAX } else { (config.generations / hall_size.max(1)).max(1) };
    let mut evader_hall: Vec<MlpPolicyModel> = Vec::new();
    let mut searcher_hall: Vec<SearcherMlpModel> = Vec::new();

    // Best-model tracking state.
    let mut best_evader_score = current.average_evader_reward;
    let mut best_evader_selection_score = initial_evader_selection_score;
    let mut best_evader_model = models.evader.clone();
    let mut best_generation = 0usize;
    let mut no_improve_count = 0usize;
    let mut stopped_early = false;
    let mut interrupted = false;

    let mut best_searcher_score = current.average_searcher_reward;
    let mut best_searcher_model = models.searcher.clone();
    let mut best_searcher_generation = 0usize;
    write_json_pretty_atomic(&history_path, &history)?;
    write_recovery_checkpoint(
        &config.output_dir,
        &models,
        &best_evader_model,
        (config.training_mode == TrainingMode::CoAgent).then_some(&best_searcher_model),
    )?;
    println!(
        "Recovery checkpoint\n  initialized: {}\n  safe resume: generation 0",
        bundle_path.display()
    );
    let _ = io::stdout().flush();

    for generation in 0..config.generations {
        if training_stop_requested() {
            println!(
                "Graceful stop\n  generation: before {}/{}\n  action: saving current best model and ending training",
                generation + 1,
                config.generations
            );
            interrupted = true;
            break;
        }

        if let Some(grow_after) = adaptive_growth_after {
            if no_improve_count > 0 && no_improve_count % grow_after == 0 {
                let old_min_nodes = active_min_nodes;
                let old_max_nodes = active_max_nodes;
                let old_population_size = active_population_size;

                let next_min_nodes = active_min_nodes
                    .saturating_add(config.stagnation_node_step)
                    .min(config.stagnation_max_nodes_cap.unwrap_or(i32::MAX));
                let next_max_nodes = active_max_nodes
                    .saturating_add(config.stagnation_node_step)
                    .min(config.stagnation_max_nodes_cap.unwrap_or(i32::MAX))
                    .max(next_min_nodes);
                let next_population_size = active_population_size
                    .saturating_add(config.stagnation_population_step)
                    .min(config.stagnation_population_cap.unwrap_or(usize::MAX));

                active_min_nodes = next_min_nodes;
                active_max_nodes = next_max_nodes;
                active_population_size = next_population_size;

                if active_min_nodes != old_min_nodes
                    || active_max_nodes != old_max_nodes
                    || active_population_size != old_population_size
                {
                    adaptive_growth_events += 1;
                    println!(
                        "Adaptive growth #{}\n  stagnant generations: {}\n  population:           {} -> {}\n  nodes:                {}..{} -> {}..{}",
                        adaptive_growth_events,
                        no_improve_count,
                        old_population_size,
                        active_population_size,
                        old_min_nodes,
                        old_max_nodes,
                        active_min_nodes,
                        active_max_nodes
                    );
                } else {
                    println!(
                        "Adaptive growth skipped\n  stagnant generations: {}\n  reason: already at configured caps",
                        no_improve_count
                    );
                }
                let _ = io::stdout().flush();
            }
        }

        let generation_start = Instant::now();
        let generation_seed = config.seed ^ generation as u64 ^ 0xA11CE;
        println!(
            "\nGeneration {}/{}\n  seed:       {}\n  heartbeat:  {}\n  population: {}\n  nodes:      {}..{}",
            generation + 1,
            config.generations,
            generation_seed,
            if training_heartbeat_enabled() {
                "enabled"
            } else {
                "disabled"
            },
            active_population_size,
            active_min_nodes,
            active_max_nodes,
        );
        let _ = io::stdout().flush();
        let episode_specs = build_episode_specs(
            config.episodes_per_eval,
            generation_seed,
            active_min_nodes,
            active_max_nodes,
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

        // Build opponent pools: current model + hall sample + best-known model.
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
        // Always include the all-time-best evader so the searcher always trains against the
        // hardest opponent it has ever faced, dampening cyclic arms-race regressions.
        if generation > 0 {
            evader_pool.push(best_evader_model.clone());
        }

        let mut searcher_pool: Vec<SearcherMlpModel> = match config.training_mode {
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
        // Mirror: always include the all-time-best searcher so the evader can't exploit regressions.
        if generation > 0 && config.training_mode == TrainingMode::CoAgent {
            searcher_pool.push(best_searcher_model.clone());
        }

        let evader_phase_start = Instant::now();
        println!(
            "  phase 1/{}: optimize evader",
            if config.training_mode == TrainingMode::CoAgent { 4 } else { 2 }
        );
        let _ = io::stdout().flush();
        let (best_evader, evader_fitness_std) = optimize_evader(
            &models.evader,
            &mut evader_es,
            config.es_lr,
            &searcher_pool,
            active_population_size,
            config.mutation_scale,
            &episode_specs,
            config.max_attempts_factor,
            config.max_attempts_ratio,
            config.max_attempts_cap,
        );
        println!(
            "  phase 1 complete | elapsed {:>7.1}s",
            evader_phase_start.elapsed().as_secs_f64()
        );
        let _ = io::stdout().flush();
        models.evader = best_evader;

        let searcher_update_interval = config.searcher_update_interval.max(1);
        let should_update_searcher = config.training_mode == TrainingMode::CoAgent
            && config.searcher_lr_scale > 0.0
            && (searcher_update_interval <= 1 || (generation + 1) % searcher_update_interval == 0);
        let mut searcher_rollback_reason: Option<String> = None;
        let searcher_baseline_eval = if should_update_searcher {
            Some(evaluate_pair_on_specs_mlp_searcher(
                &models.evader,
                &models.searcher,
                &episode_specs,
                config.max_attempts_factor,
                config.max_attempts_ratio,
                config.max_attempts_cap,
            ))
        } else {
            None
        };

        let searcher_before_update = models.searcher.clone();
        let searcher_fitness_std: Option<f64> = if should_update_searcher {
            let searcher_phase_start = Instant::now();
            println!(
                "  phase 2/4: optimize searcher (lr {:.4})",
                config.es_lr * config.searcher_lr_scale
            );
            let _ = io::stdout().flush();
            let (best_searcher, s_std) = optimize_searcher_mlp(
                &models.searcher,
                &mut searcher_es,
                config.es_lr * config.searcher_lr_scale,
                &evader_pool,
                active_population_size,
                config.mutation_scale,
                &episode_specs,
                config.max_attempts_factor,
                config.max_attempts_ratio,
                config.max_attempts_cap,
            );
            println!(
                "  phase 2 complete | elapsed {:>7.1}s",
                searcher_phase_start.elapsed().as_secs_f64()
            );
            let _ = io::stdout().flush();
            models.searcher = best_searcher;
            Some(s_std)
        } else if config.training_mode == TrainingMode::CoAgent {
            println!(
                "  phase 2/4: optimize searcher skipped (update_every={})",
                searcher_update_interval
            );
            let _ = io::stdout().flush();
            None
        } else {
            None
        };

        let eval_phase_start = Instant::now();
        println!(
            "  phase {}/{}: evaluate learned matchup",
            if config.training_mode == TrainingMode::CoAgent { 3 } else { 2 },
            if config.training_mode == TrainingMode::CoAgent { 4 } else { 2 }
        );
        let _ = io::stdout().flush();
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
        if let (Some(baseline), TrainingMode::CoAgent) = (&searcher_baseline_eval, config.training_mode) {
            let baseline_found = baseline.found_rate;
            let allowed_found_rate = if baseline_found <= config.searcher_max_found_rate {
                config.searcher_max_found_rate
            } else {
                (baseline_found + config.searcher_max_found_rate_jump * 0.5).min(1.0)
            };
            if current.found_rate > allowed_found_rate
                || current.found_rate > (baseline_found + config.searcher_max_found_rate_jump).min(1.0)
            {
                searcher_rollback_reason = Some(format!(
                    "found_rate {:.3} exceeded gate {:.3} (baseline {:.3})",
                    current.found_rate, allowed_found_rate, baseline_found
                ));
            }
            if searcher_rollback_reason.is_some() {
                models.searcher = searcher_before_update.clone();
                current = baseline.clone();
            }
        }
        println!(
            "  phase {} complete | elapsed {:>7.1}s",
            if config.training_mode == TrainingMode::CoAgent { 3 } else { 2 },
            eval_phase_start.elapsed().as_secs_f64()
        );
        if let Some(reason) = &searcher_rollback_reason {
            println!("  searcher gate: rolled back update ({reason})");
        }
        let _ = io::stdout().flush();

        // In coagent mode, also evaluate against static searchers to track hybrid performance.
        let static_eval = if config.training_mode == TrainingMode::CoAgent {
            let static_eval_start = Instant::now();
            println!("  phase 4/4: evaluate vs static searchers");
            let _ = io::stdout().flush();
            let result = evaluate_evader_against_fixed_searchers(
                &models.evader,
                &fixed_evader_only_searchers,
                &episode_specs,
                config.max_attempts_factor,
                config.max_attempts_ratio,
                config.max_attempts_cap,
            );
            println!(
                "  phase 4 complete | elapsed {:>7.1}s",
                static_eval_start.elapsed().as_secs_f64()
            );
            let _ = io::stdout().flush();
            Some(result)
        } else {
            None
        };

        // Best-model tracking: optimize for robust escaping, not raw reward alone.
        // This is the main pressure from the visualizer finding: long escapes are good,
        // but a model that still gets caught often should not be promoted over a lower-found-rate model.
        let evader_selection_score =
            robust_evader_selection_score(&current, static_eval.as_ref());
        let is_new_best = evader_selection_score > best_evader_selection_score;
        if is_new_best {
            best_evader_score = current.average_evader_reward;
            best_evader_selection_score = evader_selection_score;
            best_evader_model = models.evader.clone();
            best_generation = generation + 1;
            no_improve_count = 0;
            let _ = write_json_pretty_atomic(&best_evader_path, &best_evader_model);
            // Joint snapshot of both models at the moment the evader peaked. The promotion
            // step uses this so it never pairs a peak evader with a searcher that trained
            // further and became dominant in later generations.
            let best_pair_path = config.output_dir.join("best_pair_models.json");
            let _ = write_json_pretty_atomic(&best_pair_path, &SelfPlayModels {
                evader: models.evader.clone(),
                searcher: models.searcher.clone(),
            });
            // Snapshot to hall immediately so this peak model is available as an opponent.
            if hall_size > 0 {
                evader_hall.push(models.evader.clone());
                if evader_hall.len() > hall_size { evader_hall.remove(0); }
            }
        } else {
            no_improve_count += 1;
        }

        // Symmetric best tracking for the searcher (CoAgent only).
        if config.training_mode == TrainingMode::CoAgent
            && current.average_searcher_reward > best_searcher_score
        {
            best_searcher_score = current.average_searcher_reward;
            best_searcher_model = models.searcher.clone();
            best_searcher_generation = generation + 1;
            let _ = write_json_pretty_atomic(&best_searcher_path, &best_searcher_model);
            if hall_size > 0 {
                searcher_hall.push(models.searcher.clone());
                if searcher_hall.len() > hall_size { searcher_hall.remove(0); }
            }
        }

        let low_variance_warning = evader_fitness_std < 0.1;
        let learned_matchup_collapse = config.training_mode == TrainingMode::CoAgent
            && current.found_rate >= 0.95
            && current.survival_budget_ratio <= 0.35;
        let mut status_parts = Vec::new();
        status_parts.push(if is_new_best { "best" } else { "ok" });
        if config.training_mode == TrainingMode::CoAgent && !should_update_searcher {
            status_parts.push("searcher held");
        }
        if searcher_rollback_reason.is_some() {
            status_parts.push("searcher rollback");
        }
        if low_variance_warning {
            status_parts.push("collapse risk: low fitness variance");
        }
        if learned_matchup_collapse {
            status_parts.push("learned matchup collapse: searcher dominates");
        }
        let status = status_parts.join(", ");

        match config.training_mode {
            TrainingMode::Static => println!(
                "  summary\n    elapsed:       {:>8.1}s\n    searcher:      {:>8.2}\n    evader:        {:>8.2}\n    escape score:  {:>8.2}\n    found rate:    {:>8.3}\n    budget used:   {:>8.1}%\n    avg attempts:  {:>8.2}/{:>8.2}\n    fitness std:   {:>8.3}\n    status:        {}",
                generation_start.elapsed().as_secs_f64(),
                current.average_searcher_reward,
                current.average_evader_reward,
                current.escape_quality_score,
                current.found_rate,
                current.survival_budget_ratio * 100.0,
                current.average_attempts,
                current.average_max_attempts,
                evader_fitness_std,
                status,
            ),
            TrainingMode::CoAgent => println!(
                "  summary\n    elapsed:       {:>8.1}s\n    learned\n      searcher:    {:>8.2}\n      evader:      {:>8.2}\n      escape score:{:>8.2}\n      found rate:  {:>8.3}\n      budget used: {:>8.1}%\n      avg attempts:{:>8.2}/{:>8.2}\n    static check\n      evader:      {:>8.2}\n      escape score:{:>8.2}\n      found rate:  {:>8.3}\n      budget used: {:>8.1}%\n      avg attempts:{:>8.2}/{:>8.2}\n    promotion\n      score:       {:>8.2}\n      best score:  {:>8.2}\n    fitness\n      evader std:  {:>8.3}\n      searcher std:{:>8.3}\n    status:        {}",
                generation_start.elapsed().as_secs_f64(),
                current.average_searcher_reward,
                current.average_evader_reward,
                current.escape_quality_score,
                current.found_rate,
                current.survival_budget_ratio * 100.0,
                current.average_attempts,
                current.average_max_attempts,
                static_eval.as_ref().map(|e| e.average_evader_reward).unwrap_or(0.0),
                static_eval.as_ref().map(|e| e.escape_quality_score).unwrap_or(0.0),
                static_eval.as_ref().map(|e| e.found_rate).unwrap_or(0.0),
                static_eval.as_ref().map(|e| e.survival_budget_ratio * 100.0).unwrap_or(0.0),
                static_eval.as_ref().map(|e| e.average_attempts).unwrap_or(0.0),
                static_eval.as_ref().map(|e| e.average_max_attempts).unwrap_or(0.0),
                evader_selection_score,
                best_evader_selection_score,
                evader_fitness_std,
                searcher_fitness_std.unwrap_or(0.0),
                status,
            ),
        }
        let _ = io::stdout().flush();

        history.push(GenerationRecord {
            generation: generation + 1,
            searcher_score: current.average_searcher_reward,
            evader_score: current.average_evader_reward,
            escape_score: current.escape_quality_score,
            found_rate: current.found_rate,
            avg_attempts: current.average_attempts,
            avg_max_attempts: current.average_max_attempts,
            survival_budget_ratio: current.survival_budget_ratio,
            static_evader_score: static_eval.as_ref().map(|e| e.average_evader_reward),
            static_escape_score: static_eval.as_ref().map(|e| e.escape_quality_score),
            static_found_rate: static_eval.as_ref().map(|e| e.found_rate),
            static_survival_budget_ratio: static_eval.as_ref().map(|e| e.survival_budget_ratio),
            evader_fitness_std: Some(evader_fitness_std),
            searcher_fitness_std,
            population_size: active_population_size,
            min_nodes: active_min_nodes,
            max_nodes: active_max_nodes,
        });
        write_json_pretty_atomic(&history_path, &history)?;
        write_recovery_checkpoint(
            &config.output_dir,
            &models,
            &best_evader_model,
            (config.training_mode == TrainingMode::CoAgent).then_some(&best_searcher_model),
        )?;
        println!(
            "  checkpoint: saved after generation {}",
            generation + 1
        );
        let _ = io::stdout().flush();

        if training_stop_requested() {
            println!(
                "Graceful stop\n  generation: {}/{}\n  action: saved checkpoint and ending training with best model intact",
                generation + 1,
                config.generations
            );
            interrupted = true;
            break;
        }

        // Early stopping: bail out if evader has not improved for `patience` generations.
        if let Some(patience) = config.patience {
            if no_improve_count >= patience {
                println!(
                    "Early stopping\n  generation: {}/{}\n  reason: escape quality stalled for {} generations\n  best evader: {:.2} at generation {}\n  best escape score: {:.2}",
                    generation + 1,
                    config.generations,
                    patience,
                    best_evader_score,
                    best_generation,
                    best_evader_selection_score,
                );
                stopped_early = true;
                break;
            }
        }
    }

    write_recovery_checkpoint(
        &config.output_dir,
        &models,
        &best_evader_model,
        (config.training_mode == TrainingMode::CoAgent).then_some(&best_searcher_model),
    )?;

    println!(
        "Best evader\n  evader score:  {:.2}\n  escape score:  {:.2}\n  generation:    {}\n  saved:         {}",
        best_evader_score,
        best_evader_selection_score,
        best_generation,
        best_evader_path.display()
    );
    if config.training_mode == TrainingMode::CoAgent {
        println!(
            "Best searcher\n  score:      {:.2}\n  generation: {}\n  saved:      {}",
            best_searcher_score, best_searcher_generation, best_searcher_path.display()
        );
    }

    let actual_generations = history.len().saturating_sub(1);
    Ok((
        models,
        TrainingSummary {
            generations: actual_generations,
            final_searcher_score: current.average_searcher_reward,
            final_evader_score: current.average_evader_reward,
            final_escape_score: current.escape_quality_score,
            best_evader_score,
            best_evader_selection_score,
            best_generation,
            stopped_early,
            interrupted,
            best_searcher_score,
            best_searcher_generation,
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
