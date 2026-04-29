use crate::shell_finder::{
    derive_candidates_for_strategy, derive_target, normalize_search_strategy, HuntStep,
    MissRelocationPolicy, ShellFinder,
};
use crate::ml::{
    choose_evader_relocation_from_snapshot, choose_searcher_guess_from_snapshot, load_model_bundle,
    SelfPlayModels,
};
use crate::tree::{AdaptiveShuffleTree, NodeSnapshot, TreeNode};
use eframe::egui;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

pub const PURPLE_HISTORY: [&str; 3] = ["#cfc3df", "#a792c3", "#7a5ea8"];
pub const GREEN_HISTORY: [&str; 3] = ["#cae8ca", "#8fd18f", "#43a047"];

#[derive(Debug, Clone, PartialEq)]
pub struct LayoutPosition {
    pub x: f32,
    pub y: f32,
    pub key: i32,
}

fn snapshot_ref(node: &Option<NodeSnapshot>) -> Option<&NodeSnapshot> {
    node.as_ref()
}

fn boxed_snapshot_ref(node: &Option<Box<NodeSnapshot>>) -> Option<&NodeSnapshot> {
    node.as_deref()
}

fn tree_depth_ref(node: Option<&NodeSnapshot>) -> usize {
    match node {
        None => 0,
        Some(node) => 1 + tree_depth_ref(boxed_snapshot_ref(&node.left)).max(tree_depth_ref(boxed_snapshot_ref(&node.right))),
    }
}

fn tree_node_count_ref(node: Option<&NodeSnapshot>) -> usize {
    match node {
        None => 0,
        Some(node) => 1 + tree_node_count_ref(boxed_snapshot_ref(&node.left)) + tree_node_count_ref(boxed_snapshot_ref(&node.right)),
    }
}

fn subtree_span_ref(node: Option<&NodeSnapshot>) -> usize {
    match node {
        None => 0,
        Some(node) => {
            let left_span = subtree_span_ref(boxed_snapshot_ref(&node.left));
            let right_span = subtree_span_ref(boxed_snapshot_ref(&node.right));
            (left_span + right_span + 1).max(1)
        }
    }
}

pub fn tree_depth(node: &Option<NodeSnapshot>) -> usize {
    tree_depth_ref(snapshot_ref(node))
}

pub fn tree_node_count(node: &Option<NodeSnapshot>) -> usize {
    tree_node_count_ref(snapshot_ref(node))
}

pub fn subtree_span(node: &Option<NodeSnapshot>) -> usize {
    subtree_span_ref(snapshot_ref(node))
}

pub fn layout_tree(
    snapshot: &Option<NodeSnapshot>,
    width: f32,
    height: f32,
    top_pad: f32,
    side_pad: f32,
    bottom_pad: f32,
) -> (HashMap<String, LayoutPosition>, Vec<(String, String)>, f32) {
    let Some(root) = snapshot else {
        return (HashMap::new(), Vec::new(), 20.0);
    };

    let depth = tree_depth(snapshot).max(1) as f32;
    let usable_width = (width - (2.0 * side_pad)).max(200.0);
    let usable_height = (height - top_pad - bottom_pad).max(160.0);
    let y_step = usable_height / depth;
    let total_span = subtree_span(snapshot).max(1) as f32;
    let unit_width = usable_width / total_span;
    let radius = (unit_width * 0.3).min(y_step * 0.3).clamp(12.0, 34.0);

    let mut positions = HashMap::new();
    let mut edges = Vec::new();

    fn walk(
        node: &NodeSnapshot,
        level: usize,
        path: String,
        left_bound: f32,
        right_bound: f32,
        width: f32,
        side_pad: f32,
        top_pad: f32,
        y_step: f32,
        unit_width: f32,
        positions: &mut HashMap<String, LayoutPosition>,
        edges: &mut Vec<(String, String)>,
    ) {
        let x = (left_bound + right_bound) / 2.0;
        let y = top_pad + ((level as f32 + 0.5) * y_step);
        positions.insert(path.clone(), LayoutPosition { x, y, key: node.key });

        let left = node.left.as_deref();
        let right = node.right.as_deref();
        let left_span = subtree_span_ref(left);
        let right_span = subtree_span_ref(right);
        let total_child_span = (left_span + right_span).max(1) as f32;

        if let (Some(left_node), Some(right_node)) = (left, right) {
            let split = left_bound + ((right_bound - left_bound) * (left_span as f32 / total_child_span));
            walk(
                left_node,
                level + 1,
                format!("{path}L"),
                left_bound,
                split,
                width,
                side_pad,
                top_pad,
                y_step,
                unit_width,
                positions,
                edges,
            );
            walk(
                right_node,
                level + 1,
                format!("{path}R"),
                split,
                right_bound,
                width,
                side_pad,
                top_pad,
                y_step,
                unit_width,
                positions,
                edges,
            );
        } else if let Some(left_node) = left {
            let outward_width = ((right_bound - left_bound) * 0.82).max(unit_width * 1.5);
            let child_right = (left_bound + unit_width).max(x - (unit_width * 0.2));
            let child_left = side_pad.max(child_right - outward_width);
            walk(
                left_node,
                level + 1,
                format!("{path}L"),
                child_left,
                child_right,
                width,
                side_pad,
                top_pad,
                y_step,
                unit_width,
                positions,
                edges,
            );
        } else if let Some(right_node) = right {
            let outward_width = ((right_bound - left_bound) * 0.82).max(unit_width * 1.5);
            let child_left = (right_bound - unit_width).min(x + (unit_width * 0.2));
            let child_right = (width - side_pad).min(child_left + outward_width);
            walk(
                right_node,
                level + 1,
                format!("{path}R"),
                child_left,
                child_right,
                width,
                side_pad,
                top_pad,
                y_step,
                unit_width,
                positions,
                edges,
            );
        }

        if node.left.is_some() {
            edges.push((path.clone(), format!("{path}L")));
        }
        if node.right.is_some() {
            edges.push((path.clone(), format!("{path}R")));
        }
    }

    walk(
        root,
        0,
        "root".to_string(),
        side_pad,
        width - side_pad,
        width,
        side_pad,
        top_pad,
        y_step,
        unit_width,
        &mut positions,
        &mut edges,
    );
    (positions, edges, radius)
}

pub fn build_balanced_tree(node_count: i32, history_path: impl Into<PathBuf>) -> AdaptiveShuffleTree {
    let mut tree = AdaptiveShuffleTree::new(history_path);
    for key in 1..=node_count.max(1) {
        tree.insert(key);
    }
    tree
}

pub fn build_uneven_tree(node_count: i32, history_path: impl Into<PathBuf>) -> AdaptiveShuffleTree {
    let mut tree = AdaptiveShuffleTree::new(history_path);
    if node_count <= 0 {
        return tree;
    }

    let mut next_key = 1;

    fn opposite(side: &str) -> &'static str {
        if side == "left" { "right" } else { "left" }
    }

    fn make_node(key: i32) -> Rc<RefCell<TreeNode>> {
        Rc::new(RefCell::new(TreeNode {
            key,
            left: None,
            right: None,
            parent: None,
        }))
    }

    fn attach_subtree(
        parent: Option<Rc<RefCell<TreeNode>>>,
        side: Option<&str>,
        size: i32,
        bias: &str,
        depth: i32,
        next_key: &mut i32,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if size <= 0 {
            return None;
        }

        let node = make_node(*next_key);
        *next_key += 1;

        if let Some(parent_ref) = parent.as_ref() {
            if side == Some("left") {
                parent_ref.borrow_mut().left = Some(node.clone());
            } else {
                parent_ref.borrow_mut().right = Some(node.clone());
            }
            node.borrow_mut().parent = Some(Rc::downgrade(parent_ref));
        } else {
            // Root assignment is handled by the caller.
        }

        let remaining = size - 1;
        if remaining <= 0 {
            return Some(node);
        }

        if remaining == 1 {
            attach_subtree(Some(node.clone()), Some(bias), 1, opposite(bias), depth + 1, next_key);
            return Some(node);
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

        attach_subtree(Some(node.clone()), Some(heavy_side), heavy_size, next_bias, depth + 1, next_key);
        if light_size > 0 {
            attach_subtree(
                Some(node.clone()),
                Some(light_side),
                light_size,
                opposite(next_bias),
                depth + 1,
                next_key,
            );
        }
        Some(node)
    }

    tree.root = attach_subtree(None, None, node_count, "left", 0, &mut next_key);
    tree
}

pub fn normalize_generation_mode(mode: &str) -> &'static str {
    if mode == "uneven" { "uneven" } else { "balanced" }
}

pub fn normalize_playback_mode(mode: &str) -> &'static str {
    if mode == "thread-matching" { "thread-matching" } else { "fixed" }
}

pub fn normalize_target_mode(mode: &str) -> &'static str {
    if mode == "random" { "random" } else { "assigned" }
}

pub fn normalize_search_mode(mode: &str) -> &'static str {
    normalize_search_strategy(mode).as_str()
}

pub fn normalize_shell_behavior_mode(mode: &str) -> &'static str {
    match mode {
        "random" => "random",
        "adaptive" => "adaptive",
        _ => "static",
    }
}

pub fn normalize_search_controller_mode(mode: &str) -> &'static str {
    match mode {
        "model" => "model",
        _ => "algorithm",
    }
}

const VISUALIZER_RECENT_MEMORY: usize = 6;

fn compute_visualizer_max_attempts(
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

#[derive(Debug, Clone)]
pub struct VisualizerRunConfig {
    pub node_count: i32,
    pub generation_mode: String,
    pub shell_behavior_mode: String,
    pub shell_target: Option<i32>,
    pub search_controller_mode: String,
    pub search_algorithm: String,
    pub model_bundle_path: String,
    pub max_attempts_factor: usize,
    pub max_attempts_ratio: Option<f64>,
    pub max_attempts_cap: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizerPreset {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub node_count: i32,
    pub generation_mode: String,
    pub shell_behavior_mode: String,
    #[serde(default)]
    pub shell_target: Option<i32>,
    pub search_controller_mode: String,
    pub search_algorithm: String,
    /// Use "$CURRENT_MODEL" or an empty string to preserve the model currently loaded in the UI.
    #[serde(default)]
    pub model_bundle_path: String,
    pub max_attempts_factor: usize,
    #[serde(default)]
    pub max_attempts_ratio: Option<f64>,
    #[serde(default)]
    pub max_attempts_cap: Option<usize>,
    #[serde(default)]
    pub reveal_shell: Option<bool>,
    #[serde(default)]
    pub auto_rerun: Option<bool>,
    #[serde(default)]
    pub instant_auto_rerun: Option<bool>,
    #[serde(default)]
    pub delay_ms: Option<u64>,
}

const VISUALIZER_PRESET_FILE: &str = "visualizer_presets.json";
const CURRENT_MODEL_PLACEHOLDER: &str = "$CURRENT_MODEL";

fn preset_file_path() -> PathBuf {
    std::env::var("VISUALIZER_PRESETS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(VISUALIZER_PRESET_FILE))
}

fn default_visualizer_presets(_current_model_path: &str) -> Vec<VisualizerPreset> {
    let model = CURRENT_MODEL_PLACEHOLDER.to_string();

    vec![
        VisualizerPreset {
            name: "Champion Stress".to_string(),
            description: "Large uneven tree, adaptive shell, trained searcher, capped budget. Mirrors the current high-pressure model-vs-model read.".to_string(),
            node_count: 115,
            generation_mode: "uneven".to_string(),
            shell_behavior_mode: "adaptive".to_string(),
            shell_target: None,
            search_controller_mode: "model".to_string(),
            search_algorithm: "evasion-aware".to_string(),
            model_bundle_path: model.clone(),
            max_attempts_factor: 8,
            max_attempts_ratio: Some(2.0),
            max_attempts_cap: Some(40),
            reveal_shell: Some(true),
            auto_rerun: Some(true),
            instant_auto_rerun: Some(true),
            delay_ms: Some(5),
        },
        VisualizerPreset {
            name: "Static Generalization".to_string(),
            description: "Adaptive shell against the evasion-aware algorithm. Useful for checking whether the evader is robust beyond the learned searcher.".to_string(),
            node_count: 115,
            generation_mode: "uneven".to_string(),
            shell_behavior_mode: "adaptive".to_string(),
            shell_target: None,
            search_controller_mode: "algorithm".to_string(),
            search_algorithm: "evasion-aware".to_string(),
            model_bundle_path: model.clone(),
            max_attempts_factor: 8,
            max_attempts_ratio: Some(2.0),
            max_attempts_cap: Some(40),
            reveal_shell: Some(true),
            auto_rerun: Some(true),
            instant_auto_rerun: Some(true),
            delay_ms: Some(5),
        },
        VisualizerPreset {
            name: "Ascending Baseline".to_string(),
            description: "Adaptive shell against simple ascending search. This catches regressions where the evader forgets easy static opponents.".to_string(),
            node_count: 115,
            generation_mode: "uneven".to_string(),
            shell_behavior_mode: "adaptive".to_string(),
            shell_target: None,
            search_controller_mode: "algorithm".to_string(),
            search_algorithm: "ascending".to_string(),
            model_bundle_path: model.clone(),
            max_attempts_factor: 8,
            max_attempts_ratio: Some(2.0),
            max_attempts_cap: Some(40),
            reveal_shell: Some(true),
            auto_rerun: Some(true),
            instant_auto_rerun: Some(true),
            delay_ms: Some(5),
        },
        VisualizerPreset {
            name: "Small Sanity".to_string(),
            description: "Short, readable model-vs-model run for understanding behavior by eye before scaling up.".to_string(),
            node_count: 31,
            generation_mode: "uneven".to_string(),
            shell_behavior_mode: "adaptive".to_string(),
            shell_target: None,
            search_controller_mode: "model".to_string(),
            search_algorithm: "evasion-aware".to_string(),
            model_bundle_path: model.clone(),
            max_attempts_factor: 4,
            max_attempts_ratio: Some(0.40),
            max_attempts_cap: None,
            reveal_shell: Some(true),
            auto_rerun: Some(false),
            instant_auto_rerun: Some(false),
            delay_ms: Some(120),
        },
        VisualizerPreset {
            name: "Static Shell Probe".to_string(),
            description: "Fixed shell target with trained searcher. Useful for seeing searcher coverage patterns without relocation noise.".to_string(),
            node_count: 63,
            generation_mode: "balanced".to_string(),
            shell_behavior_mode: "static".to_string(),
            shell_target: Some(48),
            search_controller_mode: "model".to_string(),
            search_algorithm: "evasion-aware".to_string(),
            model_bundle_path: model,
            max_attempts_factor: 4,
            max_attempts_ratio: Some(0.40),
            max_attempts_cap: None,
            reveal_shell: Some(true),
            auto_rerun: Some(false),
            instant_auto_rerun: Some(false),
            delay_ms: Some(160),
        },
    ]
}

pub fn load_visualizer_presets(current_model_path: &str) -> (Vec<VisualizerPreset>, PathBuf, Option<String>) {
    let path = preset_file_path();
    match fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<Vec<VisualizerPreset>>(&text) {
            Ok(presets) if !presets.is_empty() => (presets, path, None),
            Ok(_) => (
                default_visualizer_presets(current_model_path),
                path,
                Some("Preset file is empty; loaded built-in presets.".to_string()),
            ),
            Err(err) => (
                default_visualizer_presets(current_model_path),
                path,
                Some(format!("Could not parse preset file; loaded built-ins instead: {err}")),
            ),
        },
        Err(_) => (default_visualizer_presets(current_model_path), path, None),
    }
}

fn save_visualizer_presets(path: &Path, presets: &[VisualizerPreset]) -> Result<(), String> {
    let text = serde_json::to_string_pretty(presets).map_err(|err| err.to_string())?;
    fs::write(path, text).map_err(|err| err.to_string())
}

fn recent_guess_window(already_guessed: &[i32]) -> Vec<i32> {
    let mut recent: Vec<i32> = already_guessed.to_vec();
    if recent.len() > VISUALIZER_RECENT_MEMORY {
        let start = recent.len() - VISUALIZER_RECENT_MEMORY;
        recent = recent[start..].to_vec();
    }
    recent
}

fn choose_random_shell_key(tree: &AdaptiveShuffleTree, excluded: &[i32]) -> Option<i32> {
    let mut keys = tree.node_keys();
    keys.sort();
    let candidates: Vec<i32> = keys
        .iter()
        .copied()
        .filter(|key| !excluded.contains(key))
        .collect();
    let mut rng = rand::thread_rng();
    candidates.choose(&mut rng).copied()
}

fn load_models_if_needed(
    shell_behavior_mode: &str,
    search_controller_mode: &str,
    model_bundle_path: &str,
) -> Result<Option<SelfPlayModels>, String> {
    if shell_behavior_mode == "adaptive" || search_controller_mode == "model" {
        Ok(Some(load_model_bundle(model_bundle_path)?))
    } else {
        Ok(None)
    }
}

pub fn build_visualizer_steps(config: &VisualizerRunConfig) -> Result<(Vec<HuntStep>, i32), String> {
    let mut tree = build_demo_tree(config.node_count, "evade_history.json", &config.generation_mode);
    let mut finder = ShellFinder::new("shell_finder_history.json");
    let shell_behavior_mode = normalize_shell_behavior_mode(&config.shell_behavior_mode);
    let search_controller_mode = normalize_search_controller_mode(&config.search_controller_mode);
    let max_attempts = compute_visualizer_max_attempts(
        config.node_count.max(1) as usize,
        config.max_attempts_factor,
        config.max_attempts_ratio,
        config.max_attempts_cap,
    );
    let models = load_models_if_needed(
        shell_behavior_mode,
        search_controller_mode,
        &config.model_bundle_path,
    )?;

    let initial_shell = match shell_behavior_mode {
        "static" => derive_target(config.node_count.max(1), config.shell_target),
        "random" => choose_random_shell_key(&tree, &[]).unwrap_or(1),
        "adaptive" => {
            let snapshot = tree
                .snapshot()
                .ok_or_else(|| "Cannot build adaptive shell behavior without a tree snapshot.".to_string())?;
            let models = models
                .as_ref()
                .ok_or_else(|| "Adaptive shell behavior requires a model bundle.".to_string())?;
            choose_evader_relocation_from_snapshot(&models.evader, &snapshot, &[], &[])
                .or_else(|| choose_random_shell_key(&tree, &[]))
                .unwrap_or(1)
        }
        _ => derive_target(config.node_count.max(1), config.shell_target),
    };

    let search_algorithm = normalize_search_strategy(&config.search_algorithm);
    let search_chooser = |tree: &AdaptiveShuffleTree,
                          guessed_in_order: &[i32],
                          guess_history: &[crate::shell_finder::GuessHistoryEntry]|
     -> Option<i32> {
        match search_controller_mode {
            "algorithm" => {
                let ordered = crate::shell_finder::derive_remaining_candidates_for_strategy(
                    tree,
                    None,
                    guessed_in_order,
                    search_algorithm,
                    guess_history,
                );
                ordered.first().copied()
            }
            "model" => {
                let snapshot = tree.snapshot()?;
                let models = models.as_ref()?;
                let recent = recent_guess_window(guessed_in_order);
                let excluded = if guessed_in_order.len() < tree.node_keys().len() {
                    guessed_in_order.to_vec()
                } else {
                    Vec::new()
                };
                choose_searcher_guess_from_snapshot(
                    &models.searcher,
                    &snapshot,
                    &recent,
                    &excluded,
                )
            }
            _ => None,
        }
    };

    let random_relocator =
        |tree: &AdaptiveShuffleTree, guessed_keys: &[i32]| choose_random_shell_key(tree, guessed_keys);
    let adaptive_relocator = |tree: &AdaptiveShuffleTree, guessed_keys: &[i32]| {
        let snapshot = tree.snapshot()?;
        let models = models.as_ref()?;
        let recent_guesses = recent_guess_window(guessed_keys);
        choose_evader_relocation_from_snapshot(
            &models.evader,
            &snapshot,
            &recent_guesses,
            guessed_keys,
        )
        .or_else(|| choose_random_shell_key(tree, guessed_keys))
    };
    let relocation_policy = match shell_behavior_mode {
        "static" => MissRelocationPolicy::None,
        "random" => MissRelocationPolicy::Callback(&random_relocator),
        "adaptive" => MissRelocationPolicy::Callback(&adaptive_relocator),
        _ => MissRelocationPolicy::None,
    };

    let steps = finder.iter_hunt_with_relocation_policy_limited(
        &mut tree,
        initial_shell,
        &search_chooser,
        relocation_policy,
        Some(max_attempts),
        false,
    )?;
    Ok((steps, initial_shell))
}

pub fn build_demo_tree(
    node_count: i32,
    history_path: impl Into<PathBuf>,
    generation_mode: &str,
) -> AdaptiveShuffleTree {
    if normalize_generation_mode(generation_mode) == "uneven" {
        build_uneven_tree(node_count, history_path)
    } else {
        build_balanced_tree(node_count, history_path)
    }
}

pub fn resolve_target<R: Rng + ?Sized>(
    node_count: i32,
    target_mode: &str,
    explicit_target: Option<i32>,
    rng: Option<&mut R>,
) -> i32 {
    if normalize_target_mode(target_mode) == "random" {
        if let Some(chooser) = rng {
            return chooser.gen_range(1..=node_count.max(1));
        }
        let mut thread_rng = rand::thread_rng();
        return thread_rng.gen_range(1..=node_count.max(1));
    }
    derive_target(node_count, explicit_target)
}

pub fn estimate_step_delay(base_delay_ms: u64, step: &HuntStep, playback_mode: &str) -> u64 {
    if normalize_playback_mode(playback_mode) == "fixed" {
        return base_delay_ms.max(5);
    }

    let node_count = tree_node_count(&step.tree_snapshot).max(1) as f64;
    let search_visited = step.operation_metrics.search_visited as f64;
    let shuffle_touched = step.operation_metrics.shuffle_touched as f64;

    match step.phase.as_str() {
        "search" => {
            if search_visited <= 0.0 {
                return ((base_delay_ms as f64) * 0.45).round().max(5.0) as u64;
            }
            let normalized = search_visited / node_count;
            let factor = normalized.clamp(0.35, 1.35);
            ((base_delay_ms as f64) * factor).round().max(5.0) as u64
        }
        "resolve" => {
            let work_units = if shuffle_touched > 0.0 {
                shuffle_touched
            } else {
                (search_visited / 2.0).max(1.0)
            };
            let normalized = work_units / node_count;
            let factor = (normalized + if shuffle_touched > 0.0 { 0.2 } else { 0.0 }).clamp(0.35, 2.8);
            ((base_delay_ms as f64) * factor).round().max(5.0) as u64
        }
        _ => {
            if search_visited + shuffle_touched <= 0.0 {
                return ((base_delay_ms as f64) * 0.6).round().max(5.0) as u64;
            }
            let normalized = (search_visited + shuffle_touched) / node_count;
            let factor = normalized.clamp(0.45, 2.4);
            ((base_delay_ms as f64) * factor).round().max(5.0) as u64
        }
    }
}

pub fn extract_window_snapshots(step: &HuntStep) -> Vec<Option<NodeSnapshot>> {
    let mut snapshots = vec![step.tree_snapshot.clone()];
    let mut seen = vec![format!("{:?}", step.tree_snapshot)];

    for entry in step.tree_history.iter().rev() {
        let marker = format!("{:?}", entry.snapshot);
        if entry.snapshot.is_none() || seen.contains(&marker) {
            continue;
        }
        snapshots.push(entry.snapshot.clone());
        seen.push(marker);
        if snapshots.len() == 3 {
            break;
        }
    }

    while snapshots.len() < 3 {
        snapshots.push(None);
    }
    snapshots
}

pub fn build_steps(
    node_count: i32,
    target: Option<i32>,
    candidates: Option<Vec<i32>>,
    generation_mode: &str,
    target_mode: &str,
    search_strategy: &str,
) -> Result<(Vec<HuntStep>, i32, Vec<i32>), String> {
    let mut tree = build_demo_tree(node_count, "evade_history.json", generation_mode);
    let resolved_target = resolve_target(node_count, target_mode, target, None::<&mut rand::rngs::ThreadRng>);
    let resolved_candidates = derive_candidates_for_strategy(
        &tree,
        node_count,
        candidates.clone(),
        normalize_search_strategy(search_strategy),
    );
    let mut finder = ShellFinder::new("shell_finder_history.json");
    let steps = if let Some(ref explicit_candidates) = candidates {
        finder.iter_hunt(&mut tree, resolved_target, explicit_candidates)?
    } else {
        finder.iter_hunt_with_strategy(
            &mut tree,
            resolved_target,
            None,
            normalize_search_strategy(search_strategy),
            None,
        )?
    };
    Ok((steps, resolved_target, resolved_candidates))
}

fn parse_hex_color(hex: &str) -> egui::Color32 {
    let value = hex.trim_start_matches('#');
    if value.len() != 6 {
        return egui::Color32::LIGHT_GRAY;
    }
    let r = u8::from_str_radix(&value[0..2], 16).unwrap_or(200);
    let g = u8::from_str_radix(&value[2..4], 16).unwrap_or(200);
    let b = u8::from_str_radix(&value[4..6], 16).unwrap_or(200);
    egui::Color32::from_rgb(r, g, b)
}

#[derive(Debug, Clone)]
struct VisualizationRunRecord {
    run: usize,
    epoch: usize,
    attempts: usize,
    max_attempts: usize,
    found: bool,
    escape_attempts: f64,
    survival_ratio: f64,
    running_escape_avg: f64,
    running_found_rate: f64,
    search_work: usize,
    shuffle_work: usize,
    node_keys: Vec<i32>,
    searcher_node_hits: BTreeMap<i32, usize>,
    evader_node_hits: BTreeMap<i32, usize>,
}

struct ChartSeries {
    name: &'static str,
    values: Vec<f64>,
    color: egui::Color32,
}

pub struct TreeVisualizerApp {
    pub reveal_shell: bool,
    pub delay_ms: u64,
    pub node_count: i32,
    pub generation_mode: String,
    pub shell_behavior_mode: String,
    pub shell_target: i32,
    pub search_controller_mode: String,
    pub search_algorithm: String,
    pub max_attempts_factor: usize,
    pub max_attempts_ratio: f64,
    pub use_attempt_ratio: bool,
    pub max_attempts_cap: usize,
    pub use_attempt_cap: bool,
    pub paused: bool,
    pub auto_rerun: bool,
    pub instant_auto_rerun: bool,
    pub run_count: usize,
    pub steps: Vec<HuntStep>,
    pub index: usize,
    pub current_step: Option<HuntStep>,
    pub model_bundle_path: String,
    pub status_message: String,
    pub presets: Vec<VisualizerPreset>,
    pub selected_preset: usize,
    pub preset_name_input: String,
    pub preset_file_path: PathBuf,
    pub graph_history_limit: usize,
    graph_history: Vec<VisualizationRunRecord>,
    active_parameter_signature: String,
    active_parameter_epoch: usize,
    active_max_attempts: usize,
    recorded_current_run: bool,
    last_tick: Instant,
}

impl TreeVisualizerApp {
    pub fn new(
        steps: Vec<HuntStep>,
        reveal_shell: bool,
        delay_ms: u64,
        initial_node_count: i32,
        generation_mode: &str,
        shell_behavior_mode: &str,
        shell_target: i32,
        search_controller_mode: &str,
        search_algorithm: &str,
        max_attempts_factor: usize,
        max_attempts_ratio: Option<f64>,
        max_attempts_cap: Option<usize>,
        auto_rerun: bool,
        instant_auto_rerun: bool,
        model_bundle_path: String,
    ) -> Self {
        let (presets, preset_file_path, preset_warning) = load_visualizer_presets(&model_bundle_path);
        let mut app = Self {
            reveal_shell,
            delay_ms: delay_ms.max(5),
            node_count: initial_node_count.max(1),
            generation_mode: normalize_generation_mode(generation_mode).to_string(),
            shell_behavior_mode: normalize_shell_behavior_mode(shell_behavior_mode).to_string(),
            shell_target: shell_target.max(1),
            search_controller_mode: normalize_search_controller_mode(search_controller_mode).to_string(),
            search_algorithm: normalize_search_mode(search_algorithm).to_string(),
            max_attempts_factor: max_attempts_factor.max(1),
            max_attempts_ratio: max_attempts_ratio.unwrap_or(0.45),
            use_attempt_ratio: max_attempts_ratio.is_some(),
            max_attempts_cap: max_attempts_cap.unwrap_or(10),
            use_attempt_cap: max_attempts_cap.is_some(),
            paused: false,
            auto_rerun,
            instant_auto_rerun,
            run_count: 1,
            steps,
            index: 0,
            current_step: None,
            model_bundle_path,
            status_message: preset_warning.unwrap_or_default(),
            presets,
            selected_preset: 0,
            preset_name_input: "Custom setup".to_string(),
            preset_file_path,
            graph_history_limit: 160,
            graph_history: Vec::new(),
            active_parameter_signature: String::new(),
            active_parameter_epoch: 0,
            active_max_attempts: 1,
            recorded_current_run: false,
            last_tick: Instant::now(),
        };
        app.active_parameter_signature = app.parameter_signature();
        app.active_max_attempts = app.effective_max_attempts();
        app
    }

    fn restart(&mut self) {
        let next_signature = self.parameter_signature();
        let mut reset_graph_average = false;
        if next_signature != self.active_parameter_signature {
            self.active_parameter_epoch += 1;
            self.active_parameter_signature = next_signature;
            reset_graph_average = true;
        }
        self.active_max_attempts = self.effective_max_attempts();

        let config = VisualizerRunConfig {
            node_count: self.node_count,
            generation_mode: self.generation_mode.clone(),
            shell_behavior_mode: self.shell_behavior_mode.clone(),
            shell_target: Some(self.shell_target),
            search_controller_mode: self.search_controller_mode.clone(),
            search_algorithm: self.search_algorithm.clone(),
            model_bundle_path: self.model_bundle_path.clone(),
            max_attempts_factor: self.max_attempts_factor,
            max_attempts_ratio: self.use_attempt_ratio.then_some(self.max_attempts_ratio),
            max_attempts_cap: self.use_attempt_cap.then_some(self.max_attempts_cap),
        };

        match build_visualizer_steps(&config) {
            Ok((steps, resolved_shell)) => {
                self.steps = steps;
                self.index = 0;
                self.current_step = None;
                self.shell_target = resolved_shell;
                self.last_tick = Instant::now();
                self.run_count += 1;
                self.recorded_current_run = false;
                self.status_message = if reset_graph_average {
                    "Graph averages reset because run parameters changed.".to_string()
                } else {
                    String::new()
                };
            }
            Err(err) => {
                self.status_message = err;
            }
        }
    }

    fn selected_preset_name(&self) -> String {
        self.presets
            .get(self.selected_preset)
            .map(|preset| preset.name.clone())
            .unwrap_or_else(|| "No presets".to_string())
    }

    fn apply_preset(&mut self, preset: &VisualizerPreset) {
        self.node_count = preset.node_count.max(1);
        self.generation_mode = normalize_generation_mode(&preset.generation_mode).to_string();
        self.shell_behavior_mode = normalize_shell_behavior_mode(&preset.shell_behavior_mode).to_string();
        if let Some(target) = preset.shell_target {
            self.shell_target = target.max(1);
        }
        self.search_controller_mode =
            normalize_search_controller_mode(&preset.search_controller_mode).to_string();
        self.search_algorithm = normalize_search_mode(&preset.search_algorithm).to_string();
        self.max_attempts_factor = preset.max_attempts_factor.max(1);
        self.max_attempts_ratio = preset.max_attempts_ratio.unwrap_or(self.max_attempts_ratio).max(0.05);
        self.use_attempt_ratio = preset.max_attempts_ratio.is_some();
        self.max_attempts_cap = preset.max_attempts_cap.unwrap_or(self.max_attempts_cap).max(1);
        self.use_attempt_cap = preset.max_attempts_cap.is_some();
        if let Some(reveal_shell) = preset.reveal_shell {
            self.reveal_shell = reveal_shell;
        }
        if let Some(auto_rerun) = preset.auto_rerun {
            self.auto_rerun = auto_rerun;
        }
        if let Some(instant_auto_rerun) = preset.instant_auto_rerun {
            self.instant_auto_rerun = instant_auto_rerun;
        }
        if let Some(delay_ms) = preset.delay_ms {
            self.delay_ms = delay_ms.max(5);
        }

        let model_path = preset.model_bundle_path.trim();
        if !model_path.is_empty() && model_path != CURRENT_MODEL_PLACEHOLDER {
            self.model_bundle_path = model_path.to_string();
        }

        let preset_name = preset.name.clone();
        self.restart();
        if self.status_message.is_empty() {
            self.status_message = format!("Loaded preset: {preset_name}");
        } else if self.status_message.starts_with("Graph averages reset") {
            self.status_message = format!("Loaded preset: {preset_name}. {}", self.status_message);
        }
    }

    fn preset_from_current(&self, name: String) -> VisualizerPreset {
        VisualizerPreset {
            name,
            description: "Saved from the current visualizer controls.".to_string(),
            node_count: self.node_count.max(1),
            generation_mode: self.generation_mode.clone(),
            shell_behavior_mode: self.shell_behavior_mode.clone(),
            shell_target: (self.shell_behavior_mode == "static").then_some(self.shell_target.max(1)),
            search_controller_mode: self.search_controller_mode.clone(),
            search_algorithm: self.search_algorithm.clone(),
            model_bundle_path: self.model_bundle_path.clone(),
            max_attempts_factor: self.max_attempts_factor.max(1),
            max_attempts_ratio: self.use_attempt_ratio.then_some(self.max_attempts_ratio),
            max_attempts_cap: self.use_attempt_cap.then_some(self.max_attempts_cap),
            reveal_shell: Some(self.reveal_shell),
            auto_rerun: Some(self.auto_rerun),
            instant_auto_rerun: Some(self.instant_auto_rerun),
            delay_ms: Some(self.delay_ms.max(5)),
        }
    }

    fn save_current_preset(&mut self) {
        let name = if self.preset_name_input.trim().is_empty() {
            format!("Custom setup {}", self.presets.len() + 1)
        } else {
            self.preset_name_input.trim().to_string()
        };
        let preset = self.preset_from_current(name.clone());
        if let Some(existing) = self.presets.iter_mut().find(|item| item.name == name) {
            *existing = preset;
        } else {
            self.presets.push(preset);
            self.selected_preset = self.presets.len().saturating_sub(1);
        }

        match save_visualizer_presets(&self.preset_file_path, &self.presets) {
            Ok(()) => {
                self.status_message = format!(
                    "Saved preset '{}' to {}",
                    name,
                    self.preset_file_path.display()
                );
            }
            Err(err) => {
                self.status_message = format!("Could not save preset: {err}");
            }
        }
    }

    fn delete_selected_preset(&mut self) {
        if self.presets.is_empty() {
            return;
        }
        let removed = self.presets.remove(self.selected_preset.min(self.presets.len() - 1));
        self.selected_preset = self.selected_preset.saturating_sub(1).min(self.presets.len().saturating_sub(1));
        match save_visualizer_presets(&self.preset_file_path, &self.presets) {
            Ok(()) => {
                self.status_message = format!("Deleted preset '{}'.", removed.name);
            }
            Err(err) => {
                self.status_message = format!("Deleted preset locally, but could not save file: {err}");
            }
        }
    }

    fn reload_presets(&mut self) {
        let (presets, path, warning) = load_visualizer_presets(&self.model_bundle_path);
        self.presets = presets;
        self.preset_file_path = path;
        self.selected_preset = self.selected_preset.min(self.presets.len().saturating_sub(1));
        self.status_message = warning.unwrap_or_else(|| {
            format!("Reloaded {} preset(s).", self.presets.len())
        });
    }

    fn draw_presets(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Presets")
            .default_open(true)
            .show(ui, |ui| {
            ui.label("Load repeatable visualization setups for model-vs-model, static baselines, and stress tests.");
            ui.horizontal_wrapped(|ui| {
                ui.label("Preset");
                egui::ComboBox::from_id_salt("visualizer_preset_picker")
                    .selected_text(self.selected_preset_name())
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        for (idx, preset) in self.presets.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_preset, idx, &preset.name);
                        }
                    });

                if ui.button("Apply + Restart").clicked() {
                    if let Some(preset) = self.presets.get(self.selected_preset).cloned() {
                        self.apply_preset(&preset);
                    }
                }
                if ui.button("Reload").clicked() {
                    self.reload_presets();
                }
                if ui.button("Delete").clicked() {
                    self.delete_selected_preset();
                }
            });

            if let Some(preset) = self.presets.get(self.selected_preset) {
                ui.label(format!(
                    "{} | nodes={} {} | shell={} | search={}{} | budget factor={} ratio={} cap={} | rerun={} | model={}",
                    preset.name,
                    preset.node_count,
                    normalize_generation_mode(&preset.generation_mode),
                    normalize_shell_behavior_mode(&preset.shell_behavior_mode),
                    normalize_search_controller_mode(&preset.search_controller_mode),
                    if normalize_search_controller_mode(&preset.search_controller_mode) == "algorithm" {
                        format!(":{}", normalize_search_mode(&preset.search_algorithm))
                    } else {
                        String::new()
                    },
                    preset.max_attempts_factor,
                    preset
                        .max_attempts_ratio
                        .map(|value| format!("{value:.2}"))
                        .unwrap_or_else(|| "off".to_string()),
                    preset
                        .max_attempts_cap
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "off".to_string()),
                    match (
                        preset.auto_rerun.unwrap_or(false),
                        preset.instant_auto_rerun.unwrap_or(false),
                    ) {
                        (true, true) => "instant",
                        (true, false) => "delayed",
                        (false, _) => "off",
                    },
                    if preset.model_bundle_path.trim().is_empty()
                        || preset.model_bundle_path.trim() == CURRENT_MODEL_PLACEHOLDER
                    {
                        "current model"
                    } else {
                        preset.model_bundle_path.as_str()
                    },
                ));
                if !preset.description.trim().is_empty() {
                    ui.label(egui::RichText::new(&preset.description).small());
                }
            }

            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label("Save current as");
                ui.text_edit_singleline(&mut self.preset_name_input);
                if ui.button("Save Current").clicked() {
                    self.save_current_preset();
                }
                ui.label(format!("file: {}", self.preset_file_path.display()));
            });
        });
    }

    fn advance(&mut self) {
        if self.index >= self.steps.len() {
            return;
        }
        self.current_step = self.steps.get(self.index).cloned();
        self.index += 1;
        self.last_tick = Instant::now();
    }

    fn maybe_advance(&mut self) {
        if self.paused {
            return;
        }

        if let Some(step) = &self.current_step {
            let delay = estimate_step_delay(self.delay_ms, step, "fixed");
            if self.last_tick.elapsed() >= Duration::from_millis(delay) {
                self.advance();
            }
        } else {
            let initial_delay_ms = if self.auto_rerun && self.instant_auto_rerun {
                0
            } else {
                120
            };
            if self.last_tick.elapsed() >= Duration::from_millis(initial_delay_ms) {
                self.advance();
            }
        }

        let rerun_delay_ms = if self.instant_auto_rerun { 0 } else { 600 };
        if self.index >= self.steps.len()
            && self.auto_rerun
            && self
                .current_step
                .as_ref()
                .map(|step| step.phase == "resolve")
                .unwrap_or(false)
            && self.last_tick.elapsed() >= Duration::from_millis(rerun_delay_ms)
        {
            self.record_completed_run_if_needed();
            self.restart();
        }
    }

    fn effective_max_attempts(&self) -> usize {
        compute_visualizer_max_attempts(
            self.node_count.max(1) as usize,
            self.max_attempts_factor,
            self.use_attempt_ratio.then_some(self.max_attempts_ratio),
            self.use_attempt_cap.then_some(self.max_attempts_cap),
        )
    }

    fn parameter_signature(&self) -> String {
        format!(
            "nodes={}|tree={}|shell={}|target={}|search={}|algo={}|factor={}|ratio={:?}|cap={:?}|model={}",
            self.node_count.max(1),
            self.generation_mode,
            self.shell_behavior_mode,
            if self.shell_behavior_mode == "static" { self.shell_target } else { 0 },
            self.search_controller_mode,
            self.search_algorithm,
            self.max_attempts_factor,
            self.use_attempt_ratio.then_some((self.max_attempts_ratio * 1000.0).round() as i64),
            self.use_attempt_cap.then_some(self.max_attempts_cap),
            self.model_bundle_path,
        )
    }

    fn collect_snapshot_keys(snapshot: Option<&NodeSnapshot>, keys: &mut Vec<i32>) {
        let Some(node) = snapshot else {
            return;
        };
        keys.push(node.key);
        Self::collect_snapshot_keys(node.left.as_deref(), keys);
        Self::collect_snapshot_keys(node.right.as_deref(), keys);
    }

    fn count_node_hits_in_steps(
        steps: &[HuntStep],
        step_limit: usize,
    ) -> (Vec<i32>, BTreeMap<i32, usize>, BTreeMap<i32, usize>) {
        let mut node_keys = Vec::new();
        for step in steps {
            Self::collect_snapshot_keys(step.tree_snapshot.as_ref(), &mut node_keys);
        }
        node_keys.sort_unstable();
        node_keys.dedup();

        let mut searcher_hits = BTreeMap::new();
        let mut evader_hits = BTreeMap::new();
        for key in &node_keys {
            searcher_hits.insert(*key, 0);
            evader_hits.insert(*key, 0);
        }

        for step in steps.iter().take(step_limit.min(steps.len())) {
            if step.phase == "search" {
                if let Some(guess) = step.guess {
                    *searcher_hits.entry(guess).or_insert(0) += 1;
                }
            }

            if matches!(step.phase.as_str(), "hidden" | "resolve") {
                if let Some(shell_key) = step.shell_key {
                    *evader_hits.entry(shell_key).or_insert(0) += 1;
                }
            }
        }

        (node_keys, searcher_hits, evader_hits)
    }

    fn active_node_hit_totals(&self) -> (Vec<i32>, Vec<usize>, Vec<usize>) {
        let mut node_keys = Vec::new();
        let mut searcher_hits: BTreeMap<i32, usize> = BTreeMap::new();
        let mut evader_hits: BTreeMap<i32, usize> = BTreeMap::new();

        for record in self
            .graph_history
            .iter()
            .filter(|record| record.epoch == self.active_parameter_epoch)
        {
            node_keys.extend(record.node_keys.iter().copied());
            for (key, count) in &record.searcher_node_hits {
                *searcher_hits.entry(*key).or_insert(0) += count;
            }
            for (key, count) in &record.evader_node_hits {
                *evader_hits.entry(*key).or_insert(0) += count;
            }
        }

        if !self.recorded_current_run {
            let (current_keys, current_searcher_hits, current_evader_hits) =
                Self::count_node_hits_in_steps(&self.steps, self.index.min(self.steps.len()));
            node_keys.extend(current_keys);
            for (key, count) in current_searcher_hits {
                *searcher_hits.entry(key).or_insert(0) += count;
            }
            for (key, count) in current_evader_hits {
                *evader_hits.entry(key).or_insert(0) += count;
            }
        }

        if node_keys.is_empty() {
            for step in &self.steps {
                Self::collect_snapshot_keys(step.tree_snapshot.as_ref(), &mut node_keys);
            }
        }

        node_keys.sort_unstable();
        node_keys.dedup();
        let searcher_values = node_keys
            .iter()
            .map(|key| searcher_hits.get(key).copied().unwrap_or(0))
            .collect();
        let evader_values = node_keys
            .iter()
            .map(|key| evader_hits.get(key).copied().unwrap_or(0))
            .collect();

        (node_keys, searcher_values, evader_values)
    }

    fn record_completed_run_if_needed(&mut self) {
        if self.recorded_current_run || self.index < self.steps.len() {
            return;
        }

        let Some(step) = self.current_step.as_ref() else {
            return;
        };
        if step.phase != "resolve" {
            return;
        }

        let attempts = step
            .attempt
            .max(step.guess_history.iter().map(|entry| entry.attempt).max().unwrap_or(0))
            .max(1);
        let max_attempts = self.active_max_attempts.max(attempts).max(1);
        let escape_attempts = if step.found { attempts } else { max_attempts } as f64;
        let survival_ratio = (escape_attempts / max_attempts as f64).clamp(0.0, 1.0);
        let search_work = self
            .steps
            .iter()
            .map(|step| step.operation_metrics.search_visited)
            .sum();
        let shuffle_work = self
            .steps
            .iter()
            .map(|step| step.operation_metrics.shuffle_touched)
            .sum();

        let prior_epoch_records: Vec<&VisualizationRunRecord> = self
            .graph_history
            .iter()
            .filter(|record| record.epoch == self.active_parameter_epoch)
            .collect();
        let prior_count = prior_epoch_records.len() as f64;
        let prior_escape_sum: f64 = prior_epoch_records
            .iter()
            .map(|record| record.escape_attempts)
            .sum();
        let prior_found_sum: f64 = prior_epoch_records
            .iter()
            .filter(|record| record.found)
            .count() as f64;

        let denominator = prior_count + 1.0;
        let running_escape_avg = (prior_escape_sum + escape_attempts) / denominator;
        let running_found_rate = (prior_found_sum + if step.found { 1.0 } else { 0.0 }) / denominator;
        let (node_keys, searcher_node_hits, evader_node_hits) =
            Self::count_node_hits_in_steps(&self.steps, self.steps.len());

        self.graph_history.push(VisualizationRunRecord {
            run: self.run_count,
            epoch: self.active_parameter_epoch,
            attempts,
            max_attempts,
            found: step.found,
            escape_attempts,
            survival_ratio,
            running_escape_avg,
            running_found_rate,
            search_work,
            shuffle_work,
            node_keys,
            searcher_node_hits,
            evader_node_hits,
        });
        if self.graph_history.len() > self.graph_history_limit {
            let excess = self.graph_history.len() - self.graph_history_limit;
            self.graph_history.drain(0..excess);
        }
        self.recorded_current_run = true;
    }

    fn epoch_records(&self) -> Vec<&VisualizationRunRecord> {
        self.graph_history
            .iter()
            .filter(|record| record.epoch == self.active_parameter_epoch)
            .collect()
    }

    fn draw_line_chart(
        ui: &mut egui::Ui,
        title: &str,
        series: &[ChartSeries],
        fixed_min: Option<f64>,
        fixed_max: Option<f64>,
        height: f32,
    ) {
        ui.label(egui::RichText::new(title).strong());
        let desired = egui::vec2(ui.available_width().max(220.0), height);
        let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let bg = parse_hex_color("#171a1d");
        let grid = parse_hex_color("#3b424a");
        let text = parse_hex_color("#d6d0c5");
        painter.rect_filled(rect, 8.0, bg);

        let plot_rect = rect.shrink2(egui::vec2(34.0, 22.0));
        let max_len = series.iter().map(|s| s.values.len()).max().unwrap_or(0);
        if max_len == 0 || series.iter().all(|s| s.values.iter().all(|v| !v.is_finite())) {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No completed visualization runs yet",
                egui::FontId::proportional(12.0),
                text,
            );
            return;
        }

        let mut y_min = fixed_min.unwrap_or(f64::INFINITY);
        let mut y_max = fixed_max.unwrap_or(f64::NEG_INFINITY);
        if fixed_min.is_none() || fixed_max.is_none() {
            for value in series.iter().flat_map(|s| s.values.iter().copied()) {
                if value.is_finite() {
                    if fixed_min.is_none() {
                        y_min = y_min.min(value);
                    }
                    if fixed_max.is_none() {
                        y_max = y_max.max(value);
                    }
                }
            }
        }
        if !y_min.is_finite() || !y_max.is_finite() {
            y_min = 0.0;
            y_max = 1.0;
        }
        if (y_max - y_min).abs() < f64::EPSILON {
            y_min -= 1.0;
            y_max += 1.0;
        } else if fixed_min.is_none() || fixed_max.is_none() {
            let pad = (y_max - y_min) * 0.08;
            if fixed_min.is_none() {
                y_min -= pad;
            }
            if fixed_max.is_none() {
                y_max += pad;
            }
        }

        for i in 0..=4 {
            let t = i as f32 / 4.0;
            let y = egui::lerp(plot_rect.bottom()..=plot_rect.top(), t);
            painter.line_segment(
                [egui::pos2(plot_rect.left(), y), egui::pos2(plot_rect.right(), y)],
                egui::Stroke::new(1.0, grid.linear_multiply(0.55)),
            );
            let label_value = y_min + (y_max - y_min) * t as f64;
            painter.text(
                egui::pos2(rect.left() + 4.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{label_value:.1}"),
                egui::FontId::monospace(9.0),
                text.linear_multiply(0.82),
            );
        }

        for chart in series {
            let points: Vec<egui::Pos2> = chart
                .values
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| {
                    if !value.is_finite() {
                        return None;
                    }
                    let x_t = if max_len <= 1 {
                        0.5
                    } else {
                        idx as f32 / (max_len - 1) as f32
                    };
                    let y_t = ((*value - y_min) / (y_max - y_min)).clamp(0.0, 1.0) as f32;
                    Some(egui::pos2(
                        egui::lerp(plot_rect.left()..=plot_rect.right(), x_t),
                        egui::lerp(plot_rect.bottom()..=plot_rect.top(), y_t),
                    ))
                })
                .collect();

            for window in points.windows(2) {
                painter.line_segment(
                    [window[0], window[1]],
                    egui::Stroke::new(2.0, chart.color),
                );
            }
            if let Some(last) = points.last() {
                painter.circle_filled(*last, 3.5, chart.color);
            }
        }

        let mut legend_x = plot_rect.left();
        for chart in series {
            painter.circle_filled(
                egui::pos2(legend_x + 5.0, rect.bottom() - 10.0),
                4.0,
                chart.color,
            );
            painter.text(
                egui::pos2(legend_x + 14.0, rect.bottom() - 10.0),
                egui::Align2::LEFT_CENTER,
                chart.name,
                egui::FontId::proportional(10.0),
                text,
            );
            legend_x += 92.0;
        }
    }

    fn draw_node_hit_bar_chart(
        ui: &mut egui::Ui,
        title: &str,
        node_keys: &[i32],
        counts: &[usize],
        color: egui::Color32,
        height: f32,
        min_width: f32,
    ) {
        ui.label(egui::RichText::new(title).strong());
        let desired = egui::vec2(min_width.max(ui.available_width()).max(260.0), height);
        let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let bg = parse_hex_color("#171a1d");
        let grid = parse_hex_color("#3b424a");
        let text = parse_hex_color("#d6d0c5");
        painter.rect_filled(rect, 8.0, bg);

        let plot_rect = rect.shrink2(egui::vec2(36.0, 28.0));
        if node_keys.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No spawned tree nodes yet",
                egui::FontId::proportional(12.0),
                text,
            );
            return;
        }

        let max_count = counts.iter().copied().max().unwrap_or(0).max(1);
        for i in 0..=4 {
            let t = i as f32 / 4.0;
            let y = egui::lerp(plot_rect.bottom()..=plot_rect.top(), t);
            painter.line_segment(
                [egui::pos2(plot_rect.left(), y), egui::pos2(plot_rect.right(), y)],
                egui::Stroke::new(1.0, grid.linear_multiply(0.55)),
            );
            let label_value = (max_count as f32 * t).round() as usize;
            painter.text(
                egui::pos2(rect.left() + 4.0, y),
                egui::Align2::LEFT_CENTER,
                label_value.to_string(),
                egui::FontId::monospace(9.0),
                text.linear_multiply(0.82),
            );
        }

        let n = node_keys.len().max(1);
        let slot_width = plot_rect.width() / n as f32;
        let bar_width = (slot_width * 0.72).max(1.0);
        for (idx, key) in node_keys.iter().enumerate() {
            let count = counts.get(idx).copied().unwrap_or(0);
            let x_center = plot_rect.left() + (idx as f32 + 0.5) * slot_width;
            let y_t = count as f32 / max_count as f32;
            let y_top = egui::lerp(plot_rect.bottom()..=plot_rect.top(), y_t);
            let bar_rect = egui::Rect::from_min_max(
                egui::pos2(x_center - bar_width / 2.0, y_top),
                egui::pos2(x_center + bar_width / 2.0, plot_rect.bottom()),
            );
            painter.rect_filled(bar_rect, 2.0, color);

            let label_every = ((n as f32 / 12.0).ceil() as usize).max(1);
            if idx == 0 || idx + 1 == n || idx % label_every == 0 {
                painter.text(
                    egui::pos2(x_center, rect.bottom() - 12.0),
                    egui::Align2::CENTER_CENTER,
                    key.to_string(),
                    egui::FontId::monospace(8.5),
                    text.linear_multiply(0.78),
                );
            }
        }

        if let Some(pointer) = response.hover_pos() {
            if plot_rect.contains(pointer) {
                let idx = ((pointer.x - plot_rect.left()) / slot_width)
                    .floor()
                    .clamp(0.0, (n - 1) as f32) as usize;
                let x_center = plot_rect.left() + (idx as f32 + 0.5) * slot_width;
                painter.line_segment(
                    [
                        egui::pos2(x_center, plot_rect.top()),
                        egui::pos2(x_center, plot_rect.bottom()),
                    ],
                    egui::Stroke::new(1.0, text.linear_multiply(0.6)),
                );
                painter.text(
                    egui::pos2(x_center + 6.0, plot_rect.top() + 10.0),
                    egui::Align2::LEFT_CENTER,
                    format!(
                        "node {}: {}",
                        node_keys[idx],
                        counts.get(idx).copied().unwrap_or(0)
                    ),
                    egui::FontId::monospace(10.0),
                    text,
                );
            }
        }
    }

    fn draw_graphs(&self, ui: &mut egui::Ui) {
        ui.heading("Graphs");
        ui.label("Session history from completed visualizer runs. Rolling averages reset when run parameters change.");

        let epoch_records = self.epoch_records();
        let latest = self.graph_history.last();
        let controller_label = if self.search_controller_mode == "model" {
            "model"
        } else {
            "selected searcher"
        };
        let epoch_count = epoch_records.len();
        let running_escape_avg = latest
            .filter(|record| record.epoch == self.active_parameter_epoch)
            .map(|record| record.running_escape_avg)
            .unwrap_or(0.0);
        let running_found_rate = latest
            .filter(|record| record.epoch == self.active_parameter_epoch)
            .map(|record| record.running_found_rate)
            .unwrap_or(0.0);

        ui.horizontal_wrapped(|ui| {
            ui.label(format!("Runs recorded: {}", self.graph_history.len()));
            ui.separator();
            ui.label(format!("Current parameter epoch: {epoch_count} run(s)"));
            ui.separator();
            ui.label(format!(
                "Running avg escape attempts ({controller_label}): {running_escape_avg:.2}"
            ));
            ui.separator();
            ui.label(format!("Found rate: {:.1}%", running_found_rate * 100.0));
        });

        ui.add_space(6.0);

        let all_attempts: Vec<f64> = self
            .graph_history
            .iter()
            .map(|record| record.escape_attempts)
            .collect();
        let all_running: Vec<f64> = self
            .graph_history
            .iter()
            .map(|record| {
                if record.epoch == self.active_parameter_epoch {
                    record.running_escape_avg
                } else {
                    f64::NAN
                }
            })
            .collect();
        Self::draw_line_chart(
            ui,
            "Escape Attempts Per Visualization",
            &[
                ChartSeries {
                    name: "attempts",
                    values: all_attempts,
                    color: parse_hex_color("#f4a261"),
                },
                ChartSeries {
                    name: "running avg",
                    values: all_running,
                    color: parse_hex_color("#2ec4b6"),
                },
            ],
            Some(0.0),
            None,
            130.0,
        );

        ui.add_space(8.0);

        let survival: Vec<f64> = self
            .graph_history
            .iter()
            .map(|record| record.survival_ratio * 100.0)
            .collect();
        let found_rate: Vec<f64> = self
            .graph_history
            .iter()
            .map(|record| record.running_found_rate * 100.0)
            .collect();
        Self::draw_line_chart(
            ui,
            "Survival % And Rolling Found %",
            &[
                ChartSeries {
                    name: "survival",
                    values: survival,
                    color: parse_hex_color("#ffd166"),
                },
                ChartSeries {
                    name: "found avg",
                    values: found_rate,
                    color: parse_hex_color("#ef476f"),
                },
            ],
            Some(0.0),
            Some(100.0),
            130.0,
        );

        ui.add_space(8.0);

        let search_work: Vec<f64> = self
            .graph_history
            .iter()
            .map(|record| record.search_work as f64)
            .collect();
        let shuffle_work: Vec<f64> = self
            .graph_history
            .iter()
            .map(|record| record.shuffle_work as f64)
            .collect();
        Self::draw_line_chart(
            ui,
            "Search Work vs Shuffle Work",
            &[
                ChartSeries {
                    name: "search",
                    values: search_work,
                    color: parse_hex_color("#8ecae6"),
                },
                ChartSeries {
                    name: "shuffle",
                    values: shuffle_work,
                    color: parse_hex_color("#a7c957"),
                },
            ],
            Some(0.0),
            None,
            130.0,
        );

        ui.add_space(8.0);
        ui.label(egui::RichText::new("Node Hit Counts").strong());
        ui.label(
            egui::RichText::new(concat!(
                "Auto-populated from the spawned tree. Searcher counts guesses; ",
                "evader counts shell spawn/relocation nodes for the current parameter epoch.",
            ))
            .small(),
        );
        let (node_keys, searcher_node_hits, evader_node_hits) = self.active_node_hit_totals();
        let chart_width = (node_keys.len() as f32 * 8.0 + 48.0).max(ui.available_width());
        egui::ScrollArea::horizontal()
            .id_salt("node_hit_counts_scroll")
            .auto_shrink([false, true])
            .show(ui, |ui| {
                Self::draw_node_hit_bar_chart(
                    ui,
                    "Searcher Hits By Node",
                    &node_keys,
                    &searcher_node_hits,
                    parse_hex_color("#8ecae6"),
                    140.0,
                    chart_width,
                );
                ui.add_space(8.0);
                Self::draw_node_hit_bar_chart(
                    ui,
                    "Evader Hits By Node",
                    &node_keys,
                    &evader_node_hits,
                    parse_hex_color("#f4a261"),
                    140.0,
                    chart_width,
                );
            });

        ui.add_space(8.0);
        ui.label(egui::RichText::new("Recent Visualization History").strong());
        egui::Grid::new("visualization_history_grid")
            .striped(true)
            .num_columns(6)
            .show(ui, |ui| {
                ui.strong("Run");
                ui.strong("Epoch");
                ui.strong("Outcome");
                ui.strong("Attempts");
                ui.strong("Survive");
                ui.strong("Avg");
                ui.end_row();
                for record in self.graph_history.iter().rev().take(8) {
                    ui.label(record.run.to_string());
                    ui.label(record.epoch.to_string());
                    ui.label(if record.found { "found" } else { "escaped" });
                    ui.label(format!("{}/{}", record.attempts, record.max_attempts));
                    ui.label(format!("{:.0}%", record.survival_ratio * 100.0));
                    ui.label(format!("{:.2}", record.running_escape_avg));
                    ui.end_row();
                }
            });
    }

    fn draw_snapshot(
        ui: &mut egui::Ui,
        snapshot: &Option<NodeSnapshot>,
        step: &HuntStep,
        is_current: bool,
        show_guess_history: bool,
        reveal_shell: bool,
    ) {
        let available = ui.available_size();
        let (rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
        let painter = ui.painter_at(rect);

        let (positions, edges, radius) =
            layout_tree(snapshot, rect.width(), rect.height(), 80.0, 48.0, 48.0);

        for (parent_path, child_path) in edges {
            let parent = positions.get(&parent_path).unwrap();
            let child = positions.get(&child_path).unwrap();
            painter.line_segment(
                [
                    egui::pos2(rect.left() + parent.x, rect.top() + parent.y),
                    egui::pos2(rect.left() + child.x, rect.top() + child.y),
                ],
                egui::Stroke::new((radius / 8.0).max(2.0), parse_hex_color("#b8a58f")),
            );
        }

        if show_guess_history {
            for (index, entry) in step.guess_history.iter().rev().take(3).collect::<Vec<_>>().into_iter().rev().enumerate() {
                if let Some(node) = positions.values().find(|node| node.key == entry.guess) {
                    let halo = 26.0 + (index as f32 * 6.0);
                    painter.circle_stroke(
                        egui::pos2(rect.left() + node.x, rect.top() + node.y),
                        halo,
                        egui::Stroke::new(3.0, parse_hex_color(GREEN_HISTORY[index])),
                    );
                }
            }
        }

        for node in positions.values() {
            let mut fill = parse_hex_color("#efe2c1");
            let mut outline = parse_hex_color("#614c34");
            let mut width_px = (radius / 10.0).max(2.0);

            if is_current && step.guess == Some(node.key) {
                fill = parse_hex_color("#ffd166");
                outline = parse_hex_color("#8d5b00");
                width_px = (radius / 7.0).max(3.0);
            }
            if reveal_shell && step.shell_key == Some(node.key) {
                fill = if step.guess == Some(node.key) {
                    parse_hex_color("#f4a261")
                } else {
                    parse_hex_color("#d95d39")
                };
                outline = parse_hex_color("#7f1d1d");
                width_px = (radius / 7.0).max(3.0);
            }

            let center = egui::pos2(rect.left() + node.x, rect.top() + node.y);
            painter.circle_filled(center, radius, fill);
            painter.circle_stroke(center, radius, egui::Stroke::new(width_px, outline));
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                node.key.to_string(),
                egui::FontId::proportional((radius * 0.9).max(10.0)),
                parse_hex_color("#20160f"),
            );
        }
    }
}

impl eframe::App for TreeVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.maybe_advance();
        self.record_completed_run_if_needed();

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.heading("Adaptive Shell Tree");
            if let Some(step) = &self.current_step {
                let title = match step.phase.as_str() {
                    "hidden" => "Shell hidden. Waiting for the first guess...".to_string(),
                    "search" => format!("Attempt {}: searching for guess {}", step.attempt, step.guess.unwrap_or_default()),
                    "resolve" => format!(
                        "Attempt {}: guessed {} -> {}",
                        step.attempt,
                        step.guess.unwrap_or_default(),
                        if step.found { "FOUND" } else { "MISS" }
                    ),
                    _ => "Shell finder is getting ready...".to_string(),
                };
                ui.label(title);
                let detail = match step.phase.as_str() {
                    "hidden" => {
                        if self.reveal_shell {
                            format!("Shell is hidden at node {}", step.shell_key.unwrap_or_default())
                        } else {
                            "Shell location is hidden".to_string()
                        }
                    }
                    "search" => {
                        let shell_text = if self.reveal_shell {
                            format!(" | shell={}", step.shell_key.unwrap_or_default())
                        } else {
                            String::new()
                        };
                        format!(
                            "Search phase{} | lookup visited={} nodes | search={} | shell={}",
                            shell_text,
                            step.operation_metrics.search_visited,
                            self.search_controller_mode,
                            self.shell_behavior_mode,
                        )
                    }
                    "resolve" => {
                        let shell_text = if self.reveal_shell {
                            format!(" | shell={}", step.shell_key.unwrap_or_default())
                        } else {
                            String::new()
                        };
                        let phase_text = if step.operation_metrics.shuffle_touched > 0 {
                            format!("shuffle touched={}", step.operation_metrics.shuffle_touched)
                        } else {
                            "resolved without shuffle".to_string()
                        };
                        format!(
                            "Resolve phase{} | {} | search={} | shell={}",
                            shell_text,
                            phase_text,
                            self.search_controller_mode,
                            self.shell_behavior_mode,
                        )
                    }
                    _ => format!(
                        "Run {} | nodes={} | tree={} | shell={} | search={} | delay={}ms",
                        self.run_count,
                        self.node_count,
                        self.generation_mode,
                        self.shell_behavior_mode,
                        self.search_controller_mode,
                        self.delay_ms
                    ),
                };
                ui.label(detail);
            } else {
                ui.label("Shell finder is getting ready...");
                ui.label(format!(
                    "Run {} | nodes={} | tree={} | shell={} | search={} | delay={}ms",
                    self.run_count,
                    self.node_count,
                    self.generation_mode,
                    self.shell_behavior_mode,
                    self.search_controller_mode,
                    self.delay_ms
                ));
            }
            if !self.status_message.is_empty() {
                ui.colored_label(parse_hex_color("#8b1e1e"), &self.status_message);
            }
        });

        egui::TopBottomPanel::bottom("controls").show(ctx, |ui| {
            self.draw_presets(ui);
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label("Speed");
                ui.add(egui::Slider::new(&mut self.delay_ms, 5..=1000).text("ms"));

                ui.label("Nodes");
                ui.add(egui::DragValue::new(&mut self.node_count).range(1..=999));
                ui.label("Budget");
                ui.add(egui::DragValue::new(&mut self.max_attempts_factor).range(1..=8).prefix("factor "));
                ui.checkbox(&mut self.use_attempt_ratio, "ratio");
                ui.add_enabled(
                    self.use_attempt_ratio,
                    egui::DragValue::new(&mut self.max_attempts_ratio)
                        .range(0.05..=2.0)
                        .speed(0.01)
                        .prefix("r "),
                );
                ui.checkbox(&mut self.use_attempt_cap, "cap");
                ui.add_enabled(
                    self.use_attempt_cap,
                    egui::DragValue::new(&mut self.max_attempts_cap).range(1..=999).prefix("c "),
                );

                ui.label("Generation");
                egui::ComboBox::from_id_salt("generation_mode")
                    .selected_text(&self.generation_mode)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.generation_mode, "balanced".to_string(), "balanced");
                        ui.selectable_value(&mut self.generation_mode, "uneven".to_string(), "uneven");
                    });

                ui.label("Shell");
                egui::ComboBox::from_id_salt("shell_behavior_mode")
                    .selected_text(&self.shell_behavior_mode)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.shell_behavior_mode,
                            "static".to_string(),
                            "static shell",
                        );
                        ui.selectable_value(&mut self.shell_behavior_mode, "random".to_string(), "random shell");
                        ui.selectable_value(&mut self.shell_behavior_mode, "adaptive".to_string(), "adaptive shell");
                    });

                ui.add_enabled(
                    self.shell_behavior_mode == "static",
                    egui::DragValue::new(&mut self.shell_target).range(1..=999).speed(1).prefix("node "),
                );

                ui.label("Search");
                egui::ComboBox::from_id_salt("search_controller_mode")
                    .selected_text(&self.search_controller_mode)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.search_controller_mode, "algorithm".to_string(), "algorithm");
                        ui.selectable_value(&mut self.search_controller_mode, "model".to_string(), "searcher model");
                    });

                ui.add_enabled_ui(self.search_controller_mode == "algorithm", |ui| {
                    egui::ComboBox::from_id_salt("search_algorithm")
                        .selected_text(&self.search_algorithm)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.search_algorithm, "ascending".to_string(), "ascending");
                            ui.selectable_value(&mut self.search_algorithm, "breadth-first".to_string(), "node-tree-crawl");
                            ui.selectable_value(
                                &mut self.search_algorithm,
                                "depth-first-preorder".to_string(),
                                "depth-first-preorder",
                            );
                            ui.selectable_value(&mut self.search_algorithm, "deepest-first".to_string(), "deepest-first");
                            ui.selectable_value(&mut self.search_algorithm, "evasion-aware".to_string(), "evasion-aware");
                        });
                });

                ui.label("Model");
                ui.text_edit_singleline(&mut self.model_bundle_path);

                ui.checkbox(&mut self.reveal_shell, "Reveal shell");
                ui.checkbox(&mut self.auto_rerun, "Auto-rerun");
                ui.add_enabled_ui(self.auto_rerun, |ui| {
                    ui.checkbox(&mut self.instant_auto_rerun, "Instant rerun");
                });
                let pause_label = if self.paused { "Resume" } else { "Pause" };
                if ui.button(pause_label).clicked() {
                    self.paused = !self.paused;
                    self.last_tick = Instant::now();
                }
                if ui.button("Restart").clicked() {
                    self.restart();
                }
            });
            let effective_budget = compute_visualizer_max_attempts(
                self.node_count.max(1) as usize,
                self.max_attempts_factor,
                self.use_attempt_ratio.then_some(self.max_attempts_ratio),
                self.use_attempt_cap.then_some(self.max_attempts_cap),
            );
            ui.label(format!(
                "Shell mode controls hiding behavior. Search chooses between an algorithmic walker and the trained searcher model. Active max attempts={} under the current budget settings. The last three guesses stay highlighted with the green gradient.",
                effective_budget
            ));
        });

        egui::SidePanel::right("graphs")
            .resizable(true)
            .default_width(390.0)
            .min_width(300.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.draw_graphs(ui);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let step = self.current_step.clone().unwrap_or_else(|| HuntStep {
                guess: None,
                found: false,
                tree_snapshot: None,
                shell_key: None,
                attempt: 0,
                phase: "hidden".to_string(),
                tree_history: Vec::new(),
                guess_history: Vec::new(),
                operation_metrics: crate::tree::OperationMetrics {
                    operation: "init".to_string(),
                    search_visited: 0,
                    shuffle_touched: 0,
                    found: false,
                },
            });
            let snapshots = extract_window_snapshots(&step);

            ui.columns(3, |columns| {
                columns[0].heading("Current Tree");
                Self::draw_snapshot(&mut columns[0], &snapshots[0], &step, true, true, self.reveal_shell);

                columns[1].heading("Previous Tree 1");
                Self::draw_snapshot(&mut columns[1], &snapshots[1], &step, false, false, self.reveal_shell);

                columns[2].heading("Previous Tree 2");
                Self::draw_snapshot(&mut columns[2], &snapshots[2], &step, false, false, self.reveal_shell);
            });
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}
