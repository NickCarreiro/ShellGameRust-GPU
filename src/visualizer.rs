use crate::shell_finder::{
    derive_candidates_for_strategy, derive_target, normalize_search_strategy, HuntStep, ShellFinder,
};
use crate::ml::{
    choose_evader_relocation_from_snapshot, choose_searcher_guess_from_snapshot, load_model_bundle,
    SelfPlayModels,
};
use crate::tree::{AdaptiveShuffleTree, NodeSnapshot, TreeNode};
use eframe::egui;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
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

fn recent_guess_window(guess_history: &[crate::shell_finder::GuessHistoryEntry], already_guessed: &[i32]) -> Vec<i32> {
    let mut combined: Vec<i32> = guess_history
        .iter()
        .map(|entry| entry.guess)
        .chain(already_guessed.iter().copied())
        .collect();
    if combined.len() > VISUALIZER_RECENT_MEMORY {
        let start = combined.len() - VISUALIZER_RECENT_MEMORY;
        combined = combined[start..].to_vec();
    }
    combined
}

fn choose_random_shell_key(tree: &AdaptiveShuffleTree, excluded: &[i32]) -> Option<i32> {
    let mut keys = tree.node_keys();
    keys.sort();
    let mut candidates: Vec<i32> = keys
        .iter()
        .copied()
        .filter(|key| !excluded.contains(key))
        .collect();
    if candidates.is_empty() {
        candidates = keys;
    }
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
                let recent = recent_guess_window(guess_history, guessed_in_order);
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

    let evasion: Option<&dyn Fn(&AdaptiveShuffleTree, &[i32]) -> Option<i32>> = match shell_behavior_mode {
        "static" => None,
        "random" => Some(&|tree, recent_guesses| choose_random_shell_key(tree, recent_guesses)),
        "adaptive" => Some(&|tree, recent_guesses| {
            let snapshot = tree.snapshot()?;
            let models = models.as_ref()?;
            choose_evader_relocation_from_snapshot(
                &models.evader,
                &snapshot,
                recent_guesses,
                recent_guesses,
            )
                .or_else(|| choose_random_shell_key(tree, recent_guesses))
        }),
        _ => None,
    };

    let steps = finder.iter_hunt_with_callbacks_limited(
        &mut tree,
        initial_shell,
        &search_chooser,
        evasion,
        Some(max_attempts),
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
    pub run_count: usize,
    pub steps: Vec<HuntStep>,
    pub index: usize,
    pub current_step: Option<HuntStep>,
    pub model_bundle_path: String,
    pub status_message: String,
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
        model_bundle_path: String,
    ) -> Self {
        Self {
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
            run_count: 1,
            steps,
            index: 0,
            current_step: None,
            model_bundle_path,
            status_message: String::new(),
            last_tick: Instant::now(),
        }
    }

    fn restart(&mut self) {
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
                self.status_message.clear();
            }
            Err(err) => {
                self.status_message = err;
            }
        }
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
        } else if self.last_tick.elapsed() >= Duration::from_millis(120) {
            self.advance();
        }

        if self.index >= self.steps.len()
            && self.auto_rerun
            && self
                .current_step
                .as_ref()
                .map(|step| step.phase == "resolve" && !step.found)
                .unwrap_or(false)
            && self.last_tick.elapsed() >= Duration::from_millis(600)
        {
            self.restart();
        }
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
                            "Search phase{} | visited={} nodes | search={} | shell={}",
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
