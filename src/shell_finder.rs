use crate::tree::{AdaptiveShuffleTree, HistoryEntry, NodeSnapshot, OperationMetrics};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HuntResult {
    pub found: bool,
    pub target: i32,
    pub attempts: usize,
    pub guesses: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuessHistoryEntry {
    pub guess: i32,
    pub found: bool,
    pub attempt: usize,
    pub shell_key: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HuntStep {
    pub guess: Option<i32>,
    pub found: bool,
    pub tree_snapshot: Option<NodeSnapshot>,
    pub shell_key: Option<i32>,
    pub attempt: usize,
    pub phase: String,
    pub tree_history: Vec<HistoryEntry>,
    pub guess_history: Vec<GuessHistoryEntry>,
    pub operation_metrics: OperationMetrics,
}

#[derive(Clone, Copy)]
pub enum MissRelocationPolicy<'a> {
    None,
    Heuristic,
    Callback(&'a dyn Fn(&AdaptiveShuffleTree, &[i32]) -> Option<i32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    Ascending,
    BreadthFirst,
    DepthFirstPreorder,
    DeepestFirst,
    EvasionAware,
}

impl SearchStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchStrategy::Ascending => "ascending",
            SearchStrategy::BreadthFirst => "breadth-first",
            SearchStrategy::DepthFirstPreorder => "depth-first-preorder",
            SearchStrategy::DeepestFirst => "deepest-first",
            SearchStrategy::EvasionAware => "evasion-aware",
        }
    }
}

#[derive(Debug)]
pub struct ShellFinder {
    history_path: PathBuf,
    guess_history: Vec<GuessHistoryEntry>,
}

impl ShellFinder {
    pub const HISTORY_LIMIT: usize = 3;

    pub fn new(history_path: impl Into<PathBuf>) -> Self {
        let finder = Self {
            history_path: history_path.into(),
            guess_history: Vec::new(),
        };
        finder.write_history_file();
        finder
    }

    fn write_history_file(&self) {
        let history_slice = if self.guess_history.len() > Self::HISTORY_LIMIT {
            &self.guess_history[self.guess_history.len() - Self::HISTORY_LIMIT..]
        } else {
            &self.guess_history[..]
        };
        let serialized = match serde_json::to_string_pretty(&json!({ "history": history_slice })) {
            Ok(value) => value,
            Err(_) => return,
        };

        if let Some(parent) = self.history_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let temp_name = format!(
            "{}.tmp",
            self.history_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("shell_finder_history.json")
        );
        let temp_path = self.history_path.with_file_name(temp_name);

        for attempt in 0..3 {
            if fs::write(&temp_path, &serialized).is_err() {
                let _ = fs::remove_file(&temp_path);
                return;
            }

            match fs::rename(&temp_path, &self.history_path) {
                Ok(()) => return,
                Err(_) => {
                    let _ = fs::remove_file(&temp_path);
                    if attempt < 2 {
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        }
    }

    fn record_guess(&mut self, guess: i32, found: bool, attempt: usize, shell_key: Option<i32>) {
        self.guess_history.push(GuessHistoryEntry {
            guess,
            found,
            attempt,
            shell_key,
        });
        if self.guess_history.len() > Self::HISTORY_LIMIT {
            let excess = self.guess_history.len() - Self::HISTORY_LIMIT;
            self.guess_history.drain(0..excess);
        }
        self.write_history_file();
    }

    pub fn guess_history(&self) -> Vec<GuessHistoryEntry> {
        self.guess_history.clone()
    }

    pub fn iter_hunt(
        &mut self,
        tree: &mut AdaptiveShuffleTree,
        target: i32,
        candidates: &[i32],
    ) -> Result<Vec<HuntStep>, String> {
        self.iter_hunt_with_strategy(tree, target, Some(candidates), SearchStrategy::Ascending, None)
    }

    pub fn iter_hunt_with_callbacks_limited(
        &mut self,
        tree: &mut AdaptiveShuffleTree,
        target: i32,
        chooser: &dyn Fn(&AdaptiveShuffleTree, &[i32], &[GuessHistoryEntry]) -> Option<i32>,
        evasion: Option<&dyn Fn(&AdaptiveShuffleTree, &[i32]) -> Option<i32>>,
        max_attempts: Option<usize>,
    ) -> Result<Vec<HuntStep>, String> {
        if let Some(callback) = evasion {
            let adapter = |tree: &AdaptiveShuffleTree, guessed_keys: &[i32]| callback(tree, guessed_keys);
            self.iter_hunt_with_relocation_policy_limited(
                tree,
                target,
                chooser,
                MissRelocationPolicy::Callback(&adapter),
                max_attempts,
                true,
            )
        } else {
            self.iter_hunt_with_relocation_policy_limited(
                tree,
                target,
                chooser,
                MissRelocationPolicy::Heuristic,
                max_attempts,
                true,
            )
        }
    }

    pub fn iter_hunt_with_relocation_policy_limited(
        &mut self,
        tree: &mut AdaptiveShuffleTree,
        target: i32,
        chooser: &dyn Fn(&AdaptiveShuffleTree, &[i32], &[GuessHistoryEntry]) -> Option<i32>,
        relocation_policy: MissRelocationPolicy<'_>,
        max_attempts: Option<usize>,
        splay_on_hit: bool,
    ) -> Result<Vec<HuntStep>, String> {
        tree.hide_shell(target)?;

        let mut steps = vec![HuntStep {
            guess: None,
            found: false,
            tree_snapshot: tree.snapshot(),
            shell_key: tree.shell_key(),
            attempt: 0,
            phase: "hidden".to_string(),
            tree_history: tree.tree_history(),
            guess_history: self.guess_history(),
            operation_metrics: tree.operation_metrics(),
        }];

        let mut guessed_in_order = Vec::new();

        loop {
            if let Some(limit) = max_attempts {
                if guessed_in_order.len() >= limit {
                    break;
                }
            }

            let Some(guess) = chooser(tree, &guessed_in_order, &self.guess_history) else {
                break;
            };

            let attempt = guessed_in_order.len() + 1;
            let pre_snapshot = tree.snapshot();
            let pre_shell_key = tree.shell_key();
            let pre_tree_history = tree.tree_history();
            let pre_guess_history = self.guess_history();
            let all_guess_keys: Vec<i32> = guessed_in_order
                .iter()
                .copied()
                .chain(std::iter::once(guess))
                .collect();
            let found = match relocation_policy {
                MissRelocationPolicy::None => tree.guess_shell_without_relocation(guess, splay_on_hit),
                MissRelocationPolicy::Heuristic => {
                    tree.guess_shell_with_history_and_splay(guess, &all_guess_keys, splay_on_hit)
                }
                MissRelocationPolicy::Callback(callback) => tree.guess_shell_after_miss(
                    guess,
                    splay_on_hit,
                    |tree_after_shuffle| callback(tree_after_shuffle, &all_guess_keys),
                ),
            };
            let operation_metrics = tree.operation_metrics();
            let post_snapshot = tree.snapshot();
            let post_tree_history = tree.tree_history();

            steps.push(HuntStep {
                guess: Some(guess),
                found,
                tree_snapshot: pre_snapshot,
                shell_key: pre_shell_key,
                attempt,
                phase: "search".to_string(),
                tree_history: pre_tree_history,
                guess_history: pre_guess_history,
                operation_metrics: operation_metrics.clone(),
            });

            self.record_guess(guess, found, attempt, tree.shell_key());
            steps.push(HuntStep {
                guess: Some(guess),
                found,
                tree_snapshot: post_snapshot,
                shell_key: tree.shell_key(),
                attempt,
                phase: "resolve".to_string(),
                tree_history: post_tree_history,
                guess_history: self.guess_history(),
                operation_metrics,
            });

            guessed_in_order.push(guess);
            if found {
                break;
            }
        }

        Ok(steps)
    }

    pub fn iter_hunt_with_callbacks(
        &mut self,
        tree: &mut AdaptiveShuffleTree,
        target: i32,
        chooser: &dyn Fn(&AdaptiveShuffleTree, &[i32], &[GuessHistoryEntry]) -> Option<i32>,
        evasion: Option<&dyn Fn(&AdaptiveShuffleTree, &[i32]) -> Option<i32>>,
    ) -> Result<Vec<HuntStep>, String> {
        self.iter_hunt_with_callbacks_limited(tree, target, chooser, evasion, None)
    }

    pub fn iter_hunt_with_strategy(
        &mut self,
        tree: &mut AdaptiveShuffleTree,
        target: i32,
        explicit_candidates: Option<&[i32]>,
        strategy: SearchStrategy,
        evasion: Option<&dyn Fn(&AdaptiveShuffleTree, &[i32]) -> Option<i32>>,
    ) -> Result<Vec<HuntStep>, String> {
        let chooser = |tree: &AdaptiveShuffleTree, guessed_in_order: &[i32], guess_history: &[GuessHistoryEntry]| {
            let remaining_candidates = derive_remaining_candidates_for_strategy(
                tree,
                explicit_candidates,
                guessed_in_order,
                strategy,
                guess_history,
            );
            remaining_candidates.first().copied()
        };

        self.iter_hunt_with_callbacks(tree, target, &chooser, evasion)
    }

    pub fn hunt(
        &mut self,
        tree: &mut AdaptiveShuffleTree,
        target: i32,
        candidates: &[i32],
    ) -> Result<HuntResult, String> {
        self.hunt_with_strategy(tree, target, Some(candidates), SearchStrategy::Ascending)
    }

    pub fn hunt_with_strategy(
        &mut self,
        tree: &mut AdaptiveShuffleTree,
        target: i32,
        explicit_candidates: Option<&[i32]>,
        strategy: SearchStrategy,
    ) -> Result<HuntResult, String> {
        let mut guesses = Vec::new();
        let mut found = false;

        for step in self.iter_hunt_with_strategy(tree, target, explicit_candidates, strategy, None)? {
            if step.phase != "resolve" {
                continue;
            }
            if let Some(guess) = step.guess {
                guesses.push(guess);
            }
            if step.found {
                found = true;
                break;
            }
        }

        Ok(HuntResult {
            found,
            target,
            attempts: guesses.len(),
            guesses,
        })
    }
}

pub fn build_demo_tree(node_count: i32, history_path: impl Into<PathBuf>) -> AdaptiveShuffleTree {
    let mut tree = AdaptiveShuffleTree::new(history_path);
    for key in 1..=node_count.max(1) {
        tree.insert(key);
    }
    tree
}

pub fn derive_target(node_count: i32, explicit_target: Option<i32>) -> i32 {
    if let Some(target) = explicit_target {
        return target.clamp(1, node_count.max(1));
    }
    ((node_count.max(1) * 2) / 3).max(1)
}

pub fn derive_candidates(node_count: i32, explicit_candidates: Option<Vec<i32>>) -> Vec<i32> {
    if let Some(candidates) = explicit_candidates {
        return candidates;
    }
    (1..=node_count.max(1)).collect()
}

pub fn normalize_search_strategy(mode: &str) -> SearchStrategy {
    match mode {
        "breadth-first" | "node-tree-crawl" => SearchStrategy::BreadthFirst,
        "depth-first" | "depth-first-preorder" => SearchStrategy::DepthFirstPreorder,
        "deepest-first" => SearchStrategy::DeepestFirst,
        "evasion-aware" | "adaptive" | "adaptive-hunter" => SearchStrategy::EvasionAware,
        _ => SearchStrategy::Ascending,
    }
}

#[derive(Debug, Clone)]
struct SnapshotNodeMeta {
    key: i32,
    depth: usize,
    path: String,
}

fn collect_snapshot_meta(snapshot: &NodeSnapshot, depth: usize, path: String, ordered: &mut Vec<SnapshotNodeMeta>) {
    ordered.push(SnapshotNodeMeta {
        key: snapshot.key,
        depth,
        path: path.clone(),
    });

    if let Some(left) = snapshot.left.as_deref() {
        collect_snapshot_meta(left, depth + 1, format!("{path}L"), ordered);
    }
    if let Some(right) = snapshot.right.as_deref() {
        collect_snapshot_meta(right, depth + 1, format!("{path}R"), ordered);
    }
}

fn collect_breadth_first(snapshot: &NodeSnapshot) -> Vec<i32> {
    let mut ordered = Vec::new();
    let mut queue = VecDeque::from([snapshot]);
    while let Some(node) = queue.pop_front() {
        ordered.push(node.key);
        if let Some(left) = node.left.as_deref() {
            queue.push_back(left);
        }
        if let Some(right) = node.right.as_deref() {
            queue.push_back(right);
        }
    }
    ordered
}

fn collect_depth_first_preorder(snapshot: &NodeSnapshot, ordered: &mut Vec<i32>) {
    ordered.push(snapshot.key);
    if let Some(left) = snapshot.left.as_deref() {
        collect_depth_first_preorder(left, ordered);
    }
    if let Some(right) = snapshot.right.as_deref() {
        collect_depth_first_preorder(right, ordered);
    }
}

fn collect_deepest_first(snapshot: &NodeSnapshot) -> Vec<i32> {
    let mut nodes = Vec::new();
    collect_snapshot_meta(snapshot, 0, "root".to_string(), &mut nodes);
    nodes.sort_by_key(|node| (-(node.depth as isize), node.key));
    nodes.into_iter().map(|node| node.key).collect()
}

fn path_distance(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut common = 0usize;
    while common < a_bytes.len() && common < b_bytes.len() && a_bytes[common] == b_bytes[common] {
        common += 1;
    }
    (a_bytes.len() - common) + (b_bytes.len() - common)
}

fn collect_evasion_aware(
    snapshot: &NodeSnapshot,
    guess_history: &[GuessHistoryEntry],
    already_guessed: &[i32],
) -> Vec<i32> {
    let mut nodes = Vec::new();
    collect_snapshot_meta(snapshot, 0, "root".to_string(), &mut nodes);

    let mut recent_keys: Vec<i32> = already_guessed
        .iter()
        .rev()
        .take(ShellFinder::HISTORY_LIMIT.max(3))
        .copied()
        .collect();
    if recent_keys.is_empty() {
        recent_keys = guess_history
            .iter()
            .rev()
            .take(ShellFinder::HISTORY_LIMIT.max(3))
            .map(|entry| entry.guess)
            .collect();
    }

    let recent_paths: Vec<String> = recent_keys
        .iter()
        .filter_map(|key| nodes.iter().find(|node| node.key == *key).map(|node| node.path.clone()))
        .collect();

    nodes.sort_by_key(|node| {
        let min_distance = if recent_paths.is_empty() {
            node.depth + 1
        } else {
            recent_paths
                .iter()
                .map(|path| path_distance(&node.path, path))
                .min()
                .unwrap_or(0)
        };
        let score = ((node.depth as i64) * 10) + ((min_distance as i64) * 6);
        (-score, node.key)
    });

    nodes.into_iter().map(|node| node.key).collect()
}

pub fn derive_candidates_for_strategy(
    tree: &AdaptiveShuffleTree,
    node_count: i32,
    explicit_candidates: Option<Vec<i32>>,
    strategy: SearchStrategy,
) -> Vec<i32> {
    if let Some(candidates) = explicit_candidates {
        return candidates;
    }

    match strategy {
        SearchStrategy::Ascending => derive_candidates(node_count, None),
        SearchStrategy::BreadthFirst => tree
            .snapshot()
            .map(|snapshot| collect_breadth_first(&snapshot))
            .unwrap_or_else(|| derive_candidates(node_count, None)),
        SearchStrategy::DepthFirstPreorder => tree
            .snapshot()
            .map(|snapshot| {
                let mut ordered = Vec::new();
                collect_depth_first_preorder(&snapshot, &mut ordered);
                ordered
            })
            .unwrap_or_else(|| derive_candidates(node_count, None)),
        SearchStrategy::DeepestFirst => tree
            .snapshot()
            .map(|snapshot| collect_deepest_first(&snapshot))
            .unwrap_or_else(|| derive_candidates(node_count, None)),
        SearchStrategy::EvasionAware => tree
            .snapshot()
            .map(|snapshot| collect_evasion_aware(&snapshot, &[], &[]))
            .unwrap_or_else(|| derive_candidates(node_count, None)),
    }
}

pub fn derive_remaining_candidates_for_strategy(
    tree: &AdaptiveShuffleTree,
    explicit_candidates: Option<&[i32]>,
    already_guessed: &[i32],
    strategy: SearchStrategy,
    guess_history: &[GuessHistoryEntry],
) -> Vec<i32> {
    let mut ordered = if let Some(candidates) = explicit_candidates {
        candidates.to_vec()
    } else {
        match strategy {
            SearchStrategy::Ascending => {
                let mut keys = tree.node_keys();
                keys.sort();
                keys
            }
            SearchStrategy::BreadthFirst => tree
                .snapshot()
                .map(|snapshot| collect_breadth_first(&snapshot))
                .unwrap_or_else(|| { let mut k = tree.node_keys(); k.sort(); k }),
            SearchStrategy::DepthFirstPreorder => tree
                .snapshot()
                .map(|snapshot| {
                    let mut ordered = Vec::new();
                    collect_depth_first_preorder(&snapshot, &mut ordered);
                    ordered
                })
                .unwrap_or_else(|| { let mut k = tree.node_keys(); k.sort(); k }),
            SearchStrategy::DeepestFirst => tree
                .snapshot()
                .map(|snapshot| collect_deepest_first(&snapshot))
                .unwrap_or_else(|| { let mut k = tree.node_keys(); k.sort(); k }),
            SearchStrategy::EvasionAware => tree
                .snapshot()
                .map(|snapshot| collect_evasion_aware(&snapshot, guess_history, already_guessed))
                .unwrap_or_default(),
        }
    };

    ordered.retain(|candidate| !already_guessed.contains(candidate));
    ordered
}
