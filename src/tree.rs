use rand::rngs::ThreadRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::thread;
use std::time::Duration;

#[derive(Debug)]
pub struct TreeNode {
    pub key: i32,
    pub left: Option<NodeRef>,
    pub right: Option<NodeRef>,
    pub parent: Option<WeakNodeRef>,
}

pub type NodeRef = Rc<RefCell<TreeNode>>;
pub type WeakNodeRef = Weak<RefCell<TreeNode>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeSnapshot {
    pub key: i32,
    pub left: Option<Box<NodeSnapshot>>,
    pub right: Option<Box<NodeSnapshot>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationMetrics {
    pub operation: String,
    pub search_visited: usize,
    pub shuffle_touched: usize,
    pub found: bool,
}

impl OperationMetrics {
    fn new(operation: &str, search_visited: usize, shuffle_touched: usize, found: bool) -> Self {
        Self {
            operation: operation.to_string(),
            search_visited,
            shuffle_touched,
            found,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub event: String,
    pub key: Option<i32>,
    pub shell_key: Option<i32>,
    pub snapshot: Option<NodeSnapshot>,
}

pub trait RandomSource {
    fn random_f64(&mut self) -> f64;
}

impl RandomSource for ThreadRng {
    fn random_f64(&mut self) -> f64 {
        self.gen::<f64>()
    }
}

pub struct AdaptiveShuffleTree {
    pub root: Option<NodeRef>,
    history_path: PathBuf,
    history_enabled: bool,
    tree_history: Vec<HistoryEntry>,
    shell_node: Option<NodeRef>,
    last_operation_metrics: OperationMetrics,
    rng: Box<dyn RandomSource>,
}

impl std::fmt::Debug for AdaptiveShuffleTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveShuffleTree")
            .field("root", &self.root.as_ref().map(|node| node.borrow().key))
            .field("history_path", &self.history_path)
            .field("history_enabled", &self.history_enabled)
            .field("tree_history_len", &self.tree_history.len())
            .field("shell_key", &self.shell_key())
            .field("last_operation_metrics", &self.last_operation_metrics)
            .finish()
    }
}

impl AdaptiveShuffleTree {
    pub const HISTORY_LIMIT: usize = 3;

    pub fn new(history_path: impl Into<PathBuf>) -> Self {
        Self::with_rng_and_options(rand::thread_rng(), history_path, true)
    }

    pub fn new_with_options(history_path: impl Into<PathBuf>, history_enabled: bool) -> Self {
        Self::with_rng_and_options(rand::thread_rng(), history_path, history_enabled)
    }

    pub fn with_rng(rng: impl RandomSource + 'static, history_path: impl Into<PathBuf>) -> Self {
        Self::with_rng_and_options(rng, history_path, true)
    }

    pub fn with_rng_and_options(
        rng: impl RandomSource + 'static,
        history_path: impl Into<PathBuf>,
        history_enabled: bool,
    ) -> Self {
        let tree = Self {
            root: None,
            history_path: history_path.into(),
            history_enabled,
            tree_history: Vec::new(),
            shell_node: None,
            last_operation_metrics: OperationMetrics::new("init", 0, 0, false),
            rng: Box::new(rng),
        };
        if tree.history_enabled {
            tree.write_history_file();
        }
        tree
    }

    fn make_node(key: i32) -> NodeRef {
        Rc::new(RefCell::new(TreeNode {
            key,
            left: None,
            right: None,
            parent: None,
        }))
    }

    fn parent_of(node: &NodeRef) -> Option<NodeRef> {
        node.borrow().parent.as_ref().and_then(Weak::upgrade)
    }

    fn is_left_child(child: &NodeRef, parent: &NodeRef) -> bool {
        parent
            .borrow()
            .left
            .as_ref()
            .map(|left| Rc::ptr_eq(left, child))
            .unwrap_or(false)
    }

    fn rotate(&mut self, x: NodeRef) {
        let y = Self::parent_of(&x).expect("rotation requires a parent");
        let z = Self::parent_of(&y);
        let x_is_left = Self::is_left_child(&x, &y);

        if x_is_left {
            let moved = x.borrow_mut().right.take();
            {
                let mut y_mut = y.borrow_mut();
                y_mut.left = moved.clone();
                if let Some(node) = moved {
                    node.borrow_mut().parent = Some(Rc::downgrade(&y));
                }
            }
            x.borrow_mut().right = Some(y.clone());
        } else {
            let moved = x.borrow_mut().left.take();
            {
                let mut y_mut = y.borrow_mut();
                y_mut.right = moved.clone();
                if let Some(node) = moved {
                    node.borrow_mut().parent = Some(Rc::downgrade(&y));
                }
            }
            x.borrow_mut().left = Some(y.clone());
        }

        y.borrow_mut().parent = Some(Rc::downgrade(&x));
        x.borrow_mut().parent = z.as_ref().map(Rc::downgrade);

        if let Some(z_ref) = z {
            let mut z_mut = z_ref.borrow_mut();
            let y_is_left = z_mut
                .left
                .as_ref()
                .map(|left| Rc::ptr_eq(left, &y))
                .unwrap_or(false);
            if y_is_left {
                z_mut.left = Some(x.clone());
            } else {
                z_mut.right = Some(x.clone());
            }
        } else {
            self.root = Some(x);
        }
    }

    fn splay(&mut self, x: NodeRef) {
        while let Some(y) = Self::parent_of(&x) {
            if let Some(z) = Self::parent_of(&y) {
                let zigzig = Self::is_left_child(&x, &y) == Self::is_left_child(&y, &z);
                if zigzig {
                    self.rotate(y);
                    self.rotate(x.clone());
                } else {
                    self.rotate(x.clone());
                    self.rotate(x.clone());
                }
            } else {
                self.rotate(x.clone());
            }
        }
    }

    fn iter_nodes(&self) -> Vec<NodeRef> {
        let mut result = Vec::new();
        let Some(root) = &self.root else {
            return result;
        };

        let mut queue = VecDeque::from([root.clone()]);
        while let Some(node) = queue.pop_front() {
            let (left, right) = {
                let node_ref = node.borrow();
                (node_ref.left.clone(), node_ref.right.clone())
            };
            result.push(node);
            if let Some(left_node) = left {
                queue.push_back(left_node);
            }
            if let Some(right_node) = right {
                queue.push_back(right_node);
            }
        }
        result
    }

    fn find_node_with_metrics(&self, key: i32) -> (Option<NodeRef>, usize) {
        let mut visited = 0;
        for node in self.iter_nodes() {
            visited += 1;
            if node.borrow().key == key {
                return (Some(node), visited);
            }
        }
        (None, visited)
    }

    fn find_node(&self, key: i32) -> Option<NodeRef> {
        self.find_node_with_metrics(key).0
    }

    fn node_depth(node: &NodeRef) -> usize {
        let mut depth = 0;
        let mut current = Self::parent_of(node);
        while let Some(parent) = current {
            depth += 1;
            current = Self::parent_of(&parent);
        }
        depth
    }

    fn path_to_root(node: &NodeRef) -> Vec<NodeRef> {
        let mut path = vec![node.clone()];
        let mut current = Self::parent_of(node);
        while let Some(parent) = current {
            path.push(parent.clone());
            current = Self::parent_of(&parent);
        }
        path
    }

    fn node_distance(a: &NodeRef, b: &NodeRef) -> usize {
        if Rc::ptr_eq(a, b) {
            return 0;
        }

        let a_path = Self::path_to_root(a);
        let b_path = Self::path_to_root(b);

        for (a_index, a_node) in a_path.iter().enumerate() {
            if let Some(b_index) = b_path.iter().position(|b_node| Rc::ptr_eq(a_node, b_node)) {
                return a_index + b_index;
            }
        }

        a_path.len() + b_path.len()
    }

    fn snapshot_node(node: &Option<NodeRef>) -> Option<NodeSnapshot> {
        let node_ref = node.as_ref()?;
        let node_borrow = node_ref.borrow();
        Some(NodeSnapshot {
            key: node_borrow.key,
            left: Self::snapshot_node(&node_borrow.left).map(Box::new),
            right: Self::snapshot_node(&node_borrow.right).map(Box::new),
        })
    }

    fn swap_children(node: &NodeRef) {
        let mut node_mut = node.borrow_mut();
        let tmp = node_mut.left.take();
        node_mut.left = node_mut.right.take();
        node_mut.right = tmp;
    }

    fn shuffle_subtree(&mut self, node: Option<NodeRef>) -> usize {
        let Some(node_ref) = node else {
            return 0;
        };

        if self.rng.random_f64() >= 0.5 {
            Self::swap_children(&node_ref);
        }

        let (left, right) = {
            let node_borrow = node_ref.borrow();
            (node_borrow.left.clone(), node_borrow.right.clone())
        };
        1 + self.shuffle_subtree(left) + self.shuffle_subtree(right)
    }

    fn choose_shell_relocation_candidate(&self, recent_guess_keys: &[i32]) -> Option<NodeRef> {
        let nodes = self.iter_nodes();
        if nodes.is_empty() {
            return None;
        }

        let recent_keys: Vec<i32> = recent_guess_keys
            .iter()
            .rev()
            .take(Self::HISTORY_LIMIT)
            .copied()
            .collect();
        let recent_key_set: HashSet<i32> = recent_keys.iter().copied().collect();
        let recent_nodes: Vec<NodeRef> = recent_keys
            .iter()
            .filter_map(|key| self.find_node(*key))
            .collect();

        let mut best_any: Option<(i64, i32, NodeRef)> = None;
        let mut best_fresh: Option<(i64, i32, NodeRef)> = None;

        for node in nodes {
            let key = node.borrow().key;
            let depth = Self::node_depth(&node) as i64;
            let min_distance = if recent_nodes.is_empty() {
                depth + 1
            } else {
                recent_nodes
                    .iter()
                    .map(|recent| Self::node_distance(&node, recent) as i64)
                    .min()
                    .unwrap_or(0)
            };

            let score = (depth * 10) + (min_distance * 6);
            let candidate = (score, -key, node.clone());

            if best_any
                .as_ref()
                .map(|current| (candidate.0, candidate.1) > (current.0, current.1))
                .unwrap_or(true)
            {
                best_any = Some(candidate.clone());
            }

            if !recent_key_set.contains(&key)
                && best_fresh
                    .as_ref()
                    .map(|current| (candidate.0, candidate.1) > (current.0, current.1))
                    .unwrap_or(true)
            {
                best_fresh = Some(candidate);
            }
        }

        best_fresh.or(best_any).map(|(_, _, node)| node)
    }

    fn choose_shell_relocation_key(&self, recent_guess_keys: &[i32]) -> Option<i32> {
        self.choose_shell_relocation_candidate(recent_guess_keys)
            .map(|node| node.borrow().key)
    }

    fn relocate_shell_to_existing_key(&mut self, key: i32) -> Result<(), String> {
        let Some(node) = self.find_node(key) else {
            return Err(format!("Cannot relocate shell to missing key {key}."));
        };
        self.shell_node = Some(node);
        Ok(())
    }

    fn write_history_file(&self) {
        if !self.history_enabled {
            return;
        }

        let history_slice = if self.tree_history.len() > Self::HISTORY_LIMIT {
            &self.tree_history[self.tree_history.len() - Self::HISTORY_LIMIT..]
        } else {
            &self.tree_history[..]
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
                .unwrap_or("history.json")
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

    fn record_tree_state(&mut self, event: &str, key: Option<i32>) {
        if !self.history_enabled {
            return;
        }

        self.tree_history.push(HistoryEntry {
            event: event.to_string(),
            key,
            shell_key: self.shell_key(),
            snapshot: self.snapshot(),
        });
        if self.tree_history.len() > Self::HISTORY_LIMIT {
            let excess = self.tree_history.len() - Self::HISTORY_LIMIT;
            self.tree_history.drain(0..excess);
        }
        self.write_history_file();
    }

    pub fn tree_history(&self) -> Vec<HistoryEntry> {
        self.tree_history.clone()
    }

    pub fn operation_metrics(&self) -> OperationMetrics {
        self.last_operation_metrics.clone()
    }

    pub fn search(&mut self, key: i32) -> bool {
        let (node, visited) = self.find_node_with_metrics(key);
        if let Some(found_node) = node {
            self.splay(found_node);
            self.last_operation_metrics = OperationMetrics::new("search-hit", visited, 0, true);
            self.record_tree_state("search-hit", Some(key));
            return true;
        }

        let touched = self.shuffle_subtree(self.root.clone());
        self.last_operation_metrics = OperationMetrics::new("search-miss", visited, touched, false);
        self.record_tree_state("search-miss", Some(key));
        false
    }

    pub fn hide_shell(&mut self, key: i32) -> Result<(), String> {
        let (node, visited) = self.find_node_with_metrics(key);
        let Some(shell) = node else {
            return Err(format!("Cannot hide shell at missing key {key}."));
        };
        self.shell_node = Some(shell);
        self.last_operation_metrics = OperationMetrics::new("hide-shell", visited, 0, true);
        self.record_tree_state("hide-shell", Some(key));
        Ok(())
    }

    pub fn shell_key(&self) -> Option<i32> {
        self.shell_node.as_ref().map(|node| node.borrow().key)
    }

    pub fn snapshot(&self) -> Option<NodeSnapshot> {
        Self::snapshot_node(&self.root)
    }

    pub fn node_keys(&self) -> Vec<i32> {
        self.iter_nodes()
            .into_iter()
            .map(|node| node.borrow().key)
            .collect()
    }

    pub fn relocate_shell_to_key(&mut self, key: i32) -> Result<(), String> {
        self.relocate_shell_to_existing_key(key)
    }

    pub fn guess_shell_with_history(&mut self, key: i32, recent_guess_keys: &[i32]) -> bool {
        self.guess_shell_with_history_and_splay(key, recent_guess_keys, true)
    }

    pub fn guess_shell_with_history_and_splay(
        &mut self,
        key: i32,
        recent_guess_keys: &[i32],
        splay_on_hit: bool,
    ) -> bool {
        self.guess_shell_after_miss(key, splay_on_hit, |tree| {
            tree.choose_shell_relocation_key(recent_guess_keys)
        })
    }

    pub fn guess_shell_with_relocator(
        &mut self,
        key: i32,
        recent_guess_keys: &[i32],
        relocate_to: Option<i32>,
    ) -> bool {
        match relocate_to {
            Some(relocate_key) => self.guess_shell_after_miss(key, true, |_| Some(relocate_key)),
            None => self.guess_shell_with_history(key, recent_guess_keys),
        }
    }

    pub fn guess_shell_without_relocation(&mut self, key: i32, splay_on_hit: bool) -> bool {
        self.guess_shell_after_miss(key, splay_on_hit, |_| None)
    }

    pub fn guess_shell_after_miss(
        &mut self,
        key: i32,
        splay_on_hit: bool,
        relocate_after_shuffle: impl FnOnce(&AdaptiveShuffleTree) -> Option<i32>,
    ) -> bool {
        let (guessed_node, visited) = self.find_node_with_metrics(key);
        if let (Some(node), Some(shell)) = (&guessed_node, &self.shell_node) {
            if Rc::ptr_eq(node, shell) {
                if splay_on_hit {
                    self.splay(node.clone());
                }
                self.last_operation_metrics = OperationMetrics::new("guess-hit", visited, 0, true);
                self.record_tree_state("guess-hit", Some(key));
                return true;
            }
        }

        let touched = self.shuffle_subtree(self.root.clone());
        if let Some(relocate_key) = relocate_after_shuffle(self) {
            let _ = self.relocate_shell_to_existing_key(relocate_key);
        }
        self.last_operation_metrics = OperationMetrics::new("guess-miss", visited, touched, false);
        self.record_tree_state("guess-miss", Some(key));
        false
    }

    pub fn guess_shell(&mut self, key: i32) -> bool {
        self.guess_shell_with_history(key, &[key])
    }

    pub fn insert(&mut self, key: i32) {
        if self.root.is_none() {
            self.root = Some(Self::make_node(key));
            self.last_operation_metrics = OperationMetrics::new("insert", 0, 0, true);
            self.record_tree_state("insert", Some(key));
            return;
        }

        if let Some(existing) = self.find_node(key) {
            self.splay(existing);
            self.last_operation_metrics = OperationMetrics::new("insert-existing", 0, 0, true);
            self.record_tree_state("insert-existing", Some(key));
            return;
        }

        let mut queue = VecDeque::from([self.root.as_ref().expect("root exists").clone()]);
        while let Some(node) = queue.pop_front() {
            let mut node_mut = node.borrow_mut();
            if node_mut.left.is_none() {
                let child = Self::make_node(key);
                child.borrow_mut().parent = Some(Rc::downgrade(&node));
                node_mut.left = Some(child);
                drop(node_mut);
                self.last_operation_metrics = OperationMetrics::new("insert", 0, 0, true);
                self.record_tree_state("insert", Some(key));
                return;
            }
            if node_mut.right.is_none() {
                let child = Self::make_node(key);
                child.borrow_mut().parent = Some(Rc::downgrade(&node));
                node_mut.right = Some(child);
                drop(node_mut);
                self.last_operation_metrics = OperationMetrics::new("insert", 0, 0, true);
                self.record_tree_state("insert", Some(key));
                return;
            }
            let left = node_mut.left.clone();
            let right = node_mut.right.clone();
            drop(node_mut);
            if let Some(left_node) = left {
                queue.push_back(left_node);
            }
            if let Some(right_node) = right {
                queue.push_back(right_node);
            }
        }
    }
}

pub fn history_path_or_default(path: Option<&Path>, default_name: &str) -> PathBuf {
    path.map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(default_name))
}
