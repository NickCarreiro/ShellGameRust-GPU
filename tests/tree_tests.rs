use shellgame_rust_v2::tree::AdaptiveShuffleTree;
use shellgame_rust_v2::visualizer::{
    build_demo_tree as build_visual_demo_tree, build_steps, build_visualizer_steps,
    estimate_step_delay, extract_window_snapshots, layout_tree, normalize_generation_mode,
    normalize_playback_mode, normalize_search_mode, normalize_target_mode, resolve_target,
    VisualizerRunConfig,
};
use shellgame_rust_v2::{
    build_demo_tree, derive_candidates, derive_candidates_for_strategy, derive_target, normalize_search_strategy,
    MissRelocationPolicy, SearchStrategy, ShellFinder,
};
use std::path::PathBuf;

fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("shellgame_rust_{}_{}", std::process::id(), name))
}

#[test]
fn tree_hit_splays_found_node_to_root() {
    let history = temp_file("evade_tree_hit.json");
    let mut tree = AdaptiveShuffleTree::new(&history);
    for key in [10, 5, 15, 3, 7, 12, 18] {
        tree.insert(key);
    }

    let found = tree.search(7);

    assert!(found);
    assert_eq!(tree.root.unwrap().borrow().key, 7);
}

#[test]
fn tree_snapshot_exposes_current_shape() {
    let history = temp_file("evade_tree_snapshot.json");
    let mut tree = AdaptiveShuffleTree::new(&history);
    for key in [10, 5, 15, 3, 7, 12, 18] {
        tree.insert(key);
    }

    let snapshot = tree.snapshot().unwrap();
    assert_eq!(snapshot.key, 10);
    assert_eq!(snapshot.left.as_ref().unwrap().key, 5);
    assert_eq!(snapshot.right.as_ref().unwrap().key, 15);
}

#[test]
fn shell_finder_hunt_finds_shell() {
    let tree_history = temp_file("finder_tree.json");
    let finder_history = temp_file("finder_history.json");
    let mut tree = build_demo_tree(7, &tree_history);
    let mut finder = ShellFinder::new(&finder_history);

    let result = finder.hunt(&mut tree, 5, &[5]).unwrap();

    assert!(result.found);
    assert_eq!(result.attempts, 1);
    assert_eq!(result.guesses, vec![5]);
}

#[test]
fn callback_relocation_happens_after_search_step_snapshot() {
    let tree_history = temp_file("finder_tree_callback_relocation.json");
    let finder_history = temp_file("finder_history_callback_relocation.json");
    let mut tree = build_demo_tree(5, &tree_history);
    let mut finder = ShellFinder::new(&finder_history);

    let chooser = |_tree: &AdaptiveShuffleTree,
                   guessed: &[i32],
                   _history: &[shellgame_rust_v2::shell_finder::GuessHistoryEntry]| {
        if guessed.is_empty() {
            Some(2)
        } else {
            None
        }
    };
    let relocator = |_tree: &AdaptiveShuffleTree, guessed: &[i32]| {
        assert_eq!(guessed, &[2]);
        Some(3)
    };

    let steps = finder
        .iter_hunt_with_relocation_policy_limited(
            &mut tree,
            1,
            &chooser,
            MissRelocationPolicy::Callback(&relocator),
            Some(1),
            false,
        )
        .unwrap();

    let search = steps.iter().find(|step| step.phase == "search").unwrap();
    let resolve = steps.iter().find(|step| step.phase == "resolve").unwrap();

    assert!(!search.found);
    assert_eq!(search.shell_key, Some(1));
    assert_eq!(resolve.shell_key, Some(3));
}

#[test]
fn visualizer_static_shell_does_not_relocate_on_misses() {
    let (steps, resolved_shell) = build_visualizer_steps(&VisualizerRunConfig {
        node_count: 7,
        generation_mode: "balanced".to_string(),
        shell_behavior_mode: "static".to_string(),
        shell_target: Some(7),
        search_controller_mode: "algorithm".to_string(),
        search_algorithm: "ascending".to_string(),
        model_bundle_path: "unused.json".to_string(),
        max_attempts_factor: 2,
        max_attempts_ratio: None,
        max_attempts_cap: None,
    })
    .unwrap();

    assert_eq!(resolved_shell, 7);
    let misses: Vec<_> = steps
        .iter()
        .filter(|step| step.phase == "resolve" && !step.found)
        .collect();
    assert!(!misses.is_empty());
    assert!(misses.iter().all(|step| step.shell_key == Some(7)));
    assert!(steps
        .iter()
        .any(|step| step.phase == "resolve" && step.found && step.shell_key == Some(7)));
}

#[test]
fn visualizer_hit_does_not_splay_found_node() {
    let (steps, _) = build_visualizer_steps(&VisualizerRunConfig {
        node_count: 7,
        generation_mode: "balanced".to_string(),
        shell_behavior_mode: "static".to_string(),
        shell_target: Some(4),
        search_controller_mode: "algorithm".to_string(),
        search_algorithm: "ascending".to_string(),
        model_bundle_path: "unused.json".to_string(),
        max_attempts_factor: 2,
        max_attempts_ratio: None,
        max_attempts_cap: None,
    })
    .unwrap();

    let found_index = steps
        .iter()
        .position(|step| step.phase == "resolve" && step.found)
        .unwrap();
    let search = &steps[found_index - 1];
    let resolve = &steps[found_index];

    assert_eq!(search.phase, "search");
    assert_eq!(search.tree_snapshot, resolve.tree_snapshot);
}

#[test]
fn visualizer_layout_keeps_nodes_inside_bounds() {
    let tree = build_visual_demo_tree(15, "evade_history.json", "balanced");
    let snapshot = tree.snapshot();
    let (positions, edges, radius) = layout_tree(&snapshot, 800.0, 500.0, 80.0, 48.0, 48.0);

    assert_eq!(positions.len(), 15);
    assert_eq!(edges.len(), 14);
    assert!(radius >= 12.0);

    for node in positions.values() {
        assert!(node.x - radius >= 0.0);
        assert!(node.x + radius <= 800.0);
        assert!(node.y - radius >= 0.0);
        assert!(node.y + radius <= 500.0);
    }
}

#[test]
fn uneven_generation_produces_different_shape() {
    let balanced = build_visual_demo_tree(9, "evade_history.json", "balanced").snapshot();
    let uneven = build_visual_demo_tree(9, "evade_history.json", "uneven").snapshot();

    assert_ne!(balanced, uneven);
    let uneven = uneven.unwrap();
    assert_eq!(uneven.left.as_ref().unwrap().key, 2);
    assert_eq!(uneven.right.as_ref().unwrap().key, 7);
}

#[test]
fn helper_defaults_match_python_behavior() {
    assert_eq!(derive_target(10, None), 6);
    assert_eq!(derive_candidates(5, None), vec![1, 2, 3, 4, 5]);
    assert_eq!(normalize_generation_mode("other"), "balanced");
    assert_eq!(normalize_playback_mode("other"), "fixed");
    assert_eq!(normalize_search_mode("other"), "ascending");
    assert_eq!(normalize_target_mode("other"), "assigned");
}

#[test]
fn breadth_first_tree_crawl_strategy_uses_tree_shape() {
    let tree = build_visual_demo_tree(7, "evade_history.json", "balanced");
    let candidates = derive_candidates_for_strategy(&tree, 7, None, SearchStrategy::BreadthFirst);
    assert_eq!(candidates, vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn depth_first_strategy_walks_tree_preorder() {
    let tree = build_visual_demo_tree(7, "evade_history.json", "balanced");
    let candidates = derive_candidates_for_strategy(&tree, 7, None, SearchStrategy::DepthFirstPreorder);
    assert_eq!(candidates, vec![1, 2, 4, 5, 3, 6, 7]);
}

#[test]
fn breadth_first_strategy_differs_on_uneven_tree() {
    let tree = build_visual_demo_tree(9, "evade_history.json", "uneven");
    let candidates = derive_candidates_for_strategy(&tree, 9, None, normalize_search_strategy("node-tree-crawl"));
    assert_ne!(candidates, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn deepest_first_strategy_prioritizes_leaf_nodes() {
    let tree = build_visual_demo_tree(7, "evade_history.json", "balanced");
    let candidates = derive_candidates_for_strategy(&tree, 7, None, SearchStrategy::DeepestFirst);
    assert_eq!(candidates, vec![4, 5, 6, 7, 2, 3, 1]);
}

#[test]
fn evasion_aware_strategy_targets_evasive_positions_first() {
    let tree = build_visual_demo_tree(7, "evade_history.json", "balanced");
    let candidates = derive_candidates_for_strategy(&tree, 7, None, SearchStrategy::EvasionAware);
    assert_eq!(candidates[0], 4);
    assert!(candidates[..4].iter().all(|key| [4, 5, 6, 7].contains(key)));
}

#[test]
fn evasion_aware_hunt_can_open_on_deep_target() {
    let tree_history = temp_file("finder_tree_evasion_aware.json");
    let finder_history = temp_file("finder_history_evasion_aware.json");
    let mut tree = build_demo_tree(7, &tree_history);
    let mut finder = ShellFinder::new(&finder_history);

    let result = finder
        .hunt_with_strategy(&mut tree, 4, None, SearchStrategy::EvasionAware)
        .unwrap();

    assert!(result.found);
    assert_eq!(result.attempts, 1);
    assert_eq!(result.guesses, vec![4]);
}

#[test]
fn wrong_guess_reshuffles_and_relocates_shell_toward_evasive_nodes() {
    let history = temp_file("evade_tree_relocate.json");
    let mut tree = AdaptiveShuffleTree::new(&history);
    for key in 1..=7 {
        tree.insert(key);
    }

    tree.hide_shell(1).unwrap();
    let found = tree.guess_shell_with_history(2, &[2]);

    assert!(!found);
    let relocated = tree.shell_key().unwrap();
    assert_ne!(relocated, 1);
    assert_ne!(relocated, 2);
    assert!(relocated >= 4);
}

#[test]
fn relocation_avoids_recent_guess_history_when_possible() {
    let history = temp_file("evade_tree_recent_guess_bias.json");
    let mut tree = AdaptiveShuffleTree::new(&history);
    for key in 1..=15 {
        tree.insert(key);
    }

    tree.hide_shell(1).unwrap();
    let found = tree.guess_shell_with_history(2, &[8, 9, 10, 2]);

    assert!(!found);
    assert!(![8, 9, 10, 2].contains(&tree.shell_key().unwrap()));
}

#[test]
fn estimate_delay_scales_with_work() {
    let steps = build_steps(9, Some(7), Some(vec![1, 3, 5, 7]), "balanced", "assigned", "ascending").unwrap().0;
    let hidden = &steps[0];
    let late = steps.last().unwrap();

    let hidden_delay = estimate_step_delay(300, hidden, "thread-matching");
    let late_delay = estimate_step_delay(300, late, "thread-matching");

    assert!(hidden_delay >= 5);
    assert!(late_delay >= 5);
}

#[test]
fn extract_window_snapshots_prefers_current_then_unique_history() {
    let steps = build_steps(9, Some(7), Some(vec![1, 3, 5, 7]), "balanced", "assigned", "ascending").unwrap().0;
    let snapshots = extract_window_snapshots(steps.last().unwrap());
    assert_eq!(snapshots.len(), 3);
}

#[test]
fn resolve_target_supports_assigned_mode() {
    let result = resolve_target(10, "assigned", Some(99), None::<&mut rand::rngs::ThreadRng>);
    assert_eq!(result, 10);
}
