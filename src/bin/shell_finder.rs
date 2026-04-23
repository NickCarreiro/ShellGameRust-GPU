use clap::Parser;
use shellgame_rust_v2::{build_demo_tree, derive_target, normalize_search_strategy, ShellFinder};

#[derive(Parser, Debug)]
#[command(about = "Run the shell finder against an adaptive tree.")]
struct Args {
    #[arg(long, default_value_t = 7)]
    nodes: i32,
    #[arg(long)]
    target: Option<i32>,
    #[arg(long)]
    candidates: Option<Vec<i32>>,
    #[arg(long = "search-strategy", default_value = "evasion-aware", value_parser = ["ascending", "breadth-first", "node-tree-crawl", "depth-first-preorder", "depth-first", "deepest-first", "evasion-aware", "adaptive", "adaptive-hunter"])]
    search_strategy: String,
    #[arg(long, default_value = "shell_finder_history.json")]
    finder_history: String,
    #[arg(long, default_value = "evade_history.json")]
    tree_history: String,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let node_count = args.nodes.max(1);
    let target = derive_target(node_count, args.target);
    let mut tree = build_demo_tree(node_count, &args.tree_history);
    let mut finder = ShellFinder::new(&args.finder_history);
    let result = if let Some(ref explicit_candidates) = args.candidates {
        finder.hunt(&mut tree, target, explicit_candidates)?
    } else {
        finder.hunt_with_strategy(
            &mut tree,
            target,
            None,
            normalize_search_strategy(&args.search_strategy),
        )?
    };

    let status = if result.found { "found" } else { "missed" };
    println!(
        "Shell hunt {} after {} attempts (nodes={}, target={}, strategy={}): {:?}",
        status, result.attempts, node_count, target, args.search_strategy, result.guesses
    );
    Ok(())
}
