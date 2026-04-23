use clap::Parser;
use shellgame_rust_v2::ml::{evaluate_evader_vs_naive, load_model_bundle};
use std::path::PathBuf;

/// A minimal CLI to pit the trained evader MLP against a naïve walking searcher.
///
/// The searcher walks the tree in DFS encounter order (preorder as encoded by
/// the training tree metadata) and does not react to evader moves. The evader
/// uses the trained MLP and relocation logic from the main self-play system.
#[derive(Parser, Debug)]
#[command(about = "Evaluate evader vs. naïve walking searcher")]
struct Args {
    /// Path to self_play_models.json
    #[arg(long, default_value = "models/self_play_models.json")]
    model_bundle: PathBuf,

    /// Episodes to run
    #[arg(long, default_value_t = 50)]
    episodes: usize,

    /// RNG seed
    #[arg(long, default_value_t = 2026)]
    seed: u64,

    /// Minimum nodes in generated trees
    #[arg(long, default_value_t = 7)]
    min_nodes: i32,

    /// Maximum nodes in generated trees
    #[arg(long, default_value_t = 21)]
    max_nodes: i32,

    /// Attempts factor: max_attempts = node_count * factor
    #[arg(long, default_value_t = 2)]
    max_attempts_factor: usize,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    // Load trained bundle and run evader vs. built-in naïve walker.
    let models = load_model_bundle(&args.model_bundle)?;
    let summary = evaluate_evader_vs_naive(
        &models,
        args.episodes,
        args.seed,
        args.min_nodes,
        args.max_nodes,
        args.max_attempts_factor,
    );

    println!(
        "Naive-walk eval: episodes={} found_rate={:.3} avg_attempts={:.2} searcher_reward={:.2} evader_reward={:.2}",
        summary.episodes,
        summary.found_rate,
        summary.average_attempts,
        summary.average_searcher_reward,
        summary.average_evader_reward
    );

    Ok(())
}
