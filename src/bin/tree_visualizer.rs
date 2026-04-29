use clap::Parser;
use eframe::NativeOptions;
use shellgame_rust_v2::visualizer::{build_visualizer_steps, TreeVisualizerApp, VisualizerRunConfig};

#[derive(Parser, Debug)]
#[command(about = "Visualize the adaptive shell game tree in realtime.")]
struct Args {
    #[arg(long, default_value_t = 31)]
    nodes: i32,
    #[arg(long, default_value = "balanced", value_parser = ["balanced", "uneven"])]
    generation: String,
    #[arg(long = "shell-behavior", default_value = "adaptive", value_parser = ["static", "random", "adaptive"])]
    shell_behavior: String,
    #[arg(long = "shell-target")]
    shell_target: Option<i32>,
    #[arg(long = "search-controller", default_value = "model", value_parser = ["algorithm", "model"])]
    search_controller: String,
    #[arg(long = "search-algorithm", default_value = "evasion-aware", value_parser = ["ascending", "breadth-first", "node-tree-crawl", "depth-first-preorder", "depth-first", "deepest-first", "evasion-aware", "adaptive", "adaptive-hunter"])]
    search_algorithm: String,
    #[arg(long = "delay-ms", default_value_t = 350)]
    delay_ms: u64,
    #[arg(long, default_value_t = 2)]
    max_attempts_factor: usize,
    #[arg(long)]
    max_attempts_ratio: Option<f64>,
    #[arg(long)]
    max_attempts_cap: Option<usize>,
    #[arg(long = "model-bundle", default_value = "models/self_play_models.json")]
    model_bundle: String,
    #[arg(long = "hide-shell", default_value_t = false)]
    hide_shell: bool,
    #[arg(long = "auto-rerun", default_value_t = false)]
    auto_rerun: bool,
    #[arg(long = "instant-auto-rerun", default_value_t = false)]
    instant_auto_rerun: bool,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let node_count = args.nodes.max(1);
    let (steps, resolved_shell) = build_visualizer_steps(&VisualizerRunConfig {
        node_count,
        generation_mode: args.generation.clone(),
        shell_behavior_mode: args.shell_behavior.clone(),
        shell_target: args.shell_target,
        search_controller_mode: args.search_controller.clone(),
        search_algorithm: args.search_algorithm.clone(),
        model_bundle_path: args.model_bundle.clone(),
        max_attempts_factor: args.max_attempts_factor,
        max_attempts_ratio: args.max_attempts_ratio,
        max_attempts_cap: args.max_attempts_cap,
    })?;

    let options = NativeOptions::default();
    let app = TreeVisualizerApp::new(
        steps,
        !args.hide_shell,
        args.delay_ms.max(5),
        node_count,
        &args.generation,
        &args.shell_behavior,
        resolved_shell,
        &args.search_controller,
        &args.search_algorithm,
        args.max_attempts_factor,
        args.max_attempts_ratio,
        args.max_attempts_cap,
        args.auto_rerun,
        args.instant_auto_rerun,
        args.model_bundle,
    );

    eframe::run_native(
        "Adaptive Shell Tree Visualizer",
        options,
        Box::new(|_| Ok(Box::new(app))),
    )
    .map_err(|err| err.to_string())
}
