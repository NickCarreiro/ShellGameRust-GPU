use clap::{Parser, Subcommand};
use shellgame_rust_v2::ml::{
    evaluate_model_bundle, load_model_bundle, train_self_play_models, TrainingConfig, TrainingMode,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Train and evaluate lightweight self-play ML models for evasion and searching.")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train {
        #[arg(long, default_value_t = 40)]
        generations: usize,
        #[arg(long, default_value_t = 16)]
        population_size: usize,
        #[arg(long, default_value_t = 24)]
        episodes_per_eval: usize,
        #[arg(long, default_value_t = 7)]
        min_nodes: i32,
        #[arg(long, default_value_t = 21)]
        max_nodes: i32,
        #[arg(long, default_value_t = 2)]
        max_attempts_factor: usize,
        #[arg(long)]
        max_attempts_ratio: Option<f64>,
        #[arg(long)]
        max_attempts_cap: Option<usize>,
        #[arg(long, default_value_t = 0.18)]
        mutation_scale: f64,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
        #[arg(long, default_value = "models")]
        output_dir: PathBuf,
        #[arg(long)]
        resume_from: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        hall_of_fame_size: usize,
        #[arg(long, default_value_t = 2)]
        hall_sample_count: usize,
        #[arg(long, default_value_t = 2)]
        static_opponent_sample_count: usize,
        #[arg(long = "training-mode", default_value = "static", value_parser = ["static", "coagent"])]
        training_mode: String,
        /// Adam learning rate for the OpenAI-ES gradient update.
        #[arg(long, default_value_t = 0.01)]
        es_lr: f64,
        /// Multiplier applied to es-lr for the learned searcher in coagent mode.
        #[arg(long, default_value_t = 0.25)]
        searcher_lr_scale: f64,
        /// Update the learned searcher every N generations in coagent mode.
        #[arg(long, default_value_t = 2)]
        searcher_update_interval: usize,
        /// Reject learned-searcher updates that push found-rate above this cap.
        #[arg(long, default_value_t = 0.55)]
        searcher_max_found_rate: f64,
        /// Reject learned-searcher updates that jump found-rate too far in one update.
        #[arg(long, default_value_t = 0.25)]
        searcher_max_found_rate_jump: f64,
        /// Stop training when evader score does not improve for this many consecutive generations.
        #[arg(long)]
        patience: Option<usize>,
        /// Grow nodes/population after this many consecutive stagnant generations.
        #[arg(long)]
        stagnation_grow_after: Option<usize>,
        /// Node-count increase applied to both min-nodes and max-nodes on each growth event.
        #[arg(long, default_value_t = 5)]
        stagnation_node_step: i32,
        /// Population-size increase applied on each growth event.
        #[arg(long, default_value_t = 25)]
        stagnation_population_step: usize,
        /// Optional cap for adaptively grown node counts.
        #[arg(long)]
        stagnation_max_nodes_cap: Option<i32>,
        /// Optional cap for adaptively grown population size.
        #[arg(long)]
        stagnation_population_cap: Option<usize>,
    },
    Evaluate {
        #[arg(long, default_value = "models/self_play_models.json")]
        model_bundle: PathBuf,
        #[arg(long, default_value_t = 50)]
        episodes: usize,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
        #[arg(long, default_value_t = 7)]
        min_nodes: i32,
        #[arg(long, default_value_t = 21)]
        max_nodes: i32,
        #[arg(long, default_value_t = 2)]
        max_attempts_factor: usize,
        #[arg(long)]
        max_attempts_ratio: Option<f64>,
        #[arg(long)]
        max_attempts_cap: Option<usize>,
    },
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    match args.command {
        Command::Train {
            generations,
            population_size,
            episodes_per_eval,
            min_nodes,
            max_nodes,
            max_attempts_factor,
            max_attempts_ratio,
            max_attempts_cap,
            mutation_scale,
            seed,
            output_dir,
            resume_from,
            hall_of_fame_size,
            hall_sample_count,
            static_opponent_sample_count,
            training_mode,
            es_lr,
            searcher_lr_scale,
            searcher_update_interval,
            searcher_max_found_rate,
            searcher_max_found_rate_jump,
            patience,
            stagnation_grow_after,
            stagnation_node_step,
            stagnation_population_step,
            stagnation_max_nodes_cap,
            stagnation_population_cap,
        } => {
            let (_models, summary) = train_self_play_models(&TrainingConfig {
                generations,
                population_size,
                episodes_per_eval,
                min_nodes,
                max_nodes,
                max_attempts_factor,
                max_attempts_ratio,
                max_attempts_cap,
                mutation_scale,
                seed,
                output_dir: output_dir.clone(),
                resume_from,
                hall_of_fame_size,
                hall_sample_count,
                static_opponent_sample_count,
                training_mode: match training_mode.as_str() {
                    "coagent" => TrainingMode::CoAgent,
                    _ => TrainingMode::Static,
                },
                es_lr,
                searcher_lr_scale,
                searcher_update_interval,
                searcher_max_found_rate,
                searcher_max_found_rate_jump,
                patience,
                stagnation_grow_after,
                stagnation_node_step,
                stagnation_population_step,
                stagnation_max_nodes_cap,
                stagnation_population_cap,
            })?;

            println!(
                "Training finished\n  generations:   {}\n  seed:          {}\n  final searcher:{:>8.2}\n  final evader:  {:>8.2}\n  final escape:  {:>8.2}\n  best evader:   {:>8.2} (generation {})\n  best escape:   {:>8.2}\n  best searcher: {:>8.2} (generation {})\n  stopped early: {}\n  interrupted:   {}\n  output dir:    {}",
                summary.generations,
                summary.seed,
                summary.final_searcher_score,
                summary.final_evader_score,
                summary.final_escape_score,
                summary.best_evader_score,
                summary.best_generation,
                summary.best_evader_selection_score,
                summary.best_searcher_score,
                summary.best_searcher_generation,
                summary.stopped_early,
                summary.interrupted,
                output_dir.display()
            );
        }
        Command::Evaluate {
            model_bundle,
            episodes,
            seed,
            min_nodes,
            max_nodes,
            max_attempts_factor,
            max_attempts_ratio,
            max_attempts_cap,
        } => {
            let models = load_model_bundle(&model_bundle)?;
            let summary = evaluate_model_bundle(
                &models,
                episodes,
                seed,
                min_nodes,
                max_nodes,
                max_attempts_factor,
                max_attempts_ratio,
                max_attempts_cap,
            );

            println!(
                "Evaluation\n  episodes:      {}\n  found rate:    {:>8.3}\n  budget used:   {:>8.1}%\n  avg attempts:  {:>8.2}/{:>8.2}\n  escape score:  {:>8.2}\n  searcher:      {:>8.2}\n  evader:        {:>8.2}",
                summary.episodes,
                summary.found_rate,
                summary.survival_budget_ratio * 100.0,
                summary.average_attempts,
                summary.average_max_attempts,
                summary.escape_quality_score,
                summary.average_searcher_reward,
                summary.average_evader_reward
            );
        }
    }

    Ok(())
}
