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
            })?;

            println!(
                "Training finished: generations={} seed={} searcher_score={:.2} evader_score={:.2} output_dir={}",
                summary.generations,
                summary.seed,
                summary.final_searcher_score,
                summary.final_evader_score,
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
                "Evaluation: episodes={} found_rate={:.3} avg_attempts={:.2} searcher_reward={:.2} evader_reward={:.2}",
                summary.episodes,
                summary.found_rate,
                summary.average_attempts,
                summary.average_searcher_reward,
                summary.average_evader_reward
            );
        }
    }

    Ok(())
}
