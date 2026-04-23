#[cfg(feature = "cuda")]
#[test]
fn cuda_training_uses_gpu_backend_not_cpu() {
    use shellgame_rust_v2::ml::{
        train_self_play_models, training_accelerator_is_cuda, TrainingConfig, TrainingMode,
    };

    assert!(
        training_accelerator_is_cuda(),
        "CUDA feature is enabled, but the ML training accelerator is not CUDA"
    );

    let output_dir = std::env::temp_dir().join(format!(
        "shellgame_gpu_backend_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&output_dir);

    let result = train_self_play_models(&TrainingConfig {
        generations: 1,
        population_size: 2,
        episodes_per_eval: 1,
        min_nodes: 5,
        max_nodes: 5,
        max_attempts_factor: 1,
        max_attempts_ratio: Some(0.4),
        max_attempts_cap: Some(3),
        mutation_scale: 0.01,
        seed: 20260423,
        output_dir: output_dir.clone(),
        resume_from: None,
        hall_of_fame_size: 1,
        hall_sample_count: 1,
        static_opponent_sample_count: 1,
        training_mode: TrainingMode::CoAgent,
    });

    let _ = std::fs::remove_dir_all(&output_dir);

    result.expect("tiny CUDA coagent training run should complete on the GPU backend");
    assert!(
        training_accelerator_is_cuda(),
        "training completed after falling back away from CUDA"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_backend_test_requires_cuda_feature() {
    eprintln!(
        "skipping GPU backend assertion; default builds enable CUDA, CPU-only runs use `--no-default-features`"
    );
}
