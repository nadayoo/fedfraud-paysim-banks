"""Evaluate scaled engineered-feature checkpoints on the global test split."""

from local_only.evaluate_global import main as evaluate_main


if __name__ == "__main__":
    evaluate_main(
        default_config="local_only/configs/local_only_scaled.yaml",
        expected_experiment="scaled_engineered_15",
    )
