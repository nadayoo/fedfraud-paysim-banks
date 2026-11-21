"""Train the isolated 15-feature engineered-scaling ablation."""

from local_only.train import main as train_main


if __name__ == "__main__":
    train_main(
        default_config="local_only/configs/local_only_scaled.yaml",
        expected_experiment="scaled_engineered_15",
    )
