"""Train the isolated seven-feature balance-free local-only experiment."""

from local_only.train import main as train_main


if __name__ == "__main__":
    train_main(
        default_config="local_only/configs/local_only_pretransaction.yaml",
        expected_experiment="balance_free_7",
    )
