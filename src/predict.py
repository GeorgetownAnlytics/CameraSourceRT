import pandas as pd
from models.resnet_trainer import ResNetTrainer


if __name__ == "__main__":
    trainer = ResNetTrainer.load_model("resnet18")
    test_loader = trainer.test_loader

    test_loss, test_accuracy, test_f1 = trainer._evaluate_loss_accuracy(
        trainer.test_loader, trainer.test_accuracy, trainer.test_f1
    )

    test_metrics = pd.DataFrame(
        {
            "Test Loss": [test_loss],
            "Test Accuracy": [test_accuracy],
            "Test F1": [test_f1],
        }
    )

    trainer._save_metrics_to_csv(test_metrics, file_name="testing_metrics")

    print(
        f"Test - After Training: Loss: {test_loss}, Accuracy: {test_accuracy}, F1 Score: {test_f1}"
    )
