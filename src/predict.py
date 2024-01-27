import pandas as pd
from models.resnet_trainer import ResNetTrainer
from config import paths


if __name__ == "__main__":
    trainer = ResNetTrainer.load_model()
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
    print("Saving metrics to csv...")
    trainer._save_metrics_to_csv(
        test_metrics, output_folder=paths.OUTPUTS_DIR, file_name="test_metrics.csv"
    )

    print("Saving confusion matrix...")
    test_cm = trainer._calculate_confusion_matrix(test_loader)
    trainer._plot_and_save_confusion_matrix(
        cm=test_cm,
        phase="test",
        output_folder=paths.OUTPUTS_DIR,
        class_names=trainer.train_loader.dataset.classes,
    )

    print(
        f"Test - After Training: Loss: {test_loss}, Accuracy: {test_accuracy}, F1 Score: {test_f1}"
    )
