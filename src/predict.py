import pandas as pd
from models.resnet_trainer import ResNetTrainer
from config import paths
from score import (
    calculate_confusion_matrix,
    evaluate_loss_accuracy_f1,
    plot_and_save_confusion_matrix,
    save_metrics_to_csv,
)


if __name__ == "__main__":
    trainer = ResNetTrainer.load_model()
    test_loader = trainer.test_loader

    labels, predictions, logits = trainer.predict(test_loader)

    test_loss, test_accuracy, test_f1 = evaluate_loss_accuracy_f1(
        labels=labels,
        predictions=predictions,
        logits=logits,
        loss_function=trainer.loss_function,
        accuracy_metric=trainer.test_accuracy,
        f1_metric=trainer.test_f1,
    )

    test_metrics = pd.DataFrame(
        {
            "Test Loss": [test_loss],
            "Test Accuracy": [test_accuracy],
            "Test F1": [test_f1],
        }
    )
    print("Saving metrics to csv...")
    save_metrics_to_csv(
        test_metrics, output_folder=paths.PREDICTIONS_DIR, file_name="test_metrics.csv"
    )

    print("Saving confusion matrix...")
    test_cm = calculate_confusion_matrix(all_labels=labels, all_predictions=predictions)
    plot_and_save_confusion_matrix(
        cm=test_cm,
        phase="test",
        model_name=trainer.__class__.__name__,
        output_folder=paths.PREDICTIONS_DIR,
        class_names=trainer.train_loader.dataset.classes,
    )

    print(
        f"Test - After Training: Loss: {test_loss}, Accuracy: {test_accuracy}, F1 Score: {test_f1}"
    )
