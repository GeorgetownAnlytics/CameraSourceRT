import pandas as pd
from models.resnet_trainer import ResNetTrainer
from config import paths
from score import (
    calculate_confusion_matrix,
    evaluate_metrics,
    plot_and_save_confusion_matrix,
    save_metrics_to_csv,
)
from utils import TimeAndMemoryTracker
from logger import get_logger


def predict():
    logger = get_logger(task_name="predict")

    trainer = ResNetTrainer.load_model()
    test_loader = trainer.test_loader

    logger.info("Predicting on test data...")
    with TimeAndMemoryTracker() as _:
        labels, predictions, logits = trainer.predict(test_loader)

    test_metrics = evaluate_metrics(
        labels=labels,
        predictions=predictions,
        logits=logits,
        loss_function=trainer.loss_function,
        top_k=[5],
    )

    test_metrics_df = pd.DataFrame(
        {
            "Test Loss": [test_metrics["loss"]],
            "Test Accuracy": [test_metrics["accuracy"]],
            "Test F1": [test_metrics["f1-score"]],
            "Test Recall": [test_metrics["recall"]],
            "Test Precision": [test_metrics["precision"]],
        }
    )
    logger.info("Saving metrics to csv...")
    save_metrics_to_csv(
        test_metrics_df,
        output_folder=paths.PREDICTIONS_DIR,
        file_name="test_metrics.csv",
    )

    logger.info("Saving confusion matrix...")
    test_cm = calculate_confusion_matrix(all_labels=labels, all_predictions=predictions)
    plot_and_save_confusion_matrix(
        cm=test_cm,
        phase="test",
        model_name=trainer.__class__.__name__,
        output_folder=paths.PREDICTIONS_DIR,
        class_names=trainer.train_loader.dataset.classes,
    )

    logger.info(
        f"Test - After Training: Loss: {test_metrics['loss']}, Accuracy: {test_metrics['accuracy']}, F1 Score: {test_metrics['f1-score']}"
    )


if __name__ == "__main__":
    predict()
