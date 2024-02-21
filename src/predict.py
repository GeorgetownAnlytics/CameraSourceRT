import numpy as np
import pandas as pd
from config import paths
from score import (
    calculate_confusion_matrix,
    evaluate_metrics,
    plot_and_save_confusion_matrix,
    save_metrics_to_csv,
)
from models.custom_trainer import CustomTrainer
from utils import TimeAndMemoryTracker
from logger import get_logger
from pathlib import Path


def create_prediction_df(
    ids: np.ndarray, probs: np.ndarray, predictions: np.ndarray, class_to_idx: dict
) -> pd.DataFrame:
    idx_to_class = {k: v for v, k in class_to_idx.items()}
    encoded_targets = list(range(len(class_to_idx)))
    prediction_df = pd.DataFrame({"id": ids})
    prediction_df[encoded_targets] = probs
    prediction_df["prediction"] = predictions
    prediction_df["prediction"] = prediction_df["prediction"].map(idx_to_class)
    prediction_df.rename(columns=idx_to_class, inplace=True)
    return prediction_df


def predict():
    logger = get_logger(task_name="predict")

    trainer = CustomTrainer.load_model()
    logger.info(f"Loaded model {trainer.model_name}")
    test_loader = trainer.test_loader

    logger.info("Predicting on test data...")
    with TimeAndMemoryTracker(logger) as _:
        labels, predictions, probs = trainer.predict(test_loader)

    ids = [Path(i[0]).name for i in test_loader.dataset.imgs]
    class_to_idx = test_loader.dataset.class_to_idx

    prediction_df = create_prediction_df(
        ids=ids, probs=probs, predictions=predictions, class_to_idx=class_to_idx
    )

    logger.info("Saving predictions...")
    prediction_df.to_csv(paths.PREDICTIONS_FILE, index=False)

    logger.info("Evaluating metrics...")
    test_metrics = evaluate_metrics(
        labels=labels,
        predictions=predictions,
        probabilities=probs,
        loss_function=trainer.loss_function,
        top_k=[5],
        n_classes=trainer.num_classes,
    )

    test_metrics_df = pd.DataFrame(
        {
            "test loss": [test_metrics["loss"]],
            "test accuracy": [test_metrics["accuracy"]],
            "test macro_f1": [test_metrics["macro_f1"]],
            "test weighted_f1": [test_metrics["weighted_f1"]],
            "test macro_recall": [test_metrics["macro_recall"]],
            "test weighted_recall": [test_metrics["weighted_recall"]],
            "test macro_precision": [test_metrics["macro_precision"]],
            "test weighted_precision": [test_metrics["weighted_precision"]],
            "test top_k_accuracy": [test_metrics["top_k_accuracy"]],
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
        f"Test metrics: Loss: {test_metrics['loss']}, Accuracy: {test_metrics['accuracy']}, Macro F1 Score: {test_metrics['macro_f1']}"
    )


if __name__ == "__main__":
    predict()
