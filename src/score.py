import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Union, Tuple
from torchmetrics import Accuracy, Recall, Precision, F1Score
from torch.nn.modules.loss import CrossEntropyLoss, MultiMarginLoss


def save_metrics_to_csv(
    metrics_history: Dict[str, List], output_folder: str, file_name: str
) -> None:
    """
    Converts metrics dictionary to pandas dataframe and saves it in csv file.

    Args:
        metrics_history (Dict[str, List]):
            Dictionary of metrics and its values.
            For example: {"recall": [0.9, 0.8, 0.87], "precision": [1.0, 0.33, 0.22]}

        output_folder (str): The path of the folder to store the created file.

        file_name (str): The name of the created file.

        Returns: None
    """

    # Convert all tensor values in metrics_history to CPU and then to NumPy
    for key in metrics_history:
        metrics_history[key] = [
            x.cpu().numpy() if torch.is_tensor(x) else x for x in metrics_history[key]
        ]

    metrics_df = pd.DataFrame(metrics_history)
    metrics_csv_path = os.path.join(output_folder, file_name)
    metrics_df.to_csv(metrics_csv_path, index=False)


def calculate_confusion_matrix(
    all_labels: np.ndarray, all_predictions: np.ndarray
) -> np.ndarray:
    """
    Calculated the confusion matrix for the given labels and predictions.

    Args:
        all_labels (np.ndarray): Numpy array of labels.
        all_predictions (np.ndarray): Numpy array of predictions

    Returns (np.ndarray): The confusion matric
    """
    return confusion_matrix(all_labels, all_predictions)


def plot_and_save_confusion_matrix(cm, phase, model_name, output_folder, class_names):
    plt.figure(figsize=(16, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{model_name} - {phase} Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_filename = os.path.join(
        output_folder,
        f"{phase}_confusion_matrix_{model_name}.pdf",
    )
    plt.savefig(cm_filename, format="pdf", bbox_inches="tight")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_filename = os.path.join(output_folder, f"{phase}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_filename, index_label="True Label", header="Predicted Label")
    plt.close()


def save_confusion_matrix_csv(
    all_labels: np.ndarray,
    all_predictions: np.ndarray,
    phase: str,
    class_names: List[str],
    output_folder: str,
):
    """
    Saves the confusion matrix to a CSV file.

    Args:
        all_labels (np.ndarray): Numpy array of labels.
        all_predictions (np.ndarray): Numpy array of predictions.
        class_names (List[str]): List of names of target classes.
        phase (str): The phase during which the confusion matrix was generated (e.g., 'train', 'test', 'validation').
        output_folder (str): The directory where the CSV file will be saved.

        Returns: None
    """
    confusion_matrix = calculate_confusion_matrix(
        all_labels=all_labels, all_predictions=all_predictions
    )
    cm_df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    cm_csv_filename = os.path.join(output_folder, f"{phase}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_filename, index_label="True Label", header="Predicted Label")
    print(f"Confusion matrix saved as CSV in {cm_csv_filename}")
    return confusion_matrix


def plot_metrics(metrics_history, output_folder):
    plt.figure(figsize=(16, 10))
    epochs = range(1, len(metrics_history["Epoch"]) + 1)
    plt.plot(epochs, metrics_history["Train Loss"], label="Training Loss")
    plt.plot(epochs, metrics_history["Train Accuracy"], label="Training Accuracy")
    plt.plot(epochs, metrics_history["Train F1"], label="Training F1 Score")
    plt.plot(epochs, metrics_history["Validation Loss"], label="Validation Loss")
    plt.plot(
        epochs, metrics_history["Validation Accuracy"], label="Validation Accuracy"
    )
    plt.plot(epochs, metrics_history["Validation F1"], label="Validation F1 Score")
    plt.title("Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(output_folder, "metrics_over_epochs.pdf")
    plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        metrics_history["Epoch"],
        metrics_history["Train Precision"],
        label="Train Precision",
    )
    plt.plot(
        metrics_history["Epoch"],
        metrics_history["Train Recall"],
        label="Train Recall",
    )
    plt.plot(
        metrics_history["Epoch"],
        metrics_history["Validation Precision"],
        label="Validation Precision",
    )
    plt.plot(
        metrics_history["Epoch"],
        metrics_history["Validation Recall"],
        label="Validation Recall",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training and Validation Precision and Recall")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "precision_recall_plot.pdf"))
    plt.close()


def evaluate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    logits: np.ndarray,
    loss_function: Union[CrossEntropyLoss, MultiMarginLoss],
    accuracy_metric: Accuracy,
    f1_metric: F1Score,
    recall_metric: Recall,
    precision_metric: Precision,
) -> Tuple[float, float, float, float, float]:
    """
    Evaluates loss, accuracy and f1-score on given labels and predictions.

    Args:
        labels (np.ndarray): True labels.
        predictions (np.ndarray): Predicted labels.
        loss_function (Union[CrossEntropyLoss, MultiMarginLoss]): The loss function.
        accuracy_metric (Accuracy): torchmetrics Accuracy object.
        f1_metric (F1Score): torchmetrics F1Score object.
        recall_metric (Recall): torchmetrics Recall object.
        precision_metric (Precision): torchmetrics Precision object.


    Returns (Tuple[float, float, float, float, float]): (loss, accuracy, f1-score, recall, precision)
    """
    accuracy_metric.reset()
    f1_metric.reset()
    recall_metric.reset()
    precision_metric.reset()

    labels = torch.from_numpy(labels).long()
    predictions = torch.from_numpy(predictions)
    logits = torch.from_numpy(logits)
    loss = loss_function(logits, labels).item()

    accuracy_metric.update(predictions, labels)
    f1_metric.update(predictions, labels)
    precision_metric.update(predictions, labels)
    recall_metric.update(predictions, labels)

    accuracy = accuracy_metric.compute().item()
    f1_score = f1_metric.compute().item()
    recall = recall_metric.compute().item()
    precision = precision_metric.compute().item()
    return loss, accuracy, f1_score, recall, precision


def plot_extended_metrics(
    metrics_df, output_folder, model, loader, class_names, device
):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Plot for Precision and Recall (assuming these columns are in metrics_df)
    plt.figure(figsize=(10, 5))
    plt.plot(
        metrics_df["Epoch"], metrics_df["Train Precision"], label="Train Precision"
    )
    plt.plot(metrics_df["Epoch"], metrics_df["Train Recall"], label="Train Recall")
    plt.plot(
        metrics_df["Epoch"],
        metrics_df["Validation Precision"],
        label="Validation Precision",
    )
    plt.plot(
        metrics_df["Epoch"],
        metrics_df["Validation Recall"],
        label="Validation Recall",
    )
    plt.title("Precision and Recall over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "precision_recall_plot.pdf"))
    plt.close()

    # Plot for Top-5 Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(
        metrics_df["Epoch"],
        metrics_df["Train Top-5 Accuracy"],
        label="Train Top-5 Accuracy",
    )
    plt.plot(
        metrics_df["Epoch"],
        metrics_df["Validation Top-5 Accuracy"],
        label="Validation Top-5 Accuracy",
    )
    plt.title("Top-5 Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "top5_accuracy_plot.pdf"))
    plt.close()

    # Multiclass ROC
    model.eval()
    y_true = []
    y_scores = []

    # Binarize the output labels for all classes
    num_classes = len(class_names)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    y_true = label_binarize(y_true, classes=range(num_classes))
    y_scores = np.array(y_scores)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(["blue", "red", "green", "cyan", "magenta", "yellow", "black"])
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})"
            "".format(class_names[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_folder, "multiclass_roc_curve.pdf"))
    plt.close()
