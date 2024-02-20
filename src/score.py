"""This module contains functions used in evaluating/plotting metrics"""

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Union, Tuple
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
) -> None:
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


def mean_average_precision_score(
    labels: np.ndarray, predicted_probabilities: np.ndarray, n_classes: int
) -> float:
    """
    Calculate the Mean Average Precision (mAP) across all classes for a multi-class classification task.

    This function computes the Average Precision (AP) for each class by considering the predicted probabilities
    and comparing them with the true labels. The AP scores are then averaged to obtain the mAP, which provides
    a single metric to evaluate the performance of a classifier across all classes, especially useful in
    imbalanced datasets where class distribution is uneven.

    Args:
    - labels (np.ndarray): A 1D numpy array of shape (n_samples,) containing the integer-encoded true labels
      for each sample. Each element in this array should be an integer representing the class label.
    - predicted_probabilities (np.ndarray): A 2D numpy array of shape (n_samples, n_classes) containing the
      predicted probabilities for each class for each sample. Each row in this array should sum to 1, representing
      the probability distribution across all classes for a given sample.
    - n_classes (int): The total number of unique classes in the dataset. This is used to ensure the correct
      binarization of the labels array into a one-hot encoded format.

    Returns:
    - mAP (float): The mean of the average precision scores across all classes. A higher mAP value indicates
      better overall performance of the classifier across all classes, taking into account both precision and
      recall for each class.
    """
    labels_one_hot = label_binarize(labels, classes=range(n_classes))

    # List to store average precision for each class
    ap_scores = []

    # Calculate AP for each class
    for class_index in range(n_classes):
        # Extract the true labels and predictions for the current class
        class_labels = labels_one_hot[:, class_index]
        class_predictions = predicted_probabilities[:, class_index]

        # Calculate the AP for the current class
        ap = average_precision_score(class_labels, class_predictions)

        # Store the AP score
        ap_scores.append(ap)

    # Calculate the mean of the AP scores
    mAP = np.mean(ap_scores)

    return mAP


def evaluate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    loss_function: Union[CrossEntropyLoss, MultiMarginLoss],
    top_k: List[int],
    n_classes: int,
) -> Dict[str, float]:
    """
    Evaluates loss, accuracy, recall, precision, f1-score and top k accuracy on given labels and predictions.

    Args:
        - labels (np.ndarray): True labels.
        - predictions (np.ndarray): Predicted labels.
        - probabilities (np.ndarray): Predicted probabilities of class labels.
        - loss_function (Union[CrossEntropyLoss, MultiMarginLoss]): The loss function.
        - top_k (List[int]): The values to use for k when calculating top k accuracy.
        - n_classes (int): The number of classes in the multiclass classification problem.

    Returns (Dict[str]): The calculated metrics.
    """
    torch_labels = torch.from_numpy(labels).long()
    predictions = torch.from_numpy(predictions)
    torch_probabilities = torch.from_numpy(probabilities)
    loss = loss_function(torch_probabilities, torch_labels).item()

    accuracy = accuracy_score(y_pred=predictions, y_true=labels)
    macro_recall = recall_score(y_pred=predictions, y_true=labels, average="macro")
    weighted_recall = recall_score(
        y_pred=predictions, y_true=labels, average="weighted"
    )
    macro_precision = precision_score(
        y_pred=predictions, y_true=labels, average="macro"
    )
    weighted_precision = precision_score(
        y_pred=predictions, y_true=labels, average="weighted"
    )
    macro_f1 = f1_score(y_pred=predictions, y_true=labels, average="macro")
    weighted_f1 = f1_score(y_pred=predictions, y_true=labels, average="weighted")

    mean_average_precision = mean_average_precision_score(
        labels=labels, predicted_probabilities=probabilities, n_classes=n_classes
    )
    top_k_accuracy = {}

    for k in top_k:
        score = top_k_accuracy_score(y_score=probabilities, y_true=labels, k=k)
        top_k_accuracy[k] = score

    result = {
        "loss": loss,
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top_k_accuracy": top_k_accuracy,
        "mean_average_precision": mean_average_precision,
    }
    return result


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
