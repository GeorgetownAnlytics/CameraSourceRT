import os
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torchmetrics
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from itertools import cycle
from .dataloader import CustomDataLoader
from torch.utils.data.dataloader import DataLoader
from typing import Tuple

from config import paths

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class BaseTrainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        validation_loader,
        num_samples=20,
        loss_function=torch.nn.CrossEntropyLoss(),
        output_folder=paths.OUTPUTS_DIR,
    ):
        self.model = model
        self.output_folder = output_folder
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.loss_function = loss_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_samples = num_samples
        self.model.to(self.device)
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize metrics for multiclass classification
        num_classes = len(train_loader.dataset.classes)
        self._initialize_metrics(num_classes)

        # Initialize class names from train loader if available
        if hasattr(train_loader.dataset, "classes"):
            self.class_names = train_loader.dataset.classes
        else:
            self.class_names = [str(i) for i in range(len(train_loader.dataset))]

    def _initialize_metrics(self, num_classes):
        self.train_accuracy = torchmetrics.Accuracy(
            top_k=1, task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.train_top5_accuracy = torchmetrics.Accuracy(
            top_k=5, task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.validation_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.validation_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(self.device)
        self.validation_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(self.device)
        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(self.device)
        self.validation_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(self.device)

    def set_device(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        print(f"Device set to {device}")

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def train(self, num_epochs=40, checkpoint_dir_path=paths.CHECKPOINTS_DIR):
        # Ensure at least 1 warmup epoch
        warmup_epochs = max(1, num_epochs // 5)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=self._warmup_cosine_annealing(0.001, warmup_epochs, num_epochs),
        )

        total_batches = len(self.train_loader)
        metrics_history = {
            "Epoch": [],
            "Train Loss": [],
            "Train Accuracy": [],
            "Train Recall": [],
            "Train Precision": [],
            "Train F1": [],
            "Validation Loss": [],
            "Validation Accuracy": [],
            "Validation Recall": [],
            "Validation Precision": [],
            "Validation F1": [],
        }

        best_val_accuracy = 0.0  # Initialize best validation accuracy
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            running_loss = 0.0
            self.train_accuracy.reset()
            self.train_f1.reset()
            train_progress_bar = tqdm(
                total=total_batches, desc=f"Training - Epoch {epoch + 1}/{num_epochs}"
            )

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.train_top5_accuracy.update(outputs, labels)
                self.train_precision.update(predicted, labels)
                self.train_recall.update(predicted, labels)
                optimizer.step()
                running_loss += loss.item()

                self.train_accuracy.update(predicted, labels)
                self.train_f1.update(predicted, labels)
                train_progress_bar.update(1)

            avg_loss = running_loss / total_batches
            train_accuracy = self.train_accuracy.compute()
            train_f1_score = self.train_f1.compute()
            train_precision = (
                self.train_precision.compute().item()
            )  # Convert to Python scalar
            train_recall = (
                self.train_recall.compute().item()
            )  # Convert to Python scalar

            metrics_history["Epoch"].append(epoch + 1)
            metrics_history["Train Loss"].append(avg_loss)
            metrics_history["Train Accuracy"].append(train_accuracy)
            metrics_history["Train F1"].append(train_f1_score)
            metrics_history["Train Precision"].append(train_precision)
            metrics_history["Train Recall"].append(train_recall)

            val_loss, val_accuracy, val_f1 = self._evaluate_loss_accuracy(
                self.validation_loader, self.validation_accuracy, self.validation_f1
            )

            # Checkpointing
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_checkpoint(epoch, checkpoint_dir_path)

            metrics_history["Validation Loss"].append(val_loss)
            metrics_history["Validation Accuracy"].append(val_accuracy)
            metrics_history["Validation F1"].append(val_f1)

            val_precision, val_recall = self._evaluate_additional_metrics(
                self.validation_loader,
                self.validation_precision,
                self.validation_recall,
            )
            metrics_history["Validation Precision"].append(val_precision)
            metrics_history["Validation Recall"].append(val_recall)

            print(
                f"Epoch {epoch + 1}/{num_epochs} Completed: Train Loss: {avg_loss},\
                Train Accuracy: {train_accuracy}, Train F1: {train_f1_score}, Validation Loss: {val_loss},\
                    Validation Accuracy: {val_accuracy}, Validation F1: {val_f1}"
            )
            scheduler.step()

        train_progress_bar.close()

        return metrics_history

    def _save_checkpoint(self, epoch, output_folder):
        """
        Saves a checkpoint of the model.

        Args:
            epoch (int): The current epoch number.
            output_folder (str): The directory where the checkpoint will be saved.
        """
        checkpoint_path = os.path.join(
            output_folder, f"model_checkpoint_epoch_{epoch}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _warmup_cosine_annealing(self, base_lr, warmup_epochs, num_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return base_lr * (epoch / warmup_epochs)
            else:
                return base_lr * (
                    0.5
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (epoch - warmup_epochs)
                            / (num_epochs - warmup_epochs)
                        )
                    )
                )

        return lr_lambda

    def _plot_extended_metrics(self, metrics_df, output_folder, model, loader):
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
        self.model.eval()
        y_true = []
        y_scores = []

        # Binarize the output labels for all classes
        num_classes = len(self.class_names)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
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
                "".format(self.class_names[i], roc_auc[i]),
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

    def _evaluate_additional_metrics(
        self, data_loader, precision_metric, recall_metric
    ):
        precision_metric.reset()
        recall_metric.reset()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                precision_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
        return precision_metric.compute().item(), recall_metric.compute().item()

    def predict(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts the class labels and logits for the data in a data loader.

        Args:
            data_loader (DataLoader): The input data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (Truth labels, Predicted class labels, Logits).
        """
        self.model.eval()
        with torch.no_grad():
            all_labels, all_predicted, all_logits = (
                np.array([]),
                np.array([]),
                np.array([]),
            )
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Convert tensors to numpy arrays before appending
                all_predicted = np.append(all_predicted, predicted.cpu().numpy())
                all_labels = np.append(all_labels, labels.cpu().numpy())
                all_logits = (
                    np.concatenate((all_logits, outputs.cpu().numpy()), axis=0)
                    if all_logits.size
                    else outputs.cpu().numpy()
                )

        return all_labels, all_predicted, all_logits
