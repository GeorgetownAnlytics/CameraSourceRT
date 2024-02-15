import os
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple
from config import paths
from typing import Union, Dict, List
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from score import evaluate_metrics
from torch.nn import CrossEntropyLoss, MultiMarginLoss
from torch.nn.functional import softmax


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

        # Initialize class names from train loader if available
        if hasattr(train_loader.dataset, "classes"):
            self.class_names = train_loader.dataset.classes
        else:
            self.class_names = [str(i) for i in range(len(train_loader.dataset))]

    def set_device(self, device: str) -> None:
        """
        Sets the device used for training.

        Args:
            device (str): Name of the device (ex: "cuda", "cpu").

        Returns: None
        """
        self.device = torch.device(device)
        self.model.to(self.device)

    def set_loss_function(
        self, loss_function: Union[CrossEntropyLoss, MultiMarginLoss]
    ) -> None:
        """
        Sets loss function used in training.

        Args:
            loss_function (Union[CrossEntropyLoss, MultiMarginLoss]): Loss function.

        Returns: None
        """
        self.loss_function = loss_function

    def train(
        self, num_epochs: int = 40, checkpoint_dir_path: str = paths.CHECKPOINTS_DIR
    ) -> Dict[str, List]:
        """
        Train the model on the data and calculate metrics per epoch on train and validation datasets.

        Args:
            num_epochs (int): The number of epochs.
            checkpoint_dir_path (str): Path for directory in which checkpoints are saved.


        Returns (Dict[str, List]): Train and validation metrics per epoch.
        example: {
            "Epoch": [0, 1, 2],
            "Train Loss": [1.5, 1.3, 1],
            "Validation Loss": [3, 2.5, 2.2],
        }
        """
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
            train_progress_bar = tqdm(
                total=total_batches, desc=f"Training - Epoch {epoch + 1}/{num_epochs}"
            )

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                loss = self.loss_function(outputs, labels)
                loss.backward()

                train_metrics = evaluate_metrics(
                    labels=labels.cpu().numpy(),
                    predictions=predicted.cpu().numpy(),
                    logits=outputs.data.cpu().numpy(),
                    loss_function=self.loss_function,
                    top_k=[5],
                )
                optimizer.step()
                running_loss += loss.item()
                train_progress_bar.update(1)

            metrics_history["Epoch"].append(epoch + 1)
            metrics_history["Train Loss"].append(train_metrics["loss"])
            metrics_history["Train Accuracy"].append(train_metrics["accuracy"])
            metrics_history["Train F1"].append(train_metrics["f1-score"])
            metrics_history["Train Precision"].append(train_metrics["precision"])
            metrics_history["Train Recall"].append(train_metrics["recall"])

            val_labels, val_pred, val_logits = self.predict(self.validation_loader)

            val_metrics = evaluate_metrics(
                val_labels, val_pred, val_logits, self.loss_function, top_k=[5]
            )

            # Checkpointing
            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                self._save_checkpoint(epoch, checkpoint_dir_path)

            metrics_history["Validation Loss"].append(val_metrics["loss"])
            metrics_history["Validation Accuracy"].append(val_metrics["accuracy"])
            metrics_history["Validation F1"].append(val_metrics["f1-score"])

            metrics_history["Validation Precision"].append(val_metrics["precision"])
            metrics_history["Validation Recall"].append(val_metrics["recall"])

            print(
                f"Epoch {epoch + 1}/{num_epochs} Completed: Train Loss: {train_metrics['loss']},\
                Train Accuracy: {train_metrics['accuracy']}, Train F1: {train_metrics['f1-score']}, Validation Loss: {val_metrics['loss']},\
                    Validation Accuracy: {val_metrics['accuracy']}, Validation F1: {val_metrics['f1-score']}"
            )
            scheduler.step()

        train_progress_bar.close()

        return metrics_history

    def _save_checkpoint(self, epoch: int, output_folder: str) -> None:
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

    def predict(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts the class labels and logits for the data in a data loader.

        Args:
            data_loader (DataLoader): The input data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (Truth labels, Predicted class labels, Probabilities).
        """
        self.model.eval()
        with torch.no_grad():
            all_labels, all_predicted, all_probs = (
                np.array([]),
                np.array([]),
                np.array([]),
            )
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs = softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                # Convert tensors to numpy arrays before appending
                all_predicted = np.append(all_predicted, predicted.cpu().numpy())
                all_labels = np.append(all_labels, labels.cpu().numpy())
                all_probs = (
                    np.concatenate((all_probs, probs.cpu().numpy()), axis=0)
                    if all_probs.size
                    else probs.cpu().numpy()
                )

        return all_labels, all_predicted, all_probs
