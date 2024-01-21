import os
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix
import torchmetrics
import pandas as pd
from tqdm import tqdm
from .dataloader import CustomDataLoader


class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, validation_loader, num_samples=20, loss_function=torch.nn.CrossEntropyLoss()):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.loss_function = loss_function
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_samples = num_samples
        self.model.to(self.device)

        # Initialize metrics for multiclass classification
        num_classes = len(train_loader.dataset.classes)
        self._initialize_metrics(num_classes)
        
        # Initialize class names from train loader if available
        if hasattr(train_loader.dataset, 'classes'):
            self.class_names = train_loader.dataset.classes
        else:
            self.class_names = [str(i) for i in range(len(train_loader.dataset))]

    def _initialize_metrics(self, num_classes):
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes).to(self.device)
        self.validation_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes).to(self.device)
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes).to(self.device)
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes).to(self.device)
        self.validation_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes).to(self.device)
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes).to(self.device)

    def set_device(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def train(self, num_epochs=5, warmup_epochs=1):
        if num_epochs <= warmup_epochs:
            raise ValueError("num_epochs must be greater than warmup_epochs.")
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = LambdaLR(optimizer, lr_lambda=self._warmup_cosine_annealing(
            0.001, warmup_epochs, num_epochs))

        total_batches = len(self.train_loader)
        metrics_history = {
            'Epoch': [],
            'Train Loss': [],
            'Train Accuracy': [],
            'Train F1': [],
            'Validation Loss': [],
            'Validation Accuracy': [],
            'Validation F1': []
        }

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            running_loss = 0.0
            self.train_accuracy.reset()
            self.train_f1.reset()
            train_progress_bar = tqdm(
                total=total_batches, desc=f"Training - Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                self.train_accuracy.update(predicted, labels)
                self.train_f1.update(predicted, labels)
                train_progress_bar.update(1)

            avg_loss = running_loss / total_batches
            train_accuracy = self.train_accuracy.compute()
            train_f1_score = self.train_f1.compute()

            metrics_history['Epoch'].append(epoch + 1)
            metrics_history['Train Loss'].append(avg_loss)
            metrics_history['Train Accuracy'].append(train_accuracy)
            metrics_history['Train F1'].append(train_f1_score)

            val_loss, val_accuracy, val_f1 = self._evaluate_loss_accuracy(
                self.validation_loader, self.validation_accuracy, self.validation_f1)
            metrics_history['Validation Loss'].append(val_loss)
            metrics_history['Validation Accuracy'].append(val_accuracy)
            metrics_history['Validation F1'].append(val_f1)

            print(f"Epoch {epoch + 1}/{num_epochs} Completed: Train Loss: {avg_loss},\
                Train Accuracy: {train_accuracy}, Train F1: {train_f1_score}, Validation Loss: {val_loss},\
                    Validation Accuracy: {val_accuracy}, Validation F1: {val_f1}")
            scheduler.step()

        train_progress_bar.close()

        # Evaluate on test set after training
        test_loss, test_accuracy, test_f1 = self._evaluate_loss_accuracy(
            self.test_loader, self.test_accuracy, self.test_f1)
        print(
            f"Test - After Training: Loss: {test_loss}, Accuracy: {test_accuracy}, F1 Score: {test_f1}")

        return metrics_history

    def _evaluate_loss_accuracy(self, data_loader, accuracy_metric, f1_metric):
        self.model.eval()
        total_loss = 0.0
        accuracy_metric.reset()
        f1_metric.reset()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                accuracy_metric.update(predicted, labels)
                f1_metric.update(predicted, labels)

        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = accuracy_metric.compute()
        f1_score = f1_metric.compute()
        return avg_loss, accuracy, f1_score

    def _warmup_cosine_annealing(self, base_lr, warmup_epochs, num_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return base_lr * (epoch / warmup_epochs)
            else:
                return base_lr * (0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs))))
        return lr_lambda

    def _calculate_confusion_matrix(self, loader):
        all_preds, all_labels = [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return confusion_matrix(all_labels, all_preds)
    
    def save_confusion_matrix_csv(self, confusion_matrix, phase, output_folder):
        """
        Saves the confusion matrix to a CSV file.

        Args:
            confusion_matrix (numpy.ndarray): The confusion matrix to be saved.
            phase (str): The phase during which the confusion matrix was generated (e.g., 'train', 'test', 'validation').
            output_folder (str): The directory where the CSV file will be saved.
        """
        cm_df = pd.DataFrame(confusion_matrix, index=self.class_names, columns=self.class_names)
        cm_csv_filename = os.path.join(output_folder, f'{phase}_confusion_matrix.csv')
        cm_df.to_csv(cm_csv_filename, index_label='True Label', header='Predicted Label')
        print(f"Confusion matrix saved as CSV in {cm_csv_filename}")

    def _plot_and_save_confusion_matrix(self, cm, phase, output_folder, class_names):
        plt.figure(figsize=(16, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{self.model.__class__.__name__} - {phase} Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        cm_filename = os.path.join(
            output_folder, f'{phase}_confusion_matrix_{self.model.__class__.__name__}.pdf')
        plt.savefig(cm_filename, format='pdf', bbox_inches='tight')
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_csv_filename = os.path.join(
            output_folder, f'{phase}_confusion_matrix.csv')
        cm_df.to_csv(cm_csv_filename, index_label='True Label',
                     header='Predicted Label')
        plt.close()

    def _plot_metrics(self, metrics_history, output_folder):
        plt.figure(figsize=(16, 10))
        epochs = range(1, len(metrics_history['Epoch']) + 1)
        plt.plot(epochs, metrics_history['Train Loss'], label='Training Loss')
        plt.plot(
            epochs, metrics_history['Train Accuracy'], label='Training Accuracy')
        plt.plot(
            epochs, metrics_history['Train F1'], label='Training F1 Score')
        plt.plot(
            epochs, metrics_history['Validation Loss'], label='Validation Loss')
        plt.plot(
            epochs, metrics_history['Validation Accuracy'], label='Validation Accuracy')
        plt.plot(
            epochs, metrics_history['Validation F1'], label='Validation F1 Score')
        plt.title('Metrics Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join(output_folder, 'metrics_over_epochs.pdf')
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Initialize data loaders
    custom_data_loader = CustomDataLoader()
    train_loader, test_loader, validation_loader = custom_data_loader.train_loader, custom_data_loader.test_loader, custom_data_loader.validation_loader

    # Initialize the model
    # Note: Replace 'YourModel' with your actual model class
    model = BaseTrainer()

    # Create an instance of the BaseTrainer
    trainer = BaseTrainer(model=model, train_loader=train_loader,
                          test_loader=test_loader, validation_loader=validation_loader)

    # Set device and loss function for the trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer.set_device(device)
    trainer.set_loss_function(torch.nn.CrossEntropyLoss())

    # Specify the output folder to save metrics and plots
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    # Start training and evaluation
    trainer.train(num_epochs=10)

    # Optionally, you can generate and plot confusion matrices
    class_names = train_loader.dataset.classes  # Or however you obtain class names
    train_cm = trainer._calculate_confusion_matrix(train_loader)
    trainer._plot_and_save_confusion_matrix(train_cm, 'train',
                                            output_folder, class_names)

    test_cm = trainer._calculate_confusion_matrix(test_loader)
    trainer._plot_and_save_confusion_matrix(test_cm, 'test',
                                            output_folder, class_names)

    validation_cm = trainer._calculate_confusion_matrix(validation_loader)
    trainer._plot_and_save_confusion_matrix(validation_cm, 'validation',
                                            output_folder, class_names)
