import os
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torchmetrics
import pandas as pd

from .dataloader import CustomDataLoader


class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, validation_loader, num_samples=20):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_samples = num_samples

        # Initialize metrics for multiclass classification
        num_classes = len(train_loader.dataset.classes)
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

    def train_and_evaluate(self, num_epochs=10, output_folder=None, warmup_epochs=2):
        if num_epochs <= warmup_epochs:
            raise ValueError("num_epochs must be greater than warmup_epochs.")
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = LambdaLR(optimizer, lr_lambda=self._warmup_cosine_annealing(
            0.001, warmup_epochs, num_epochs))

        # Data structure to store metrics
        metrics_history = {
            'Epoch': [],
            'Train Loss': [],
            'Train Accuracy': [],
            'Train F1': [],
            'Test Loss': [],
            'Test Accuracy': [],
            'Test F1': [],
            'Validation Loss': [],
            'Validation Accuracy': [],
            'Validation F1': []
        }

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            running_loss = 0.0
            self.train_accuracy.reset()
            self.train_f1.reset()

            for inputs, labels in tqdm(self.train_loader, desc=f"Training - Epoch {epoch + 1}/{num_epochs}"):
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

                # Calculate average loss, accuracy, and F1 score for the epoch
                avg_loss = running_loss / len(self.train_loader)
                accuracy = self.train_accuracy.compute()
                f1_score = self.train_f1.compute()

                # Append training metrics to history
                metrics_history['Epoch'].append(epoch + 1)
                metrics_history['Train Loss'].append(avg_loss)
                metrics_history['Train Accuracy'].append(accuracy)
                metrics_history['Train F1'].append(f1_score)

                # Evaluate on test set
                test_loss, test_accuracy, test_f1 = self._evaluate_loss_accuracy(
                    self.test_loader, self.test_accuracy, self.test_f1)
                metrics_history['Test Loss'].append(test_loss)
                metrics_history['Test Accuracy'].append(test_accuracy)
                metrics_history['Test F1'].append(test_f1)

                # Evaluate on validation set
                val_loss, val_accuracy, val_f1 = self._evaluate_loss_accuracy(
                    self.validation_loader, self.validation_accuracy, self.validation_f1)
                metrics_history['Validation Loss'].append(val_loss)
                metrics_history['Validation Accuracy'].append(val_accuracy)
                metrics_history['Validation F1'].append(val_f1)

                print(f"Completed epoch {epoch + 1}/{num_epochs}")
                scheduler.step()

            # Save metrics to a DataFrame and CSV
            metrics_df = pd.DataFrame({
                'Epoch': metrics_history['Epoch'],
                'Train Loss': metrics_history['Train Loss'],
                'Train Accuracy': [acc.item() for acc in metrics_history['Train Accuracy']],
                'Train F1': [f1.item() for f1 in metrics_history['Train F1']],
                'Test Loss': metrics_history['Test Loss'],
                'Test Accuracy': [acc.item() for acc in metrics_history['Test Accuracy']],
                'Test F1': [f1.item() for f1 in metrics_history['Test F1']],
                'Validation Loss': metrics_history['Validation Loss'],
                'Validation Accuracy': [acc.item() for acc in metrics_history['Validation Accuracy']],
                'Validation F1': [f1.item() for f1 in metrics_history['Validation F1']]
            })

            metrics_df.to_csv(os.path.join(
                output_folder, 'training_metrics.csv'), index=False)

            # Plotting metrics
            self._plot_metrics(metrics_history, output_folder)

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

    def _plot_and_save_confusion_matrix(self, cm, phase, output_folder, class_names):
        plt.figure(figsize=(16, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{self.model.__class__.__name__} - {phase} Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        cm_filename = os.path.join(output_folder,
                                   f'{phase}_confusion_matrix_{self.model.__class__.__name__}.pdf')
        plt.savefig(cm_filename, format='pdf', bbox_inches='tight')
        # Save the confusion matrix as a CSV file
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_csv_filename = os.path.join(
            output_folder, f'{phase}_confusion_matrix.csv')
        cm_df.to_csv(cm_csv_filename, index_label='True Label',
                     header='Predicted Label')
        plt.close()

    def _plot_metrics(self, metrics_history, output_folder):
        plt.figure(figsize=(16, 10))
        epochs = range(1, len(metrics_history['Epoch']) + 1)

        plt.plot(epochs, metrics_history['Train Loss'],
                 label='Training Loss')
        plt.plot(epochs, metrics_history['Test Loss'],
                 label='Test Loss')
        plt.plot(epochs, metrics_history['Validation Loss'],
                 label='Validation Loss')
        plt.plot(epochs, metrics_history['Train Accuracy'],
                 label='Training Accuracy')
        plt.plot(epochs, metrics_history['Test Accuracy'],
                 label='Test Accuracy')
        plt.plot(epochs, metrics_history['Validation Accuracy'],
                 label='Validation Accuracy')
        plt.plot(epochs, metrics_history['Train F1'],
                 label='Training F1 Score')
        plt.plot(epochs, metrics_history['Test F1'],
                 label='Test F1 Score')
        plt.plot(epochs, metrics_history['Validation F1'],
                 label='Validation F1 Score')

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
    # Note: Replace 'YourDataLoader' with your actual data loader class or function
    custom_data_loader = CustomDataLoader()
    train_loader, test_loader, validation_loader = custom_data_loader.train_loader, custom_data_loader.test_loader, custom_data_loader.validation_loader


    # Initialize the model
    # Note: Replace 'YourModel' with your actual model class
    model = BaseTrainer()

    # Create an instance of the BaseTrainer
    trainer = BaseTrainer(model, train_loader, test_loader, validation_loader)

    # Set device and loss function for the trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer.set_device(device)
    trainer.set_loss_function(torch.nn.CrossEntropyLoss())

    # Specify the output folder to save metrics and plots
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    # Start training and evaluation
    trainer.train_and_evaluate(num_epochs=10, output_folder=output_folder)

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
