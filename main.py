import os
import numpy as np
import pandas as pd
import torch
from models.dataloader import CustomDataLoader
from models.resnet_trainer import ResNetTrainer


def validate_input(prompt, default, valid_options=None):
    while True:
        user_input = input(prompt) or default
        if valid_options and user_input not in valid_options:
            print(f"Invalid input. Valid options are: {valid_options}")
        else:
            return user_input


def main():
    num_epochs = int(validate_input(
        "Enter the number of total epochs (default is 40): ", "40"))
    device = validate_input(
        "Enter 'cuda' for GPU or 'cpu' for CPU (default is 'cuda'): ", "cuda", ["cuda", "cpu"])
    loss_choice = validate_input("Choose the loss function: 'crossentropy' or 'multiclass_hinge' (default is 'crossentropy'): ", "crossentropy", [
                                 "crossentropy", "multiclass_hinge"])

    loss_function = torch.nn.CrossEntropyLoss(
    ) if loss_choice == 'crossentropy' else torch.nn.MultiMarginLoss()

    output_folder = 'output/model_outputs'
    model_names = ['resnet18', 'resnet34',
                   'resnet50', 'resnet101', 'resnet152']

    custom_data_loader = CustomDataLoader()
    for model_name in model_names:
        print(f"\nWorking on model: {model_name}")

        model_output_folder = os.path.join(output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)

        train_loader, test_loader, validation_loader = custom_data_loader.train_loader, custom_data_loader.test_loader, custom_data_loader.validation_loader

        num_classes = len(train_loader.dataset.classes)
        trainer = ResNetTrainer(train_loader, test_loader,
                                validation_loader, num_classes, model_name)
        trainer.set_device(device)
        trainer.set_loss_function(loss_function)

        metrics_history = trainer.train(num_epochs=num_epochs)

        # Save metrics to CSV and plot extended metrics
        trainer._save_metrics_to_csv(metrics_history)
        trainer._plot_extended_metrics(pd.DataFrame(
            metrics_history), model_output_folder, trainer.model, validation_loader)

        print(
            f"Training Accuracy (Last Epoch): {metrics_history['Train Accuracy'][-1]}")

        print(f"Training and evaluation for model {model_name} completed.\n")

    print("All models have been processed.")


if __name__ == '__main__':
    main()
