import os
import torch
from models.dataloader import CustomDataLoader
from models.resnet_trainer import ResNetTrainer
import pandas as pd

from utils import read_json_as_dict
from config import paths


def main():
    params = read_json_as_dict(paths.CONFIG_FILE)
    num_epochs = params.get("num_epochs")
    device = params.get("device")
    loss_choice = params.get("loss_function")
    num_workers = params.get("num_workers")
    batch_size = params.get("batch_size")
    loss_function = (
        torch.nn.CrossEntropyLoss()
        if loss_choice == "crossentropy"
        else torch.nn.MultiMarginLoss()
    )

    output_folder = paths.OUTPUTS_DIR
    model_names = params.get("model_names")

    custom_data_loader = CustomDataLoader(
        batch_size=batch_size, num_workers=num_workers
    )

    for model_name in model_names:
        print(f"\nWorking on model: {model_name}")

        model_output_folder = os.path.join(output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)

        train_loader, test_loader, validation_loader = (
            custom_data_loader.train_loader,
            custom_data_loader.test_loader,
            custom_data_loader.validation_loader,
        )

        num_classes = len(train_loader.dataset.classes)

        trainer = ResNetTrainer(
            train_loader, test_loader, validation_loader, num_classes, model_name
        )
        trainer.set_device(device)
        trainer.set_loss_function(loss_function)

        print("Training model...")
        metrics_history = trainer.train(num_epochs=num_epochs)

        print("Saving metrics to csv...")
        trainer._save_metrics_to_csv(metrics_history, file_name="training_metrics.csv")

        print("Saving model...")
        trainer.save_model()

        # trainer._plot_extended_metrics(
        #     pd.DataFrame(metrics_history),
        #     model_output_folder,
        #     trainer.model,
        #     validation_loader,
        # )

        print(
            f"Training Accuracy (Last Epoch): {metrics_history['Train Accuracy'][-1]}"
        )

        print(f"Training and evaluation for model {model_name} completed.\n")

    print("All models have been processed.")


if __name__ == "__main__":
    main()
