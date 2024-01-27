import torch
from models.dataloader import CustomDataLoader
from models.resnet_trainer import ResNetTrainer

from utils import read_json_as_dict, set_seeds
from config import paths


def main():
    params = read_json_as_dict(paths.HPT_FILE)["default_hyperparameters"]
    config = read_json_as_dict(paths.CONFIG_FILE)
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
    print("Setting seeds to:", params["seed"])
    set_seeds(params["seed"])
    model_name = config.get("model_name")

    custom_data_loader = CustomDataLoader(
        batch_size=batch_size, num_workers=num_workers
    )

    print(f"\nWorking on model: {model_name}")

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

    print("Saving model...")
    trainer.save_model()

    print("Saving metrics to csv...")
    trainer._save_metrics_to_csv(
        metrics_history,
        output_folder=paths.MODEL_ARTIFACTS_DIR,
        file_name="train_validation_metrics.csv",
    )

    print("Saving confusion matrix...")
    train_cm = trainer._calculate_confusion_matrix(train_loader)
    validation_cm = trainer._calculate_confusion_matrix(validation_loader)

    trainer._plot_and_save_confusion_matrix(
        cm=train_cm,
        phase="train",
        output_folder=paths.MODEL_ARTIFACTS_DIR,
        class_names=trainer.train_loader.dataset.classes,
    )

    trainer._plot_and_save_confusion_matrix(
        cm=validation_cm,
        phase="validation",
        output_folder=paths.MODEL_ARTIFACTS_DIR,
        class_names=trainer.train_loader.dataset.classes,
    )

    print(f"Training Accuracy (Last Epoch): {metrics_history['Train Accuracy'][-1]}")

    print(f"Training and evaluation for model {model_name} completed.\n")

    print("All models have been processed.")


if __name__ == "__main__":
    main()
