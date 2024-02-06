import torch
from models.dataloader import CustomDataLoader
from models.resnet_trainer import ResNetTrainer

from utils import read_json_as_dict, set_seeds, get_model_parameters
from config import paths
from score import (
    save_metrics_to_csv,
    plot_and_save_confusion_matrix,
    calculate_confusion_matrix,
)

from utils import TimeAndMemoryTracker


def main():
    config = read_json_as_dict(paths.CONFIG_FILE)

    model_name = config.get("model_name")
    params = get_model_parameters(
        model_name=model_name,
        hyperparameters_file_path=paths.HYPERPARAMETERS_FILE,
        hyperparameter_tuning=config["hyperparameter_tuning"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = config.get("num_epochs")
    loss_choice = config.get("loss_function")
    num_workers = config.get("num_workers")
    batch_size = params.get("batch_size")
    loss_function = (
        torch.nn.CrossEntropyLoss()
        if loss_choice == "crossentropy"
        else torch.nn.MultiMarginLoss()
    )
    print("Setting seeds to:", config["seed"])
    set_seeds(config["seed"])

    custom_data_loader = CustomDataLoader(
        base_folder=paths.INPUTS_DIR, batch_size=batch_size, num_workers=num_workers
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
    with TimeAndMemoryTracker() as _:
        metrics_history = trainer.train(num_epochs=num_epochs)

    print("Saving model...")
    trainer.save_model()

    print("Saving metrics to csv...")
    save_metrics_to_csv(
        metrics_history,
        output_folder=paths.MODEL_ARTIFACTS_DIR,
        file_name="train_validation_metrics.csv",
    )

    print("Predicting train and validation labels...")
    train_labels, train_pred, _ = trainer.predict(train_loader)
    validiation_labels, validation_pred, _ = trainer.predict(validation_loader)

    print("Saving confusion matrix...")
    train_cm = calculate_confusion_matrix(
        all_labels=train_labels, all_predictions=train_pred
    )
    validation_cm = calculate_confusion_matrix(
        all_labels=validiation_labels, all_predictions=validation_pred
    )

    plot_and_save_confusion_matrix(
        cm=train_cm,
        phase="train",
        model_name=trainer.__class__.__name__,
        output_folder=paths.MODEL_ARTIFACTS_DIR,
        class_names=trainer.train_loader.dataset.classes,
    )

    plot_and_save_confusion_matrix(
        cm=validation_cm,
        phase="validation",
        model_name=model_name,
        output_folder=paths.MODEL_ARTIFACTS_DIR,
        class_names=trainer.train_loader.dataset.classes,
    )

    print(f"Training Accuracy (Last Epoch): {metrics_history['Train Accuracy'][-1]}")

    print(f"Training and evaluation for model {model_name} completed.\n")

    print("All models have been processed.")


if __name__ == "__main__":
    main()
