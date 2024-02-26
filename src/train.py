import torch
from models.dataloader import CustomDataLoader
from models.custom_trainer import CustomTrainer
from utils import (
    read_json_as_dict,
    set_seeds,
    get_model_parameters,
    TimeAndMemoryTracker,
)
from config import paths
from score import (
    save_metrics_to_csv,
    plot_and_save_confusion_matrix,
    calculate_confusion_matrix,
)
from logger import get_logger

logger = get_logger(__file__)


def run_training(
    train_folder_path: str = paths.INPUTS_DIR,
    hyperparameters_file_path: str = paths.HYPERPARAMETERS_FILE,
    config_file_path: str = paths.CONFIG_FILE,
) -> None:
    logger.info("Starting training...")

    logger.info("Loading config file...")
    config = read_json_as_dict(config_file_path)

    model_name = config.get("model_name")
    params = get_model_parameters(
        model_name=model_name,
        hyperparameters_file_path=hyperparameters_file_path,
        hyperparameter_tuning=config["hyperparameter_tuning"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = config.get("num_epochs")
    loss_choice = config.get("loss_function")
    num_workers = config.get("num_workers")
    validation_size = config.get("validation_size")

    batch_size = params.get("batch_size")
    image_size = params.get("image_size")
    optimizer = params.get("optimizer")
    lr = params.get("lr")

    loss_function = (
        torch.nn.CrossEntropyLoss()
        if loss_choice == "crossentropy"
        else torch.nn.MultiMarginLoss()
    )
    logger.info(f"Setting seeds to: {config['seed']}")
    set_seeds(config["seed"])

    logger.info("Creating data loader...")
    data_loader = CustomDataLoader(
        base_folder=train_folder_path,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        validation_size=validation_size,
    )

    trainer = CustomTrainer(
        data_loader.train_loader,
        data_loader.test_loader,
        data_loader.validation_loader,
        num_classes=data_loader.num_classes,
        model_name=model_name,
        optimizer=optimizer,
        lr=lr,
    )

    logger.info(f"Using device {device}")
    trainer.set_device(device)
    trainer.set_loss_function(loss_function)

    logger.info("Training model...")
    with TimeAndMemoryTracker(logger) as _:
        metrics_history = trainer.train(num_epochs=num_epochs)

    logger.info("Saving model...")
    trainer.save_model()

    logger.info("Saving training metrics to csv...")
    save_metrics_to_csv(
        metrics_history,
        output_folder=paths.MODEL_ARTIFACTS_DIR,
        file_name="train_validation_metrics.csv",
    )

    logger.info("Predicting labels on training data...")
    train_labels, train_pred, _ = trainer.predict(data_loader.train_loader)

    logger.info("Saving confusion matrix for training data...")
    train_cm = calculate_confusion_matrix(
        all_labels=train_labels, all_predictions=train_pred
    )

    logger.info("Saving confusion matrix plot for training data...")
    plot_and_save_confusion_matrix(
        cm=train_cm,
        phase="train",
        model_name=trainer.__class__.__name__,
        output_folder=paths.MODEL_ARTIFACTS_DIR,
        class_names=trainer.train_loader.dataset.classes,
    )

    if data_loader.validation_loader:
        logger.info("Predicting validation labels...")
        validiation_labels, validation_pred, _ = trainer.predict(
            data_loader.validation_loader
        )

        logger.info("Saving validation confusion matrix...")
        validation_cm = calculate_confusion_matrix(
            all_labels=validiation_labels, all_predictions=validation_pred
        )

        logger.info("Saving validation confusion matrix plot...")
        plot_and_save_confusion_matrix(
            cm=validation_cm,
            phase="validation",
            model_name=model_name,
            output_folder=paths.MODEL_ARTIFACTS_DIR,
            class_names=trainer.train_loader.dataset.classes,
        )

        logger.info(
            f"Validation Accuracy (Last Epoch): {metrics_history['validation accuracy'][-1]}"
        )

    logger.info(
        f"Training Accuracy (Last Epoch): {metrics_history['train accuracy'][-1]}"
    )

    logger.info(f"Training and evaluation for model {model_name} completed.\n")


if __name__ == "__main__":
    run_training()
