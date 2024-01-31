import os
import torch
from models.dataloader import CustomDataLoader
from models.resnet_trainer import ResNetTrainer
import pandas as pd

from utils import read_json_as_dict, set_seeds
from config import paths

if __name__ == "__main__":
    config = read_json_as_dict(paths.CONFIG_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = config.get("num_epochs")
    loss_choice = config.get("loss_function")
    num_workers = config.get("num_workers")
    batch_size = config.get("run_all_batch_size")
    loss_function = (
        torch.nn.CrossEntropyLoss()
        if loss_choice == "crossentropy"
        else torch.nn.MultiMarginLoss()
    )
    print("Setting seeds to:", config["seed"])
    set_seeds(config["seed"])

    predictions_folder = paths.RUN_ALL_PREDICTIONS_DIR
    artifacts_folder = paths.RUN_ALL_ARTIFACTS_DIR
    model_names = config.get("run_all_model_names")

    custom_data_loader = CustomDataLoader(
        batch_size=batch_size, num_workers=num_workers
    )

    for model_name in model_names:
        print(f"\nWorking on model: {model_name}")

        model_predictions_folder = os.path.join(predictions_folder, model_name)
        model_artifacts_folder = os.path.join(artifacts_folder, model_name)
        os.makedirs(model_artifacts_folder, exist_ok=True)
        os.makedirs(model_predictions_folder, exist_ok=True)

        train_loader, test_loader, validation_loader = (
            custom_data_loader.train_loader,
            custom_data_loader.test_loader,
            custom_data_loader.validation_loader,
        )

        num_classes = len(train_loader.dataset.classes)

        trainer = ResNetTrainer(
            train_loader,
            test_loader,
            validation_loader,
            num_classes,
            model_name,
            model_artifacts_folder,
        )
        trainer.set_device(device)
        trainer.set_loss_function(loss_function)

        checkpoint_dir_path = os.path.join(model_artifacts_folder, "checkpoints")
        os.makedirs(checkpoint_dir_path, exist_ok=True)

        print("Training model...")
        metrics_history = trainer.train(
            num_epochs=num_epochs, checkpoint_dir_path=checkpoint_dir_path
        )

        predictor_path = os.path.join(model_artifacts_folder, "predictor")

        print("Saving model...")
        trainer.save_model(predictor_path=predictor_path)

        print("Saving metrics to csv...")
        trainer._save_metrics_to_csv(
            metrics_history,
            output_folder=model_artifacts_folder,
            file_name="train_validation_metrics.csv",
        )

        print("Saving confusion matrix...")
        train_cm = trainer._calculate_confusion_matrix(train_loader)
        validation_cm = trainer._calculate_confusion_matrix(validation_loader)

        trainer._plot_and_save_confusion_matrix(
            cm=train_cm,
            phase="train",
            output_folder=model_artifacts_folder,
            class_names=trainer.train_loader.dataset.classes,
        )

        trainer._plot_and_save_confusion_matrix(
            cm=validation_cm,
            phase="validation",
            output_folder=model_artifacts_folder,
            class_names=trainer.train_loader.dataset.classes,
        )

        test_loss, test_accuracy, test_f1 = trainer._evaluate_loss_accuracy(
            trainer.test_loader, trainer.test_accuracy, trainer.test_f1
        )

        test_metrics = pd.DataFrame(
            {
                "Test Loss": [test_loss],
                "Test Accuracy": [test_accuracy],
                "Test F1": [test_f1],
            }
        )
        print("Saving test metrics to csv...")
        trainer._save_metrics_to_csv(
            test_metrics,
            output_folder=model_predictions_folder,
            file_name="test_metrics.csv",
        )

        print("Saving test confusion matrix...")
        test_cm = trainer._calculate_confusion_matrix(test_loader)
        trainer._plot_and_save_confusion_matrix(
            cm=test_cm,
            phase="test",
            output_folder=model_predictions_folder,
            class_names=trainer.train_loader.dataset.classes,
        )

        print(
            f"Training Accuracy (Last Epoch): {metrics_history['Train Accuracy'][-1]}"
        )

        print(f"Training and evaluation for model {model_name} completed.\n")

    print("All models have been processed.")
