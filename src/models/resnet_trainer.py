import torch
import torch.nn as nn
from torchvision import models
import os
from .dataloader import CustomDataLoader
from .base_trainer import BaseTrainer
import joblib
from config import paths


# Dictionary of supported model names with their corresponding model functions
supported_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

supported_weights = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "resnet101": models.ResNet101_Weights.IMAGENET1K_V2,
    "resnet152": models.ResNet152_Weights.IMAGENET1K_V2,
}


class ResNetTrainer(BaseTrainer):
    """
    A trainer class for ResNet models.

    This class inherits from BaseTrainer and is specialized for training various
    ResNet models with custom configurations.

    Attributes:
        model (nn.Module): The ResNet model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
    """

    def __init__(
        self,
        train_loader,
        test_loader,
        validation_loader,
        num_classes,
        model_name="resnet18",
        output_folder=paths.OUTPUTS_DIR,
    ):
        """
        Initializes the ResNetTrainer with the specified model, data loaders, and number of classes.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            validation_loader (DataLoader): DataLoader for the validation dataset.
            num_classes (int): Number of classes in the dataset.
            model_name (str, optional): Name of the ResNet model to be used. Defaults to 'resnet18'.
        """

        if model_name not in supported_models:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.num_classes = num_classes
        self.model_name = model_name
        self.output_folder = output_folder

        model_fn = supported_models[model_name]
        model_weights = supported_weights[model_name]
        model = model_fn(weights=model_weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        super().__init__(
            model,
            train_loader,
            test_loader,
            validation_loader,
            output_folder=output_folder,
        )

    def save_model(self, predictor_path=paths.PREDICTOR_DIR):
        model_params = {
            "train_loader": self.train_loader,
            "test_loader": self.test_loader,
            "validation_loader": self.validation_loader,
            "num_classes": self.num_classes,
            "model_name": self.model_name,
            "output_folder": self.output_folder,
        }
        params_path = os.path.join(predictor_path, "model_params.joblib")
        model_path = os.path.join(predictor_path, "model_state.pth")
        joblib.dump(model_params, params_path)
        torch.save(self.model.state_dict(), model_path)

    @staticmethod
    def load_model(predictor_path=paths.PREDICTOR_DIR):
        params_path = os.path.join(predictor_path, "model_params.joblib")
        model_path = os.path.join(predictor_path, "model_state.pth")
        params = joblib.load(params_path)
        model_state = torch.load(model_path)

        model_name = params["model_name"]
        num_classes = params["num_classes"]
        model_fn = supported_models[model_name]
        model_weights = supported_weights[model_name]
        model = model_fn(weights=model_weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        model.load_state_dict(model_state)

        trainer = ResNetTrainer(**params)
        trainer.model = model
        return trainer


if __name__ == "__main__":
    # Example usage of ResNetTrainer.
    custom_data_loader = CustomDataLoader()
    train_loader = custom_data_loader.train_loader
    test_loader = custom_data_loader.test_loader
    validation_loader = custom_data_loader.validation_loader

    num_classes = len(train_loader.dataset.classes)
    resnet_trainer = ResNetTrainer(
        train_loader, test_loader, validation_loader, num_classes, model_name="resnet18"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet_trainer.set_device(device)
    resnet_trainer.set_loss_function(torch.nn.CrossEntropyLoss())

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    resnet_trainer.train(num_epochs=10)
