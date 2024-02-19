import os
import torch
import joblib
import torch.nn as nn
from torchvision import models
from .base_trainer import BaseTrainer
from config import paths


supported_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "inceptionV1": models.googlenet,
    "inceptionV3": models.inception_v3,
    "mnasnet0_5": models.mnasnet0_5,
    "mnasnet1_0": models.mnasnet1_0,
    "mnasnet1_3": models.mnasnet1_3,
}

supported_weights = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "resnet101": models.ResNet101_Weights.IMAGENET1K_V2,
    "resnet152": models.ResNet152_Weights.IMAGENET1K_V2,
    "inceptionV1": models.GoogLeNet_Weights.IMAGENET1K_V1,
    "inceptionV3": models.Inception_V3_Weights.IMAGENET1K_V1,
    "mnasnet0_5": models.MNASNet0_5_Weights.IMAGENET1K_V1,
    "mnasnet1_0": models.MNASNet1_0_Weights.IMAGENET1K_V1,
    "mnasnet1_3": models.MNASNet1_3_Weights.IMAGENET1K_V1,
}


class CustomTrainer(BaseTrainer):
    """
    A trainer class for custom models.

    This class inherits from BaseTrainer and is specialized for training various
    ResNet and Inception models with custom configurations.

    Attributes:
        model (nn.Module): The custom model to be trained.
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
        model_name,
        output_folder=paths.OUTPUTS_DIR,
    ):
        """
        Initializes the CustomTrainer with the specified model, data loaders, and number of classes.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            validation_loader (DataLoader): DataLoader for the validation dataset.
            num_classes (int): Number of classes in the dataset.
            model_name (str, optional): Name of the Custom model to be used.
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

        if self.model_name.startswith("mnasnet"):
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(model.classifier[1].in_features, num_classes),
            )
        else:
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
        os.makedirs(predictor_path, exist_ok=True)
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

        trainer = CustomTrainer(**params)
        trainer.model = model
        return trainer
