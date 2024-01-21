import os
import json
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


class CustomDataLoader:
    OUTPUT_FOLDER = "output/image_output"
    IMAGE_SIZE = (224, 224)
    MEAN_STD_FILE = "mean_std.json"

    def __init__(self, base_folder="./datasets/Vision_data", batch_size=256, num_workers=6):
        self.base_folder = base_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize mean and std
        self.MEAN = [0.485, 0.456, 0.406]  # Default values
        self.STD_DEV = [0.229, 0.224, 0.225]  # Default values

        # Define transformations using the initialized or loaded mean and std
        self.transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD_DEV)
        ])
        
        # Create data loaders
        self.train_loader, self.test_loader, self.validation_loader = self.create_data_loaders()

        # Initialize or load mean and std
        self._initialize_or_load_mean_std()

    def _initialize_or_load_mean_std(self):
        if os.path.exists(self.MEAN_STD_FILE):
            with open(self.MEAN_STD_FILE, 'r') as file:
                mean_std = json.load(file)
            self.MEAN = mean_std['mean']
            self.STD_DEV = mean_std['std']
        else:
            self.calculate_and_save_mean_std()

    def create_data_loaders(self):
        train_folder = os.path.join(self.base_folder, "train")
        test_folder = os.path.join(self.base_folder, "test")
        validation_folder = os.path.join(self.base_folder, "validation")

        train_dataset = datasets.ImageFolder(
            root=train_folder, transform=self.transform)
        test_dataset = datasets.ImageFolder(
            root=test_folder, transform=self.transform)
        validation_dataset = datasets.ImageFolder(
            root=validation_folder, transform=self.transform)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, test_loader, validation_loader

    def calculate_and_save_mean_std(self):
        mean = torch.zeros(3)
        std = torch.zeros(3)
        num_samples = 0

        for images, _ in tqdm(self.train_loader, desc="Calculating Mean and Std"):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            num_samples += batch_samples

        mean /= num_samples
        std /= num_samples

        self.MEAN = mean.tolist()
        self.STD_DEV = std.tolist()

        mean_std_dict = {"mean": self.MEAN, "std": self.STD_DEV}
        with open(self.MEAN_STD_FILE, "w") as file:
            json.dump(mean_std_dict, file)


# Example usage
if __name__ == "__main__":
    data_loader = CustomDataLoader()
    print("Data loaders have been created.")
