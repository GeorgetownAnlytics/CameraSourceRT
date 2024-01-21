import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class CustomDataLoader:
    """
    A data loader for loading images from a specified directory structure.
    This class prepares data loaders for training, testing, and validation datasets,
    using predefined ImageNet mean and standard deviation for normalization.

    Attributes:
        OUTPUT_FOLDER (str): Path to the folder where output will be saved.
        IMAGE_SIZE (tuple): The target size of images (height, width) after resizing.
        MEAN (list): The mean values for each channel (RGB) used for normalization.
        STD_DEV (list): The standard deviation values for each channel (RGB) used for normalization.
    """

    OUTPUT_FOLDER = "output/image_output"
    IMAGE_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    STD_DEV = [0.229, 0.224, 0.225]  # ImageNet std dev

    def __init__(self, base_folder="./datasets/Vision_data", batch_size=256, num_workers=6):
        """
        Initializes the CustomDataLoader with the specified directory,
        batch size, and number of workers.

        Args:
            base_folder (str): Path to the base image directory.
            batch_size (int): The size of the batches of data.
            num_workers (int): The number of subprocesses for data loading.
        """
        self.base_folder = base_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD_DEV)
        ])

        self.train_loader, self.test_loader, self.validation_loader = self.create_data_loaders()

    def create_data_loaders(self):
        """
        Creates data loaders for the training, testing, and validation datasets.

        Returns:
            tuple: A tuple containing the train, test, and validation DataLoader objects.
        """
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


# Example usage
if __name__ == "__main__":
    data_loader = CustomDataLoader()
    print("Data loaders have been created.")
