import os
import random
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars
from src.config import paths

# Configure logging
logging.basicConfig(
    filename=paths.TRAINING_LOGS_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)


class CustomDataLoader:
    OUTPUT_FOLDER = "output/image_output"
    IMAGE_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD_DEV = [0.229, 0.224, 0.225]

    def __init__(
        self, base_folder=paths.VISION_DATA_DIR, batch_size=256, num_workers=6
    ):
        self.base_folder = base_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD_DEV),
            ]
        )
        (
            self.train_loader,
            self.test_loader,
            self.validation_loader,
        ) = self.create_data_loaders()

    

    def create_data_loaders(self):
        train_folder = os.path.join(self.base_folder, "train")
        test_folder = os.path.join(self.base_folder, "test")
        validation_folder = os.path.join(self.base_folder, "validation")

        self.num_classes = len(os.listdir(train_folder))

        train_dataset = ImageFolder(root=train_folder, transform=self.transform)
        test_dataset = ImageFolder(root=test_folder, transform=self.transform)
        validation_dataset = ImageFolder(
            root=validation_folder, transform=self.transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, test_loader, validation_loader

    @staticmethod
    def select_random_images(dataset, num_images=6):
        return random.sample(dataset.samples, num_images)

    @staticmethod
    def save_images(dataset_name, images):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
        for i, (image_path, _) in enumerate(images):
            img = Image.open(image_path)
            axes[i // 3, i % 3].imshow(img)
            axes[i // 3, i % 3].set_title(
                os.path.basename(image_path),
                fontweight="bold",
                fontname="Times New Roman",
            )
            axes[i // 3, i % 3].axis("off")
        plt.tight_layout()

        # Create the output directory if it doesn't exist
        os.makedirs(CustomDataLoader.OUTPUT_FOLDER, exist_ok=True)

        output_path = os.path.join(
            CustomDataLoader.OUTPUT_FOLDER,
            f'{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
        )
        plt.savefig(output_path)
        plt.close()

    def process_datasets(self):
        for dataset_name, loader in zip(
            ["train", "test", "validation"],
            [self.train_loader, self.test_loader, self.validation_loader],
        ):
            # Wrap the loader with tqdm for progress tracking
            for images, labels in tqdm(loader, desc=f"Processing {dataset_name} Data"):
                # Placeholder for processing code
                pass

            selected_images = self.select_random_images(loader.dataset)
            # Call save_images method to actually save the images
            self.save_images(dataset_name, selected_images)

            # Log progress
            for image_path, _ in selected_images:
                logging.info(
                    f"Saved {os.path.basename(image_path)} for {dataset_name} dataset"
                )


# Script runs only when executed directly
if __name__ == "__main__":
    # Example usage
    data_loader = CustomDataLoader()
    data_loader.process_datasets()
