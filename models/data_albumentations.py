import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

class ImagePadding:
    def __init__(self, input_folder, output_folder, padding_color=(128, 128, 128)):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.padding_color = padding_color

    def add_padding_to_images(self):
        for folder_name in ['train', 'test', 'validation']:
            input_folder = os.path.join(self.input_folder, folder_name)
            output_folder = os.path.join(self.output_folder, folder_name)
            
            image_files = []
            for root, _, files in os.walk(input_folder):
                image_files.extend([os.path.join(root, file) for file in files if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])

            for input_path in tqdm(image_files, desc=f"Adding Padding to {folder_name}", unit="image"):
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Open the image
                img = Image.open(input_path)

                # Calculate padding size to make the image square
                width, height = img.size
                max_dim = max(width, height)
                padding = ImageOps.expand(img, ((max_dim - width) // 2, (max_dim - height) // 2), fill=self.padding_color)

                # Save the modified image
                padding.save(output_path)

    @staticmethod
    def create_random_image_plot(images_folder, num_images=8):
        image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

        # Randomly select num_images from the list
        selected_images = random.sample(image_files, num_images)

        # Create a 2-row, 4-column plot
        fig, axes = plt.subplots(2, 4, figsize=(16, 10), dpi=300)
        fig.tight_layout()

        for i, image_file in enumerate(selected_images):
            img = Image.open(os.path.join(images_folder, image_file))
            class_name = image_file.split('.')[0]  # Get class name from the file name
            axes[i // 4, i % 4].imshow(img)
            axes[i // 4, i % 4].set_title(class_name)
            axes[i // 4, i % 4].set_xlabel(image_file)

        # Save the plot as a JPEG with 300 DPI
        plt.savefig(os.path.join("D:/Repositories/CameraSourceRT/documentation", "random_image_plot.jpg"), dpi=300, format='jpg')

if __name__ == "__main__":
    import argparse

    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description="Add padding to images to make them square")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing 'train', 'test', and 'validation' subfolders")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for modified images")
    parser.add_argument("--padding_color", nargs=3, type=int, default=[128, 128, 128],
                        help="RGB values for padding color (default: 128 128 128)")

    args = parser.parse_args()

    # Create an instance of the ImagePadding class
    image_padder = ImagePadding(args.input_folder, args.output_folder, tuple(args.padding_color))

    # Add padding to images with progress bar
    image_padder.add_padding_to_images()

    # Create a random image plot and save it as JPEG with 300 DPI
    image_padder.create_random_image_plot(args.output_folder)

    print("Padding added to images and random image plot saved successfully.")
