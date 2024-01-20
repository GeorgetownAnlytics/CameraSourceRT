import os
import torch
from models.dataloader import CustomDataLoader
from models.resnet_trainer import ResNet18Trainer, ResNet34Trainer 
from models.resnet_trainer import ResNet50Trainer, ResNet101Trainer
from models.resnet_trainer import ResNet152Trainer


def initialize_trainer(model_name, train_loader, test_loader, validation_loader, device, loss_function, num_epochs=10):
    # Get num_classes from the data loader
    num_classes = len(train_loader.dataset.classes)

    # Initialize the correct trainer based on the ResNet model variant
    if model_name == 'resnet18':
        trainer = ResNet18Trainer(train_loader, test_loader, validation_loader, num_classes)
    elif model_name == 'resnet34':
        trainer = ResNet34Trainer(train_loader, test_loader, validation_loader, num_classes)
    elif model_name == 'resnet50':
        trainer = ResNet50Trainer(train_loader, test_loader, validation_loader, num_classes)
    elif model_name == 'resnet101':
        trainer = ResNet101Trainer(train_loader, test_loader, validation_loader, num_classes)
    elif model_name == 'resnet152':
        trainer = ResNet152Trainer(train_loader, test_loader, validation_loader, num_classes)
    else:
        raise ValueError(f"Unsupported ResNet model name: {model_name}")

    # Set the loss function for the trainer
    trainer.set_loss_function(loss_function)

    return trainer


if __name__ == '__main__':
    num_epochs = int(input("Enter the number of total epochs (default is 10): ") or 10)
    device = input("Enter 'cuda' for GPU or 'cpu' for CPU (default is 'cuda'): ") or 'cuda'
    loss_choice = input("Choose the loss function: 'crossentropy' or 'multiclass_hinge' (default is 'crossentropy'): ") or 'crossentropy'
    loss_function = torch.nn.CrossEntropyLoss() if loss_choice == 'crossentropy' else torch.nn.MultiMarginLoss()

    output_folder = 'output/model_outputs'
    model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    # Initialize the data loader
    custom_data_loader = CustomDataLoader()

    for model_name in model_names:
        print(f"\nWorking on model: {model_name}")

        model_output_folder = os.path.join(output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)

        train_loader, test_loader, validation_loader = custom_data_loader.train_loader, custom_data_loader.test_loader, custom_data_loader.validation_loader
        
        print("Initializing and configuring the trainer...")
        trainer = initialize_trainer(model_name, train_loader, test_loader, validation_loader, device, loss_function, num_epochs)
        trainer.set_device(device)
        trainer.set_loss_function(loss_function)

        print("Starting training and evaluation...")
        metrics_history = trainer.train_and_evaluate(num_epochs, model_output_folder)

        print("Calculating and saving confusion matrices...")
        class_names = train_loader.dataset.classes  # Adjust this to your dataset

        train_cm = trainer._calculate_confusion_matrix(train_loader)
        trainer._plot_and_save_confusion_matrix(train_cm, 'train', model_output_folder, class_names)

        test_cm = trainer._calculate_confusion_matrix(test_loader)
        trainer._plot_and_save_confusion_matrix(test_cm, 'test', model_output_folder, class_names)

        validation_cm = trainer._calculate_confusion_matrix(validation_loader)
        trainer._plot_and_save_confusion_matrix(validation_cm, 'validation', model_output_folder, class_names)

        # Saving confusion matrices as CSV (if the method is implemented)
        print("Saving confusion matrices as CSV...")
        trainer.save_confusion_matrix_csv(train_cm, 'train', model_output_folder)
        trainer.save_confusion_matrix_csv(test_cm, 'test', model_output_folder)
        trainer.save_confusion_matrix_csv(validation_cm, 'validation', model_output_folder)

        print(f"Training and evaluation for model {model_name} completed.\n")

    print("All models have been processed.")

