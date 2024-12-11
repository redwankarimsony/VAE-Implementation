import os
import torch
import matplotlib.pyplot as plt

def save_model(model, path):
    """
    Saves the model's state dictionary to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_loss_curves(train_losses, val_losses, output_dir):
    """
    Saves the training and validation loss curves as a PNG file.

    Args:
        train_losses (list): List of training loss values.
        val_losses (list): List of validation loss values.
        output_dir (str): Directory to save the loss curves plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Curves")
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.close()
    print(f"Loss curves saved to {os.path.join(output_dir, 'loss_curves.png')}")
