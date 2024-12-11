import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
import argparse
import matplotlib.pyplot as plt
from models import VAE_CIFAR
from datasets import get_cifar10_datasets
from utils import save_loss_curves, save_model

def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    latent_dim = args.latent_dim
    epochs = args.epochs

    # Load datasets
    train_loader, val_loader = get_cifar10_datasets(batch_size)

    # Instantiate the model
    model = VAE_CIFAR(latent_dim).to(device)

    # Loss function
    # Reconstruction loss + KL divergence
    criterion = nn.BCELoss(reduction="sum")
    
    def loss_function(recon_x, x, mu, log_var):
        # Ensure input to BCELoss is in range [0, 1]
        recon_loss = criterion(recon_x, x.view(-1, 3 * 32 * 32))
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_divergence

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop with validation and saving best model
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # Forward pass
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, log_var = model(data)
                loss = loss_function(recon_batch, data, mu, log_var)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(checkpoint_dir, "vae_cifar_best.pth"))

        scheduler.step()

    # Save the final model
    save_model(model, os.path.join(checkpoint_dir, "vae_cifar.pth"))

    # Save training and validation loss curves
    save_loss_curves(train_losses, val_losses, checkpoint_dir)

    # Export model to ONNX format
    onnx_model_path = os.path.join(checkpoint_dir, "vae_cifar.onnx")
    # Set the model to inference mode
    model.eval()
    # Dummy input for tracing the model
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    # Export the model
    torch.onnx.export(model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"], opset_version=11)
    print(f"Model saved to {onnx_model_path} in ONNX format")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder on CIFAR-10 dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples in a batch")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--latent_dim", type=int, default=32, help="Dimension of the latent space")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args)
