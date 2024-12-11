import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models import VAE_MNIST
import argparse
import os

def load_model(model_path, latent_dim, device):
    """
    Load the pre-trained VAE model.

    Args:
        model_path (str): Path to the saved model.
        latent_dim (int): Latent dimension size.
        device (torch.device): Device to load the model onto.

    Returns:
        model (VAE_MNIST): Loaded VAE model.
    """
    model = VAE_MNIST(latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def reconstruct_images(model, data_loader, device, output_dir):
    """
    Reconstruct images using the VAE model and save the original and reconstructed images.

    Args:
        model (VAE_MNIST): Pre-trained VAE model.
        data_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to perform inference.
        output_dir (str): Directory to save the reconstructed images.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.to(device)

    # Reconstruct images
    with torch.no_grad():
        reconstructed, _, _ = model(images)

    # Save original and reconstructed images
    fig, axs = plt.subplots(2, len(images), figsize=(15, 4))
    for i in range(len(images)):
        # Original image
        axs[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
        axs[0, i].axis("off")
        # Reconstructed image
        axs[1, i].imshow(reconstructed[i].cpu().view(28, 28), cmap="gray")
        axs[1, i].axis("off")
    plt.savefig(os.path.join(output_dir, "reconstructed_images_mnist.png"))
    plt.close()



def main(args):
    # Device configuration
    device = args.device

    # Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the pre-trained model
    model = load_model(args.model_path, args.latent_dim, device)

    # Reconstruct and save images
    reconstruct_images(model, test_loader, device, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with a trained Variational Autoencoder on MNIST dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained VAE model")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save reconstructed images")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args)
