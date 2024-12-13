# %% Import libraries
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import VAE_MNIST

# %% Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% Load MNIST Test Dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# %% Load Pre-Trained VAE Model
model = VAE_MNIST(latent_dim=32).to(device)
model_path = "checkpoints/vae_mnist_best.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# %% Helper Functions


def extract_latent_vectors(model, images, device):
    """
    Extract latent vectors (mu, var) from the encoder.
    Args:
        model: Trained VAE model.
        images: Input images.
        device: Device to use (CPU or GPU).

    Returns:
        mu: Mean of the latent distribution.
        var: Variance of the latent distribution.
    """
    with torch.no_grad():
        x = images.to(device).view(-1, 28 * 28)
        h = model.encoder(x)
        mu, var = torch.chunk(h, 2, dim=1)
        return mu, var


def generate_new_images(model, mu, var):
    """
    Generate new images from the decoder using latent vectors.
    Args:
        model: Trained VAE model.
        mu: Mean of the latent distribution.
        var: Variance of the latent distribution.

    Returns:
        new_images: Generated images.
    """
    with torch.no_grad():
        # Sample latent vectors
        z = mu + torch.randn_like(mu)*2 * torch.exp(0.5 * var)
        new_images = model.decoder(z)
        return new_images.view(-1, 1, 28, 28)


def visualize_images(original, generated, label, output_path):
    """
    Visualize and save original and generated images.
    Args:
        original: Original input images.
        generated: Generated images from the decoder.
        label: Class label for visualization.
        output_path: Path to save the visualization.
    """
    N_SAMPLES = len(original)
    fig, axs = plt.subplots(2, N_SAMPLES, figsize=(10, 4))

    for i in range(N_SAMPLES):
        axs[0, i].imshow(original[i].cpu().squeeze(), cmap="gray")
        axs[0, i].axis("off")
        axs[1, i].imshow(generated[i].cpu().squeeze(), cmap="gray")
        axs[1, i].axis("off")

    plt.savefig(output_path)
    plt.show()

# %% Sampling and Reconstruction


def process_and_visualize(model, label1, label2, n_samples, device, output_dir="results"):
    """
    Process and visualize images for two labels.
    Args:
        model: Trained VAE model.
        label1: First label for comparison.
        label2: Second label for comparison.
        n_samples: Number of samples per label.
        device: Device to use (CPU or GPU).
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample images with selected labels
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images1 = images[labels == label1][:n_samples].to(device)
    images2 = images[labels == label2][:n_samples].to(device)

    # Extract latent vectors
    mu1, var1 = extract_latent_vectors(model, images1, device)
    mu2, var2 = extract_latent_vectors(model, images2, device)

    # Generate new images
    new_images1 = generate_new_images(model, mu1, var1)
    new_images2 = generate_new_images(model, mu2, var2)

    # Visualize and save results
    visualize_images(images1, new_images1, label1,
                     os.path.join(output_dir, f"label_{label1}.png"))
    visualize_images(images2, new_images2, label2,
                     os.path.join(output_dir, f"label_{label2}.png"))


# %% Generate Transition Images between Two Labels (Interpolation)
def interpolate_images(model, label1, label2, n_samples, device):
    """
    Generate transition images between two labels.
    Args:
        model: Trained VAE model.
        label1: First label for interpolation.
        label2: Second label for interpolation.
        n_samples: Number of samples per label.
        device: Device to use (CPU or GPU).

    Returns:
        transition_images: Transition images between two labels.
    """
    # Sample images with selected labels
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images1 = images[labels == label1][:n_samples].to(device)
    images2 = images[labels == label2][:n_samples].to(device)

    # Extract latent vectors
    mu1, var1 = extract_latent_vectors(model, images1, device)
    mu2, var2 = extract_latent_vectors(model, images2, device)

    mu1, mu2 = mu1[0], mu2[0]

    # Generate transition images
    transition_images = []
    with torch.no_grad():
        for alpha in torch.linspace(0, 1., n_samples):
            z = alpha * mu2 + (1. - alpha) * mu1
            new_images = model.decoder(z)
            transition_images.append(new_images.view(-1, 1, 28, 28))

    # Display transition images
    transition_images = torch.cat(transition_images, dim=0)

    # Show the transition images
    fig, axs = plt.subplots(1, n_samples, figsize=(10, 2))
    for i in range(n_samples):
        axs[i].imshow(transition_images[i].cpu().squeeze(), cmap="gray")
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(f"results/transition_mnist/from_{label1}_to_{label2}.png")
    plt.show()

    return transition_images


# Extract all the latent vectors and use t-SNE to visualize them in 3dimensions
def visualize_latent_space(model, test_loader, device):
    """
    Visualize the latent space of the VAE using t-SNE.
    Args:
        model: Trained VAE model.
        test_loader: DataLoader for test data.
        device: Device to use (CPU or GPU).
    """
    # Extract latent vectors for all images
    mu_list, label_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader)):
            mu, _ = extract_latent_vectors(model, images, device)
            mu_list.append(mu)
            label_list.append(labels)

    mu = torch.cat(mu_list, dim=0).cpu().numpy()
    labels = torch.cat(label_list, dim=0).cpu().numpy()

    # Perform t-SNE on the latent vectors
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, random_state=0)
    mu_tsne = tsne.fit_transform(mu)

    # Create a DataFrame for visualization
    df = pd.DataFrame({
        "x": mu_tsne[:, 0],
        "y": mu_tsne[:, 1],
        "z": mu_tsne[:, 2],
        "label": labels
    })

    # Plot the latent space in 3D with transparency
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for label in range(10):
        df_label = df[df["label"] == label]
        ax.scatter(df_label["x"], df_label["y"], df_label["z"],
                   label=label, alpha=0.6)  # Set transparency with alpha
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")

    plt.legend(title="Digit Label")
    plt.savefig("results/latent_space_mnist.png")
    plt.show()


def zero_shot_classification(model, test_loader, device):
    """
    Perform zero-shot classification using the VAE model.
    Args:
        model: Trained VAE model.
        test_loader: DataLoader for test data.
        device: Device to use (CPU or GPU).
    """

    # Extract latent vectors for all images
    mu_list, label_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader)):
            mu, _ = extract_latent_vectors(model, images, device)
            mu_list.append(mu)
            label_list.append(labels)

    mu = torch.cat(mu_list, dim=0).cpu().numpy()
    label_list = torch.cat(label_list, dim=0).cpu().numpy()

    # Train a simple classifier using the latent vectors
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(mu, label_list)

    # Predict the labels using the latent vectors
    y_pred = clf.predict(mu)

    # Calculate the accuracy
    accuracy = (y_pred == label_list).mean()
    print(f"Zero-shot classification accuracy: {accuracy:.2f}")



def sample_around_digit(model, label, n_samples, device, output_dir="results"):
    """
    Generate new samples by sampling the z-space around a given digit.
    Args:
        model: Trained VAE model.
        label: Digit label to sample around.
        n_samples: Number of samples to generate.
        device: Device to use (CPU or GPU).
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample images with the selected label
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images = images[labels == label][:1].to(device)  # Take one image of the given label

    # Extract latent vectors
    mu, var = extract_latent_vectors(model, images, device)
    mu, var = mu[0], var[0]

    # Generate new images by sampling around the latent vector
    new_images = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * var)
            new_image = model.decoder(z)
            new_images.append(new_image.view(-1, 1, 28, 28))
    # Use matplotlib in Agg mode to save the images
    plt.switch_backend("Agg")
    # Visualize and save results
    new_images = torch.cat(new_images, dim=0)
    fig, axs = plt.subplots(1, n_samples, figsize=(10, 1))
    for i in range(n_samples):
        axs[i].imshow(new_images[i].cpu().squeeze(), cmap="gray")
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_around_digit_{label}.png"))
    plt.close()
    # plt.show()

    # return new_images
# %% Main Execution
if __name__ == "__main__":
    LABEL1, LABEL2 = 0, 4
    N_SAMPLES = 12
    OUTPUT_DIR = "results"

    process_and_visualize(model, LABEL1, LABEL2, N_SAMPLES, device, OUTPUT_DIR)
    interpolate_images(model, LABEL1, LABEL2, N_SAMPLES, device)
    visualize_latent_space(model, test_loader, device)
    zero_shot_classification(model, test_loader, device)
    
    for digit in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        sample_around_digit(model, label=digit, n_samples=10, device=device, output_dir=OUTPUT_DIR)
