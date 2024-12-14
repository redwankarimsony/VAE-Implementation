# %% Import libraries
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import VAE_CIFAR_CNN

# %% Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% Load CIFAR-10 Test Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = datasets.CIFAR10(
    root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# %% Load Pre-Trained VAE Model
model = VAE_CIFAR_CNN(latent_dim=256).to(device)
model_path = "checkpoints/vae_cifar_cnn_best.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# %% Helper Functions


def extract_latent_vectors(model, images, device):
    with torch.no_grad():
        x = images.to(device)
        mu, var = model.encoder_function(x)
        return mu, var


def generate_new_images(model, mu, var):
    with torch.no_grad():
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * var)
        new_images = model.decoder_function(z)
        return new_images


def visualize_images(original, generated, label, output_path):
    N_SAMPLES = len(original)
    fig, axs = plt.subplots(2, N_SAMPLES, figsize=(15, 6))

    for i in range(N_SAMPLES):
        axs[0, i].imshow(original[i].cpu().permute(1, 2, 0))
        axs[0, i].axis("off")
        axs[1, i].imshow(generated[i].cpu().permute(1, 2, 0))
        axs[1, i].axis("off")

    plt.savefig(output_path)
    plt.show()

# %% Sampling and Reconstruction


def process_and_visualize(model, label1, label2, n_samples, device, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images1 = images[labels == label1][:n_samples].to(device)
    images2 = images[labels == label2][:n_samples].to(device)

    mu1, var1 = extract_latent_vectors(model, images1, device)
    mu2, var2 = extract_latent_vectors(model, images2, device)
    
    # Print the mean and variance of the latent vectors shape
    print(f"mu1 shape: {mu1.shape}, var1 shape: {var1.shape}")
    

    new_images1 = generate_new_images(model, mu1, var1)
    new_images2 = generate_new_images(model, mu2, var2)

    visualize_images(images1, new_images1, label1,
                     os.path.join(output_dir, f"label_{label1}.png"))
    visualize_images(images2, new_images2, label2,
                     os.path.join(output_dir, f"label_{label2}.png"))

# %% Latent Space Visualization


def visualize_latent_space(model, test_loader, device):
    mu_list, label_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader)):
            mu, _ = extract_latent_vectors(model, images, device)
            mu_list.append(mu)
            label_list.append(labels)

    mu = torch.cat(mu_list, dim=0).cpu().numpy()
    labels = torch.cat(label_list, dim=0).cpu().numpy()

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, random_state=0)
    mu_tsne = tsne.fit_transform(mu)

    df = pd.DataFrame({"x": mu_tsne[:, 0], "y": mu_tsne[:, 1], "z" :mu_tsne[:, 2],  "label": labels})
    # Convert the matplotlib to Tkinter for the GUI
    plt.switch_backend('TkAgg')
    
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'], c=df['label'], cmap='tab10')
    
    # CIFAR class names for the legend
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Legend the plot
    for i in range(10):
        ax.text(df['x'][i], df['y'][i], df['z'][i], CLASS_NAMES[i])
    
    plt.tight_layout()
    plt.savefig("results/latent_space_3d_cifar.png")
    
    # 2D plot
# %% Zero-Shot Classification


def zero_shot_classification(model, test_loader, device):
    """
    Perform zero-shot classification using the VAE model.
    Args:
        model: Trained VAE model.
        test_loader: DataLoader for test data.
        device: Device to use (CPU or GPU).
    """
    mu_list, label_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader)):
            mu, _ = extract_latent_vectors(model, images, device)
            mu_list.append(mu)
            label_list.append(labels)

    mu = torch.cat(mu_list, dim=0).cpu().numpy()
    label_list = torch.cat(label_list, dim=0).cpu().numpy()

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(mu, label_list)

    y_pred = clf.predict(mu)
    accuracy = (y_pred == label_list).mean()
    print(f"Zero-shot classification accuracy: {accuracy:.2f}")

# %% Sampling Around a Class


def sample_around_class(model, label, n_samples, device, output_dir="results"):
    """
    Generate new samples by sampling the z-space around a given class.
    Args:
        model: Trained VAE model.
        label: Class label to sample around.
        n_samples: Number of samples to generate.
        device: Device to use (CPU or GPU).
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images = images[labels == label][:1].to(device)

    mu, var = extract_latent_vectors(model, images, device)
    mu, var = mu[0], var[0]

    new_images = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * var)
            new_image = model.decoder(z.view(1, -1, 1, 1))
            new_images.append(new_image)

    new_images = torch.cat(new_images, dim=0)
    fig, axs = plt.subplots(1, n_samples, figsize=(15, 3))
    for i in range(n_samples):
        axs[i].imshow(new_images[i].cpu().permute(1, 2, 0))
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_around_class_{label}.png"))
    plt.show()


# %% Main Execution
if __name__ == "__main__":
    LABEL1, LABEL2 = 1, 5  # Example: Airplane and Deer
    N_SAMPLES = 12
    OUTPUT_DIR = "results"

    process_and_visualize(model, LABEL1, LABEL2, N_SAMPLES, device, OUTPUT_DIR)
    
    visualize_latent_space(model, test_loader, device)
    
    
    # zero_shot_classification(model, test_loader, device)

    # for label in range(10):
    #     sample_around_class(model, label, n_samples=10,
    #                         device=device, output_dir=OUTPUT_DIR)
