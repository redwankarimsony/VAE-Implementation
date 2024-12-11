from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_datasets(batch_size, train_split=0.8):
    """
    Loads the MNIST dataset and splits it into training and validation sets.

    Args:
        batch_size (int): Batch size for the data loaders.
        train_split (float): Fraction of the dataset to use for training (default: 0.8).

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
    """
    # Define the data transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the full MNIST dataset
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

    # Calculate the sizes of the training and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
