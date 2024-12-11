from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_datasets(batch_size, train_split=0.8):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def get_cifar10_datasets(batch_size, train_split=0.8):
    transform = transforms.Compose([
        transforms.ToTensor()  # Remove normalization to keep data in [0, 1]
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
