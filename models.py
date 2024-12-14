import torch
import torch.nn as nn

class VAE_MNIST(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_MNIST, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)  # Mean and log-variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var


class VAE_CIFAR(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_CIFAR, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * latent_dim)  # Mean and log-variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 32 * 32),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var
import torch
import torch.nn as nn

class VAE_CIFAR_CNN(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_CIFAR_CNN, self).__init__()
        
        # Encoder: Convolutional Neural Network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 4, 4)
            nn.ReLU()
        )
        
        # Fully Connected Layers for Latent Variables
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)  # Mean
        self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)  # Log-variance
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # Decoder: Transposed Convolutional Neural Network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Output: (3, 32, 32)
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample z ~ N(mu, var) from the learned distribution.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        """
        # Encode
        mu, log_var = self.encoder_function(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        out = self.decoder_function(z)
        
        return out, mu, log_var
    
    def encoder_function(self, x):
        """
        Encoder function to generate latent variables.
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var
    
        
    def decoder_function(self, z):
        """
        Decoder function to generate images from latent space.
        """
        z = self.fc_decode(z)
        z = z.view(z.size(0), 128, 4, 4)
        out = self.decoder(z)
        
        return out
        

