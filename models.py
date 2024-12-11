import torch
import torch.nn as nn

class VAE_MNIST(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_MNIST, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)  # Outputs mean and log-variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()  # Output is in the range [0, 1]
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample z from N(mu, var) using z = mu + std * epsilon
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE
        """
        x = x.view(-1, 28 * 28)  # Flatten the input
        h = self.encoder(x)      # Encode to latent space
        mu, log_var = torch.chunk(h, 2, dim=1)  # Split into mean and log-variance
        z = self.reparameterize(mu, log_var)    # Sample latent vector
        out = self.decoder(z)    # Decode back to input space
        return out, mu, log_var



