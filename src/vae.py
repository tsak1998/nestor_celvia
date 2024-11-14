import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import os
import math

class VAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),  # [88, 128, 32, 32]
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),   # [88, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),    # [88, 32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1),    # [88, 16, 4, 4]
            nn.ReLU()
        )

        # Compute the size of the flattened layer output
        self.fc_input_dim = 16 * 4 * 4  # From the shape [88, 16, 4, 4]

        # Latent space
        self.fc_mu = nn.Linear(self.fc_input_dim, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(self.fc_input_dim, latent_dim)  # Log-variance

        # Decoder layers
        self.decoder_fc = nn.Linear(latent_dim, self.fc_input_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),  # [88, 32, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # [88, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1), # [88, 128, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1) # [88, 256, 64, 64]
        )

        self.aggregation_layer = nn.Linear(latent_dim, latent_dim)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 16, 4, 4)  # Reshape for ConvTranspose2d layers
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        aggregated_z = self.aggregation_layer(z.mean(dim=0))
        return self.decode(z), mu, logvar, aggregated_z

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (e.g., MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def train_model(id, concat_t):
    device = 'cuda:0'
    save_path = os.path.abspath('F:\latent_vectors')
    vae = VAE(latent_dim=1024).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    concat_t = concat_t.to(device)
    min_loss = float('inf')
    min_loss_idx = 0
    num_epochs = 100
    for i in range(num_epochs):
        # Forward pass
        reconstructed, mu, logvar, _ = vae(concat_t)  # concat_t shape: [88, 256, 64, 64]

        # Loss computation
        loss = vae_loss(reconstructed, concat_t, mu, logvar)
        loss_np = loss.detach().cpu().numpy().item()
        
        # Keep best model
        if loss_np < min_loss:
            min_loss = loss_np
            min_loss_idx = i
            torch.save(vae.state_dict(), f'best_vae/{id}.pth')  # Save the best model

        # Backpropagation
        loss.backward()
        optimizer.step()
    # Load best model and extract latent vector and save
    vae = VAE(latent_dim=1024).to(device)
    vae.load_state_dict(torch.load(f'best_vae/{id}.pth'))
    reconstructed, mu, logvar, agrregated = vae(concat_t)
    np.save(os.path.join(save_path, f"{id}.npy"), agrregated.detach().cpu().numpy())
    tqdm.write(f"{id}, loss: {min_loss:.2f}, epoch: {min_loss_idx}")


def round_to_nearest_5(number):
    return math.ceil(number / 5) * 5

def read_and_concat(full_path, file_num, id):
    t_list = []
    for i in range(0, round_to_nearest_5(int(file_num)), 5):
        t_list.append(torch.load(os.path.join(full_path, f"{id}_{i}_{i+5}.pt"), map_location="cpu"))
    return torch.concat(t_list, dim=0)


path_feature_vector = os.path.abspath('E:/extracted_embeddings_pretrained/')
for dir_item in tqdm(os.listdir(path_feature_vector)):
    full_path = os.path.join(path_feature_vector, dir_item)
    id = dir_item.split('_')[0]
    file_num = dir_item.split('_')[1]
    
    concat_t = read_and_concat(full_path, file_num, id)
    train_model(id, concat_t)