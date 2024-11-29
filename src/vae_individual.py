import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

path_feature = os.path.abspath('F:/extracted_embeddings_pretrained_concat')
save_path = os.path.abspath('F:/latent_vectors_individual/')
model_save_path = os.path.abspath('C:/Users/user/Documents/nestor_celvia/best_vae_all/')
device = 'cuda:0'


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
        aggregated_z = self.aggregation_layer(z)
        return self.decode(z), mu, logvar, aggregated_z
    
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (e.g., MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_names = os.listdir(data_dir)
    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_dir+f"/{self.data_names[idx]}"))
        return {'data':data, 'id':self.data_names[idx].split('.')[0]}

# Train
train_dataset = CustomDataset(path_feature)
trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
vae = VAE(latent_dim=1024).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
min_loss = float('inf')
num_epochs = 100
for epoch in tqdm(range(num_epochs)):
    for data in trainloader:
        # Forward pass
        data = data['data'].squeeze(0).to(device)
        reconstructed, mu, logvar, _ = vae(data)  # concat_t shape: [88, 256, 64, 64]
        # Loss computation
        loss = vae_loss(reconstructed, data, mu, logvar)
        loss_np = loss.detach().cpu().numpy().item()
        # Keep best model
        if loss_np < min_loss:
            min_loss = loss_np
            torch.save(vae.state_dict(),os.path.join(model_save_path+f'/best_vae.pth'))
        # Backpropagation
        loss.backward()
        optimizer.step()
print('Training finished.')

# Infer
vae = VAE(latent_dim=1024).to(device)
vae.load_state_dict(torch.load(os.path.join(model_save_path+f'/best_vae.pth')))
vae.eval()
with torch.no_grad():
    for data_t in tqdm(trainloader):
        data = data_t['data'].squeeze(0).to(device)
        reconstructed, mu, logvar, agrregated = vae(data)
        torch.save(agrregated, os.path.join(save_path, f"{data_t['id'][0]}.pt"))
print('VAE finished.')