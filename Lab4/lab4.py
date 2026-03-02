import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- ПАРАМЕТРЫ ---
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
LATENT_DIM = 32
FC_WEIGHTS = 'fc_autoencoder_weights.pth'
CONV_WEIGHTS = 'conv_autoencoder_weights.pth'

# --- МОДЕЛИ ---

class FCAutoEncoder(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), latent_dim=32):
        super(FCAutoEncoder, self).__init__()
        self.img_shape = img_shape
        inp_dim = img_shape[1] * img_shape[2]
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, inp_dim), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.decoder(self.encoder(x)).view(-1, *self.img_shape)

class ConvAutoEncoder(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), latent_dim=32):
        super(ConvAutoEncoder, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.fc_encode = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=3), nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        return self.fc_encode(x.view(x.size(0), -1))

    def decode(self, z):
        x = self.fc_decode(z).view(-1, 128, 4, 4)
        return self.decoder_cnn(x)

    def forward(self, x):
        return self.decode(self.encode(x))

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def train_model(model, trainloader, testloader, criterion, optimizer, name, device):
    print(f"\n>>> Training {name}...")
    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss = 0.0
        for data, _ in trainloader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), data)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for data, _ in testloader:
                data = data.to(device)
                v_loss += criterion(model(data), data).item()
        
        train_losses.append(t_loss/len(trainloader))
        val_losses.append(v_loss/len(testloader))
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")
    return train_losses, val_losses

def save_reconstruction_plot(model, testloader, device, filename):
    model.eval()
    data, _ = next(iter(testloader))
    with torch.no_grad():
        rec = model(data[:5].to(device)).cpu()
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        ax[0, i].imshow(data[i].squeeze(), cmap='gray')
        ax[1, i].imshow(rec[i].squeeze(), cmap='gray')
        ax[0, i].axis('off'); ax[1, i].axis('off')
    plt.savefig(filename)
    print(f"Saved reconstruction to {filename}")

# --- MAIN ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


    fc_net = FCAutoEncoder().to(device)
    conv_net = ConvAutoEncoder().to(device)


    for model, path, name in [(fc_net, FC_WEIGHTS, "FC_AE"), (conv_net, CONV_WEIGHTS, "Conv_AE")]:
        if os.path.exists(path):
            print(f"\n[INFO] Found weights for {name}. Loading...")
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"\n[INFO] No weights for {name}. Starting training...")
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            train_model(model, train_loader, test_loader, nn.MSELoss(), optimizer, name, device)
            torch.save(model.state_dict(), path)
            print(f"Weights saved to {path}")
        
        save_reconstruction_plot(model, test_loader, device, f"results_{name}.png")


    print("\nGenerating new fashion samples from latent space...")
    conv_net.eval()
    with torch.no_grad():
        noise = torch.randn(10, LATENT_DIM).to(device)
        gen_imgs = conv_net.decode(noise).cpu()
    
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(gen_imgs[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.savefig("latent_generation.png")
    print("Generation result saved to 'latent_generation.png'")

if __name__ == "__main__":
    main()