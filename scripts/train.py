import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torch.utils.data import DataLoader

# Configuration
IMAGE_SIZE = 128
INPUT_CHANNELS = 3
Z_DIM = 128
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4  # Karpathy constant
a = 1
b = 0.7  # weights of loss functions
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(f"Model is running on: {device}")
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize images to 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] instead of [0,255]
])
dataset = datasets.CelebA(root="../dataset/", split="train", transform=transform, download=True)


# Dataset Loading
def train():
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = VariationalAutoEncoder(IMAGE_SIZE, Z_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", ncols=100,
                    dynamic_ncols=True)
        for i, (x, _) in loop:
            x = x.to(device)
            x_reconstructed, mu, sd = model.forward(x)
            # compute loss
            x = (x + 1) / 2  # DENORMALIZE
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_divergence = - torch.sum(1 + torch.log(sd.pow(2)) - mu.pow(2) - sd.pow(2))

            # backpropagation
            loss = a * reconstruction_loss + b * kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model, "../models/vae_model_faces_10_epochs.pth")


# Train
if __name__ == "__main__":
    train()
