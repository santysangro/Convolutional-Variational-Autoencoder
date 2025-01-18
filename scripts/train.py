import torch
import os
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Configuration
IMAGE_SIZE = 64
INPUT_CHANNELS = 3
INPUT_DIM = INPUT_CHANNELS * IMAGE_SIZE * IMAGE_SIZE
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 1
BATCH_SIZE = 32
LR_RATE = 3e-4  # Karpathy constant
a = 1
b = 0.7  # weights of loss functions


# Dataset Loading
def train():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize images to 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    dataset = datasets.CelebA(root="../dataset/", split="train", transform=transform, download=True)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", ncols=100,
                    dynamic_ncols=True)
        for i, (x, _) in loop:
            x = x.view(x.shape[0], INPUT_DIM)
            x = x.to("cuda" if torch.cuda.is_available() else "cpu")
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

    torch.save(model, "vae_model_faces.pth")


def inference(model, num_examples=5):
    with torch.no_grad():
        for example in range(num_examples):
            z = torch.randn(1, Z_DIM).to("cuda" if torch.cuda.is_available() else "cpu")
            out = model.decode(z)
            out = out.view(-1, 3, 64, 64)
            # out = (out + 1) / 2  # De-normalize to [0, 1] for saving
            save_image(out, f"generated_face_{example}.png")


# Train and infer
if __name__ == "__main__":
    trained_model = "/vae_model_faces.pth"
    if not os.path.exists(trained_model):
        train()

    model = torch.load("vae_model.pth")
    inference(model, num_examples=5)
