import torch
import numpy as np
from torchvision.utils import save_image
from torchvision import datasets, transforms
import argparse

# Set device
device = "cpu"

# Define transformation for the input images (assuming images are 128x128 and normalized)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load dataset (you can modify this based on where your dataset is located)
dataset = datasets.CelebA(root="../dataset/", split="train", transform=transform, download=False)

# Attributes to index
attributes_arr = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

# Create a dictionary mapping attributes to indices
attribute_to_index = {attr: idx for idx, attr in enumerate(attributes_arr)}


# Define inference function
def inference(model, num_examples=5, attribute="Bald"):
    with torch.no_grad():
        for example in range(num_examples):
            idx = np.random.randint(0, len(dataset))  # Random index from dataset
            input_image, attributes = dataset[idx]
            if attribute:
                index = attribute_to_index[attribute]
                while attributes[index] != 1:
                    idx = np.random.randint(0, len(dataset))  # Random index from dataset
                    input_image, attributes = dataset[idx]

            input_image = input_image.to(device)
            input_image = input_image.view(1, 3, 128, 128)  # Ensure the input shape is correct

            # Get the mean and sd from the encoder
            mu, sd = model.encode(input_image)
            z = mu + sd * torch.randn_like(sd)
            # Decode the latent vector z
            out = model.decode(z)

            # Save the generated image
            save_image(out, f"generated_{example}.png")


# Main function to load model and call inference
def main(num_images, attribute):
    # Load the trained model (ensure it's the correct path)
    try:
        model = torch.load('../models/vae_model_faces_10_epochs.pth')
    except FileNotFoundError:
        print("No model found to load. Please ensure the model file exists in the '/models/' directory.")
        return

    model.to(device)
    model.eval()  # Set the model to evaluation mode
    # Run inference
    inference(model, num_images, attribute)


if __name__ == "__main__":
    # Argument parser to specify number of images to generate
    parser = argparse.ArgumentParser(description="Generate images using a trained VAE model.")
    parser.add_argument('--num-images', type=int, default=5, help='Number of generated images.')
    parser.add_argument('--attribute', type=str, default=None,
                        help='Filter by attribute (e.g., "Eyeglasses, Smiling, Heavy_Makeup,etc.").')
    args = parser.parse_args()

    # Call main with the specified number of images
    main(args.num_images, args.attribute)
