# Convolutional Variational Autoencoder (VAE)

This repository contains an implementation of a Convolutional Variational Autoencoder (VAE) using PyTorch. The project trains the VAE on the CelebA dataset and generates face images.

## Features
- Convolution Variational Autoencoder implementation in PyTorch
- Trained on CelebA dataset
- Customizable hyperparameters for flexibility
- Progress tracking with `tqdm`
- Possible to specify inference parameter attribute in order to generate images with one of the 40 specified attributes form the dataset.  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/santysangro/Convolutional-Variational-Autoencoder.git
   cd Convolutional-Variational-Autoencoder  
   pip install -r requirements.txt 
   python scripts/train.py #To train the model
   python scripts/inference.py --num-images 5 --attribute Similing #To generate images


## Dataset
Download the CelebA dataset from [this link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place it in the `dataset/` directory.