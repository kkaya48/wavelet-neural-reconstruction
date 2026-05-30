# Wavelet-Based Image Reconstruction with Neural Networks

This project explores image reconstruction using Haar wavelet decomposition and neural-network-based reconstruction methods.

RGB images are decomposed into wavelet subbands using the Haar transform. A PyTorch-based MLP model is then used to reconstruct subband information, and reconstruction quality is evaluated with metrics such as MSE and PSNR.

## Main Features

- Haar wavelet decomposition of RGB images

- Subband-based image representation

- PyTorch-based MLP reconstruction model

- Positional encoding and latent-grid based learning approach

- Reconstruction quality evaluation with MSE and PSNR

## Technologies

- Python

- PyTorch

- NumPy

- PyWavelets

- Pillow

- Matplotlib

## Purpose

The project combines classical image processing with neural-network-based reconstruction methods. It was developed as part of my work on technical and perceptual image reconstruction quality analysis.

## How to Run

Install the required packages:

```bash

pip install -r requirements.txt

```

Then run:

```bash

python wavelet_reconstruction.py

```

The script expects input images to be available locally. Dataset files are not included in this repository.

## Author

Kerem Kaya
