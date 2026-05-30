


import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image


def load_rgb_image(image_path, crop_size=None):
    """
    Resmi RGB olarak yükler.
    İstenirse merkezden crop yapar.
    Çıktı: float32 tipinde nump array, shape = (H, W, 3)
    """
    img = Image.open(image_path).convert("RGB")
    img = np.array(img).astype(np.float32)

    if crop_size is not None:
        h, w, _ = img.shape
        ch, cw = crop_size

        start_h = (h - ch) // 2
        start_w = (w - cw) // 2
        img = img[start_h:start_h+ch, start_w:start_w+cw, :]
    return img

def haar_decompose_rgb(image):
    """
    RGB görüntüye kanal kanal tek seviyeli 2D Haar decomposition uygular.
    Çıktı: LL, LH, HL, HH
    Her biri shape olarak (H/2, W/2, 3)
    """
    ll_channels = []
    lh_channels = []
    hl_channels = []
    hh_channels = []

    for c in range(3):
        channel = image[:, :, c]

        # cA = approximation = LL
        # (cH, cV, cD) = detail.coefficients
        cA, (cH, cV, cD) = pywt.dwt2(channel, 'haar')

        ll_channels.append(cA)
        lh_channels.append(cH)
        hl_channels.append(cV)
        hh_channels.append(cD)

    LL = np.stack(ll_channels, axis=2)
    LH = np.stack(lh_channels, axis=2)
    HL = np.stack(hl_channels, axis=2)
    HH = np.stack(hh_channels, axis=2)

    return LL, LH, HL, HH


def haar_reconstruct_rgb(LL, LH, HL, HH):
    """
    RGB görüntüyü subband'lardan geri kurar.
    """
    reconstructed_channels = []
    for c in range(3):
        cA = LL[:, :, c]
        cH = LH[:, :, c]
        cV = HL[:, :, c]
        cD = HH[:, :, c]

        channel_reconstructed = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        reconstructed_channels.append(channel_reconstructed)
    reconstructed = np.stack(reconstructed_channels, axis=2)

    return reconstructed

def normalize_for_display(arr):

    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr)

    return(arr - arr_min) / (arr_max - arr_min)

def denormalize_with_stats(normalized, arr_min, arr_max):
    if arr_max - arr_min < 1e-8:
        return np.full_like(normalized, arr_min)
    return normalized * ((arr_max - arr_min) + arr_min)


def clip_to_uint8(image):
    # Görüntüyü [0, 255] aralığına sıkıştırır ve uint8 yapar.

    return np.clip(image, 0, 255).astype(np.uint8)

def visualize_subbands(original, LL, LH, HL, HH, title_prefix=""):

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(clip_to_uint8(original))
    axes[0].set_title(f"{title_prefix} Original")
    axes[0].axis("off")

    axes[1].imshow(normalize_for_display(LL))
    axes[1].set_title("LL")
    axes[1].axis("off")

    axes[2].imshow(normalize_for_display(LH))
    axes[2].set_title("LH")
    axes[2].axis("off")

    axes[3].imshow(normalize_for_display(HL))
    axes[3].set_title("HL")
    axes[3].axis("off")

    axes[4].imshow(normalize_for_display(HH))
    axes[4].set_title("HH")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


def compare_original_and_reconstruction(original, reconstructed, title_prefix=""):

    reconstructed_clipped = clip_to_uint8(reconstructed)

    mse = np.mean((original - reconstructed) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(clip_to_uint8(original))
    axes[0].set_title(f"{title_prefix} Original")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_clipped)
    axes[1].set_title(f"Reconstructed\nMSE = {mse:.6f}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"{title_prefix} Reconstruction MSE: {mse:.6f}")


def process_one_image(image_path, crop_size=(256, 256)):

    print(f"\nProcessing: {image_path}")

    image= load_rgb_image(image_path, crop_size=crop_size)
    print("Original image shape:", image.shape)

    LL, LH, HL, HH = haar_decompose_rgb(image)

    print("LL shape:", LL.shape)
    print("LH shape:", LH.shape)
    print("HL shape:", HL.shape)
    print("HH shape:", HH.shape)

    visualize_subbands(image, LL, LH, HL, HH, title_prefix=os.path.basename(image_path))

    reconstructed = haar_reconstruct_rgb(LL, LH, HL, HH)
    print("Reconstructed shape:", reconstructed.shape)

    compare_original_and_reconstruction(image, reconstructed, title_prefix=os.path.basename(image_path))


def main():
    
    image_paths = [f"kodim{i:02d}.png" for i in range (1, 25)]

    for path in image_paths:
        if os.path.exists(path):
            process_one_image(path, crop_size=(256, 256))

        else:
            print(f"File not found: {path}")


#if __name__ =="__main__":
#    main()


def make_coordinate_grid(height, width):
    """
    [-1, 1] aralığında normalize edilmiş 2D coordinate grid üretir.
    Çıktı shape: (height*width, 2)
    """
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)

    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([grid_x, grid_y], axis=-1)   # shape: (H, W, 2)
    coords = coords.reshape(-1, 2)                 # shape: (H*W, 2)

    return coords

def positional_encoding(coords, num_frequencies=6):
    """
    coords: shape (N, 2)
    çıktı: shape (N, 2 + 4*num_frequencies)
    Her x, y koordinatını sin/cos tabanlı daha zengin bir vektöre çevirir.
    """
    encoded_parts = [coords]

    for k in range(num_frequencies):
        freq = 2.0 ** k
        encoded_parts.append(np.sin(freq * np.pi * coords))
        encoded_parts.append(np.cos(freq * np.pi * coords))

    encoded = np.concatenate(encoded_parts, axis=1).astype(np.float32)
    return encoded
    
        
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self,x):
        return self.net(x)

def compute_psnr(target, predicted, eps=1e-12):
    """
    target, predicted: shape (H, W, C) veya benzeri
    PSNR'yi subband'in kendi dinamik aralığına göre hesaplar.
    """
    mse = np.mean((target - predicted) ** 2)

    if mse < eps:
        return 100.0

    dynamic_range = target.max() - target.min()

    if dynamic_range < eps:
        dynamic_range = 1.0

    psnr = 10.0 * np.log10((dynamic_range ** 2) / mse)
    return psnr

def train_inr_on_subband(subband, subband_name="subband",
                         num_frequencies=10,
                         latent_dim=4,
                         hidden_dim=64,
                         num_epochs=1000,
                         lr=1e-3,
                         show_plot=True,
                         record_every=50):
    """
    Tek bir subband üzerinde PE + latent grid + MLP ile INR eğitir.
    Girdi:
        subband: shape (H, W, C)
    Çıktı:
        predicted_np: shape (H, W, C)
    """
    h, w, c = subband.shape

    coords = make_coordinate_grid(h, w)
    targets = subband.reshape(-1, c).astype(np.float32)

    encoded_coords = positional_encoding(coords, num_frequencies=num_frequencies)

    coords_torch = torch.from_numpy(encoded_coords)
    targets_torch = torch.from_numpy(targets)

    latent_grid = nn.Parameter(torch.randn(h, w, latent_dim) * 0.01)

    input_dim = coords_torch.shape[1] + latent_dim
    model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=c)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(model.parameters()) + [latent_grid],
        lr=lr
    )

    print(f"\nTraining on {subband_name}")
    print(f"{subband_name} shape: {subband.shape}")
    print(f"Encoded coords shape: {encoded_coords.shape}")
    print(f"Latent grid shape: {latent_grid.shape}")

    history = {
        "iterations": [],
        "psnr": [],
        "loss": []
    }

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        latent_features = latent_grid.reshape(-1, latent_dim)
        model_input = torch.cat([coords_torch, latent_features], dim=1)

        outputs = model(model_input)
        loss = criterion(outputs, targets_torch)

        loss.backward()
        optimizer.step()

        if (epoch % record_every == 0) or (epoch == num_epochs - 1):
            with torch.no_grad():
                pred_np = outputs.detach().numpy().reshape(h, w, c)
                psnr = compute_psnr(subband, pred_np)

            history["iterations"].append(epoch)
            history["psnr"].append(psnr)
            history["loss"].append(loss.item())

            print(f"{subband_name} | Epoch {epoch}: loss = {loss.item():.6f}, PSNR = {psnr:.3f} dB")

    with torch.no_grad():
        latent_features = latent_grid.reshape(-1, latent_dim)
        model_input = torch.cat([coords_torch, latent_features], dim=1)
        predicted = model(model_input)

    predicted_np = predicted.numpy().reshape(h, w, c)

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(normalize_for_display(subband))
        axes[0].set_title(f"True {subband_name}")
        axes[0].axis("off")

        axes[1].imshow(normalize_for_display(predicted_np))
        axes[1].set_title(f"Predicted {subband_name}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    return predicted_np, history

def plot_psnr_histories(histories_dict, title="PSNR vs Training Iteration"):
    plt.figure(figsize=(8, 5))

    for name, history in histories_dict.items():
        plt.plot(history["iterations"], history["psnr"], label=name)

    plt.xlabel("Training Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close('all')


def plot_loss_histories(histories_dict, title="Loss vs Training Iteration"):
    plt.figure(figsize=(8, 5))

    for name, history in histories_dict.items():
        plt.plot(history["iterations"], history["loss"], label=name)

    plt.xlabel("Training Iteration")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close('all')

def compute_image_psnr(original, reconstructed, eps=1e-12):
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    mse = np.mean((original - reconstructed) ** 2)

    if mse < eps:
        return 100.0

    max_val = 255.0
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return psnr



def main_part3():
    image_path = "kodim17.png"

    image = load_rgb_image(image_path, crop_size=(256, 256))
    LL, LH, HL, HH = haar_decompose_rgb(image)

    print("Original image shape:", image.shape)
    print("LL shape:", LL.shape)
    print("LH shape:", LH.shape)
    print("HL shape:", HL.shape)
    print("HH shape:", HH.shape)

    pred_LL, history_LL = train_inr_on_subband(LL, subband_name="LL",num_epochs=1000, record_every=50, show_plot=False)
    pred_LH, history_LH = train_inr_on_subband(LH, subband_name="LH",num_epochs=1000, record_every=50, show_plot=False)
    pred_HL, history_HL = train_inr_on_subband(HL, subband_name="HL",num_epochs=1000, record_every=50, show_plot=False)
    pred_HH, history_HH = train_inr_on_subband(HH, subband_name="HH",num_epochs=1000, record_every=50, show_plot=False)

    reconstructed = haar_reconstruct_rgb(pred_LL, pred_LH, pred_HL, pred_HH)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(clip_to_uint8(image))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(clip_to_uint8(reconstructed))
    axes[1].set_title("Reconstructed from Predicted Subbands")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    mse = np.mean((image - reconstructed) ** 2)
    print(f"Final reconstruction MSE: {mse:.6f}")

    histories = {
        "LL": history_LL,
        "LH": history_LH,
        "HL": history_HL,
        "HH": history_HH,
    }


    plot_psnr_histories(
        histories,
        title=f"Experiment A - PSNR vs Training Iteration ({image_path})"
    )

    plot_loss_histories(
    histories,
    title=f"Experiment A - Loss vs Training Iteration ({image_path})"
    )

    final_psnr = compute_image_psnr(image, reconstructed)
    print(f"Final full-image PSNR: {final_psnr:.3f} dB")

if __name__ == "__main__":
    main_part3()
    

   
    













































    
