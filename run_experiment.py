#!/usr/bin/env python3
"""
CLI equivalent of experiment_runner.ipynb

Pipeline:
  1. Generate dataset
  2. Save sample images
  3. Train convolutional autoencoder
  4. Save loss curve and reconstruction comparison
  5. Extract latent representations
  6. PCA and UMAP 3D projections (saved as interactive HTML)
  7. Save latent data (.npz) for offline visualization

Usage:
    python run_experiment.py --experiment 1
    python run_experiment.py --experiment 5 --epochs 100 --latent-dim 128 --image-size 64
    python run_experiment.py --experiment 1 --no-umap   # skip UMAP (faster)
"""

import argparse
import importlib.util
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')  # headless backend — no display required
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import OrderedDict
from sklearn.decomposition import PCA
import umap

from plotting import build_3d_figure


# ------------------------------------------------------------------ #
#  Autoencoder                                                        #
# ------------------------------------------------------------------ #

class ConvAutoencoder(nn.Module):
    """Symmetric Conv Autoencoder — adapts to any power-of-2 image size."""

    def __init__(self, latent_dim, activation=nn.ReLU, n=64,
                 use_batch_norm=True, dropout_rate=0.0):
        super().__init__()
        self.n = n
        self.latent_dim = latent_dim

        # ---------- ENCODER ----------
        enc = []
        channels = [1, 32, 64, 128, 256]
        for i in range(len(channels) - 1):
            enc.append((f'conv{i+1}', nn.Conv2d(channels[i], channels[i+1],
                         kernel_size=3, stride=2, padding=1)))
            if use_batch_norm:
                enc.append((f'bn{i+1}', nn.BatchNorm2d(channels[i+1])))
            enc.append((f'act{i+1}', activation()))
            if dropout_rate > 0:
                enc.append((f'drop{i+1}', nn.Dropout2d(p=dropout_rate)))
        self.conv_encoder = nn.Sequential(OrderedDict(enc))

        spatial = n // 16
        flattened = 256 * spatial * spatial
        self.flatten = nn.Flatten()
        self.fc_encoder = nn.Linear(flattened, latent_dim)

        # ---------- DECODER ----------
        self.fc_decoder = nn.Linear(latent_dim, flattened)
        self.act_fc     = activation()
        self.unflatten  = nn.Unflatten(1, (256, spatial, spatial))

        dec = []
        dec_ch = list(reversed(channels))
        for i in range(len(dec_ch) - 1):
            is_last = (i == len(dec_ch) - 2)
            dec.append((f'deconv{i+1}', nn.ConvTranspose2d(
                dec_ch[i], dec_ch[i+1], kernel_size=3, stride=2,
                padding=1, output_padding=1)))
            if is_last:
                dec.append(('sigmoid', nn.Sigmoid()))
            else:
                if use_batch_norm:
                    dec.append((f'bn_dec{i+1}', nn.BatchNorm2d(dec_ch[i+1])))
                dec.append((f'act_dec{i+1}', activation()))
                if dropout_rate > 0:
                    dec.append((f'drop_dec{i+1}', nn.Dropout2d(p=dropout_rate)))
        self.conv_decoder = nn.Sequential(OrderedDict(dec))

    def encode(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, self.n, self.n)
        x = self.conv_encoder(x)
        x = self.flatten(x)
        return self.fc_encoder(x)

    def decode(self, latent):
        x = self.act_fc(self.fc_decoder(latent))
        x = self.unflatten(x)
        x = self.conv_decoder(x)
        return x.view(-1, self.n * self.n)

    def forward(self, x):
        return self.decode(self.encode(x))


# ------------------------------------------------------------------ #
#  Main pipeline                                                      #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description='Run an activation-space-projections experiment.')
    parser.add_argument('--experiment', '-e', type=int, required=True,
                        help='Experiment ID (1-45)')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Image resolution (default: 64)')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of images to generate (default: 10000)')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Autoencoder bottleneck dimension (default: 64)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout rate (default: 0.05)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputs/experiment_<ID>)')
    parser.add_argument('--no-umap', action='store_true',
                        help='Skip UMAP projection (faster)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    exp_id     = args.experiment
    img_size   = args.image_size
    n_samples  = args.n_samples
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr         = args.lr
    dropout    = args.dropout
    output_dir = args.output_dir or f'outputs/experiment_{exp_id}'

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment {exp_id} | size={img_size} | n={n_samples} | "
          f"latent={latent_dim} | epochs={num_epochs}")
    print(f"Outputs -> {output_dir}")

    # ------------------------------------------------------------ #
    # 1. Generate dataset                                            #
    # ------------------------------------------------------------ #
    print("\n[1/7] Generating dataset...")
    ds_path = os.path.join('Experiments', str(exp_id), 'datasets.py')
    if not os.path.isfile(ds_path):
        sys.exit(f"Error: {ds_path} not found. Valid experiment IDs: 1-45")

    spec = importlib.util.spec_from_file_location('exp_datasets', ds_path)
    exp_datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_datasets)

    raw_images, raw_labels = exp_datasets.generate_dataset(n=n_samples, size=img_size)
    print(f"  shape={raw_images.shape}  dtype={raw_images.dtype}  "
          f"range=[{raw_images.min()}, {raw_images.max()}]")

    # ------------------------------------------------------------ #
    # 2. Save sample images                                          #
    # ------------------------------------------------------------ #
    print("\n[2/7] Saving sample images...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    indices = random.sample(range(len(raw_images)), 10)
    for idx, ax in zip(indices, axes.flat):
        ax.imshow(raw_images[idx], cmap='gray')
        ax.axis('off')
    fig.suptitle(f'Experiment {exp_id} — Sample Images ({img_size}x{img_size})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved samples.png")

    # ------------------------------------------------------------ #
    # 3. Prepare DataLoaders                                         #
    # ------------------------------------------------------------ #
    print("\n[3/7] Preparing data loaders...")
    dataset_flat = raw_images.astype(np.float32) / 255.0
    dataset_flat = dataset_flat.reshape(len(dataset_flat), -1)

    dataset_tensor = torch.tensor(dataset_flat, dtype=torch.float32)
    labels_tensor  = torch.tensor(raw_labels, dtype=torch.long)
    full_dataset   = TensorDataset(dataset_tensor, labels_tensor)

    train_size = int(0.8 * len(full_dataset))
    test_size  = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    print(f"  Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")

    # ------------------------------------------------------------ #
    # 4. Train autoencoder                                           #
    # ------------------------------------------------------------ #
    print("\n[4/7] Training autoencoder...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = ConvAutoencoder(
        latent_dim=latent_dim, activation=nn.ReLU, n=img_size,
        use_batch_norm=True, dropout_rate=dropout
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses  = []

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        running = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), data)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_losses.append(running / len(train_loader))

        # --- Eval ---
        model.eval()
        running = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                running += criterion(model(data), data).item()
        test_losses.append(running / len(test_loader))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:>3}/{num_epochs}]  "
                  f"Train: {train_losses[-1]:.4f}  Test: {test_losses[-1]:.4f}")

    print("  Training complete!")

    # Loss curve
    fig = plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses,  label='Test')
    plt.xlabel('Epoch'); plt.ylabel('BCE Loss')
    plt.title('Training & Validation Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved loss_curve.png")

    # Reconstruction comparison
    model.eval()
    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data.to(device)
    with torch.no_grad():
        recon = model(sample_data).cpu().numpy()
    orig = sample_data.cpu().numpy()

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(orig[i].reshape(img_size, img_size), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].reshape(img_size, img_size), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    plt.suptitle(f'Experiment {exp_id} — Reconstructions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstructions.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved reconstructions.png")

    # ------------------------------------------------------------ #
    # 5. Extract latent representations                              #
    # ------------------------------------------------------------ #
    print("\n[5/7] Extracting latent representations...")
    model.eval()
    all_latents = []
    all_images  = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            all_latents.append(model.encode(data).cpu().numpy())
            all_images.append(data.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_images  = np.concatenate(all_images,  axis=0)
    print(f"  {len(all_latents)} latent vectors of dim {all_latents.shape[1]}")

    # ------------------------------------------------------------ #
    # 6. PCA 3D Projection                                           #
    # ------------------------------------------------------------ #
    print("\n[6/7] Computing PCA projection...")
    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(all_latents)
    var = pca.explained_variance_ratio_
    print(f"  Variance explained: {var.sum():.2%}  "
          f"(PC1={var[0]:.2%}, PC2={var[1]:.2%}, PC3={var[2]:.2%})")

    pca_labels = [
        f'PC1 ({var[0]:.1%})',
        f'PC2 ({var[1]:.1%})',
        f'PC3 ({var[2]:.1%})'
    ]
    _, pca_html = build_3d_figure(
        pca_3d, all_images, 'PCA', pca_labels,
        latent_dim=all_latents.shape[1],
        title_extra=f' | Var={var.sum():.1%}',
        img_size=img_size
    )
    pca_path = os.path.join(output_dir, 'pca_3d.html')
    with open(pca_path, 'w') as f:
        f.write(pca_html)
    print(f"  Saved {pca_path}")

    # ------------------------------------------------------------ #
    # 7. UMAP 3D Projection                                         #
    # ------------------------------------------------------------ #
    umap_3d = None
    if not args.no_umap:
        print("\n[7/7] Computing UMAP projection (this may take a while)...")
        reducer = umap.UMAP(n_components=3, random_state=42)
        umap_3d = reducer.fit_transform(all_latents)
        print(f"  UMAP shape: {umap_3d.shape}")

        umap_labels = ['UMAP1', 'UMAP2', 'UMAP3']
        _, umap_html = build_3d_figure(
            umap_3d, all_images, 'UMAP', umap_labels,
            latent_dim=all_latents.shape[1],
            img_size=img_size
        )
        umap_path = os.path.join(output_dir, 'umap_3d.html')
        with open(umap_path, 'w') as f:
            f.write(umap_html)
        print(f"  Saved {umap_path}")
    else:
        print("\n[7/7] Skipping UMAP (--no-umap)")

    # ------------------------------------------------------------ #
    # Save data for offline visualization                            #
    # ------------------------------------------------------------ #
    save_dict = dict(
        latents    = all_latents,
        images     = all_images,
        pca_3d     = pca_3d,
        pca_var    = pca.explained_variance_ratio_,
        experiment = np.array([exp_id]),
        image_size = np.array([img_size])
    )
    if umap_3d is not None:
        save_dict['umap_3d'] = umap_3d

    npz_path = os.path.join(output_dir, 'latent_data.npz')
    np.savez_compressed(npz_path, **save_dict)
    print(f"\nSaved: {npz_path}")
    print("Done!  Open pca_3d.html / umap_3d.html in a browser for interactive 3D plots.")


if __name__ == '__main__':
    main()
