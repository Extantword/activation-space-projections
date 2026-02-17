#!/usr/bin/env python3
"""
CLI equivalent of visualize_results.ipynb

Loads the latent_data.npz saved by run_experiment.py and regenerates
the interactive 3D PCA and UMAP HTML plots without re-training.

Usage:
    python visualize.py --experiment 1
    python visualize.py --experiment 5 --data-dir outputs/experiment_5
"""

import argparse
import os
import sys

import numpy as np

from plotting import build_3d_figure


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate 3D visualizations from saved experiment data.')
    parser.add_argument('--experiment', '-e', type=int, default=None,
                        help='Experiment ID (used to locate default data dir)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing latent_data.npz '
                             '(default: outputs/experiment_<ID>)')
    args = parser.parse_args()

    if args.data_dir is None and args.experiment is None:
        parser.error('Provide at least --experiment or --data-dir')

    data_dir = args.data_dir or f'outputs/experiment_{args.experiment}'
    npz_path = os.path.join(data_dir, 'latent_data.npz')

    if not os.path.isfile(npz_path):
        sys.exit(f"Error: {npz_path} not found. Run run_experiment.py first.")

    # ------------------------------------------------------------ #
    # Load saved data                                                #
    # ------------------------------------------------------------ #
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    all_latents = data['latents']
    all_images  = data['images']
    pca_3d      = data['pca_3d']
    pca_var     = data['pca_var']
    exp_id      = int(data['experiment'][0])
    img_size    = int(data['image_size'][0])
    latent_dim  = all_latents.shape[1]

    has_umap = 'umap_3d' in data.files
    if has_umap:
        umap_3d = data['umap_3d']

    print(f"Experiment {exp_id} | {len(all_latents)} samples | "
          f"latent dim={latent_dim} | image {img_size}x{img_size}")

    # ------------------------------------------------------------ #
    # PCA 3D                                                         #
    # ------------------------------------------------------------ #
    print("\nGenerating PCA plot...")
    pca_labels = [
        f'PC1 ({pca_var[0]:.1%})',
        f'PC2 ({pca_var[1]:.1%})',
        f'PC3 ({pca_var[2]:.1%})'
    ]
    _, pca_html = build_3d_figure(
        pca_3d, all_images, 'PCA', pca_labels,
        latent_dim=latent_dim,
        title_extra=f' | Var={pca_var.sum():.1%}',
        img_size=img_size
    )
    pca_path = os.path.join(data_dir, 'pca_3d.html')
    with open(pca_path, 'w') as f:
        f.write(pca_html)
    print(f"  Saved {pca_path}")

    # ------------------------------------------------------------ #
    # UMAP 3D                                                        #
    # ------------------------------------------------------------ #
    if has_umap:
        print("\nGenerating UMAP plot...")
        umap_labels = ['UMAP1', 'UMAP2', 'UMAP3']
        _, umap_html = build_3d_figure(
            umap_3d, all_images, 'UMAP', umap_labels,
            latent_dim=latent_dim,
            img_size=img_size
        )
        umap_path = os.path.join(data_dir, 'umap_3d.html')
        with open(umap_path, 'w') as f:
            f.write(umap_html)
        print(f"  Saved {umap_path}")
    else:
        print("\nNo UMAP data found in .npz — skipping UMAP plot.")
        print("  (Re-run run_experiment.py without --no-umap to include it)")

    print("\nDone!  Open the HTML files in a browser for interactive 3D plots.")


if __name__ == '__main__':
    main()
