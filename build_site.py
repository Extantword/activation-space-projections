#!/usr/bin/env python3
"""
Generates a static GitHub-Pages site from experiment outputs.

Reads outputs/experiment_*/  and Experiments/*/  to build:
  docs/
  ├── index.html            # gallery of all experiments
  ├── style.css
  └── experiment/<ID>/
      ├── index.html         # detail page
      ├── samples.png        # copied from outputs
      ├── loss_curve.png
      ├── reconstructions.png
      ├── pca.html           # copied Plotly standalone
      └── umap.html          # (if present)

Usage:
    python build_site.py                    # build from outputs/
    python build_site.py --output-dir docs  # explicit output dir
"""

import argparse
import os
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
EXPERIMENTS_SRC = ROOT / "Experiments"
DOCS = ROOT / "docs"

# Files to copy from each experiment output
COPY_FILES = {
    "samples.png":         "samples.png",
    "loss_curve.png":      "loss_curve.png",
    "reconstructions.png": "reconstructions.png",
    "pca_3d.html":         "pca.html",
    "umap_3d.html":        "umap.html",
}

# Fallback sample images from Experiments/<ID>/
SAMPLE_FALLBACK = "sample_1.png"


# ── Helpers ──────────────────────────────────────────────────


def get_experiment_meta(exp_id: int) -> dict:
    """Parse name and description from Experiments/<ID>/datasets.py header."""
    ds_path = EXPERIMENTS_SRC / str(exp_id) / "datasets.py"
    name = f"Experiment {exp_id}"
    description = ""
    if ds_path.exists():
        first_line = ds_path.read_text().split("\n", 1)[0]
        # Pattern: # Experiment 1: Kaleidoscope Patterns (D6 Dihedral Symmetry)
        m = re.match(r"#\s*Experiment\s+\d+:\s*(.+)", first_line)
        if m:
            raw = m.group(1).strip()
            # Split name from parenthetical description
            pm = re.match(r"(.+?)\s*\((.+)\)\s*$", raw)
            if pm:
                name = pm.group(1).strip()
                description = pm.group(2).strip()
            else:
                name = raw
        # Try remaining comment lines for extra description
        lines = ds_path.read_text().split("\n")
        desc_lines = []
        for line in lines[1:]:
            if line.startswith("#"):
                part = line.lstrip("#").strip()
                if part:
                    desc_lines.append(part)
            else:
                break
        if desc_lines:
            description = " ".join(desc_lines)
    return {"id": exp_id, "name": name, "description": description}


def discover_experiments() -> list[dict]:
    """Find all experiments that have outputs."""
    experiments = []
    if not OUTPUTS.exists():
        return experiments
    for d in sorted(OUTPUTS.iterdir()):
        m = re.match(r"experiment_(\d+)$", d.name)
        if m and d.is_dir():
            exp_id = int(m.group(1))
            meta = get_experiment_meta(exp_id)
            meta["output_dir"] = d
            # Check which files exist
            meta["has_pca"] = (d / "pca_3d.html").exists()
            meta["has_umap"] = (d / "umap_3d.html").exists()
            meta["has_samples"] = (d / "samples.png").exists()
            meta["has_loss"] = (d / "loss_curve.png").exists()
            meta["has_recon"] = (d / "reconstructions.png").exists()
            experiments.append(meta)
    return experiments


def discover_all_experiments() -> list[dict]:
    """Find ALL 45 experiments, marking which have outputs."""
    experiments = []
    completed_ids = set()

    # First gather completed ones
    if OUTPUTS.exists():
        for d in sorted(OUTPUTS.iterdir()):
            m = re.match(r"experiment_(\d+)$", d.name)
            if m and d.is_dir():
                completed_ids.add(int(m.group(1)))

    for exp_id in range(1, 46):
        src_dir = EXPERIMENTS_SRC / str(exp_id)
        if not src_dir.exists():
            continue
        meta = get_experiment_meta(exp_id)
        meta["completed"] = exp_id in completed_ids
        if meta["completed"]:
            d = OUTPUTS / f"experiment_{exp_id}"
            meta["output_dir"] = d
            meta["has_pca"] = (d / "pca_3d.html").exists()
            meta["has_umap"] = (d / "umap_3d.html").exists()
            meta["has_samples"] = (d / "samples.png").exists()
            meta["has_loss"] = (d / "loss_curve.png").exists()
            meta["has_recon"] = (d / "reconstructions.png").exists()
        else:
            meta["has_pca"] = False
            meta["has_umap"] = False
            meta["has_samples"] = False
            meta["has_loss"] = False
            meta["has_recon"] = False
        # Check for preview image from source
        meta["has_preview"] = (src_dir / SAMPLE_FALLBACK).exists()
        meta["preview_src"] = src_dir / SAMPLE_FALLBACK
        experiments.append(meta)
    return experiments


def copy_experiment_assets(meta: dict, dest: Path):
    """Copy output files to the site directory."""
    dest.mkdir(parents=True, exist_ok=True)
    if not meta["completed"]:
        return
    out = meta["output_dir"]
    for src_name, dst_name in COPY_FILES.items():
        src = out / src_name
        if src.exists():
            shutil.copy2(src, dest / dst_name)

    # Also copy preview from Experiments/ source for the thumbnail
    if meta["has_preview"]:
        shutil.copy2(meta["preview_src"], dest / "preview.png")


# ── HTML templates ───────────────────────────────────────────


def css() -> str:
    return """\
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --accent: #58a6ff;
  --accent-hover: #79c0ff;
  --success: #3fb950;
  --pending: #8b949e;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 15px; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.6; min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-hover); text-decoration: underline; }

/* ── Header ── */
.site-header {
  border-bottom: 1px solid var(--border);
  padding: 2rem 1.5rem 1.5rem;
  text-align: center;
}
.site-header h1 { font-size: 1.8rem; font-weight: 600; }
.site-header p { color: var(--text-muted); margin-top: 0.4rem; font-size: 0.95rem; }
.stats {
  display: flex; justify-content: center; gap: 2rem;
  margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted);
}
.stats span { display: flex; align-items: center; gap: 0.3rem; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-done { background: var(--success); }
.dot-pending { background: var(--pending); }

/* ── Grid ── */
.container { max-width: 1280px; margin: 0 auto; padding: 1.5rem; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
}

/* ── Card ── */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden;
  transition: border-color 0.15s, transform 0.15s;
}
.card:hover { border-color: var(--accent); transform: translateY(-2px); }
.card.pending { opacity: 0.45; pointer-events: none; }
.card-img {
  width: 100%; aspect-ratio: 1; object-fit: cover;
  background: #000; display: block;
}
.card-img-placeholder {
  width: 100%; aspect-ratio: 1;
  display: flex; align-items: center; justify-content: center;
  background: #000; color: var(--text-muted); font-size: 0.8rem;
}
.card-body { padding: 0.75rem 1rem; }
.card-body h3 {
  font-size: 0.95rem; font-weight: 600;
  display: flex; align-items: center; gap: 0.5rem;
}
.card-body h3 .id { color: var(--text-muted); font-weight: 400; font-size: 0.85rem; }
.card-body .desc { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }
.card-body .links {
  display: flex; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap;
}
.badge {
  font-size: 0.7rem; padding: 0.2rem 0.5rem;
  border-radius: 12px; border: 1px solid var(--border);
  color: var(--text-muted); background: var(--bg);
  transition: border-color 0.15s, color 0.15s;
}
.badge:hover { border-color: var(--accent); color: var(--accent); text-decoration: none; }

/* ── Detail page ── */
.back { display: inline-block; margin-bottom: 1rem; font-size: 0.9rem; }
.detail-header { margin-bottom: 1.5rem; }
.detail-header h1 { font-size: 1.5rem; }
.detail-header .desc { color: var(--text-muted); margin-top: 0.3rem; }
.section { margin-bottom: 2rem; }
.section h2 {
  font-size: 1.1rem; font-weight: 600;
  margin-bottom: 0.75rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border);
}
.image-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
}
.image-row figure { margin: 0; }
.image-row img { width: 100%; border-radius: 6px; border: 1px solid var(--border); }
.image-row figcaption {
  font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem;
}
.viz-links { display: flex; gap: 1rem; flex-wrap: wrap; }
.viz-btn {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.6rem 1.2rem; border-radius: 6px;
  background: var(--surface); border: 1px solid var(--border);
  color: var(--text); font-size: 0.9rem; font-weight: 500;
  transition: border-color 0.15s, background 0.15s;
}
.viz-btn:hover { border-color: var(--accent); background: var(--bg); text-decoration: none; }
"""


def index_html(experiments: list[dict]) -> str:
    n_done = sum(1 for e in experiments if e["completed"])
    n_total = len(experiments)

    cards = []
    for e in experiments:
        exp_id = e["id"]
        cls = "" if e["completed"] else " pending"

        if e["completed"] and e["has_samples"]:
            img = f'<img class="card-img" src="experiment/{exp_id}/samples.png" alt="{e["name"]}" loading="lazy">'
        elif e["has_preview"]:
            img = f'<img class="card-img" src="experiment/{exp_id}/preview.png" alt="{e["name"]}" loading="lazy">'
        else:
            img = '<div class="card-img-placeholder">No preview</div>'

        links = ""
        if e["completed"]:
            badges = []
            if e["has_pca"]:
                badges.append(f'<a class="badge" href="experiment/{exp_id}/pca.html">PCA 3D</a>')
            if e["has_umap"]:
                badges.append(f'<a class="badge" href="experiment/{exp_id}/umap.html">UMAP 3D</a>')
            badges.append(f'<a class="badge" href="experiment/{exp_id}/">Details</a>')
            links = f'<div class="links">{"".join(badges)}</div>'

        desc_html = f'\n        <div class="desc">{e["description"]}</div>' if e["description"] else ""
        links_html = f'\n        {links}' if links else ""

        href = f'experiment/{exp_id}/' if e["completed"] else '#'
        cards.append(
            f'    <a href="{href}" style="text-decoration:none;color:inherit">\n'
            f'    <div class="card{cls}">\n'
            f'      {img}\n'
            f'      <div class="card-body">\n'
            f'        <h3><span class="id">#{exp_id}</span> {e["name"]}</h3>'
            f'{desc_html}{links_html}\n'
            f'      </div>\n'
            f'    </div>\n'
            f'    </a>'
        )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activation Space Projections</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="site-header">
  <h1>Activation Space Projections</h1>
  <p>Interactive 3D latent-space visualizations of convolutional autoencoder embeddings</p>
  <div class="stats">
    <span><span class="dot dot-done"></span> {n_done} completed</span>
    <span><span class="dot dot-pending"></span> {n_total - n_done} pending</span>
  </div>
</header>
<main class="container">
  <div class="grid">
{"".join(cards)}
  </div>
</main>
</body>
</html>
"""


def experiment_html(e: dict) -> str:
    exp_id = e["id"]

    # Images section
    images = []
    if e["has_samples"]:
        images.append(("samples.png", "Dataset samples"))
    if e["has_loss"]:
        images.append(("loss_curve.png", "Training / validation loss"))
    if e["has_recon"]:
        images.append(("reconstructions.png", "Reconstructions (top: original, bottom: decoded)"))

    img_html = "\n".join(
        f'      <figure><img src="{fname}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
        for fname, cap in images
    )

    # Viz links
    viz_btns = []
    if e["has_pca"]:
        viz_btns.append(
            '<a class="viz-btn" href="pca.html">'
            '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">'
            '<circle cx="3" cy="13" r="1.5"/><circle cx="6" cy="8" r="1.5"/>'
            '<circle cx="10" cy="10" r="1.5"/><circle cx="13" cy="4" r="1.5"/>'
            '</svg>PCA 3D Projection</a>'
        )
    if e["has_umap"]:
        viz_btns.append(
            '<a class="viz-btn" href="umap.html">'
            '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">'
            '<circle cx="4" cy="5" r="1.5"/><circle cx="7" cy="12" r="1.5"/>'
            '<circle cx="12" cy="7" r="1.5"/><circle cx="9" cy="3" r="1.5"/>'
            '</svg>UMAP 3D Projection</a>'
        )

    desc_html = f'<p class="desc">{e["description"]}</p>' if e["description"] else ""

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>#{exp_id} {e["name"]} — Activation Space Projections</title>
<link rel="stylesheet" href="../../style.css">
</head>
<body>
<main class="container">
  <a class="back" href="../../">&larr; All experiments</a>
  <div class="detail-header">
    <h1>#{exp_id} &mdash; {e["name"]}</h1>
    {desc_html}
  </div>

  <div class="section">
    <h2>Interactive 3D Visualizations</h2>
    <div class="viz-links">
      {"".join(viz_btns) if viz_btns else "<p>No visualizations available yet.</p>"}
    </div>
  </div>

  <div class="section">
    <h2>Training Results</h2>
    <div class="image-row">
{img_html}
    </div>
  </div>
</main>
</body>
</html>
"""


# ── Main ─────────────────────────────────────────────────────


def build(docs_dir: Path):
    print(f"Building site into {docs_dir}/")
    docs_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_all_experiments()
    if not experiments:
        print("No experiments found. Run some experiments first, or check that")
        print("Experiments/ directory contains subdirectories 1-45.")
        return

    # Write CSS
    (docs_dir / "style.css").write_text(css())

    # Copy assets and build per-experiment pages
    for e in experiments:
        dest = docs_dir / "experiment" / str(e["id"])
        copy_experiment_assets(e, dest)
        # Copy preview image for non-completed experiments too
        if not e["completed"] and e["has_preview"]:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(e["preview_src"], dest / "preview.png")
        # Write detail page only for completed experiments
        if e["completed"]:
            (dest / "index.html").write_text(experiment_html(e))

    # Write index
    (docs_dir / "index.html").write_text(index_html(experiments))

    n_done = sum(1 for e in experiments if e["completed"])
    print(f"Done. {n_done}/{len(experiments)} experiments included.")
    print(f"Open {docs_dir / 'index.html'} in a browser to preview.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GitHub Pages site from experiment outputs")
    parser.add_argument("--output-dir", type=Path, default=DOCS,
                        help="Directory to write the site into (default: docs/)")
    args = parser.parse_args()
    build(args.output_dir)
