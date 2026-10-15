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
        if not src.exists():
            continue
        if src_name.endswith("_3d.html"):
            (dest / dst_name).write_text(patch_plot(src.read_text()))
        else:
            shutil.copy2(src, dest / dst_name)

    # Also copy preview from Experiments/ source for the thumbnail
    if meta["has_preview"]:
        shutil.copy2(meta["preview_src"], dest / "preview.png")


# Injected into every PCA/UMAP page: the hovered sample's image follows the
# cursor, just above it, instead of sitting in a panel under the plot, and
# the plot fills its window so the detail page can show two side by side.
# Works on pages from any version of plotting.py: it hides their own image
# panel (imgdiv_*) or tooltip (tooltip_*) and Plotly's text label.
PLOT_PATCH = """
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #fff; }
  [id^="imgdiv_"], [id^="tooltip_"] { display: none !important; }
  .plotly-graph-div { width: 100vw !important; height: 100vh !important; }
  #sample-tip {
    position: fixed; z-index: 9999; display: none; pointer-events: none;
    background: #000; border: 1px solid #333; border-radius: 6px; padding: 6px;
    box-shadow: 0 6px 20px rgba(0,0,0,.35);
    font: 11px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #8b949e; text-align: center;
  }
  #sample-tip img { display: block; width: 128px; height: 128px; image-rendering: pixelated; border-radius: 3px; }
  #sample-tip div { margin-top: 4px; }
</style>
<div id="sample-tip"></div>
<script>
(function () {
  var tip = document.getElementById('sample-tip');
  var mx = 0, my = 0;
  function fit(gd) { Plotly.relayout(gd, { width: window.innerWidth, height: window.innerHeight }); }
  function place() {
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var x = Math.min(Math.max(mx - w / 2, 4), window.innerWidth - w - 4);
    var y = my - h - 14;
    if (y < 4) y = my + 18;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  document.addEventListener('mousemove', function (e) {
    mx = e.clientX; my = e.clientY;
    if (tip.style.display === 'block') place();
  });
  document.querySelectorAll('.plotly-graph-div').forEach(function (gd) {
    gd.on('plotly_hover', function (data) {
      var pt = data.points[0], cd = pt.customdata;
      if (!cd || !cd[3]) return;
      tip.innerHTML = '<img src="' + cd[3] + '" alt=""><div>Sample #' + pt.pointNumber + '</div>';
      tip.style.display = 'block';
      place();
    });
    gd.on('plotly_unhover', function () { tip.style.display = 'none'; });
    gd.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
    try {
      Plotly.restyle(gd, { hovertemplate: null, hoverinfo: 'none' });
      fit(gd);
    } catch (e) { window.__patchError = String(e); }
  });
  window.addEventListener('resize', function () {
    document.querySelectorAll('.plotly-graph-div').forEach(fit);
  });
})();
</script>
"""


def patch_plot(html: str) -> str:
    """Add PLOT_PATCH just before the page's last </body>."""
    i = html.rfind("</body>")
    return html + PLOT_PATCH if i < 0 else html[:i] + PLOT_PATCH + html[i:]


# ── HTML templates ───────────────────────────────────────────


def css() -> str:
    return """\
:root {
  color-scheme: dark;
  --bg: #000;
  --surface: #000;
  --border: #262626;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --accent: #58a6ff;
  --accent-hover: #79c0ff;
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
.stats { margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted); }

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
.card-main, .card-main:hover { display: block; color: inherit; text-decoration: none; }
.card-links { padding-top: 0; }
.card-body h3 {
  font-size: 0.95rem; font-weight: 600;
  display: flex; align-items: center; gap: 0.5rem;
}
.card-body h3 .id { color: var(--text-muted); font-weight: 400; font-size: 0.85rem; }
.card-body .desc { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }
.card-body .links {
  display: flex; gap: 0.5rem; flex-wrap: wrap;
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
.viz-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 1rem;
}
.viz { margin: 0; }
.viz figcaption {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 0.9rem; font-weight: 600; margin-bottom: 0.4rem;
}
.viz figcaption a { font-size: 0.8rem; font-weight: 400; }
.viz iframe {
  display: block; width: 100%; height: 560px;
  border: 1px solid var(--border); border-radius: 6px; background: #fff;
}
@media (max-width: 520px) {
  .viz-row { grid-template-columns: 1fr; }
  .viz iframe { height: 420px; }
}
"""


def index_html(experiments: list[dict]) -> str:
    # Only completed experiments get a card.
    experiments = [e for e in experiments if e["completed"]]

    cards = []
    for e in experiments:
        exp_id = e["id"]

        if e["has_samples"]:
            img = f'<img class="card-img" src="experiment/{exp_id}/samples.png" alt="{e["name"]}" loading="lazy">'
        elif e["has_preview"]:
            img = f'<img class="card-img" src="experiment/{exp_id}/preview.png" alt="{e["name"]}" loading="lazy">'
        else:
            img = '<div class="card-img-placeholder">No preview</div>'

        badges = []
        if e["has_pca"]:
            badges.append(f'<a class="badge" href="experiment/{exp_id}/pca.html">PCA 3D</a>')
        if e["has_umap"]:
            badges.append(f'<a class="badge" href="experiment/{exp_id}/umap.html">UMAP 3D</a>')
        badges.append(f'<a class="badge" href="experiment/{exp_id}/">Details</a>')
        links = f'<div class="links">{"".join(badges)}</div>'

        desc_html = f'\n        <div class="desc">{e["description"]}</div>' if e["description"] else ""
        links_html = f'\n        {links}'

        cards.append(
            f'    <div class="card">\n'
            f'      <a class="card-main" href="experiment/{exp_id}/">\n'
            f'      {img}\n'
            f'      <div class="card-body">\n'
            f'        <h3><span class="id">#{exp_id}</span> {e["name"]}</h3>'
            f'{desc_html}\n'
            f'      </div>\n'
            f'      </a>\n'
            f'      <div class="card-body card-links">{links_html}\n'
            f'      </div>\n'
            f'    </div>\n'
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
  <div class="stats">{len(experiments)} experiments</div>
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

    # Both projections, side by side
    views = []
    for has, fname, label in ((e["has_pca"], "pca.html", "PCA"), (e["has_umap"], "umap.html", "UMAP")):
        if has:
            views.append(
                f'      <figure class="viz">\n'
                f'        <figcaption>{label} <a href="{fname}">Full screen &#8599;</a></figcaption>\n'
                f'        <iframe src="{fname}" title="{label} 3D projection" loading="lazy"></iframe>\n'
                f'      </figure>'
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
    <div class="viz-row">
{chr(10).join(views) if views else "      <p>No visualizations available yet.</p>"}
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
