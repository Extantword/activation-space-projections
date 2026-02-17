"""Shared plotting utilities for activation-space-projections CLI scripts."""

import io
import base64
import numpy as np
from PIL import Image
import plotly.graph_objects as go


def build_3d_figure(coords_3d, all_imgs, method_name, axis_labels,
                    latent_dim, title_extra='', img_size=64):
    """
    Build an interactive Plotly 3D scatter with hover-image display.
    Colors are mapped to the x-axis values using Viridis.
    Returns (plotly Figure, full HTML string).
    """

    def _to_b64(arr):
        img = Image.fromarray((arr.reshape(img_size, img_size) * 255).astype(np.uint8))
        img = img.resize((200, 200), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    b64 = [_to_b64(im) for im in all_imgs]

    x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]

    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4, opacity=0.7,
            color=x,
            colorscale='Viridis',
            colorbar=dict(title=axis_labels[0]),
            line=dict(width=0.3, color='white')
        ),
        customdata=np.column_stack([x, y, z, b64]),
        hovertemplate=(
            '<b>Sample #%{pointNumber}</b><br>'
            f'<b>{axis_labels[0]}:</b>' + ' %{customdata[0]}<br>'
            f'<b>{axis_labels[1]}:</b>' + ' %{customdata[1]}<br>'
            f'<b>{axis_labels[2]}:</b>' + ' %{customdata[2]}<br>'
            '<extra></extra>'
        ),
        showlegend=False
    ))

    fig.update_layout(
        title=dict(
            text=f'3D {method_name} of Latent Space '
                 f'({latent_dim}D -> 3D){title_extra}',
            x=0.5, xanchor='center', font=dict(size=18)
        ),
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='#f8f9fa'
        ),
        width=1000, height=800,
        template='plotly_white'
    )

    div_id = f'plot3d_{method_name.lower().replace(" ", "_")}'
    func_id = div_id.replace('-', '_')
    plot_div = fig.to_html(include_plotlyjs='cdn', div_id=div_id,
                           full_html=False)
    full_html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D {method_name} — Activation Space Projections</title>
<style>
  body {{ margin: 0; background: #0d1117; color: #e6edf3;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
</style>
</head>
<body>
<div id="container_{div_id}" style="display:flex;align-items:flex-start;gap:24px;padding:10px;">
    <div style="flex:1;min-width:0;">{plot_div}</div>
    <div id="imgdiv_{div_id}" style="width:280px;min-height:300px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <p style="color:#666;text-align:center;">Hover over a point to see its image.</p>
    </div>
</div>
<script>
var p = document.getElementById('{div_id}');
function showImg_{func_id}(data) {{
    var pt = data.points[0];
    var cd = pt.customdata;
    document.getElementById('imgdiv_{div_id}').innerHTML =
        '<div style="padding:20px;background:#f0f0f0;border-radius:12px;text-align:center;">' +
        '<h3 style="margin:0 0 10px 0;color:#333;">Sample #' + pt.pointNumber + '</h3>' +
        '<img src="' + cd[3] + '" width="200" height="200" style="border:2px solid #444;border-radius:6px;display:block;margin:0 auto;">' +
        '<div style="margin-top:10px;font-size:12px;color:#555;line-height:1.6;">' +
        '{axis_labels[0]}: ' + Number(cd[0]).toFixed(4) + '<br>' +
        '{axis_labels[1]}: ' + Number(cd[1]).toFixed(4) + '<br>' +
        '{axis_labels[2]}: ' + Number(cd[2]).toFixed(4) + '</div></div>';
}}
p.on('plotly_hover',  showImg_{func_id});
p.on('plotly_click',  showImg_{func_id});
</script>
</body>
</html>
"""
    return fig, full_html
