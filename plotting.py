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
        img = img.resize((128, 128), Image.LANCZOS)
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
    plot_html = fig.to_html(include_plotlyjs='cdn', div_id=div_id)
    full_html = f"""
<div id="container_{div_id}">
    {plot_html}
    <div id="imgdiv_{div_id}" style="margin-top:20px;text-align:center;min-height:200px;">
        <p style="color:#666">Hover over a point to see its image.</p>
    </div>
</div>
<script>
var p = document.getElementById('{div_id}');
function showImg_{div_id.replace('-','_')}(data) {{
    var pt = data.points[0];
    var cd = pt.customdata;
    document.getElementById('imgdiv_{div_id}').innerHTML =
        '<div style="display:inline-block;padding:20px;background:#f0f0f0;border-radius:12px;">' +
        '<h3>Sample #' + pt.pointNumber + '</h3>' +
        '<img src="' + cd[3] + '" style="border:2px solid #444;border-radius:6px;">' +
        '<div style="margin-top:8px;font-size:13px;">' +
        '{axis_labels[0]}: ' + Number(cd[0]).toFixed(4) + '  |  ' +
        '{axis_labels[1]}: ' + Number(cd[1]).toFixed(4) + '  |  ' +
        '{axis_labels[2]}: ' + Number(cd[2]).toFixed(4) + '</div></div>';
}}
p.on('plotly_hover',  showImg_{div_id.replace('-','_')});
p.on('plotly_click',  showImg_{div_id.replace('-','_')});
</script>
"""
    return fig, full_html
