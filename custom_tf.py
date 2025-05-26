import base64
import json

import numpy as np
from matplotlib import cm


def generate_tf_json(
    filename, cmap_name="rainbow", resolution=1024, scalar_min=0.0, scalar_max=1.0
):
    # Generate alpha array linearly increasing from 0 to 1
    alpha_array = np.linspace(0, 1, resolution, dtype=np.float32)
    alpha_encoded = base64.b64encode(alpha_array.tobytes()).decode("ascii")

    # Get colormap from Matplotlib
    cmap = cm.get_cmap(cmap_name).resampled(resolution)
    positions = np.linspace(0, 1, num=resolution)
    colors = [cmap(p)[:3] for p in positions]

    # Construct JSON structure
    tf_json = {
        "version": "VIDI3D",
        "view": {
            "volume": {
                "scalarMappingRange": {"minimum": scalar_min, "maximum": scalar_max},
                "transferFunction": {
                    "alphaArray": {"data": alpha_encoded, "encoding": "BASE64"},
                    "colorControls": [
                        {"color": {"r": r, "g": g, "b": b}, "position": float(pos)}
                        for (r, g, b), pos in zip(colors, positions)
                    ],
                    "resolution": resolution,
                },
            }
        },
    }

    with open(filename, "w") as f:
        json.dump(tf_json, f, indent=4)

    print(f"Saved TF JSON to {filename}")


# Example usage:
generate_tf_json(
    "custom_tf.json",
    cmap_name="plasma",
    resolution=1024,
    scalar_min=0.0,
    scalar_max=1.0,
)
