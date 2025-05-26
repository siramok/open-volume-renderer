#!/usr/bin/env python3
import json
import math
import os
import sys
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from vtk import vtkMatrix3x3, vtkMatrix4x4


def arrayFromVTKMatrix(vmatrix):
    """Convert a vtkMatrix3x3 or vtkMatrix4x4 into a NumPy array."""
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize, dtype=np.float32)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class ReferenceCamera(NamedTuple):
    """Store reference camera parameters with unmodified mesh."""

    azimuth: float
    elevation: float
    position: np.ndarray
    focal_point: np.ndarray
    view_up: np.ndarray
    view_angle: float
    clipping_range: tuple


def capture_reference_cameras(filepath, spacing, resolution):
    """
    Capture reference camera parameters using an unmodified mesh.

    This function loads the volume data without applying scaling or
    translation transformations, then captures the camera parameters
    at each position in the orbit.

    Args:
        filepath: Path to the raw volume data file
        spacing: Original spacing tuple for the volume
        resolution: Resolution for the output images

    Returns:
        List of ReferenceCamera objects containing camera parameters at each position
    """
    # Parse the filename
    filename = os.path.basename(filepath)[:-4]
    tokens = filename.split("_")
    dimensions = tuple(map(int, tokens[1].split("x")))
    data_type = np.dtype(tokens[2])

    # Setup plotter with original mesh
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [resolution, resolution]

    # Load the raw data without scaling or translating
    values = np.fromfile(filepath, dtype=data_type)
    mesh = pv.ImageData(dimensions=dimensions, spacing=spacing)
    mesh.point_data["value"] = values

    # Setup the camera
    camera = pl.camera
    camera.clipping_range = (0.001, 10000.0)  # Use a large clipping range for safety

    # Controls the camera orbit
    # azimuth_steps = 18
    # elevation_steps = 7
    azimuth_steps = 2
    elevation_steps = 2
    azimuth_range = np.linspace(0, 360, azimuth_steps, endpoint=False)
    elevation_range = np.linspace(-35, 35, elevation_steps, endpoint=True)

    # Add the volume without scaling/transforming
    pl.add_volume(
        mesh,
        show_scalar_bar=False,
        scalars="value",
        cmap=plt.get_cmap("rainbow"),
        opacity="linear",
        blending="composite",
        shade=False,
        diffuse=0.0,
        specular=0.0,
        specular_power=0.0,
        ambient=1.0,
        culling=True,
        pickable=False,
        render=False,
    )

    # Initialize view to XY plane
    pl.view_xy(render=False)

    # Store camera information for each position
    reference_cameras = []

    for elevation in elevation_range:
        for azimuth in azimuth_range:
            # Update camera position
            camera.elevation = elevation
            camera.azimuth = azimuth

            # Render to update camera matrices
            pl.render()

            # Get original camera position and focal point
            original_position = np.array(camera.position)
            original_focal_point = np.array(camera.focal_point)

            # Apply the same offset that volume.py would apply
            # This ensures the cameras are in the same relative position
            # to the centered volume
            position = original_position
            focal_point = original_focal_point

            # Store camera parameters
            reference_cameras.append(
                ReferenceCamera(
                    azimuth=azimuth,
                    elevation=elevation,
                    position=position,
                    focal_point=focal_point,
                    view_up=np.array(camera.up, dtype=np.float32),
                    view_angle=camera.view_angle,
                    clipping_range=camera.clipping_range,
                )
            )

    # Close the plotter
    pl.close()

    # Save the reference camera data
    output_path = os.path.join(os.path.dirname(filepath), "reference_cameras.json")
    with open(output_path, "w") as f:
        json.dump(
            [
                {
                    "azimuth": cam.azimuth,
                    "elevation": cam.elevation,
                    "position": cam.position.tolist(),
                    "focal_point": cam.focal_point.tolist(),
                    "view_up": cam.view_up.tolist(),
                    "view_angle": cam.view_angle,
                    "clipping_range": list(cam.clipping_range),
                }
                for cam in reference_cameras
            ],
            f,
            indent=2,
        )

    print(f"Saved {len(reference_cameras)} reference camera positions to {output_path}")
    return reference_cameras


if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python camera_calibrator.py <filepath> <resolution>")
        print(
            "Example: python camera_calibrator.py volume-data/chameleon/chameleon_1024x1024x1080_uint16.raw 256"
        )
        sys.exit(1)

    filepath = sys.argv[1]
    resolution = int(sys.argv[2])

    if filepath.endswith("chameleon_1024x1024x1080_uint16.raw"):
        spacing = (0.09228515625, 0.09228515625, 0.105)
    elif filepath.endswith("skull_256x256x256_uint8.raw"):
        spacing = (1.0, 1.0, 1.0)
    else:
        print("Unknown dataset, using default spacing (1.0, 1.0, 1.0)")
        spacing = (1.0, 1.0, 1.0)

    print(f"Loading {filepath} with spacing {spacing} and resolution {resolution}")
    capture_reference_cameras(filepath, spacing, resolution)
