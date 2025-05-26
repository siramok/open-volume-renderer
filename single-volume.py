import base64
import gc
import json
import math
import os
import shutil
import subprocess
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


def transform_point(M, p):
    """Transform a point by a 4x4 matrix."""
    p_hom = np.ones(4, dtype=np.float32)
    p_hom[:3] = p
    transformed = M @ p_hom
    return transformed[:3] / transformed[3]


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    mvt_matrix: np.ndarray
    proj_matrix: np.ndarray
    center: np.ndarray
    # fields to hold the mesh→world transform and its inverse
    mesh_transform: np.ndarray
    mesh_transform_inv: np.ndarray
    # fields for original space camera parameters
    cam_pos_orig: np.ndarray
    cam_at_orig: np.ndarray
    cam_up_orig: np.ndarray
    # reference to original camera params from unmodified mesh
    reference_position: np.ndarray
    reference_focal_point: np.ndarray
    reference_up: np.ndarray


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def ensure_reference_cameras(filepath, resolution):
    """
    Ensure that reference camera data exists. If not, generate it by running camera_calibrator.py
    in a separate process.

    Returns:
        List of reference camera data dictionaries
    """
    reference_file = os.path.join(os.path.dirname(filepath), "reference_cameras.json")
    print(reference_file)

    if not os.path.exists(reference_file):
        print("Generating reference camera data...")
        calibrator_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "camera_calibrator.py"
        )
        subprocess.run(
            ["python", calibrator_path, filepath, str(resolution)],
            check=True,
        )

    with open(reference_file, "r") as f:
        reference_cameras = json.load(f)

    print(f"Loaded {len(reference_cameras)} reference camera positions")
    return reference_cameras


class ReferenceCamera(NamedTuple):
    """Store reference camera parameters with unmodified mesh."""

    azimuth: float
    elevation: float
    position: np.ndarray
    focal_point: np.ndarray
    view_up: np.ndarray
    view_angle: float
    clipping_range: tuple


def print_camera_info(camera, label="Camera Info"):
    """Print detailed camera information for debugging."""
    print(f"\n=== {label} ===")
    print(f"Position:     {camera.position}")
    print(f"Focal Point:  {camera.focal_point}")
    print(f"Up Vector:    {camera.up}")
    print(f"View Angle:   {camera.view_angle}")
    print(f"Azimuth:      {camera.azimuth}")
    print(f"Elevation:    {camera.elevation}")
    print(f"Distance:     {camera.distance}")
    print(f"Clipping:     {camera.clipping_range}")

    # Calculate and print direction vector
    direction = np.array(camera.focal_point) - np.array(camera.position)
    norm_direction = direction / np.linalg.norm(direction)
    print(f"Direction:    {norm_direction}")

    # Calculate and print right vector (cross product of direction and up)
    up_vector = np.array(camera.up)
    right_vector = np.cross(norm_direction, up_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)
    print(f"Right Vector: {right_vector}")
    print("=" * (len(label) + 8))


def create_camera_indicators(plotter, camera, scale=1.0):
    """Create visual indicators for the camera orientation."""
    # Camera position
    pos = np.array(camera.position)

    # Calculate normalized direction vectors
    focal = np.array(camera.focal_point)
    direction = focal - pos
    dir_length = np.linalg.norm(direction)
    direction = direction / dir_length if dir_length > 0 else np.array([0, 0, 1])

    up = np.array(camera.up)
    right = np.cross(direction, up)
    right = right / np.linalg.norm(right)

    # Recalculate up to ensure orthogonality
    up = np.cross(right, direction)
    up = up / np.linalg.norm(up)

    # Create arrows for each direction
    arrow_length = scale
    plotter.add_arrows(
        np.array([pos]),
        np.array([direction * arrow_length]),
        color="red",
        label="Direction",
    )
    plotter.add_arrows(
        np.array([pos]), np.array([up * arrow_length * 0.5]), color="green", label="Up"
    )
    plotter.add_arrows(
        np.array([pos]),
        np.array([right * arrow_length * 0.5]),
        color="blue",
        label="Right",
    )

    # Add a small sphere at the camera position
    sphere = pv.Sphere(radius=arrow_length * 0.1, center=pos)
    plotter.add_mesh(sphere, color="yellow", label="Camera")

    # Add a small sphere at the origin (0,0,0) for reference
    origin_sphere = pv.Sphere(radius=arrow_length * 0.05, center=[0, 0, 0])
    plotter.add_mesh(origin_sphere, color="white", label="Origin (0,0,0)")


def center_volume_at_origin(mesh):
    """Center the volume at the origin (0,0,0)."""
    # Calculate the current center of the volume in world coordinates
    extents = np.array(mesh.bounds)
    current_center = np.array(
        [
            (extents[0] + extents[1]) / 2,  # X center
            (extents[2] + extents[3]) / 2,  # Y center
            (extents[4] + extents[5]) / 2,  # Z center
        ]
    )

    # Calculate the offset needed to move to origin
    offset_to_origin = -current_center

    print(f"Current volume center: {current_center}")
    print(f"Applying offset to center at origin: {offset_to_origin}")

    # Apply the offset to the mesh origin
    mesh.origin = tuple(np.array(mesh.origin) + offset_to_origin)

    return mesh


def test_camera(
    filepath,
    spacing,
    resolution,
    position=None,
    focal_point=None,
    up_vector=None,
    show_ui=True,
    center_at_origin=False,
):
    """
    Test specific camera parameters and visualize the volume.

    Args:
        filepath: Path to the raw volume data file
        spacing: Volume spacing tuple (x, y, z)
        resolution: Image resolution for rendering
        position: Camera position as (x, y, z) tuple, or None for default
        focal_point: Camera focal point as (x, y, z) tuple, or None for default
        up_vector: Camera up vector as (x, y, z) tuple, or None for default
        show_ui: Whether to show the interactive UI (False for headless rendering)
        center_at_origin: If True, center the volume at (0,0,0)

    Returns:
        The reference camera information
    """
    filename = os.path.basename(filepath)[:-4]
    tokens = filename.split("_")
    dimensions = tuple(map(int, tokens[1].split("x")))
    data_type = np.dtype(tokens[2])

    # Setup plotter
    pl = pv.Plotter(off_screen=not show_ui)
    pl.window_size = [resolution, resolution]

    # Load the raw data without scaling or translating
    values = np.fromfile(filepath, dtype=data_type)
    mesh = pv.ImageData(dimensions=dimensions, spacing=spacing)
    mesh.point_data["value"] = values

    # Calculate volume extents before any transformations
    points_min = np.array(mesh.origin)
    points_max = points_min + (np.array(mesh.dimensions) - 1) * np.array(mesh.spacing)
    max_extent = np.max(points_max - points_min)

    print(f"Volume dimensions: {dimensions}")
    print(f"Volume extents: Min: {points_min}, Max: {points_max}")
    print(f"Volume max extent: {max_extent}")

    # Scale the volume if it's too large or too small
    # This is similar to what's done in volume.py
    scale_factor = 1.0 / max_extent
    mesh.spacing = tuple(np.array(mesh.spacing) * scale_factor)

    print(f"Applied scale factor: {scale_factor}")
    print(f"New spacing: {mesh.spacing}")

    # Center at origin if requested
    if center_at_origin:
        mesh = center_volume_at_origin(mesh)
        # When centered at origin, set the focal point to exactly (0,0,0)
        if focal_point is None and position is not None:
            focal_point = (0.0, 0.0, 0.0)

    # Setup the camera
    camera = pl.camera
    camera.clipping_range = (0.001, 10000.0)  # Use a large clipping range for safety

    # Add the volume without additional transformations
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

    # Initialize view to XY plane as default
    pl.view_xy(render=False)

    # Print initial camera settings
    print_camera_info(camera, "Initial Camera Settings")

    # Apply custom camera parameters if provided
    if position is not None:
        # Store original settings to later print what changed
        original_position = np.array(camera.position)
        original_focal_point = np.array(camera.focal_point)
        original_up = np.array(camera.up)

        # Apply new settings
        camera.position = position

        # If focal_point is provided, use it, otherwise keep existing focal point
        if focal_point is not None:
            camera.focal_point = focal_point

        # If up_vector is provided, use it, otherwise keep existing up vector
        if up_vector is not None:
            camera.up = up_vector

        # Render to update camera matrices
        pl.render()

        # Print updated camera settings
        print_camera_info(camera, "Updated Camera Settings")

        # Print what changed
        print("\n=== Changes Applied ===")
        print(f"Position:    {original_position} -> {np.array(camera.position)}")
        print(f"Focal Point: {original_focal_point} -> {np.array(camera.focal_point)}")
        print(f"Up Vector:   {original_up} -> {np.array(camera.up)}")
        print("=" * 20)

    # Create visual indicators for the camera
    create_camera_indicators(pl, camera, scale=max_extent * scale_factor * 0.2)

    # Add world coordinate axes for reference
    pl.add_axes(interactive=True, line_width=2)

    # Render the scene
    pl.render()

    # Get camera information after the rendering is complete
    reference_camera = ReferenceCamera(
        azimuth=camera.azimuth,
        elevation=camera.elevation,
        position=np.array(camera.position),
        focal_point=np.array(camera.focal_point),
        view_up=np.array(camera.up, dtype=np.float32),
        view_angle=camera.view_angle,
        clipping_range=camera.clipping_range,
    )

    # Save a screenshot of the current view if not in interactive mode
    if not show_ui:
        output_dir = "camera_debug_renders"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(output_dir, f"camera_debug_{timestamp}.png")
        pl.screenshot(screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")
    else:
        # If interactive, start the UI
        pl.show()

    return reference_camera


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(
        description="Debug tool for testing camera positions in volume rendering"
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="volume-data/chameleon/chameleon_1024x1024x1080_uint16.raw",
        help="Path to the volume data file",
    )
    parser.add_argument(
        "--resolution", type=int, default=768, help="Resolution for rendering"
    )
    parser.add_argument(
        "--position", type=float, nargs=3, help="Camera position (x y z)"
    )
    parser.add_argument(
        "--target", type=float, nargs=3, help="Camera look-at point (x y z)"
    )
    parser.add_argument("--up", type=float, nargs=3, help="Camera up vector (x y z)")
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no UI)"
    )
    parser.add_argument("--spacing", type=float, nargs=3, help="Volume spacing (x y z)")
    parser.add_argument(
        "--center-at-origin",
        action="store_true",
        help="Center the volume at the origin (0,0,0)",
    )

    args = parser.parse_args()

    # Set default spacing based on dataset if not specified
    if args.spacing:
        spacing = tuple(args.spacing)
    elif args.filepath.endswith("chameleon_1024x1024x1080_uint16.raw"):
        spacing = (0.09228515625, 0.09228515625, 0.105)
    elif args.filepath.endswith("skull_256x256x256_uint8.raw"):
        spacing = (1.0, 1.0, 1.0)
    else:
        spacing = (1.0, 1.0, 1.0)

    # Check if all camera params are provided or none
    camera_params_count = sum(
        1 for param in [args.position, args.target, args.up] if param is not None
    )
    if 0 < camera_params_count < 3 and not (args.position and args.center_at_origin):
        parser.error(
            "If specifying camera parameters, must provide all three: --position, --target, and --up"
        )

    position = tuple(args.position) if args.position else None
    target = tuple(args.target) if args.target else None
    up = tuple(args.up) if args.up else None

    print(f"Volume: {args.filepath}")
    print(f"Spacing: {spacing}")
    print(f"Resolution: {args.resolution}")
    print(f"Center at Origin: {args.center_at_origin}")

    if position:
        print(f"Camera Position: {position}")
    if target:
        print(f"Camera Target: {target}")
    elif args.center_at_origin and position:
        print(f"Camera Target: (0.0, 0.0, 0.0) (using origin as focal point)")
    if up:
        print(f"Camera Up Vector: {up}")

    # Run the camera test
    camera_info = test_camera(
        args.filepath,
        spacing,
        args.resolution,
        position=position,
        focal_point=target,
        up_vector=up,
        show_ui=not args.headless,
        center_at_origin=args.center_at_origin,
    )

    print("\n=== Final Camera Configuration ===")
    print(f"Position:     {camera_info.position}")
    print(f"Focal Point:  {camera_info.focal_point}")
    print(f"Up Vector:    {camera_info.view_up}")
    print(f"View Angle:   {camera_info.view_angle}")
    print(f"Clipping:     {camera_info.clipping_range}")
    print("=" * 35)
