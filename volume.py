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


def orbitVolume(filepath, spacing, resolution):
    reference_data_path = os.path.join(
        os.path.dirname(filepath), "reference_cameras.json"
    )
    if os.path.exists(reference_data_path):
        os.remove(reference_data_path)
    camera_data_path = os.path.join(os.path.dirname(filepath), "camera_data.json")
    if os.path.exists(camera_data_path):
        os.remove(camera_data_path)

    # Ensure we have reference camera data from unmodified mesh
    reference_cameras = ensure_reference_cameras(filepath, resolution)

    # Directory setup
    image_dir = os.path.join(os.path.dirname(filepath), "images")
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    # Window setup
    width = resolution
    height = resolution
    ratio = width / height
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [width, height]

    # Parse the filename
    filename = os.path.basename(filepath)[:-4]
    tokens = filename.split("_")
    dimensions = tuple(map(int, tokens[1].split("x")))
    data_type = np.dtype(tokens[2])

    # Load the raw data
    values = np.fromfile(filepath, dtype=data_type)
    mesh = pv.ImageData(dimensions=dimensions, spacing=spacing)
    mesh.point_data["value"] = values

    # --- compute the uniform scale factor and offset exactly once ---
    # original extents in world units
    points_min = np.array(mesh.origin)
    points_max = points_min + (np.array(mesh.dimensions) - 1) * np.array(mesh.spacing)
    max_extent = np.max(points_max - points_min)
    scale_factor = 1.0 / max_extent
    print(f"scale_factor: {scale_factor}")

    # workaround for camera-too-close bug
    offset = list(pl.camera.focal_point)
    offset[2] -= 3
    offset = [-x for x in offset]
    print(f"offset: {offset}")

    # apply to mesh: uniform scale via spacing, then translate via origin
    mesh.spacing = tuple(np.array(mesh.spacing) * scale_factor)
    mesh.origin = tuple(offset)

    # build the 4×4 that maps original_data_coords → transformed/world coords
    mesh_spacing = np.array(mesh.spacing, dtype=np.float32)
    mesh_origin = np.array(mesh.origin, dtype=np.float32)
    print(f"mesh_origin: {mesh_origin}")
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = np.diag(mesh_spacing)  # scale
    M[:3, 3] = mesh_origin  # translate

    # invert once for later use
    invM = np.linalg.inv(M)
    # -------------------------------------------------------------------

    print(f"mesh_origin: {mesh_origin}")

    cam_infos = []
    image_counter = 0

    pl.background_color = "black"
    camera = pl.camera
    camera.clipping_range = (0.001, 1000.0)
    opacity_unit_distance = 1.0 / 128.0

    # Controls the camera orbit and capture frequency
    # azimuth_steps = 18
    # elevation_steps = 7
    azimuth_steps = 2
    elevation_steps = 2
    azimuth_range = np.linspace(0, 360, azimuth_steps, endpoint=False)
    elevation_range = np.linspace(-35, 35, elevation_steps, endpoint=True)

    pl.add_volume(
        mesh,
        name="volume_actor",
        show_scalar_bar=False,
        scalars="value",
        cmap=plt.get_cmap("rainbow"),
        blending="composite",
        shade=False,
        diffuse=0.0,
        specular=0.0,
        specular_power=0.0,
        ambient=1.0,
        culling=True,
        pickable=False,
        render=False,
        opacity_unit_distance=opacity_unit_distance,
    )
    pl.view_xy(render=False)

    ref_index = 0  # Index to track our position in the reference cameras list

    # Prepare the simplified camera data for JSON export
    simplified_camera_data = []

    for elevation in elevation_range:
        for azimuth in azimuth_range:
            # Get the corresponding reference camera
            ref_camera = reference_cameras[ref_index]
            ref_position = np.array(ref_camera["position"], dtype=np.float32)
            ref_focal_point = np.array(ref_camera["focal_point"], dtype=np.float32)
            ref_up = np.array(ref_camera["view_up"], dtype=np.float32)

            image_counter += 1
            ref_index += 1

            # update camera spherical angles
            camera.elevation = elevation
            camera.azimuth = azimuth

            # render & snapshot
            pl.render()
            img = pl.screenshot(None, return_img=True)

            image_name = f"{image_counter:05d}.png"
            image_path = os.path.join(image_dir, image_name)
            plt.imsave(image_path, img)

            # grab VTK matrices and invert the modelview to get world→camera
            mvt_matrix = np.linalg.inv(
                arrayFromVTKMatrix(camera.GetModelViewTransformMatrix())
            )
            # correct handedness
            mvt_matrix[:3, 1:3] *= -1

            # extract R, T
            R = mvt_matrix[:3, :3].T
            T = mvt_matrix[:3, 3]

            # FOVs
            FovY = np.radians(camera.view_angle)
            FovX = focal2fov(fov2focal(FovY, height), width)

            # projection matrix
            proj_matrix = arrayFromVTKMatrix(
                camera.GetCompositeProjectionTransformMatrix(ratio, 0.001, 1000.0)
            )
            proj_matrix[1:3, :] *= -1

            # fix sign if flipped
            if camera.position[1] < 0:
                mvt_matrix[2, 1] *= -1
            mvt_matrix[2, 3] = abs(mvt_matrix[2, 3])

            # camera center in world space
            center = mvt_matrix[:3, 3]

            # Calculate original space camera parameters by inverting the mesh transformation
            # Get camera position and focal point (at) in transformed space
            cam_pos = np.array(camera.position, dtype=np.float32)
            cam_at = np.array(camera.focal_point, dtype=np.float32)
            cam_up = np.array(camera.up, dtype=np.float32)

            # Transform back to original space using the inverse mesh transformation
            cam_pos_orig = transform_point(invM, cam_pos)
            cam_at_orig = transform_point(invM, cam_at)

            # For up vector, we need to transform it as a direction vector (not a point)
            # Extract just the rotation/scaling part of the inverse transform
            invM_rot = invM[:3, :3]
            cam_up_orig = invM_rot @ cam_up
            # Normalize the up vector
            cam_up_orig = cam_up_orig / np.linalg.norm(cam_up_orig)

            # Add just the essential information to the simplified camera data
            simplified_camera_data.append(
                {
                    "uid": image_counter,
                    "image_name": image_name,
                    "width": width,
                    "height": height,
                    "reference_position": ref_position.tolist(),
                    "reference_focal_point": ref_focal_point.tolist(),
                    "reference_up": ref_up.tolist(),
                    "FovY": float(FovY),
                    "FovX": float(FovX),
                }
            )

            # store everything in our CameraInfo for internal use
            cam_infos.append(
                CameraInfo(
                    uid=image_counter,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    depth_params=None,
                    image_path=image_path,
                    image_name=image_name,
                    depth_path="",
                    width=width,
                    height=height,
                    is_test=False,
                    mvt_matrix=mvt_matrix,
                    proj_matrix=proj_matrix,
                    center=center,
                    mesh_transform=M,
                    mesh_transform_inv=invM,
                    cam_pos_orig=cam_pos_orig,
                    cam_at_orig=cam_at_orig,
                    cam_up_orig=cam_up_orig,
                    reference_position=ref_position,
                    reference_focal_point=ref_focal_point,
                    reference_up=ref_up,
                )
            )

    # Write the simplified camera data to JSON
    with open(camera_data_path, "w") as f:
        json.dump(simplified_camera_data, f, indent=2)

    # Copy camera_data.json to /home/siramok/code/open-volume-renderer
    target_path = "./camera_data.json"
    shutil.copy(camera_data_path, target_path)
    print(f"Copied simplified camera_data.json to {target_path}")

    gc.collect()
    pl.close()
    return cam_infos


if __name__ == "__main__":
    # filepath = "volume-data/skull/skull_256x256x256_uint8.raw"
    filepath = "volume-data/chameleon/chameleon_1024x1024x1080_uint16.raw"
    # spacing = (1.0, 1.0, 1.0)
    spacing = (0.09228515625, 0.09228515625, 0.105)
    resolution = 768
    infos = orbitVolume(filepath, spacing, resolution)
    print(f"Captured {len(infos)} camera poses.")
