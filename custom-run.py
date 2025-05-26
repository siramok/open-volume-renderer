#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def rotate_point(point, azimuth, elevation):
    """
    Rotate a point around the origin by azimuth and elevation angles.

    Args:
        point: [x, y, z] point coordinates
        azimuth: Angle around y-axis in degrees
        elevation: Angle around x-axis in degrees

    Returns:
        Rotated point coordinates
    """
    # Convert angles to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    # Convert point to numpy array
    p = np.array(point)

    # Create rotation matrices
    # Rotate around y-axis (azimuth)
    rot_y = np.array(
        [
            [np.cos(azimuth_rad), 0, np.sin(azimuth_rad)],
            [0, 1, 0],
            [-np.sin(azimuth_rad), 0, np.cos(azimuth_rad)],
        ]
    )

    # Rotate around x-axis (elevation)
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(elevation_rad), -np.sin(elevation_rad)],
            [0, np.sin(elevation_rad), np.cos(elevation_rad)],
        ]
    )

    # Apply rotations
    p_rotated = rot_y @ rot_x @ p

    return p_rotated.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Render images from orbital camera positions"
    )
    parser.add_argument(
        "--neural-volume", type=str, required=True, help="Path to neural volume data"
    )
    parser.add_argument(
        "--tfn", type=str, required=True, help="Path to transfer function config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="renders",
        help="Directory to save rendered images",
    )
    parser.add_argument(
        "--vnr-cmd-path",
        type=str,
        default="./build/vnr_cmd_render",
        help="Path to the vnr_cmd_render executable",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to render per camera",
    )
    parser.add_argument(
        "--base-distance",
        type=float,
        default=100.0,
        help="Base distance from camera to origin",
    )
    parser.add_argument(
        "--azimuth-steps",
        type=int,
        default=8,
        help="Number of steps around the azimuth (horizontal orbit)",
    )
    parser.add_argument(
        "--elevation-steps",
        type=int,
        default=4,
        help="Number of steps in elevation (vertical orbit)",
    )
    parser.add_argument(
        "--elevation-range",
        type=str,
        default="-45,45",
        help="Min and max elevation angles in degrees (e.g. '-45,45')",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Parse elevation range
    elev_min, elev_max = map(float, args.elevation_range.split(","))

    # Generate orbit camera positions
    base_point = [0.0, 0.0, -args.base_distance]  # Starting at negative z-axis
    up_vector = [0.0, 1.0, 0.0]  # Y-axis is up
    focal_point = [0.0, 0.0, 0.0]  # Looking at origin

    azimuth_steps = 2
    elevation_steps = 2
    azimuth_range = np.linspace(0, 360, azimuth_steps, endpoint=False)
    elevation_range = np.linspace(-35, 35, elevation_steps, endpoint=True)

    # Generate all camera positions
    camera_count = 0
    camera_positions = []

    for elevation in elevation_range:
        for azimuth in azimuth_range:
            camera_count += 1
            # Rotate the base point to the new position
            position = rotate_point(base_point, azimuth, elevation)

            # Also rotate the up vector for consistent camera orientation
            rotated_up = rotate_point(
                up_vector, azimuth, 0
            )  # Don't apply elevation to up vector

            camera_positions.append(
                {
                    "uid": camera_count,
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "position": position,
                    "up_vector": rotated_up,
                }
            )

    print(f"Generated {len(camera_positions)} camera positions")

    # Save camera positions to file for reference
    with open(os.path.join(args.output_dir, "camera_positions.json"), "w") as f:
        json.dump(camera_positions, f, indent=2)

    # Process each camera
    for camera in camera_positions:
        uid = camera["uid"]
        position = camera["position"]
        up_vector = camera["up_vector"]
        azimuth = camera["azimuth"]
        elevation = camera["elevation"]

        position_str = f"{position[0]},{position[1]},{position[2]}"
        focal_point_str = "0,0,0"  # Always looking at origin
        up_vector_str = f"{up_vector[0]},{up_vector[1]},{up_vector[2]}"

        output_path = os.path.join(args.output_dir, f"{uid:05d}")

        cmd = [
            args.vnr_cmd_path,
            "--neural-volume",
            args.neural_volume,
            "--tfn",
            args.tfn,
            "--num-frames",
            str(args.num_frames),
            "--camera-from",
            position_str,
            "--camera-at",
            focal_point_str,
            "--camera-up",
            up_vector_str,
            "--exp",
            output_path,
            "--rendering-mode",
            str(5),
        ]

        print(
            f"Rendering camera {uid}: azimuth={azimuth:.1f}° elevation={elevation:.1f}°"
        )
        try:
            subprocess.run(cmd, check=True)
            print(f"Saved render to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error rendering camera {uid}: {e}")
            continue


if __name__ == "__main__":
    main()
