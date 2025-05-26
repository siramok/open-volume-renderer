#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Render images from camera positions in JSON file"
    )
    parser.add_argument(
        "--camera-data",
        type=str,
        default="camera_data.json",
        help="Path to camera data JSON file",
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
        "--camera-range",
        type=str,
        default=None,
        help='Range of cameras to render (e.g., "0-10" or "5,10,15")',
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Validate the camera data file exists
    if not os.path.isfile(args.camera_data):
        raise FileNotFoundError(f"Camera data file not found: {args.camera_data}")

    # Load all cameras
    with open(args.camera_data, "r") as f:
        cameras = json.load(f)

    # Handle camera range argument if provided
    selected_cameras = cameras
    if args.camera_range:
        indices = []
        # Parse ranges like "0-10" or comma-separated values like "5,10,15"
        if "-" in args.camera_range:
            start, end = map(int, args.camera_range.split("-"))
            indices = list(range(start, end + 1))
        else:
            indices = [int(i) for i in args.camera_range.split(",")]

        selected_cameras = [cam for cam in cameras if cam["uid"] in indices]
        print(
            f"Selected {len(selected_cameras)} cameras from range: {args.camera_range}"
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each selected camera
    for camera in selected_cameras:
        uid = camera["uid"]
        position = camera["reference_position"]
        focal_point = camera["reference_focal_point"]
        up_vector = camera["reference_up"]

        position_str = f"{position[0]},{position[1]},{position[2]}"
        # focal_point_str = f"{focal_point[0]},{focal_point[1]},{focal_point[2]}"
        focal_point_str = "0,0,0"
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
            f"Rendering camera {uid}/{len(selected_cameras)} ({position_str} → {focal_point_str})..."
        )
        try:
            subprocess.run(cmd, check=True)
            print(f"Saved render to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error rendering camera {uid}: {e}")
            continue


if __name__ == "__main__":
    main()
