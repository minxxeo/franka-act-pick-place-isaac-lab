#!/usr/bin/env python3
"""Convert the Isaac Lab pick-and-place demonstrations to LeRobotDataset v3.

The policy state is the absolute Franka joint position (7 arm + 2 fingers).
The policy action is the next absolute arm joint position (7) followed by the
binary gripper command recorded during teleoperation (1=open, -1=close).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERA_KEYS = ("gripper_cam", "side_cam", "top_cam")
TASK = "Pick up the red cube and place it beside the blue cube."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=None,
        help="One or more source HDF5 files. Overrides the legacy --fixed/--random inputs.",
    )
    parser.add_argument("--fixed", type=Path, default=Path("datasets/pick_place_fixed_20.hdf5"))
    parser.add_argument("--random", type=Path, default=Path("datasets/pick_place_random_20.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("datasets/lerobot/pick_place_fixed_random_40"))
    parser.add_argument("--repo-id", default="local/isaac_pick_place_fixed_random_40")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--pre-motion-frames",
        type=int,
        default=5,
        help="Keep this many frames immediately before the first non-zero arm command.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def episode_names(group: h5py.Group) -> list[str]:
    return sorted(group.keys(), key=lambda name: int(name.rsplit("_", 1)[1]))


def first_motion_frame(raw_actions: np.ndarray, pre_motion_frames: int) -> int:
    moving = np.flatnonzero(np.linalg.norm(raw_actions[:, :6], axis=1) > 1.0e-6)
    if len(moving) == 0:
        return 0
    return max(0, int(moving[0]) - pre_motion_frames)


def main() -> None:
    args = parse_args()
    sources = tuple(path.resolve() for path in args.inputs) if args.inputs else (
        args.fixed.resolve(),
        args.random.resolve(),
    )
    output = args.output.resolve()

    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "panda_joint1_target",
                "panda_joint2_target",
                "panda_joint3_target",
                "panda_joint4_target",
                "panda_joint5_target",
                "panda_joint6_target",
                "panda_joint7_target",
                "gripper_command",
            ],
        },
    }
    for camera in CAMERA_KEYS:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": (84, 84, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output,
        robot_type="franka_panda_isaac_lab",
        features=features,
        use_videos=False,
        image_writer_threads=8,
    )

    episode_count = 0
    input_frames = 0
    output_frames = 0
    try:
        for source in sources:
            with h5py.File(source, "r") as h5_file:
                for episode_name in episode_names(h5_file["data"]):
                    episode = h5_file[f"data/{episode_name}"]
                    raw_actions = episode["actions"][:].astype(np.float32)
                    joint_positions = episode["states/articulation/robot/joint_position"][:].astype(np.float32)
                    start = first_motion_frame(raw_actions, args.pre_motion_frames)
                    length = len(raw_actions)
                    input_frames += length

                    for index in range(start, length):
                        next_index = min(index + 1, length - 1)
                        action = np.concatenate(
                            (joint_positions[next_index, :7], raw_actions[index, 6:7]), axis=0
                        ).astype(np.float32)
                        frame = {
                            "observation.state": joint_positions[index],
                            "action": action,
                            "task": TASK,
                        }
                        for camera in CAMERA_KEYS:
                            frame[f"observation.images.{camera}"] = episode[f"obs/{camera}"][index]
                        dataset.add_frame(frame)

                    dataset.save_episode()
                    episode_count += 1
                    output_frames += length - start
                    print(
                        f"[{episode_count:02d}] {source.name}/{episode_name}: "
                        f"{length} -> {length - start} frames"
                    )
    finally:
        dataset.finalize()

    print(f"Converted {episode_count} episodes: {input_frames} -> {output_frames} frames")
    print(f"LeRobot dataset: {output}")


if __name__ == "__main__":
    main()
