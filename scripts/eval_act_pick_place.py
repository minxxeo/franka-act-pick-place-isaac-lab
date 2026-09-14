#!/usr/bin/env python3
"""Evaluate a LeRobot ACT server in the Isaac Lab pick-and-place task."""

from __future__ import annotations

import argparse
import csv
import pickle
import socket
import struct
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-PickPlace-Cube-Franka-ACT-JointPos-Fixed-v0",
    help="Registered ACT JointPos environment ID.",
)
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--replan_interval", type=int, default=10)
parser.add_argument("--server_host", default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=5555)
parser.add_argument("--seed", type=int, default=42, help="Base seed used for reproducible evaluation episodes.")
parser.add_argument("--results", type=Path, default=Path("outputs/act_pick_place_eval/results.csv"))
parser.add_argument("--no_save_results", action="store_true", help="Run inference without writing a CSV file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


HEADER = struct.Struct("!Q")
CAMERAS = ("gripper_cam", "side_cam", "top_cam")


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("ACT server disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def exchange(connection: socket.socket, request: dict) -> dict:
    payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(HEADER.pack(len(payload)) + payload)
    (size,) = HEADER.unpack(receive_exact(connection, HEADER.size))
    response = pickle.loads(receive_exact(connection, size))
    if "error" in response:
        raise RuntimeError(f"ACT server error: {response['error']}")
    return response


def policy_request(env, observation: dict) -> dict:
    policy_obs = observation["policy"]
    state = env.scene["robot"].data.joint_pos[0, :9].detach().cpu().numpy().astype(np.float32)
    images = {}
    for camera in CAMERAS:
        image = policy_obs[camera][0].detach().cpu().numpy()
        images[camera] = np.asarray(image[..., :3], dtype=np.uint8)
    return {"command": "infer", "state": state, "images": images}


def main() -> None:
    if args_cli.num_episodes < 1:
        raise ValueError("--num_episodes must be positive")
    if args_cli.replan_interval < 0 or args_cli.replan_interval > 100:
        raise ValueError("--replan_interval must be between 0 and 100; 0 disables forced replanning")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.recorders = {}
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"ACT evaluation environment ready: action_space={env.action_space.shape}", flush=True)
    if env.action_space.shape[-1] != 8:
        raise RuntimeError(f"Expected an 8D ACT action space, got {env.action_space.shape}")

    rows = []
    if not args_cli.no_save_results:
        args_cli.results = args_cli.results.expanduser().resolve()
        args_cli.results.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Connecting to ACT server at {args_cli.server_host}:{args_cli.server_port}...", flush=True)
        with socket.create_connection((args_cli.server_host, args_cli.server_port), timeout=30.0) as connection:
            print("Connected to ACT server.", flush=True)
            connection.settimeout(120.0)
            for episode in range(1, args_cli.num_episodes + 1):
                print(f"Resetting episode {episode}...", flush=True)
                episode_seed = args_cli.seed + episode - 1
                observation, _ = env.reset(seed=episode_seed)
                exchange(connection, {"command": "reset"})
                print(f"Episode {episode} started.", flush=True)
                success = False
                termination = "max_steps"
                steps_taken = 0

                for step in range(args_cli.max_steps):
                    # Discard the remaining queued actions periodically so ACT
                    # incorporates a fresh camera observation.
                    if args_cli.replan_interval > 0 and step > 0 and step % args_cli.replan_interval == 0:
                        exchange(connection, {"command": "reset"})

                    if step == 0:
                        print("Preparing first ACT observation...", flush=True)
                    request = policy_request(env, observation)
                    if step == 0:
                        print("Sending first ACT observation...", flush=True)
                    response = exchange(connection, request)
                    if step == 0:
                        print("Received first ACT action.", flush=True)
                    action = torch.as_tensor(response["action"], device=env.device, dtype=torch.float32).unsqueeze(0)
                    observation, _, terminated, truncated, _ = env.step(action)
                    steps_taken = step + 1
                    success = bool(env.termination_manager.get_term("success")[0].item())

                    if success or bool(terminated[0].item()) or bool(truncated[0].item()):
                        if success:
                            termination = "success"
                        elif bool(truncated[0].item()):
                            termination = "timeout"
                        else:
                            termination = "failure_termination"
                        break

                rows.append(
                    {
                        "task": args_cli.task,
                        "episode": episode,
                        "success": int(success),
                        "steps": steps_taken,
                        "termination": termination,
                        "replan_interval": args_cli.replan_interval,
                        "seed": episode_seed,
                    }
                )
                print(
                    f"Episode {episode:03d}/{args_cli.num_episodes}: "
                    f"{'SUCCESS' if success else 'FAIL'} ({termination}, {steps_taken} steps)",
                    flush=True,
                )
    finally:
        env.close()

    successes = sum(row["success"] for row in rows)
    print(f"Success rate: {successes}/{len(rows)} = {100.0 * successes / len(rows):.1f}%")
    if args_cli.no_save_results:
        print("Results were not saved (--no_save_results).")
    else:
        with args_cli.results.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results: {args_cli.results}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
