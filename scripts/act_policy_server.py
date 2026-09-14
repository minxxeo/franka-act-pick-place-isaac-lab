#!/usr/bin/env python3
"""Serve a trained LeRobot ACT policy to an Isaac Lab process over localhost."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
from pathlib import Path

import numpy as np
import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


HEADER = struct.Struct("!Q")


def receive_message(connection: socket.socket):
    header = receive_exact(connection, HEADER.size)
    if not header:
        return None
    (size,) = HEADER.unpack(header)
    return pickle.loads(receive_exact(connection, size))


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            if not chunks:
                return b""
            raise ConnectionError("Connection closed during a message")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(connection: socket.socket, value) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(HEADER.pack(len(payload)) + payload)


def build_observation(request: dict) -> dict[str, torch.Tensor]:
    observation = {
        "observation.state": torch.from_numpy(np.asarray(request["state"], dtype=np.float32)),
    }
    for camera in ("gripper_cam", "side_cam", "top_cam"):
        image = np.asarray(request["images"][camera], dtype=np.uint8)
        if image.shape != (84, 84, 3):
            raise ValueError(f"{camera} has shape {image.shape}, expected (84, 84, 3)")
        observation[f"observation.images.{camera}"] = (
            torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        )
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Override how many predicted actions are executed before the next policy query.",
    )
    parser.add_argument(
        "--temporal_ensemble_coeff",
        type=float,
        default=None,
        help="Enable ACT temporal ensembling with this exponential weighting coefficient.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Invalid LeRobot checkpoint: {checkpoint}")

    config = ACTConfig.from_pretrained(checkpoint)
    config.device = args.device
    if args.n_action_steps is not None:
        if args.n_action_steps < 1 or args.n_action_steps > config.chunk_size:
            raise ValueError(
                f"--n_action_steps must be between 1 and the trained chunk size ({config.chunk_size})"
            )
        config.n_action_steps = args.n_action_steps
    if args.temporal_ensemble_coeff is not None:
        if args.temporal_ensemble_coeff <= 0.0:
            raise ValueError("--temporal_ensemble_coeff must be positive")
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.reset()

    print(f"Loaded ACT checkpoint: {checkpoint}")
    print(f"Chunk size: {config.chunk_size}, queued action steps: {config.n_action_steps}")
    print(
        "Temporal ensemble: "
        + ("OFF" if config.temporal_ensemble_coeff is None else f"ON (coeff={config.temporal_ensemble_coeff})")
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"ACT server listening on {args.host}:{args.port}", flush=True)
        while True:
            connection, address = server.accept()
            print(f"Isaac Lab client connected: {address}", flush=True)
            with connection:
                try:
                    while True:
                        request = receive_message(connection)
                        if request is None:
                            break
                        command = request.get("command")
                        if command == "reset":
                            policy.reset()
                            send_message(connection, {"ok": True})
                        elif command == "infer":
                            observation = preprocessor(build_observation(request))
                            with torch.inference_mode():
                                action = policy.select_action(observation)
                            action = postprocessor(action).squeeze(0).cpu().numpy().astype(np.float32)
                            # Send plain Python floats so different NumPy versions in
                            # the LeRobot and Isaac Sim environments can interoperate.
                            send_message(connection, {"action": action.tolist()})
                        elif command == "shutdown":
                            send_message(connection, {"ok": True})
                            return
                        else:
                            raise ValueError(f"Unknown command: {command}")
                except (ConnectionError, EOFError):
                    pass
                except Exception as error:
                    try:
                        send_message(connection, {"error": f"{type(error).__name__}: {error}"})
                    except OSError:
                        pass
                    print(f"Client error: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
