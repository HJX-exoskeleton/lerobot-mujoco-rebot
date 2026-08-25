"""Replay a recorded episode using the exact stored action targets."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from rebot_act_sim.config import (
    DEFAULT_CONFIG,
    load_config,
    object_randomization_kwargs,
    resolve_project_path,
    tactile_processing_kwargs,
)
from rebot_act_sim.environment import SimACTEnvironment
from rebot_act_sim.timing import WallClockRate
from rebot_act_sim.visualization import ReplaySensorVisualizer


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, episode: int):
        if not 0 <= episode < dataset.num_episodes:
            raise ValueError(f"episode must be in [0, {dataset.num_episodes - 1}]")
        start = dataset.episode_data_index["from"][episode].item()
        stop = dataset.episode_data_index["to"][episode].item()
        self.indices = range(start, stop)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--xml", help="Override the environment XML used for replay.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--no-sensors",
        action="store_true",
        help="Disable the synchronized IMU/tactile panel.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier; 1.0 replays at the dataset's wall-clock rate.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render MuJoCo viewer every N control steps (1 = every step).",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    raw = load_config(args.config)
    dataset_cfg = raw["dataset"]
    dataset = LeRobotDataset(
        str(dataset_cfg["repo_id"]), root=resolve_project_path(dataset_cfg["root"])
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=EpisodeSampler(dataset, args.episode), num_workers=0
    )
    env = SimACTEnvironment(
        resolve_project_path(args.xml or raw["environment"]["xml"]),
        seed=None,
        **object_randomization_kwargs(raw),
        **tactile_processing_kwargs(raw),
    )
    iterator = iter(loader)
    sensor_visualizer = ReplaySensorVisualizer(
        history_frames=int(dataset_cfg["fps"]) * 2,
        tactile_color_max=float(
            raw["environment"].get("tactile_processing", {}).get(
                "visualization_color_max", 15.0
            )
        ),
    )
    rate = WallClockRate(float(dataset_cfg["fps"]) * args.speed)
    render_every = max(1, int(args.render_every))
    first = True
    frame_index = 0
    try:
        while env.is_alive():
            env.advance_physics()
            if not env.is_control_tick(int(dataset_cfg["fps"])):
                continue
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if first:
                env.set_object_initial_position(
                    batch["episode.object_initial_position"][0].numpy()
                )
                first = False
            env.command(batch["action"][0].numpy())
            env.task.rgb_agent = np.transpose(
                (batch["observation.image"][0].numpy() * 255).astype(np.uint8), (1, 2, 0)
            )
            env.task.rgb_ego = np.transpose(
                (batch["observation.wrist_image"][0].numpy() * 255).astype(np.uint8), (1, 2, 0)
            )
            env.task.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
            if frame_index % render_every == 0:
                if not args.no_sensors:
                    sensor_panel = sensor_visualizer.render(
                        batch["sensor.imu"][0].numpy(),
                        batch["sensor.tactile_left"][0].numpy(),
                        batch["sensor.tactile_right"][0].numpy(),
                        frame_index=frame_index,
                        timestamp=float(batch["timestamp"][0]),
                    )
                    env.parser.viewer_rgb_overlay(sensor_panel, loc="top left")
                env.render()
            frame_index += 1
            rate.wait()
    finally:
        env.close()


if __name__ == "__main__":
    main()
