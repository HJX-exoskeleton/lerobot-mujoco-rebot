"""Replay a language-conditioned episode (from 6.visualize_data.ipynb)."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from scripts.common import EpisodeSampler, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="demo_data_language")
    parser.add_argument("--repo-id", default="omy_pnp_language")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    from mujoco_env.y_env2 import SimpleEnv2

    dataset = LeRobotDataset(args.repo_id, root=resolve_path(args.root))
    if args.push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)
        return
    sampler = EpisodeSampler(dataset, args.episode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
    env = SimpleEnv2(str(resolve_path("asset_rebot/example_scene_rebot_language.xml")), action_type="joint_angle")
    iterator, step = iter(loader), 0
    env.reset()
    try:
        while env.env.is_viewer_alive():
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue
            try:
                data = next(iterator)
            except StopIteration:
                iterator, step = iter(loader), 0
                env.reset()
                continue
            if step == 0:
                env.set_instruction(data["task"][0])
                pose = data["obj_init"][0]
                env.set_obj_pose(pose[:3], pose[3:6], pose[6:9])
            env.step(data["action"][0].numpy())
            env.rgb_agent = np.transpose((data["observation.image"][0].numpy() * 255).astype(np.uint8), (1, 2, 0))
            env.rgb_ego = np.transpose((data["observation.wrist_image"][0].numpy() * 255).astype(np.uint8), (1, 2, 0))
            env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
            env.render()
            step += 1
    finally:
        env.env.close_viewer()


if __name__ == "__main__":
    main()
