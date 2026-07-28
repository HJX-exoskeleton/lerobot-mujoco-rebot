"""Collect language-conditioned demonstrations (from 5.language_env.ipynb)."""

from __future__ import annotations

import argparse
import shutil

import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from scripts.common import resolve_path


FEATURES = {
    "observation.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
    "observation.wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    "obj_init": {"dtype": "float32", "shape": (9,), "names": ["obj_init"]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="demo_data_language")
    parser.add_argument("--repo-id", default="omy_pnp_language")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Use -1 for randomized initialization")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--success-hold-seconds", type=float, default=1.0,
        help="Seconds that all success conditions must remain true before saving the episode.",
    )
    args = parser.parse_args()
    from mujoco_env.y_env2 import SimpleEnv2

    root = resolve_path(args.root)
    seed = None if args.seed < 0 else args.seed
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    dataset = (
        LeRobotDataset(args.repo_id, root=root)
        if root.exists()
        else LeRobotDataset.create(
            repo_id=args.repo_id, root=root, robot_type="omy", fps=20, features=FEATURES,
            image_writer_threads=10, image_writer_processes=5,
        )
    )
    env = SimpleEnv2(
        str(resolve_path("asset_rebot/example_scene_rebot_language.xml")),
        seed=seed,
        state_type="joint_angle",
        success_hold_seconds=args.success_hold_seconds,
    )
    action, episode_id, recording = np.zeros(7), 0, False
    try:
        while env.env.is_viewer_alive() and episode_id < args.episodes:
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue
            if env.check_success():
                dataset.save_episode()
                episode_id += 1
                print(f"Saved episode {episode_id}/{args.episodes}")
                env.reset(seed=seed)
                recording = False
                continue
            action, reset = env.teleop_robot()
            if not recording and np.any(action):
                recording = True
                print(f"Recording: {env.instruction}")
            if reset:
                env.reset(seed=seed)
                dataset.clear_episode_buffer()
                recording = False
                continue
            agent, wrist = env.grab_image()
            agent = np.asarray(Image.fromarray(agent).resize((256, 256)))
            wrist = np.asarray(Image.fromarray(wrist).resize((256, 256)))
            state = env.step(action)[:6].astype(np.float32)
            joint_action = env.q[:7].astype(np.float32)
            if recording:
                dataset.add_frame(
                    {
                        "observation.image": agent,
                        "observation.wrist_image": wrist,
                        "observation.state": state,
                        "action": joint_action,
                        "obj_init": np.asarray(env.obj_init_pose, dtype=np.float32),
                    },
                    task=env.instruction,
                )
            env.render(teleop=True, idx=episode_id)
    finally:
        env.env.close_viewer()
    images = dataset.root / "images"
    if images.exists():
        shutil.rmtree(images)


if __name__ == "__main__":
    main()
