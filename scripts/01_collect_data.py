"""Collect single-task keyboard demonstrations (from 1.collect_data.ipynb)."""

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
    "obj_init": {"dtype": "float32", "shape": (6,), "names": ["obj_init"]},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="demo_data")
    parser.add_argument("--repo-id", default="rebot_pnp")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Use -1 for randomized resets")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task", default="Put mug cup on the plate")
    parser.add_argument(
        "--success-hold-seconds", type=float, default=1.0,
        help="Seconds that all success conditions must remain true before saving the episode.",
    )
    return parser.parse_args()


def open_dataset(args, root):
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    if root.exists():
        required_metadata = (
            "meta/info.json",
            "meta/tasks.jsonl",
            "meta/episodes.jsonl",
            "meta/episodes_stats.jsonl",
        )
        missing = [name for name in required_metadata if not (root / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Dataset directory {root} is incomplete; missing {missing}. "
                "This commonly happens when collection stops before the first episode is saved. "
                "Use --overwrite to discard it, or choose a different --root."
            )
        print(f"Appending to existing dataset: {root}")
        return LeRobotDataset(args.repo_id, root=root)
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type="rebot",
        fps=20,
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def main():
    args = parse_args()
    root = resolve_path(args.root)
    seed = None if args.seed < 0 else args.seed
    dataset = open_dataset(args, root)
    from mujoco_env.y_env import SimpleEnv

    env = SimpleEnv(
        str(resolve_path("asset_rebot/example_scene_rebot.xml")),
        seed=seed,
        state_type="joint_angle",
        success_hold_seconds=args.success_hold_seconds,
    )
    action = np.zeros(7)
    episode_id = 0
    recording = False
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
                print("Recording started")
            if reset:
                env.reset(seed=seed)
                dataset.clear_episode_buffer()
                recording = False
                continue
            state = env.get_ee_pose().astype(np.float32)
            agent, wrist = env.grab_image()
            agent = np.asarray(Image.fromarray(agent).resize((256, 256)))
            wrist = np.asarray(Image.fromarray(wrist).resize((256, 256)))
            joint_action = env.step(action).astype(np.float32)
            if recording:
                dataset.add_frame(
                    {
                        "observation.image": agent,
                        "observation.wrist_image": wrist,
                        "observation.state": state,
                        "action": joint_action,
                        "obj_init": np.asarray(env.obj_init_pose, dtype=np.float32),
                    },
                    task=args.task,
                )
            env.render(teleop=True)
    finally:
        env.env.close_viewer()
    images = dataset.root / "images"
    if images.exists():
        shutil.rmtree(images)


if __name__ == "__main__":
    main()
