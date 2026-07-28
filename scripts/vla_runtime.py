"""Shared train/deploy runtime for Pi0 and SmolVLA scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from torchvision.transforms.functional import pil_to_tensor

from scripts.common import PROJECT_ROOT, resolve_path, select_device


def add_subcommands(parser: argparse.ArgumentParser, default_config: str, default_checkpoint: str):
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run train_model.py")
    train.add_argument("--config", default=default_config)
    train.add_argument(
        "--pretrained",
        help="Hugging Face repo ID (for example lerobot/pi0), local cache root, or snapshot directory",
    )
    train.add_argument(
        "--cache-dir", default=str(PROJECT_ROOT / "models"),
        help="Hugging Face model cache (default: project models/ directory)",
    )
    deploy = commands.add_parser("deploy", help="Run the trained policy in MuJoCo")
    deploy.add_argument("--dataset-root", default="demo_data_language")
    deploy.add_argument("--repo-id", default="omy_pnp_language")
    deploy.add_argument("--checkpoint", default=default_checkpoint)
    deploy.add_argument("--hub-model", help="Use a Hugging Face model instead of --checkpoint")
    deploy.add_argument("--device", default="cuda")
    deploy.add_argument("--seed", type=int, default=0)


def run_train(config: str, pretrained: str | None, policy_type: str, cache_dir: str):
    env = os.environ.copy()
    cache_dir = str(resolve_path(cache_dir))
    # Set these before the training subprocess imports huggingface_hub,
    # transformers or datasets. This keeps large downloads out of $HOME.
    env["HF_HOME"] = str(resolve_path("models/.hf_home"))
    env["HF_HUB_CACHE"] = cache_dir
    env["HF_DATASETS_CACHE"] = str(resolve_path("models/datasets"))
    env["HF_XET_CACHE"] = str(resolve_path("models/.xet"))
    env["HF_ASSETS_CACHE"] = str(resolve_path("models/.assets"))
    env["TORCH_HOME"] = str(resolve_path("models/torch"))
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    if pretrained:
        candidate = PROJECT_ROOT / pretrained
        # A simple "owner/repository" value is a Hub ID. Existing paths and
        # explicitly path-like values continue to be resolved locally.
        is_hub_id = (
            pretrained.count("/") == 1
            and not pretrained.startswith((".", "/"))
            and not candidate.exists()
        )
        env[f"{policy_type.upper()}_PRETRAINED_PATH"] = (
            pretrained if is_hub_id else str(resolve_path(pretrained))
        )
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "train_model.py"), "--config_path", str(resolve_path(config))],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def image_tensor(array):
    image = Image.fromarray(array).resize((256, 256))
    return pil_to_tensor(image).float() / 255


class GripperCommandAdapter:
    """Convert dataset-space gripper positions to the env's binary command."""

    def __init__(self, open_position: float, closed_position: float):
        self.open_position = float(open_position)
        self.closed_position = float(closed_position)
        span = abs(self.open_position - self.closed_position)
        midpoint = (self.open_position + self.closed_position) / 2
        # A small dead band prevents continuous policy noise from rapidly
        # toggling the binary gripper command near the decision boundary.
        margin = span * 0.1
        self.close_threshold = midpoint - margin
        self.open_threshold = midpoint + margin
        self.is_closed = False

    def reset(self):
        self.is_closed = False

    def __call__(self, predicted_position: float) -> float:
        predicted_position = float(predicted_position)
        if predicted_position <= self.close_threshold:
            self.is_closed = True
        elif predicted_position >= self.open_threshold:
            self.is_closed = False
        return float(self.is_closed)


def deploy(args, policy_class, *, adapt_physical_gripper=False):
    from mujoco_env.y_env2 import SimpleEnv2

    device = select_device(args.device)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=resolve_path(args.dataset_root))
    model_source = args.hub_model or str(resolve_path(args.checkpoint))
    policy = policy_class.from_pretrained(model_source, dataset_stats=metadata.stats).to(device).eval()
    env = SimpleEnv2(str(resolve_path("asset_rebot/example_scene_rebot_language.xml")), action_type="joint_angle")
    env.reset(seed=args.seed)
    policy.reset()
    gripper_adapter = (
        GripperCommandAdapter(env.gripper_open, env.gripper_closed)
        if adapt_physical_gripper
        else None
    )
    try:
        with torch.inference_mode():
            while env.env.is_viewer_alive():
                env.step_env()
                if not env.env.loop_every(HZ=20):
                    continue
                state = env.get_joint_state()[:6]
                agent, wrist = env.grab_image()
                batch = {
                    "observation.state": torch.as_tensor(state, device=device).float().unsqueeze(0),
                    "observation.image": image_tensor(agent).unsqueeze(0).to(device),
                    "observation.wrist_image": image_tensor(wrist).unsqueeze(0).to(device),
                    "task": [env.instruction],
                }
                action = policy.select_action(batch)[0, :7].cpu().numpy()
                if gripper_adapter is not None:
                    # SmolVLA was trained on the physical gripper joint positions
                    # stored in env.q (open=0.05, closed=0.001), while env.step()
                    # expects a logical close fraction (open=0, closed=1).
                    action[-1] = gripper_adapter(action[-1])
                env.step(action)
                env.render()
                if env.check_success():
                    print("Success")
                    break
    finally:
        env.env.close_viewer()
