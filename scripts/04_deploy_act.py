"""Deploy an ACT checkpoint in MuJoCo (from 4.deploy.ipynb)."""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from torchvision.transforms.functional import pil_to_tensor

from scripts.common import resolve_path, select_device


def to_tensor(image):
    return pil_to_tensor(Image.fromarray(image).resize((256, 256))).float() / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="demo_data")
    parser.add_argument("--repo-id", default="rebot_pnp")
    parser.add_argument("--checkpoint", default="ckpt/act_y")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from mujoco_env.y_env import SimpleEnv

    device = select_device(args.device)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=resolve_path(args.root))
    features = dataset_to_policy_features(metadata.features)
    outputs = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    inputs = {k: v for k, v in features.items() if k not in outputs}
    inputs.pop("observation.wrist_image", None)
    cfg = ACTConfig(
        input_features=inputs, output_features=outputs, chunk_size=10,
        n_action_steps=1, temporal_ensemble_coeff=0.9,
    )
    policy = ACTPolicy.from_pretrained(
        resolve_path(args.checkpoint), config=cfg, dataset_stats=metadata.stats
    ).to(device).eval()
    env = SimpleEnv(str(resolve_path("asset_rebot/example_scene_rebot.xml")), action_type="joint_angle")
    env.reset(seed=args.seed)
    policy.reset()
    step = 0
    try:
        with torch.inference_mode():
            while env.env.is_viewer_alive():
                env.step_env()
                if not env.env.loop_every(HZ=20):
                    continue
                state = env.get_ee_pose()
                agent, wrist = env.grab_image()
                batch = {
                    "observation.state": torch.as_tensor(state, device=device).float().unsqueeze(0),
                    "observation.image": to_tensor(agent).unsqueeze(0).to(device),
                    "observation.wrist_image": to_tensor(wrist).unsqueeze(0).to(device),
                    "task": ["Put mug cup on the plate"],
                    "timestamp": torch.tensor([step / 20], device=device),
                }
                action = policy.select_action(batch)[0].cpu().numpy()
                env.step(action)
                env.render()
                step += 1
                if env.check_success():
                    print("Success")
                    break
    finally:
        env.env.close_viewer()


if __name__ == "__main__":
    main()
