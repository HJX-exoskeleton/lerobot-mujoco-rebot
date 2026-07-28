"""Train and evaluate ACT (from 3.train.ipynb)."""

from __future__ import annotations

import argparse
from collections import deque
import json

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

from scripts.common import EpisodeSampler, resolve_path, select_device


class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="demo_data")
    parser.add_argument("--repo-id", default="rebot_pnp")
    parser.add_argument("--output", default="ckpt/act_y")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--eval-episode", type=int, default=0)
    parser.add_argument(
        "--exclude-episodes", type=int, nargs="*", default=[],
        help="Episode indices to exclude from training and normalization stats, e.g. 0 3",
    )
    args = parser.parse_args()
    device = select_device(args.device)
    root = resolve_path(args.root)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=root)
    excluded = set(args.exclude_episodes)
    invalid = sorted(i for i in excluded if i < 0 or i >= metadata.total_episodes)
    if invalid:
        raise ValueError(f"Invalid excluded episode indices: {invalid}")
    selected_episodes = [i for i in range(metadata.total_episodes) if i not in excluded]
    if not selected_episodes:
        raise ValueError("Cannot exclude every episode")
    if args.eval_episode in excluded:
        raise ValueError("--eval-episode cannot also appear in --exclude-episodes")
    features = dataset_to_policy_features(metadata.features)
    outputs = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    inputs = {k: v for k, v in features.items() if k not in outputs}
    inputs.pop("observation.wrist_image", None)
    cfg = ACTConfig(
        input_features=inputs, output_features=outputs, chunk_size=10,
        n_action_steps=10, device=str(device),
    )
    dataset = LeRobotDataset(
        args.repo_id,
        root=root,
        delta_timestamps=resolve_delta_timestamps(cfg, metadata),
        image_transforms=AddGaussianNoise(),
    )
    # LeRobot v2.1 has an indexing bug when `episodes` contains non-contiguous
    # original IDs. Keep the full dataset index, but sample only frames from the
    # selected episodes and aggregate normalization stats from those episodes.
    selected_stats = aggregate_stats([metadata.episodes_stats[i] for i in selected_episodes])
    policy = ACTPolicy(cfg, dataset_stats=selected_stats).to(device).train()
    print(f"Using {len(selected_episodes)}/{metadata.total_episodes} episodes; excluded={sorted(excluded)}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    training_frame_ids = []
    for episode_index in selected_episodes:
        start = dataset.episode_data_index["from"][episode_index].item()
        stop = dataset.episode_data_index["to"][episode_index].item()
        training_frame_ids.extend(range(start, stop))
    train_sampler = torch.utils.data.SubsetRandomSampler(training_frame_ids)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=True,
    )
    step = 0
    running_loss = 0.0
    recent_losses = deque(maxlen=100)
    loss_history = []
    avg100_history = []
    progress = tqdm(total=args.steps, desc="Training ACT", unit="step", dynamic_ncols=True)
    try:
        while step < args.steps:
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss, _ = policy(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_value = loss.item()
                running_loss += loss_value
                recent_losses.append(loss_value)
                loss_history.append(loss_value)
                avg100_history.append(sum(recent_losses) / len(recent_losses))
                step += 1
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg100=f"{sum(recent_losses) / len(recent_losses):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )
                if step % args.log_freq == 0:
                    progress.write(
                        f"step={step} loss={loss_value:.4f} "
                        f"avg_loss={running_loss / step:.4f}"
                    )
                if step >= args.steps:
                    break
    finally:
        progress.close()
    output_dir = resolve_path(args.output)
    policy.save_pretrained(output_dir)

    loss_fig, loss_axis = plt.subplots(figsize=(10, 5))
    loss_axis.plot(loss_history, color="tab:blue", alpha=0.25, linewidth=0.8, label="step loss")
    loss_axis.plot(avg100_history, color="tab:blue", linewidth=2, label="100-step average")
    loss_axis.set(title="ACT Training Loss", xlabel="Training step", ylabel="Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    loss_fig.tight_layout()
    loss_plot_path = output_dir / "loss_curve.png"
    loss_fig.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close(loss_fig)

    policy.eval().reset()
    sampler = EpisodeSampler(dataset, args.eval_episode)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            predictions.append(policy.select_action(batch))
            targets.append(batch["action"][:, 0, :])
    predictions, targets = torch.cat(predictions), torch.cat(targets)
    absolute_error = (predictions - targets).abs()
    mean_action_error = absolute_error.mean().item()
    per_dimension_mae = absolute_error.mean(dim=0).cpu().tolist()
    print(f"Mean action error: {mean_action_error:.4f}")
    fig, axes = plt.subplots(7, 1, figsize=(10, 10))
    for dim, axis in enumerate(axes):
        axis.plot(predictions[:, dim].cpu(), label="pred")
        axis.plot(targets[:, dim].cpu(), label="gt")
        axis.legend()
    plot_path = output_dir / "action_error.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "final_loss": loss_history[-1],
            "final_avg100_loss": avg100_history[-1],
            "best_avg100_loss": min(avg100_history),
            "loss": loss_history,
            "avg100_loss": avg100_history,
        },
        "evaluation": {
            "episode": args.eval_episode,
            "mean_action_error": mean_action_error,
            "per_dimension_mae": per_dimension_mae,
        },
    }
    metrics_path = output_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(
        f"Saved checkpoint and visualizations to {output_dir}: "
        f"{loss_plot_path.name}, {plot_path.name}, {metrics_path.name}"
    )


if __name__ == "__main__":
    main()
