"""Train visual or multimodal ACT with one explicit, reproducible loop."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parents[2] / "models"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / ".matplotlib"))
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from tqdm.auto import tqdm

from rebot_act_sim.config import (
    DEFAULT_CONFIG,
    build_act_config,
    load_config,
    resolve_project_path,
)
from rebot_act_sim.policy import make_policy
from rebot_act_sim.sensors import (
    validate_processing_spec,
    write_or_validate_processing_spec,
)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _save_metrics(output, history: list[float]) -> None:
    average = []
    window = deque(maxlen=100)
    for value in history:
        window.append(value)
        average.append(sum(window) / len(window))
    payload = {
        "steps": len(history),
        "final_loss": history[-1],
        "final_avg100_loss": average[-1],
        "best_avg100_loss": min(average),
        "loss": history,
        "avg100_loss": average,
    }
    (output / "training_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(history, alpha=0.25, linewidth=0.8, label="step loss")
    axis.plot(average, linewidth=2, label="100-step average")
    axis.set(xlabel="step", ylabel="loss", title="ACT training loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "loss_curve.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--imu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tactile", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    raw = load_config(args.config)
    training = raw["training"]
    policy_raw = raw["policy"]
    if args.device:
        policy_raw["device"] = args.device
    multimodal = raw.get("multimodal", {})
    use_imu = bool(multimodal.get("use_imu", False) if args.imu is None else args.imu)
    use_tactile = bool(
        multimodal.get("use_tactile", False) if args.tactile is None else args.tactile
    )
    steps = int(args.steps or training["steps"])
    batch_size = int(args.batch_size or training["batch_size"])
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    seed = int(training.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_cfg = raw["dataset"]
    root = resolve_project_path(dataset_cfg["root"])
    tactile_processing = raw["environment"].get("tactile_processing")
    validate_processing_spec(root, tactile_processing)
    metadata = LeRobotDatasetMetadata(str(dataset_cfg["repo_id"]), root=root)
    if int(metadata.fps) != int(dataset_cfg["fps"]):
        raise ValueError(
            f"dataset fps={metadata.fps} does not match configured "
            f"control frequency {dataset_cfg['fps']} Hz"
        )
    config = build_act_config(raw, metadata)
    device = _device(str(policy_raw["device"]))
    dataset = LeRobotDataset(
        str(dataset_cfg["repo_id"]),
        root=root,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
    )
    if dataset.num_episodes <= 0:
        raise ValueError("dataset has no episodes")
    policy = make_policy(
        config,
        metadata.stats,
        use_imu=use_imu,
        use_tactile=use_tactile,
        sensor_embed_dim=int(multimodal.get("sensor_embed_dim", 64)),
    ).to(device).train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("dataset is smaller than one batch; reduce batch_size")
    output = resolve_project_path(training["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    history: list[float] = []
    iterator = iter(loader)
    progress = tqdm(range(1, steps + 1), desc="Training ACT", dynamic_ncols=True)
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        loss, _ = policy(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        history.append(value)
        progress.set_postfix(loss=f"{value:.4f}")
        save_freq = int(training.get("save_freq", 0))
        if save_freq > 0 and step % save_freq == 0 and step < steps:
            checkpoint = output / "checkpoints" / f"step_{step:06d}"
            policy.save_pretrained(checkpoint)
            if use_tactile:
                write_or_validate_processing_spec(checkpoint, tactile_processing)
        if step % int(training.get("log_freq", 50)) == 0:
            progress.write(f"step={step} loss={value:.5f}")

    # Always persist the final training step regardless of save_freq alignment.
    final_step_ckpt = output / "checkpoints" / f"step_{steps:06d}"
    final_step_ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_step_ckpt)
    if use_tactile:
        write_or_validate_processing_spec(final_step_ckpt, tactile_processing)

    policy.eval()
    final_checkpoint = output / "pretrained_model"
    policy.save_pretrained(final_checkpoint)
    if use_tactile:
        write_or_validate_processing_spec(final_checkpoint, tactile_processing)
    _save_metrics(output, history)
    run = {
        "config": raw,
        "use_imu": use_imu,
        "use_tactile": use_tactile,
        "dataset_episodes": dataset.num_episodes,
        "dataset_frames": len(dataset),
    }
    (output / "run_config.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved final checkpoint to {output / 'pretrained_model'}")


if __name__ == "__main__":
    main()
