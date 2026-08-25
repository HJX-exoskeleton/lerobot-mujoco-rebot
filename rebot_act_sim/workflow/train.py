"""Train visual or multimodal ACT with one explicit, reproducible loop."""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from collections import deque
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parents[2] / "models"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / ".matplotlib"))
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_XET_CACHE", str(_CACHE_ROOT / ".xet"))
os.environ.setdefault("HF_ASSETS_CACHE", str(_CACHE_ROOT / ".assets"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
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
from lerobot.common.datasets.utils import dataset_to_policy_features
from torch.amp import GradScaler
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


def _preflight(raw: dict, metadata, dataset, config, *, use_imu: bool, use_tactile: bool) -> None:
    """Fail before a long run when collection and policy schemas diverge."""
    expected = {"observation.image", "observation.wrist_image", "observation.state", "action"}
    policy_features = set(dataset_to_policy_features(metadata.features))
    if policy_features != expected:
        raise ValueError(
            "dataset ACT feature mismatch: "
            f"expected={sorted(expected)}, actual={sorted(policy_features)}"
        )
    episodes_file = Path(dataset.root) / "meta" / "episodes.jsonl"
    lengths = [
        int(json.loads(line)["length"])
        for line in episodes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lengths or min(lengths) < int(config.chunk_size):
        raise ValueError(
            f"shortest episode must contain at least chunk_size={config.chunk_size} frames; "
            f"got {min(lengths) if lengths else 0}"
        )
    sample = dataset[0]
    expected_shapes = {
        "observation.image": (3, 256, 256),
        "observation.wrist_image": (3, 256, 256),
        "observation.state": (6,),
        "action": (int(config.chunk_size), 7),
    }
    if use_imu:
        expected_shapes["sensor.imu"] = (10,)
    if use_tactile:
        expected_shapes["sensor.tactile_left"] = (8, 16)
        expected_shapes["sensor.tactile_right"] = (8, 16)
    for key, shape in expected_shapes.items():
        actual = tuple(sample[key].shape)
        if actual != shape:
            raise ValueError(f"{key} shape mismatch: expected={shape}, actual={actual}")
    print(
        "[TRAIN PREFLIGHT] "
        f"episodes={dataset.num_episodes} frames={len(dataset)} "
        f"chunk={config.chunk_size} action_steps={config.n_action_steps} "
        f"IMU={use_imu} tactile={use_tactile}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--imu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tactile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--check-only", action="store_true")
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    dataset = LeRobotDataset(
        str(dataset_cfg["repo_id"]),
        root=root,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
    )
    if dataset.num_episodes <= 0:
        raise ValueError("dataset has no episodes")
    _preflight(raw, metadata, dataset, config, use_imu=use_imu, use_tactile=use_tactile)
    if args.check_only:
        return
    device = _device(str(policy_raw["device"]))
    policy = make_policy(
        config,
        metadata.stats,
        use_imu=use_imu,
        use_tactile=use_tactile,
        sensor_embed_dim=int(multimodal.get("sensor_embed_dim", 64)),
        sensor_dropout=float(multimodal.get("sensor_dropout", 0.0)),
        tactile_fusion_gain=float(multimodal.get("tactile_fusion_gain", 1.0)),
    ).to(device).train()
    use_preset = bool(training.get("use_policy_training_preset", True))
    preset = config.get_optimizer_preset()
    optimizer = torch.optim.AdamW(
        policy.get_optim_params() if use_preset else policy.parameters(),
        lr=float(preset.lr if use_preset else training.get("learning_rate", 1e-5)),
        weight_decay=float(
            preset.weight_decay if use_preset else training.get("weight_decay", 1e-4)
        ),
    )
    use_amp = bool(training.get("use_amp", False)) and device.type == "cuda"
    grad_clip_norm = float(training.get("grad_clip_norm", 10.0))
    scaler = GradScaler(device.type, enabled=use_amp)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=int(training.get("num_workers", 0)) > 0,
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
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type) if use_amp else nullcontext():
            loss, _ = policy(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
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
            progress.write(
                f"step={step} loss={value:.5f} grad_norm={float(grad_norm):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

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
