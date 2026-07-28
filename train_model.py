#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import time
import json
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

# Hugging Face tokenizers uses its own Rayon thread pool. DataLoader workers
# are forked later, after the tokenizer has been initialized by VLA policies;
# inheriting that thread pool is unsafe and produces repeated deadlock warnings.
# Tokenization remains available, while DataLoader parallelism is controlled by
# cfg.num_workers independently.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from tqdm.auto import tqdm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


PROJECT_ROOT = Path(__file__).resolve().parent


def save_local_training_artifacts(output_dir: Path, history: dict[str, list[float]]) -> None:
    """Persist VLA training metrics and plots without requiring WandB."""
    if not history["step"]:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window = 100
    losses = history["loss"]
    avg100_loss = [
        sum(losses[max(0, i - window + 1) : i + 1]) / min(i + 1, window)
        for i in range(len(losses))
    ]
    metrics = {
        "summary": {
            "completed_steps": history["step"][-1],
            "final_loss": losses[-1],
            "final_avg100_loss": avg100_loss[-1],
            "best_avg100_loss": min(avg100_loss),
        },
        "history": {**history, "avg100_loss": avg100_loss},
    }
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(history["step"], losses, alpha=0.25, linewidth=0.8, label="step loss")
    axis.plot(history["step"], avg100_loss, linewidth=2, label="100-step average")
    axis.set(title="VLA Training Loss", xlabel="Training step", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(history["step"], history["grad_norm"])
    axes[0].set_ylabel("Gradient norm")
    axes[1].plot(history["step"], history["lr"])
    axes[1].set_ylabel("Learning rate")
    axes[2].plot(history["step"], history["step_s"], label="total")
    axes[2].plot(history["step"], history["dataloading_s"], alpha=0.7, label="data loading")
    axes[2].set(xlabel="Training step", ylabel="Seconds / step")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("VLA Training Diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "training_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_local_hf_snapshot(cache_root: str | Path) -> Path:
    """Resolve a Hugging Face cache repository root to a loadable snapshot."""
    cache_root = Path(cache_root).expanduser().resolve()
    if (cache_root / "model.safetensors").is_file():
        return cache_root
    ref = cache_root / "refs" / "main"
    if ref.is_file():
        snapshot = cache_root / "snapshots" / ref.read_text().strip()
        if (snapshot / "model.safetensors").is_file():
            return snapshot
    snapshots = sorted(
        p.parent for p in (cache_root / "snapshots").glob("*/model.safetensors")
    )
    if len(snapshots) == 1:
        return snapshots[0]
    if not snapshots:
        raise FileNotFoundError(f"No model.safetensors found under {cache_root}")
    raise RuntimeError(f"Multiple snapshots found under {cache_root}; set an explicit snapshot path")


def get_pretrained_path(policy_type: str) -> str:
    """Prefer explicit/local model weights and fall back to the Hub model id."""
    if policy_type == "smolvla":
        env_name = "SMOLVLA_PRETRAINED_PATH"
        local_defaults = [PROJECT_ROOT / "models" / "models--lerobot--smolvla_base"]
        hub_id = "lerobot/smolvla_base"
    elif policy_type == "pi0":
        env_name = "PI0_PRETRAINED_PATH"
        # The tutorial's published Pi0 checkpoint is the practical default for
        # adapting a newly collected OMY dataset. Keep the upstream cache name
        # as a secondary option for users who downloaded lerobot/pi0 directly.
        local_defaults = [
            PROJECT_ROOT / "models" / "models--Jeongeun--omy_pnp_pi0",
            PROJECT_ROOT / "models" / "models--lerobot--pi0",
        ]
        hub_id = "lerobot/pi0"
    else:
        raise ValueError(f"Unsupported pretrained policy type: {policy_type}")

    configured = os.environ.get(env_name)
    if configured:
        configured_path = Path(configured).expanduser()
        is_hub_id = configured.count("/") == 1 and not configured.startswith((".", "/"))
        if is_hub_id and not configured_path.exists():
            logging.info(f"Using Hugging Face pretrained model {configured}")
            return configured
        snapshot = resolve_local_hf_snapshot(configured_path)
        logging.info(f"Using pretrained weights from {env_name}={snapshot}")
        return str(snapshot)
    for local_default in local_defaults:
        if local_default.exists():
            snapshot = resolve_local_hf_snapshot(local_default)
            logging.info(f"Using project-local pretrained weights: {snapshot}")
            return str(snapshot)
    logging.info("No local pretrained weights found; using Hugging Face Hub model %s", hub_id)
    return hub_id


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    if cfg.policy.type == "pi0":
        cfg.policy.pretrained_path = get_pretrained_path("pi0")
    elif cfg.policy.type == "smolvla":
        cfg.policy.pretrained_path = get_pretrained_path("smolvla")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
        # Keep workers alive across epochs. Besides avoiding repeated process
        # startup, this prevents a new tokenizer/fork warning on every epoch.
        persistent_workers=cfg.num_workers > 0,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )
    local_history = {
        "step": [],
        "loss": [],
        "grad_norm": [],
        "lr": [],
        "step_s": [],
        "dataloading_s": [],
        "gpu_memory_gb": [],
    }

    logging.info("Start offline training on a fixed dataset")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(
        total=cfg.steps,
        initial=step,
        desc=f"Training {cfg.policy.type}",
        unit="step",
        dynamic_ncols=True,
        mininterval=1.0,
        smoothing=0.1,
    )
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        elapsed_step_s = train_tracker.update_s.val + train_tracker.dataloading_s.val
        postfix = {
            "loss": f"{train_tracker.loss.val:.4f}",
            "lr": f"{train_tracker.lr.val:.2e}",
            "step_s": f"{elapsed_step_s:.2f}",
            "sample_s": f"{cfg.batch_size / max(elapsed_step_s, 1e-9):.2f}",
        }
        if device.type == "cuda":
            postfix["gpu_GB"] = f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}"
        local_history["step"].append(step)
        local_history["loss"].append(float(train_tracker.loss.val))
        local_history["grad_norm"].append(float(train_tracker.grad_norm.val))
        local_history["lr"].append(float(train_tracker.lr.val))
        local_history["step_s"].append(float(elapsed_step_s))
        local_history["dataloading_s"].append(float(train_tracker.dataloading_s.val))
        local_history["gpu_memory_gb"].append(
            float(torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0
        )
        progress.set_postfix(postfix, refresh=False)
        progress.update(1)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            save_local_training_artifacts(cfg.output_dir, local_history)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    progress.close()
    save_local_training_artifacts(cfg.output_dir, local_history)
    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
