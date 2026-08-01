"""Structured ACT simulation workflow for the reBot MuJoCo task."""

import os
from pathlib import Path

# Configure writable, project-local caches before any submodule imports LeRobot,
# Datasets, Transformers or Matplotlib.
_MODEL_CACHE = Path(__file__).resolve().parents[1] / "models"
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE / ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(_MODEL_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_MODEL_CACHE / "datasets"))
os.environ.setdefault("HF_XET_CACHE", str(_MODEL_CACHE / ".xet"))
os.environ.setdefault("HF_ASSETS_CACHE", str(_MODEL_CACHE / ".assets"))
os.environ.setdefault("MPLCONFIGDIR", str(_MODEL_CACHE / ".matplotlib"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .schema import ACTION_DIM, FPS, STATE_DIM, TACTILE_SHAPE

__all__ = ["ACTION_DIM", "FPS", "STATE_DIM", "TACTILE_SHAPE"]
