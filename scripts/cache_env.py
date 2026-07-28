"""Configure project-local caches before importing Hugging Face libraries."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"


def configure_project_caches() -> Path:
    """Keep model, dataset, Xet, asset, and Torch caches outside $HOME."""
    os.environ["HF_HOME"] = str(MODELS_ROOT / ".hf_home")
    os.environ["HF_HUB_CACHE"] = str(MODELS_ROOT)
    os.environ["HF_DATASETS_CACHE"] = str(MODELS_ROOT / "datasets")
    os.environ["HF_XET_CACHE"] = str(MODELS_ROOT / ".xet")
    os.environ["HF_ASSETS_CACHE"] = str(MODELS_ROOT / ".assets")
    os.environ["TORCH_HOME"] = str(MODELS_ROOT / "torch")
    # Xet discarded the partial Pi0 download after an unstable CDN connection.
    # Plain HTTP uses a persistent .incomplete file and can resume on retry.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    return MODELS_ROOT
