"""Machine-specific runtime settings for optional PyTorch training.

Call ``prepare_torch_runtime()`` before constructing a model or dataloader.
It reads the project-root ``.env`` without requiring an additional package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}


@dataclass(frozen=True)
class TorchRuntime:
    """Resolved PyTorch device and optional per-process CPU thread limit."""

    device: str
    num_threads: int | None


def load_project_env(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE entries from .env without overwriting shell values."""
    if not path.is_file():
        return

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}의 환경변수 형식이 올바르지 않습니다.")
        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"{path}:{line_number}의 환경변수 이름이 올바르지 않습니다.")
        os.environ.setdefault(key, value.strip("'\""))


def _torch_module():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch가 설치되지 않았습니다. requirements.txt를 먼저 설치하세요.") from error
    return torch


def resolve_torch_device(requested: str | None = None) -> str:
    """Resolve an explicit device or automatically choose cuda, mps, then cpu."""
    choice = (requested or os.getenv("TORCH_DEVICE", "auto")).strip().lower()
    if choice not in VALID_DEVICES:
        choices = ", ".join(sorted(VALID_DEVICES))
        raise ValueError(f"TORCH_DEVICE는 다음 중 하나여야 합니다: {choices}")

    torch = _torch_module()
    mps_backend = getattr(torch.backends, "mps", None)
    cuda_available = torch.cuda.is_available()
    mps_available = mps_backend is not None and mps_backend.is_available()

    if choice == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if choice == "cuda" and not cuda_available:
        raise RuntimeError("TORCH_DEVICE=cuda이지만 사용할 CUDA GPU를 찾지 못했습니다.")
    if choice == "mps" and not mps_available:
        raise RuntimeError("TORCH_DEVICE=mps이지만 MPS 지원 장치를 찾지 못했습니다.")
    return choice


def prepare_torch_runtime(env_path: Path = ENV_PATH) -> TorchRuntime:
    """Load local settings, resolve a device, and apply ML_NUM_THREADS if present."""
    load_project_env(env_path)
    thread_value = os.getenv("ML_NUM_THREADS")
    num_threads = int(thread_value) if thread_value else None
    if num_threads is not None and num_threads < 1:
        raise ValueError("ML_NUM_THREADS는 1 이상의 정수여야 합니다.")

    device = resolve_torch_device()
    if num_threads is not None:
        _torch_module().set_num_threads(num_threads)
    return TorchRuntime(device=device, num_threads=num_threads)
