"""Backend factory. mock stays in-process; Darwin Qwen aliases load mlx_qwen."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kev.backends.base import Backend
from kev.backends.mock import MockBackend

HF_DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
MLX_DEFAULT_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MLX_REPO_3B = "mlx-community/Qwen2.5-3B-Instruct-4bit"
HF_REPO_3B = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_REPO = MLX_DEFAULT_REPO if sys.platform == "darwin" else HF_DEFAULT_REPO

MOCK_ALIASES = {"mock"}
MLX_ALIASES = {
    "mlx",
    "mlx_qwen",
    "kev-latest",
    "qwen2.5-1.5b",
    "qwen2.5-instruct",
    "qwen2.5-3b",
}
HF_ALIASES = {
    "hf",
    "hf_qwen",
    "qwen",
    "qwen2.5-7b",
    "qwen3-8b",
    "qwen3-instruct",
}


def on_darwin() -> bool:
    return sys.platform == "darwin"


def is_mlx_repo(repo: str | None) -> bool:
    if not repo:
        return False
    lowered = repo.lower()
    return "mlx-community" in lowered or lowered.startswith("mlx/")


def default_backend_for(alias: str, repo: str | None = None) -> str:
    key = alias.lower().replace("_", "-")
    if key == "mock":
        return "mock"
    if is_mlx_repo(repo) or key in {"mlx", "mlx-qwen"}:
        return "mlx_qwen"
    if on_darwin() and key in MLX_ALIASES:
        return "mlx_qwen"
    return "hf_qwen"


def default_repo_for(alias: str) -> str:
    key = alias.lower().replace("_", "-")
    if key in {"qwen2.5-3b"}:
        return MLX_REPO_3B if on_darwin() else HF_REPO_3B
    if default_backend_for(key) == "mlx_qwen":
        return MLX_DEFAULT_REPO
    return HF_DEFAULT_REPO


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    backend: str
    repo: str | None
    chat_format: str
    available: bool


def _qwen_spec(alias: str, repo: str | None = None, chat_format: str = "qwen2.5") -> ModelSpec:
    backend = default_backend_for(alias, repo)
    resolved = repo or default_repo_for(alias)
    if backend == "mlx_qwen" and alias in {"kev-latest", "qwen2.5-1.5b", "qwen2.5-instruct"}:
        resolved = MLX_DEFAULT_REPO if on_darwin() or is_mlx_repo(resolved) else resolved
        if on_darwin():
            resolved = MLX_DEFAULT_REPO
    if backend == "mlx_qwen" and alias == "qwen2.5-3b":
        resolved = MLX_REPO_3B if on_darwin() else (repo or MLX_REPO_3B)
    return ModelSpec(alias, backend, resolved, chat_format, True)


_BUILTIN: dict[str, ModelSpec] = {
    "mock": ModelSpec("mock", "mock", None, "qwen2.5", True),
    "kev-latest": _qwen_spec("kev-latest"),
    "qwen2.5-1.5b": _qwen_spec("qwen2.5-1.5b"),
    "qwen2.5-instruct": _qwen_spec("qwen2.5-instruct"),
    "qwen2.5-3b": _qwen_spec("qwen2.5-3b", MLX_REPO_3B if on_darwin() else HF_REPO_3B),
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", "hf_qwen", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5", True),
    "qwen3-8b": ModelSpec("qwen3-8b", "hf_qwen", "Qwen/Qwen3-8B", "qwen3", True),
    "qwen3-instruct": ModelSpec("qwen3-instruct", "hf_qwen", "Qwen/Qwen3-8B", "qwen3", True),
    "mlx": _qwen_spec("mlx", MLX_DEFAULT_REPO),
}


def _config_path() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "models.yaml"
        if candidate.exists():
            return candidate
    cwd = Path.cwd() / "configs" / "models.yaml"
    return cwd if cwd.exists() else None


def _coerce_spec(alias: str, backend: str | None, repo: str | None, chat_format: str) -> ModelSpec:
    key = alias.lower().replace("_", "-")
    raw_backend = (backend or "auto").lower()
    if raw_backend in {"auto", ""}:
        chosen = default_backend_for(key, repo)
    elif on_darwin() and key in {"kev-latest", "qwen2.5-1.5b", "qwen2.5-instruct", "qwen2.5-3b"}:
        chosen = "mlx_qwen"
    else:
        chosen = raw_backend.replace("-", "_")
        if chosen == "mlx":
            chosen = "mlx_qwen"
    resolved = repo
    mlx_1_5 = key in {"kev-latest", "qwen2.5-1.5b", "qwen2.5-instruct"}
    if chosen == "mlx_qwen" and on_darwin() and mlx_1_5:
        resolved = MLX_DEFAULT_REPO
    if chosen == "mlx_qwen" and on_darwin() and key == "qwen2.5-3b":
        resolved = MLX_REPO_3B
    if resolved is None:
        resolved = default_repo_for(key)
    return ModelSpec(str(alias), chosen, resolved, chat_format, True)


def _load_yaml_specs() -> dict[str, ModelSpec]:
    path = _config_path()
    if path is None:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    specs: dict[str, ModelSpec] = {}
    models = data.get("models") or {}
    aliases = data.get("aliases") or {}
    for alias, raw in models.items():
        if not isinstance(raw, dict):
            continue
        repo = raw.get("repo")
        chat_format = str(raw.get("chat_format") or ("qwen3" if "qwen3" in alias else "qwen2.5"))
        specs[str(alias)] = _coerce_spec(str(alias), raw.get("backend"), repo, chat_format)
    for alias, target in aliases.items():
        if target in specs:
            src = specs[str(target)]
            specs[str(alias)] = _coerce_spec(str(alias), src.backend, src.repo, src.chat_format)
        elif isinstance(target, str) and "/" in target:
            fmt = "qwen3" if "qwen3" in target.lower() else "qwen2.5"
            specs[str(alias)] = _coerce_spec(str(alias), None, target, fmt)
    return specs


def resolve_model(name: str) -> ModelSpec:
    key = name.lower().replace("_", "-")
    specs = {**_BUILTIN, **_load_yaml_specs()}
    if key in specs:
        return specs[key]
    if "/" in name:
        fmt = "qwen3" if "qwen3" in name.lower() else "qwen2.5"
        return _coerce_spec(name, None, name, fmt)
    raise ValueError(f"Unknown model: {name}")


def display_model_id(name: str) -> str:
    """Wire model id shown on responses. kev-latest resolves to qwen2.5-1.5b."""
    try:
        spec = resolve_model(name)
    except ValueError:
        return name
    if spec.backend == "mock":
        return "mock"
    repo = spec.repo or ""
    if spec.alias in {"kev-latest", "qwen2.5-instruct", "qwen2.5-1.5b"} or "1.5B" in repo:
        return "qwen2.5-1.5b"
    if spec.alias in {"qwen2.5-3b"} or "Qwen2.5-3B" in repo:
        return "qwen2.5-3b"
    if spec.alias == "qwen3-instruct":
        return "qwen3-8b"
    return spec.alias


def same_backend(left: str, right: str) -> bool:
    return resolve_model(left).backend == resolve_model(right).backend


def list_models() -> list[dict[str, object]]:
    specs = {**_BUILTIN, **_load_yaml_specs()}
    cards: list[dict[str, object]] = []
    seen: set[str] = set()
    for spec in specs.values():
        if spec.alias in seen:
            continue
        seen.add(spec.alias)
        cards.append(
            {
                "id": spec.alias,
                "backend": spec.backend,
                "repo": spec.repo,
                "available": spec.available,
            }
        )
    return cards


def get_backend(
    name: str = "mock",
    seed: int | None = None,
    device: str = "auto",
    dtype: str = "auto",
    adapter: str | None = None,
) -> Backend:
    spec = resolve_model(name)
    if spec.backend == "mock":
        return MockBackend(seed=seed)
    if spec.backend == "mlx_qwen":
        from kev.backends.mlx_qwen import MLXQwenBackend

        return MLXQwenBackend(
            repo=spec.repo or MLX_DEFAULT_REPO,
            device=device,
            dtype=dtype,
            chat_format=spec.chat_format,
            alias=spec.alias,
            adapter=adapter,
        )
    if spec.backend == "hf_qwen":
        from kev.backends.hf_qwen import HFQwenBackend

        return HFQwenBackend(
            repo=spec.repo or HF_DEFAULT_REPO,
            device=device,
            dtype=dtype,
            chat_format=spec.chat_format,
            alias=spec.alias,
            adapter=adapter,
        )
    raise ValueError(f"Unknown backend for model: {name}")
