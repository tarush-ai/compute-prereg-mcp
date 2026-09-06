"""Provider adapters. Import lazily: each may need an optional package or a credential."""

from __future__ import annotations

import importlib
import os

from ...errors import AdapterUnavailable
from ..base import ProviderAdapter

REGISTRY = {
    "mock": ("compute_prereg.broker.adapters.mock", "MockAdapter"),
    "runpod": ("compute_prereg.broker.adapters.runpod", "RunPodAdapter"),
    "modal": ("compute_prereg.broker.adapters.modal", "ModalAdapter"),
    "skypilot": ("compute_prereg.broker.adapters.skypilot", "SkyPilotAdapter"),
}


def build_adapter(name: str, **kwargs) -> ProviderAdapter:
    if name not in REGISTRY:
        raise AdapterUnavailable(f"unknown adapter {name!r}; known: {sorted(REGISTRY)}")
    mod, cls = REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def autodetect() -> list[str]:
    """Adapters that look usable from the environment (credential present / package installed)."""
    names = []
    if os.environ.get("RUNPOD_API_KEY"):
        names.append("runpod")
    if os.environ.get("MODAL_TOKEN_ID") or os.path.exists(os.path.expanduser("~/.modal.toml")):
        if importlib.util.find_spec("modal"):
            names.append("modal")
    if importlib.util.find_spec("sky"):
        names.append("skypilot")
    return names
