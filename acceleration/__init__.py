"""Frozen LLaDA generation baselines, loaded only when first used."""

from .backend import LLaDABackend
from .presets import MODES, describe_config

__all__ = ["LLaDABackend", "MODES", "describe_config"]
