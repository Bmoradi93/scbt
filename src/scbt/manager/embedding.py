"""Failure-context embedding phi(c) (paper Sec. IV-C).

phi(c) = [ text(s) | onehot(failed node tag) | onehot(class) | numeric(b) ]

Only the text+tag block is used for retrieval similarity; the class and
numeric blocks are part of the bandit context. Text embedding uses
sentence-transformers (all-MiniLM-L6-v2) when available, with a
deterministic feature-hashing fallback so the pipeline runs offline.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Dict, List, Optional

import numpy as np

from ..bt.registry import LEAF_REGISTRY
from ..observer.events import FailureEvent
from ..sim.config import DEFAULT_CLASSES

LOG = logging.getLogger("scbt.manager")

TAGS = sorted(LEAF_REGISTRY)
CLASSES = sorted(DEFAULT_CLASSES)
NUMERIC_KEYS = ["alt", "speed", "wind_speed", "battery_pct", "link_quality"]
NUMERIC_SCALE = {"alt": 50.0, "speed": 10.0, "wind_speed": 30.0, "battery_pct": 100.0, "link_quality": 1.0}


class HashTextEmbedder:
    """Feature hashing of word uni+bigrams into ``dim`` buckets, L2-normalized."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        toks = re.findall(r"[a-z0-9_.]+", text.lower())
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class STTextEmbedder:
    name = "minilm"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=np.float32)


def make_text_embedder(kind: str = "auto"):
    if kind in ("auto", "minilm"):
        try:
            return STTextEmbedder()
        except Exception as exc:  # noqa: BLE001
            if kind == "minilm":
                raise
            LOG.warning(f"sentence-transformers unavailable ({exc}); using hashing embedder")
    return HashTextEmbedder()


class ContextEmbedder:
    def __init__(self, text_embedder=None):
        self.text = text_embedder or make_text_embedder()
        self.text_dim = self.text.dim
        self.dim = self.text_dim + len(TAGS) + len(CLASSES) + len(NUMERIC_KEYS)

    def __call__(self, ev: FailureEvent) -> np.ndarray:
        t = self.text(ev.context_text())
        tag = np.zeros(len(TAGS), dtype=np.float32)
        if ev.failed_tag in TAGS:
            tag[TAGS.index(ev.failed_tag)] = 1.0
        cls = np.zeros(len(CLASSES), dtype=np.float32)
        if ev.vehicle_class in CLASSES:
            cls[CLASSES.index(ev.vehicle_class)] = 1.0
        num = np.array([float(ev.blackboard.get(k, 0.0)) / NUMERIC_SCALE[k] for k in NUMERIC_KEYS], dtype=np.float32)
        return np.concatenate([t, tag, cls, num])

    def retrieval_part(self, phi: np.ndarray) -> np.ndarray:
        """Text + failed-tag block, L2-normalized (class is NOT part of
        retrieval so cross-class candidates are visible)."""
        v = phi[: self.text_dim + len(TAGS)].copy()
        v[self.text_dim:] *= 0.5   # tag block weight relative to text
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
