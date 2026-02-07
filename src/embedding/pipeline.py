"""
High-level helpers for embedding generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .dataloader import DataSpec, DEFAULT_DATA_BASE_DIR, load_and_select
from .embeddings import EmbeddingConfig, encode, get_default_model_paths


def get_embeddings(
    *,
    data_name: str = "clinc",
    data_choice: str = "large",
    data_path: Optional[str] = None,
    model_name: str = "instructor",
    class_num: int = 1,
    select_all: bool = False,
    class_names: Optional[Sequence[Any]] = None,
    seed: int = 42,
    batch_size: int = 32,
    normalize: bool = True,
    instruction: str = "Represent the intent of this utterance for clustering",
    model_paths: Optional[Dict[str, str]] = None,
    data_base_dir: str = DEFAULT_DATA_BASE_DIR,
) -> Tuple[np.ndarray, list, list]:
    """
    Load data and compute embeddings for selected classes.

    Returns:
        embeddings, labels, chosen_class_names
    """
    data = DataSpec(name=data_name, choice=data_choice, path=data_path)
    sentences, labels, chosen = load_and_select(
        data,
        base_dir=data_base_dir,
        class_num=class_num,
        class_names=class_names,
        select_all=select_all,
        seed=seed,
    )

    cfg = EmbeddingConfig(batch_size=batch_size, normalize=normalize, instruction=instruction)
    paths = model_paths or get_default_model_paths()
    emb = encode(sentences, model_name=model_name, model_paths=paths, config=cfg)
    return emb, labels, chosen


def get_all_embeddings(
    *,
    data_name: str = "clinc",
    data_choice: str = "large",
    data_path: Optional[str] = None,
    model_name: str = "instructor",
    seed: int = 42,
    batch_size: int = 32,
    normalize: bool = True,
    instruction: str = "Represent the intent of this utterance for clustering",
    model_paths: Optional[Dict[str, str]] = None,
    data_base_dir: str = DEFAULT_DATA_BASE_DIR,
) -> Tuple[np.ndarray, list]:
    """
    Embed the whole dataset (all labels). No file output.

    Returns:
        embeddings, labels
    """
    emb, labels, _ = get_embeddings(
        data_name=data_name,
        data_choice=data_choice,
        data_path=data_path,
        model_name=model_name,
        class_num=1,
        select_all=True,
        class_names=None,
        seed=seed,
        batch_size=batch_size,
        normalize=normalize,
        instruction=instruction,
        model_paths=model_paths,
        data_base_dir=data_base_dir,
    )
    return emb, labels


def run_and_save_embeddings(
    *,
    data_name: str = "clinc",
    data_choice: str = "large",
    data_path: Optional[str] = None,
    model_name: str = "instructor",
    seed: int = 42,
    batch_size: int = 32,
    normalize: bool = True,
    instruction: str = "Represent the intent of this utterance for clustering",
    model_paths: Optional[Dict[str, str]] = None,
    data_base_dir: str = DEFAULT_DATA_BASE_DIR,
) -> Tuple[np.ndarray, list]:
    """
    Embed the whole dataset (all labels). No file output.

    Returns:
        embeddings, labels
    """
    emb, labels, _ = get_embeddings(
        data_name=data_name,
        data_choice=data_choice,
        data_path=data_path,
        model_name=model_name,
        class_num=1,
        select_all=True,
        class_names=None,
        seed=seed,
        batch_size=batch_size,
        normalize=normalize,
        instruction=instruction,
        model_paths=model_paths,
        data_base_dir=data_base_dir,
    )
    return emb, labels
