"""
Embedding/encoding utilities.

This module keeps all model-specific encoding logic (Instructor, E5, Gemma, Qwen, SBERT)
in one place so the main script can stay simple.

All functions return numpy arrays of shape (n_samples, dim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np


@dataclass(frozen=True)
class EmbeddingConfig:
    """Common embedding configuration."""
    batch_size: int = 32
    normalize: bool = True
    instruction: str = "Represent the intent of this utterance for clustering"
    use_prompt: str = "query"  # for models that support prompt selection


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def encode_instructor(
    sentences: Sequence[str],
    model_path: str,
    config: Optional[EmbeddingConfig] = None,
) -> np.ndarray:
    """
    Encode texts using InstructorEmbedding.

    Args:
        sentences: list of texts.
        model_path: local path to Instructor model directory.
        config: embedding configuration (instruction, batch_size, normalize).

    Returns:
        embeddings: np.ndarray (n, d)
    """
    cfg = config or EmbeddingConfig()
    try:
        from InstructorEmbedding import INSTRUCTOR
    except Exception as e:
        raise ImportError(
            "InstructorEmbedding is required for 'instructor' model. "
            "Install it (and its deps) or choose another model."
        ) from e

    model = INSTRUCTOR(model_path)
    pairs = [[cfg.instruction, s] for s in sentences]
    emb = model.encode(pairs, batch_size=cfg.batch_size, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    emb = _l2_normalize(emb) if cfg.normalize else emb
    print("INSTRUCTOR embeddings shape:", emb.shape)
    return emb



def _load_sentence_transformer(model_path: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers is required for E5/Gemma/Qwen/SBERT encoders. "
            "Install it or choose 'instructor'."
        ) from e
    return SentenceTransformer(model_path)


def encode_sbert(
    sentences: Sequence[str],
    model_path: str,
    config: Optional[EmbeddingConfig] = None,
) -> np.ndarray:
    """
    Encode texts using a SentenceTransformer model (classic SBERT usage).

    Args:
        sentences: list of texts.
        model_path: local path or HF id.
        config: embedding configuration (batch_size, normalize). instruction is ignored.

    Returns:
        embeddings: np.ndarray (n, d)
    """
    cfg = config or EmbeddingConfig()
    model = _load_sentence_transformer(model_path)
    emb = model.encode(
        list(sentences),
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.normalize,
    )
    emb = emb.astype(np.float32)
    print("SBERT embeddings shape:", emb.shape)
    return emb


def encode_e5(
    sentences: Sequence[str],
    model_path: str,
    config: Optional[EmbeddingConfig] = None,
    prefix: str = "query: ",   # E5 models often use 'query: ' or 'passage: '
) -> np.ndarray:
    """
    Encode texts using an E5-style SentenceTransformer model.

    Many E5 checkpoints expect prefixed inputs (e.g., 'query: ' or 'passage: ').

    Args:
        sentences: list of texts.
        model_path: local path or HF id for an E5 model.
        config: embedding configuration (batch_size, normalize). instruction is ignored.
        prefix: text prefix applied to each sentence.

    Returns:
        embeddings: np.ndarray (n, d)
    """
    cfg = config or EmbeddingConfig()
    model = _load_sentence_transformer(model_path)
    inputs = [f"{prefix}{s}" for s in sentences]
    emb = model.encode(
        inputs,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.normalize,
    )
    emb = emb.astype(np.float32)
    print("E5 embeddings shape:", emb.shape)
    return emb


def encode_embedding_gemma(
    sentences: Sequence[str],
    model_path: str,
    config: Optional[EmbeddingConfig] = None,
) -> np.ndarray:
    """
    Encode texts using EmbeddingGemma (SentenceTransformer format).

    A common prompt format is:
        "Instruct: <instruction>\nQuery: <sentence>"

    Args:
        sentences: list of texts.
        model_path: local path to EmbeddingGemma model directory.
        config: embedding configuration (instruction, batch_size, normalize).

    Returns:
        embeddings: np.ndarray (n, d)
    """
    cfg = config or EmbeddingConfig()
    model = _load_sentence_transformer(model_path)
    inputs = [f"Instruct: {cfg.instruction}\nQuery: {s}" for s in sentences]
    emb = model.encode(
        inputs,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.normalize,
    )
    emb = emb.astype(np.float32)
    print("EmbeddingGemma embeddings shape:", emb.shape)
    return emb


def encode_qwen_embedding(
    sentences: Sequence[str],
    model_path: str,
    config: Optional[EmbeddingConfig] = None,
) -> np.ndarray:
    """
    Encode texts using Qwen3 embedding model (typically loaded via SentenceTransformer).

    Note: Some Qwen embedding checkpoints may have their own recommended prompting.
    This implementation keeps it simple and uses plain sentences.

    Args:
        sentences: list of texts.
        model_path: local path to Qwen3 embedding model directory.
        config: embedding configuration (batch_size, normalize). instruction is ignored.

    Returns:
        embeddings: np.ndarray (n, d)
    """
    cfg = config or EmbeddingConfig()
    model = _load_sentence_transformer(model_path)
    emb = model.encode(
        list(sentences),
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.normalize,
        prompt_name=cfg.use_prompt,
    )
    emb = emb.astype(np.float32)
    print("Qwen3-Embedding embeddings shape:", emb.shape)
    return emb


def get_default_model_paths() -> Dict[str, str]:
    """
    Default local model paths consistent with the original notebook.
    Update these paths to match your environment if needed.
    """
    return {
        "instructor": "./code/torch_test/local_models/instructor-large",
        "e5": "./code/torch_test/local_models/e5-large-v2",
        "gemma": "./code/torch_test/local_models/embeddinggemma-300m",
        "qwen": "./code/torch_test/local_models/Qwen3-Embedding-0.6B",
        "sbert": "./code/torch_test/local_models/Sbert",
    }


def encode(
    sentences: Sequence[str],
    model_name: str,
    model_paths: Optional[Dict[str, str]] = None,
    config: Optional[EmbeddingConfig] = None,
) -> np.ndarray:
    """
    Unified entry point to encode sentences with a chosen model.

    Args:
        sentences: list of texts.
        model_name: one of {'instructor','e5','gemma','qwen','sbert'} (case-insensitive).
        model_paths: mapping from model_name -> local model path.
        config: embedding configuration.

    Returns:
        embeddings: np.ndarray (n, d)
    """
    name = model_name.strip().lower()
    paths = model_paths or get_default_model_paths()
    if name not in paths:
        raise ValueError(f"Unknown model_name='{model_name}'. Supported: {sorted(paths.keys())}")

    model_path = paths[name]

    if name == "instructor":
        print(f"Loading INSTRUCTOR model from: {model_path}")
        print(f"Encoding {len(sentences)} sentences with INSTRUCTOR...")
        return encode_instructor(sentences, model_path=model_path, config=config)
    if name == "e5":
        print(f"Loading E5 model from: {model_path}")
        print(f"Encoding {len(sentences)} sentences with E5...")
        return encode_e5(sentences, model_path=model_path, config=config)
    if name == "gemma":
        print(f"Loading EmbeddingGemma model from: {model_path}")
        print(f"Encoding {len(sentences)} sentences with EmbeddingGemma (query prompt)...")
        return encode_embedding_gemma(sentences, model_path=model_path, config=config)
    if name == "qwen":
        print(f"Loading Qwen3-Embedding model from: {model_path}")
        print(f"Encoding {len(sentences)} sentences with Qwen3-Embedding...")
        return encode_qwen_embedding(sentences, model_path=model_path, config=config)
    if name == "sbert":
        print(f"Loading SBERT model from: {model_path}")
        print(f"Encoding {len(sentences)} sentences with SBERT...")
        return encode_sbert(sentences, model_path=model_path, config=config)

    # Defensive (should be unreachable)
    raise ValueError(f"Unhandled model_name='{model_name}'")
