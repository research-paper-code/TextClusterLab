"""
Text clustering test on CLINC small.jsonl using Instructor embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _add_project_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(10):
        if (cur / "src").is_dir():
            sys.path.insert(0, str(cur))
            return
        cur = cur.parent


_add_project_root_to_syspath()

from src.embedding.pipeline import get_all_embeddings  # noqa: E402
from src.metrics.cluster_metrics import compute_clustering_metrics  # noqa: E402

from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import calinski_harabasz_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402


DATA_NAME = "test"
# DATA_CHOICE = "ext_small_labels10_n0_plabel"
DATA_CHOICE = "Qwen_compact_10_test_ext_small_massive_intent_labels10_n0_prelabel_40_e5"
# DATA_CHOICE = "extract_10_mtop_intent_small"

# DATA_NAME = "test"
# DATA_CHOICE = "ext_clinc_large_10s_prelabel"

# DATA_NAME = "clinc"
# DATA_CHOICE = "small"


MODEL_NAME = "gemma"  # use lower-case to match pipeline defaults
OUT_DIR = Path("./code/DataGene/outputs/new")


def run():
    emb, labels = get_all_embeddings(
        data_name=DATA_NAME,
        data_choice=DATA_CHOICE,
        model_name=MODEL_NAME,
    )
    y_pred, ari, nmi, sil, db = compute_clustering_metrics(emb, labels)
    ch = None
    if y_pred is not None and len(set(y_pred)) > 1 and emb.shape[0] > 1:
        ch = calinski_harabasz_score(emb, y_pred)

    print("data:", f"{DATA_NAME}/{DATA_CHOICE}.jsonl")
    print("embeddings:", emb.shape)
    print("labels:", len(set(labels)))
    print("ARI:", f"{ari:.4f}")
    print("NMI:", f"{nmi:.4f}")
    print("Silhouette:", "None" if sil is None else f"{sil:.4f}")
    print("Davies-Bouldin:", "None" if db is None else f"{db:.4f}")
    print("Calinski-Harabasz:", "None" if ch is None else f"{ch:.4f}")

    # t-SNE plot

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_samples = emb.shape[0]
    perplexity = min(30, max(5, (n_samples - 1) // 3)) if n_samples > 1 else 1
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    emb_2d = tsne.fit_transform(emb)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("t-SNE plot skipped (matplotlib not available):", e)
        return y_pred, ari, nmi, sil, db, ch

    plt.figure(figsize=(7, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y, cmap="tab20", s=8, alpha=0.8)
    plt.grid(False)
    plt.axis("off")
    out_path = OUT_DIR / f"tsne_{DATA_NAME}_{DATA_CHOICE}_{MODEL_NAME}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("t-SNE saved to:", out_path)

    return y_pred, ari, nmi, sil, db, ch


if __name__ == "__main__":
    run()
