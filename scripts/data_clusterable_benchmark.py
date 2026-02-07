"""
Clusterability / suitability checks for text datasets using embeddings.

Implements criteria:
  1) Neighborhood signal vs null baseline
  2) Graph community structure (mutual kNN + Louvain/greedy modularity) + conductance
  3) Bootstrap stability (KMeans + Agglomerative): ARI across runs + co-assignment stability (sampled pairs)
  4) Intra vs inter separation ratio (cosine distance)
  5) Cross-embedding agreement (kNN Jaccard + optional ARI between clusterings)
  6) Embedding-based semantic coherence proxies (within-cluster similarity, silhouette per cluster, exemplar indices)
  7) Class imbalance ratio (label distribution)
  8) Hopkins statistic (cluster tendency)

Dependencies:
  - numpy
  - scikit-learn
  - networkx

pip install numpy scikit-learn networkx
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import argparse
import numpy as np
import re

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize

import networkx as nx
import sys
from pathlib import Path

def _add_project_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(10):
        if (cur / "src").is_dir():
            sys.path.insert(0, str(cur))
            return
        cur = cur.parent


_add_project_root_to_syspath()

from src.synthetic.dataprocesser import load_jsonl  # noqa: E402
from src.embedding.pipeline import run_and_save_embeddings  # noqa: E402

DA = "m"
DC = "small"
# DATA_NAME = "extracted_data"
# DATA_CHOICE = f"extract_10_{DA}_{DC}"
DATA_NAME = "banking77"
DATA_CHOICE = "small"
EMBED_MODEL_NAME1 = "instructor"
EMBED_MODEL_NAME2 = "e5"

# DATA_PATH = Path(f"./code/DataGene/datasets/{DATA_NAME}/{DATA_CHOICE}.jsonl")
DATA_PATH = Path(f"/home/wdm/code/DataGene/comparecluster/compact.jsonl")
# DATA_PATH = Path(f"./code/DataGene/result/Qwen_diversity_102_mtop_intent_small_5_gemma.jsonl")
OUT_DIR = Path("./code/DataGene/synthetic_data")

LABEL_LIST: List[Any] | None = None  # e.g. ["change volume", "timer"]



# -----------------------------
# Helpers
# -----------------------------

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def l2_normalize(E: np.ndarray) -> np.ndarray:
    """Return L2-normalized embeddings (row-wise)."""
    E = np.asarray(E, dtype=np.float32)
    return normalize(E, norm="l2", axis=1)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity for normalized vectors."""
    return A @ B.T


def _choose_k_default(n: int) -> int:
    return int(min(50, max(10, round(np.sqrt(n)))))


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _is_finite(val: Optional[float]) -> bool:
    try:
        return val is not None and np.isfinite(val)
    except Exception:
        return False


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_unit(x: Optional[float]) -> float:
    if not _is_finite(x):
        return float("nan")
    return 100.0 * _clamp01(float(x))


def _score_minus1_to1(x: Optional[float]) -> float:
    if not _is_finite(x):
        return float("nan")
    return 100.0 * _clamp01((float(x) + 1.0) / 2.0)


def _score_sigmoid(x: Optional[float]) -> float:
    if not _is_finite(x):
        return float("nan")
    return float(100.0 / (1.0 + np.exp(-float(x))))


def _score_ratio_high(ratio: Optional[float]) -> float:
    if not _is_finite(ratio) or ratio <= 0:
        return float("nan")
    return _score_sigmoid(np.log(float(ratio)))


def _score_ratio_low(ratio: Optional[float]) -> float:
    if not _is_finite(ratio) or ratio <= 0:
        return float("nan")
    return 100.0 * _clamp01(1.0 / float(ratio))


def _score_avg(scores: List[Optional[float]]) -> float:
    vals = [float(s) for s in scores if _is_finite(s)]
    return float(np.mean(vals)) if vals else float("nan")


# -----------------------------
# Legacy (unused): Duplicate / near-duplicate dominance
# -----------------------------

@dataclass
class DuplicateDominanceResult:
    n: int
    unique_texts: int
    exact_duplicate_pct: float
    near_duplicate_pct: float
    near_duplicate_threshold: float
    per_point_nn_sim: np.ndarray


def _normalize_text(text: str, lowercase: bool = True, boilerplate_regex: Optional[str] = None) -> str:
    text = text.strip()
    if boilerplate_regex:
        text = re.sub(boilerplate_regex, "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.lower() if lowercase else text


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def duplicate_dominance(
    texts: List[str],
    E: np.ndarray,
    near_duplicate_threshold: float = 0.98,
    lowercase: bool = True,
    boilerplate_regex: Optional[str] = None,
) -> DuplicateDominanceResult:
    """
    Legacy (unused):
      - exact duplicate rate
      - near-duplicate rate via nearest-neighbor cosine similarity
    """
    n = len(texts)
    if n == 0:
        return DuplicateDominanceResult(
            n=0,
            unique_texts=0,
            exact_duplicate_pct=float("nan"),
            near_duplicate_pct=float("nan"),
            near_duplicate_threshold=near_duplicate_threshold,
            per_point_nn_sim=np.array([], dtype=np.float32),
        )

    norm_texts = [_normalize_text(t, lowercase=lowercase, boilerplate_regex=boilerplate_regex) for t in texts]
    unique = len(set(norm_texts))
    exact_dup_pct = 100.0 * (1.0 - (unique / max(1, n)))

    E = l2_normalize(E)
    if E.shape[0] != n:
        raise ValueError(f"Embeddings rows ({E.shape[0]}) must match texts ({n}).")

    if n < 2:
        per_point_nn_sim = np.array([np.nan], dtype=np.float32)
        near_dup_pct = 0.0
    else:
        nn = NearestNeighbors(n_neighbors=2, metric="cosine")
        nn.fit(E)
        dists, _ = nn.kneighbors(E, return_distance=True)
        # nearest neighbor similarity (excluding self)
        per_point_nn_sim = 1.0 - dists[:, 1]
        near_dup_pct = 100.0 * float(np.mean(per_point_nn_sim >= near_duplicate_threshold))

    return DuplicateDominanceResult(
        n=n,
        unique_texts=unique,
        exact_duplicate_pct=float(exact_dup_pct),
        near_duplicate_pct=float(near_dup_pct),
        near_duplicate_threshold=near_duplicate_threshold,
        per_point_nn_sim=per_point_nn_sim,
    )


# -----------------------------
# Legacy (unused): Ambiguity proxy via length & lexical variety
# -----------------------------

@dataclass
class AmbiguityProxyResult:
    n: int
    length_median: float
    pct_below_short_threshold: float
    short_length_threshold: int
    unique_token_ratio_median: float
    lengths: np.ndarray
    unique_token_ratios: np.ndarray


def ambiguity_proxy(
    texts: List[str],
    short_length_threshold: int = 5,
    lowercase: bool = True,
    boilerplate_regex: Optional[str] = None,
) -> AmbiguityProxyResult:
    """
    Legacy (unused):
      - length distribution
      - unique token ratio (lexical variety proxy)
    """
    n = len(texts)
    if n == 0:
        empty = np.array([], dtype=np.float32)
        return AmbiguityProxyResult(
            n=0,
            length_median=float("nan"),
            pct_below_short_threshold=float("nan"),
            short_length_threshold=short_length_threshold,
            unique_token_ratio_median=float("nan"),
            lengths=empty,
            unique_token_ratios=empty,
        )

    lengths = np.empty(n, dtype=np.float32)
    ratios = np.empty(n, dtype=np.float32)

    for i, t in enumerate(texts):
        norm_t = _normalize_text(t, lowercase=lowercase, boilerplate_regex=boilerplate_regex)
        tokens = _tokenize(norm_t)
        L = len(tokens)
        lengths[i] = L
        ratios[i] = (len(set(tokens)) / max(1, L)) if L > 0 else 0.0

    length_median = float(np.median(lengths))
    pct_short = 100.0 * float(np.mean(lengths < short_length_threshold))
    ratio_median = float(np.median(ratios))

    return AmbiguityProxyResult(
        n=n,
        length_median=length_median,
        pct_below_short_threshold=pct_short,
        short_length_threshold=short_length_threshold,
        unique_token_ratio_median=ratio_median,
        lengths=lengths,
        unique_token_ratios=ratios,
    )


# -----------------------------
# Criterion 7: Class imbalance ratio
# -----------------------------

@dataclass
class ClassImbalanceResult:
    n: int
    num_classes: int
    max_count: int
    min_count: int
    ratio_max_over_min: float
    per_class_counts: Dict[str, int]


def class_imbalance_ratio(labels: List[str]) -> ClassImbalanceResult:
    """
    Criterion 7:
      - max/min label count ratio (ignores empty labels)
    """
    cleaned = [str(lab).strip() for lab in labels if str(lab).strip()]
    n = len(cleaned)
    if n == 0:
        return ClassImbalanceResult(
            n=0,
            num_classes=0,
            max_count=0,
            min_count=0,
            ratio_max_over_min=float("nan"),
            per_class_counts={},
        )

    counts: Dict[str, int] = {}
    for lab in cleaned:
        counts[lab] = counts.get(lab, 0) + 1

    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = float(max_count / min_count) if min_count > 0 else float("inf")

    return ClassImbalanceResult(
        n=n,
        num_classes=len(counts),
        max_count=max_count,
        min_count=min_count,
        ratio_max_over_min=ratio,
        per_class_counts=counts,
    )


# -----------------------------
# Criterion 8: Hopkins statistic (cluster tendency)
# -----------------------------

@dataclass
class HopkinsResult:
    n: int
    sample_size: int
    hopkins: float
    mean_u: float
    mean_w: float


def hopkins_statistic(
    E: np.ndarray,
    sample_size: Optional[int] = None,
    seed: Optional[int] = 0,
) -> HopkinsResult:
    """
    Criterion 8:
      - Hopkins statistic for cluster tendency (0.5 ~ random; >0.5 clusterable).
      - Uses L2-normalized embeddings and Euclidean distance.
    """
    E = l2_normalize(E)
    n, d = E.shape
    if n < 2:
        return HopkinsResult(
            n=n,
            sample_size=0,
            hopkins=float("nan"),
            mean_u=float("nan"),
            mean_w=float("nan"),
        )

    if sample_size is None:
        sample_size = int(min(1000, max(10, round(np.sqrt(n)))))
    sample_size = max(1, min(sample_size, n - 1))

    g = _rng(seed)
    idx = g.choice(n, size=sample_size, replace=False)

    mins = E.min(axis=0)
    maxs = E.max(axis=0)
    U = g.random((sample_size, d)) * (maxs - mins) + mins

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(E)

    # Distances from real points to nearest neighbor (excluding self)
    w = nn.kneighbors(E[idx], n_neighbors=2, return_distance=True)[0][:, 1]
    # Distances from random points to nearest neighbor in data
    u = nn.kneighbors(U, n_neighbors=1, return_distance=True)[0].reshape(-1)

    sum_u = float(u.sum())
    sum_w = float(w.sum())
    hopkins = float(sum_u / (sum_u + sum_w)) if (sum_u + sum_w) > 0 else float("nan")

    return HopkinsResult(
        n=n,
        sample_size=sample_size,
        hopkins=hopkins,
        mean_u=float(np.mean(u)) if u.size else float("nan"),
        mean_w=float(np.mean(w)) if w.size else float("nan"),
    )


# -----------------------------
# Criterion 1: Neighborhood signal vs null
# -----------------------------

@dataclass
class NeighborhoodSignalResult:
    k: int
    mean_knn_sim: float
    mean_rand_sim: float
    delta: float
    effect_size: float  # delta / std(rand_sim)
    per_point_knn_sim: np.ndarray  # shape (N,)
    per_point_rand_sim: np.ndarray  # shape (N,)


def neighborhood_signal_vs_null(
    E: np.ndarray,
    k: Optional[int] = None,
    null_pairs_per_point: Optional[int] = None,
    seed: Optional[int] = 0,
    max_points_for_null: int = 20000,
) -> NeighborhoodSignalResult:
    """
    Criterion 1:
      Compare mean cosine similarity to k-nearest neighbors vs mean cosine similarity to random points.

    Notes:
      - Assumes E are embeddings; will L2-normalize internally.
      - For huge N, null baseline can be computed on a subsample (max_points_for_null).
    """
    E = l2_normalize(E)
    n = E.shape[0]
    if k is None:
        k = _choose_k_default(n)
    k = min(k, n - 1)

    # kNN (cosine distance = 1 - cosine similarity)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="auto")
    nn.fit(E)
    dists, idxs = nn.kneighbors(E, return_distance=True)
    # first neighbor is self (distance 0); convert distances to similarity
    knn_sims = 1.0 - dists[:, 1:]  # shape (N, k)
    per_point_knn_sim = knn_sims.mean(axis=1)

    # Null baseline: random pairs
    g = _rng(seed)
    if null_pairs_per_point is None:
        null_pairs_per_point = k

    # For very large N, compute null on a subsample of points to control time/memory
    if n > max_points_for_null:
        sample_idx = g.choice(n, size=max_points_for_null, replace=False)
        E_null = E[sample_idx]
    else:
        sample_idx = np.arange(n)
        E_null = E

    n_null = E_null.shape[0]
    m = null_pairs_per_point

    # Sample random indices for each point; avoid self (best-effort)
    rand_j = g.integers(0, n, size=(n_null, m))
    self_global = sample_idx[:, None]
    # if any equals self, shift by 1 mod n (simple avoidance)
    mask = rand_j == self_global
    rand_j[mask] = (rand_j[mask] + 1) % n

    # compute random similarities in chunks
    per_point_rand_sim = np.empty(n_null, dtype=np.float32)
    chunk = 2048
    for start in range(0, n_null, chunk):
        end = min(n_null, start + chunk)
        A = E_null[start:end]  # (c, d)
        # shape (c, m, d): m random points per row in A
        B = E[rand_j[start:end]]  # (c, m, d)
        # row-wise cosine similarity to each of the m random points
        sims = np.einsum("id,ijd->ij", A, B)
        per_point_rand_sim[start:end] = sims.mean(axis=1)

    mean_knn = float(per_point_knn_sim.mean())
    mean_rand = float(per_point_rand_sim.mean())
    delta = mean_knn - mean_rand
    std_rand = float(per_point_rand_sim.std(ddof=1)) if per_point_rand_sim.size > 1 else 0.0
    effect = float(delta / (std_rand + 1e-12))

    # If we subsampled for null, expand per_point_rand_sim to size N with NaN for unsampled
    if n > max_points_for_null:
        full_rand = np.full(n, np.nan, dtype=np.float32)
        full_rand[sample_idx] = per_point_rand_sim
        per_point_rand_sim_full = full_rand
    else:
        per_point_rand_sim_full = per_point_rand_sim

    return NeighborhoodSignalResult(
        k=k,
        mean_knn_sim=mean_knn,
        mean_rand_sim=mean_rand,
        delta=float(delta),
        effect_size=effect,
        per_point_knn_sim=per_point_knn_sim,
        per_point_rand_sim=per_point_rand_sim_full,
    )


# -----------------------------
# Criterion 2: Mutual kNN graph communities + modularity + conductance
# -----------------------------

@dataclass
class GraphCommunityResult:
    k: int
    n_nodes: int
    n_edges: int
    communities: List[List[int]]
    modularity: float
    conductance_mean: float
    conductance_per_community: List[float]


def _mutual_knn_graph(E: np.ndarray, k: int) -> nx.Graph:
    """
    Build undirected mutual kNN graph with edge weight = cosine similarity.
    """
    E = l2_normalize(E)
    n = E.shape[0]
    k = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(E)
    dists, idxs = nn.kneighbors(E, return_distance=True)

    # neighbor lists excluding self
    neigh = idxs[:, 1:]  # (N, k)
    neigh_set = [set(neigh[i].tolist()) for i in range(n)]

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add mutual edges; weight = cosine similarity = 1 - cosine distance
    for i in range(n):
        for j_pos, j in enumerate(neigh[i]):
            if i < j and i in neigh_set[j]:
                sim = 1.0 - float(dists[i, j_pos + 1])
                # guard: occasionally numerical weirdness
                sim = max(-1.0, min(1.0, sim))
                G.add_edge(i, j, weight=sim)
    return G


def _detect_communities(G: nx.Graph, seed: Optional[int] = 0) -> List[List[int]]:
    """
    Try Louvain (networkx) if available; otherwise greedy modularity communities.
    """
    # NetworkX Louvain exists in newer versions: nx.algorithms.community.louvain_communities
    try:
        from networkx.algorithms.community import louvain_communities  # type: ignore
        comms = louvain_communities(G, weight="weight", seed=seed)
        return [sorted(list(c)) for c in comms]
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G, weight="weight")
        return [sorted(list(c)) for c in comms]


def graph_community_structure(
    E: np.ndarray,
    k: Optional[int] = None,
    seed: Optional[int] = 0,
) -> GraphCommunityResult:
    """
    Criterion 2:
      Build mutual kNN graph, detect communities, compute modularity and conductance.

    Conductance uses networkx.algorithms.cuts.conductance.
    """
    E = l2_normalize(E)
    n = E.shape[0]
    if k is None:
        k = _choose_k_default(n)
    k = min(k, n - 1)

    G = _mutual_knn_graph(E, k=k)
    comms = _detect_communities(G, seed=seed)

    # modularity
    from networkx.algorithms.community.quality import modularity
    mod = float(modularity(G, [set(c) for c in comms], weight="weight"))

    # conductance per community (lower is better)
    from networkx.algorithms.cuts import conductance
    conds: List[float] = []
    all_nodes = set(G.nodes())
    for c in comms:
        S = set(c)
        # conductance requires non-trivial cut
        if len(S) == 0 or len(S) == len(all_nodes):
            conds.append(np.nan)
        else:
            try:
                conds.append(float(conductance(G, S, T=all_nodes - S, weight="weight")))
            except Exception:
                conds.append(np.nan)

    cond_mean = float(np.nanmean(conds)) if np.any(~np.isnan(conds)) else float("nan")

    return GraphCommunityResult(
        k=k,
        n_nodes=n,
        n_edges=G.number_of_edges(),
        communities=comms,
        modularity=mod,
        conductance_mean=cond_mean,
        conductance_per_community=conds,
    )


# -----------------------------
# Criterion 3: Bootstrap stability (ARI + co-assignment stability)
# -----------------------------

@dataclass
class BootstrapStabilityResult:
    k: int
    n_bootstrap: int
    sample_frac: float
    ari_kmeans: float
    ari_agglomerative: float
    ari_kmeans_all: List[float]
    ari_agglomerative_all: List[float]
    coassign_mean: float
    coassign_hist: Tuple[np.ndarray, np.ndarray]  # (bin_edges, counts)


def infer_k_by_silhouette(
    E: np.ndarray,
    k_min: int = 2,
    k_max: int = 30,
    sample_size: int = 5000,
    seed: Optional[int] = 42,
) -> int:
    """
    Heuristic to pick k using KMeans silhouette on a subsample.
    """
    E = l2_normalize(E)
    n = E.shape[0]
    g = _rng(seed)
    idx = np.arange(n)
    if n > sample_size:
        idx = g.choice(n, size=sample_size, replace=False)
    X = E[idx]

    best_k, best_s = k_min, -1.0
    for k in range(k_min, min(k_max, X.shape[0] - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        s = silhouette_score(X, labels, metric="cosine")
        if s > best_s:
            best_s, best_k = s, k
    return int(best_k)


def _cluster_kmeans(E: np.ndarray, k: int, seed: Optional[int]) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(E)


def _cluster_agglomerative(E: np.ndarray, k: int) -> np.ndarray:
    # cosine in AgglomerativeClustering requires sklearn>=1.2 with metric='cosine' and linkage='average'
    # We'll try cosine; fallback to euclidean on normalized embeddings.
    try:
        return AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average"
        ).fit_predict(E)
    except Exception:
        return AgglomerativeClustering(
            n_clusters=k,
            affinity="euclidean",
            linkage="average"
        ).fit_predict(E)


def bootstrap_stability(
    E: np.ndarray,
    k: Optional[int] = None,
    n_bootstrap: int = 5,
    sample_frac: float = 0.85,
    seed: Optional[int] = 0,
    pair_samples_per_bootstrap: int = 20000,
) -> BootstrapStabilityResult:
    """
    Criterion 3:
      - Bootstrap resample points
      - Cluster with KMeans and Agglomerative
      - Compute ARI across bootstrap runs (pairwise adjacent runs for simplicity)
      - Co-assignment stability: sample random pairs from overlap and compute P(same cluster)

    Notes:
      - If k is None, infer k by silhouette.
      - Co-assignment is approximated by sampled pairs, not all O(N^2) pairs.
    """
    E = l2_normalize(E)
    n = E.shape[0]
    if k is None:
        k = infer_k_by_silhouette(E, k_min=2, k_max=min(30, max(2, n // 50)), seed=seed)

    g = _rng(seed)
    m = int(round(sample_frac * n))
    m = max(2, min(m, n))

    # Store labels on each bootstrap sample as dict: idx -> label
    km_runs: List[Tuple[np.ndarray, np.ndarray]] = []
    ag_runs: List[Tuple[np.ndarray, np.ndarray]] = []

    for b in range(n_bootstrap):
        idx = g.choice(n, size=m, replace=False)
        Xb = E[idx]
        km_lab = _cluster_kmeans(Xb, k=k, seed=None if seed is None else seed + b)
        ag_lab = _cluster_agglomerative(Xb, k=k)
        km_runs.append((idx, km_lab))
        ag_runs.append((idx, ag_lab))

    # ARI across runs on overlaps (adjacent pairs is cheap and usually enough)
    def _adjacent_ari(runs: List[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
        aris: List[float] = []
        for b in range(len(runs) - 1):
            idx1, lab1 = runs[b]
            idx2, lab2 = runs[b + 1]
            common, i1, i2 = np.intersect1d(idx1, idx2, assume_unique=False, return_indices=True)
            if common.size < 2:
                continue
            aris.append(float(adjusted_rand_score(lab1[i1], lab2[i2])))
        return aris

    ari_km_all = _adjacent_ari(km_runs)
    ari_ag_all = _adjacent_ari(ag_runs)

    ari_km = float(np.mean(ari_km_all)) if len(ari_km_all) else float("nan")
    ari_ag = float(np.mean(ari_ag_all)) if len(ari_ag_all) else float("nan")

    # Co-assignment stability (sampled pairs from overlap across all adjacent pairs, using KMeans runs by default)
    same_probs: List[float] = []
    for b in range(len(km_runs) - 1):
        idx1, lab1 = km_runs[b]
        idx2, lab2 = km_runs[b + 1]
        common, i1, i2 = np.intersect1d(idx1, idx2, return_indices=True)
        c = common.size
        if c < 2:
            continue

        # sample pairs in [0, c)
        P = min(pair_samples_per_bootstrap, c * (c - 1) // 2)
        a = g.integers(0, c, size=P)
        b2 = g.integers(0, c, size=P)
        mask = a != b2
        a, b2 = a[mask], b2[mask]
        if a.size == 0:
            continue

        # co-assignment in run 1 and run 2
        same1 = (lab1[i1[a]] == lab1[i1[b2]])
        same2 = (lab2[i2[a]] == lab2[i2[b2]])
        same_probs.append(float(np.mean(same1 == same2)))

    co_mean = float(np.mean(same_probs)) if len(same_probs) else float("nan")
    # histogram for quick inspection
    vals = np.array(same_probs, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, 11)
    counts, edges = np.histogram(vals[~np.isnan(vals)], bins=bins)
    return BootstrapStabilityResult(
        k=int(k),
        n_bootstrap=n_bootstrap,
        sample_frac=sample_frac,
        ari_kmeans=ari_km,
        ari_agglomerative=ari_ag,
        ari_kmeans_all=ari_km_all,
        ari_agglomerative_all=ari_ag_all,
        coassign_mean=co_mean,
        coassign_hist=(edges, counts),
    )


# -----------------------------
# Criterion 4: Intra vs inter separation ratio
# -----------------------------

@dataclass
class SeparationRatioResult:
    intra_mean: float
    inter_mean: float
    ratio_inter_over_intra: float
    centroids: np.ndarray  # (C, d)
    cluster_sizes: Dict[int, int]


def separation_ratio(
    E: np.ndarray,
    labels: np.ndarray,
) -> SeparationRatioResult:
    """
    Criterion 4:
      Compute mean cosine distance to cluster centroid (intra) and mean cosine distance between centroids (inter).
      ratio = inter / intra (bigger is better)
    """
    E = l2_normalize(E)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    # ignore noise label if user uses -1
    uniq = uniq[uniq != -1]

    centroids = []
    sizes: Dict[int, int] = {}
    for c in uniq:
        idx = np.where(labels == c)[0]
        sizes[int(c)] = int(idx.size)
        if idx.size == 0:
            continue
        mu = E[idx].mean(axis=0, keepdims=True)
        mu = l2_normalize(mu)[0]
        centroids.append(mu)
    centroids = np.vstack(centroids) if len(centroids) else np.zeros((0, E.shape[1]), dtype=np.float32)

    # Intra: mean cosine distance to centroid
    intra_dists = []
    for c_i, c in enumerate(uniq):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        sim = (E[idx] @ centroids[c_i])
        dist = 1.0 - sim
        intra_dists.append(dist)
    intra = float(np.mean(np.concatenate(intra_dists))) if len(intra_dists) else float("nan")

    # Inter: mean cosine distance between centroids
    if centroids.shape[0] >= 2:
        sims = cosine_sim_matrix(centroids, centroids)
        # upper triangle without diagonal
        iu = np.triu_indices_from(sims, k=1)
        inter = float(np.mean(1.0 - sims[iu]))
    else:
        inter = float("nan")

    ratio = float(inter / (intra + 1e-12)) if np.isfinite(inter) and np.isfinite(intra) else float("nan")
    return SeparationRatioResult(
        intra_mean=intra,
        inter_mean=inter,
        ratio_inter_over_intra=ratio,
        centroids=centroids,
        cluster_sizes=sizes,
    )


# -----------------------------
# Criterion 5: Cross-embedding agreement
# -----------------------------

@dataclass
class CrossEmbeddingAgreementResult:
    k: int
    knn_jaccard_mean: float
    knn_jaccard_median: float
    per_point_jaccard: np.ndarray
    ari_between_clusterings: Optional[float]


def _knn_indices(E: np.ndarray, k: int) -> np.ndarray:
    E = l2_normalize(E)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(E)
    _, idxs = nn.kneighbors(E, return_distance=True)
    return idxs[:, 1:]  # exclude self


def cross_embedding_agreement(
    E_a: np.ndarray,
    E_b: np.ndarray,
    k: Optional[int] = None,
    compute_cluster_ari: bool = True,
    cluster_k: Optional[int] = None,
    seed: Optional[int] = 0,
) -> CrossEmbeddingAgreementResult:
    """
    Criterion 5:
      - kNN overlap (Jaccard) between embeddings A and B
      - optional ARI between KMeans clusterings computed on each embedding
    """
    E_a = l2_normalize(E_a)
    E_b = l2_normalize(E_b)
    n = E_a.shape[0]
    assert E_b.shape[0] == n, "E_a and E_b must have same number of rows."

    if k is None:
        k = _choose_k_default(n)
    k = min(k, n - 1)

    knn_a = _knn_indices(E_a, k)
    knn_b = _knn_indices(E_b, k)

    per_j = np.empty(n, dtype=np.float32)
    for i in range(n):
        sa = set(knn_a[i].tolist())
        sb = set(knn_b[i].tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        per_j[i] = inter / max(1, union)

    ari_val: Optional[float] = None
    if compute_cluster_ari:
        if cluster_k is None:
            cluster_k = infer_k_by_silhouette(E_a, k_min=2, k_max=min(30, max(2, n // 50)), seed=seed)
        la = _cluster_kmeans(E_a, k=cluster_k, seed=seed)
        lb = _cluster_kmeans(E_b, k=cluster_k, seed=seed)
        ari_val = float(adjusted_rand_score(la, lb))

    return CrossEmbeddingAgreementResult(
        k=k,
        knn_jaccard_mean=float(np.mean(per_j)),
        knn_jaccard_median=float(np.median(per_j)),
        per_point_jaccard=per_j,
        ari_between_clusterings=ari_val,
    )


# -----------------------------
# Criterion 6: Embedding-based semantic coherence proxies (no raw text needed)
# -----------------------------

@dataclass
class CoherenceProxyResult:
    overall_mean_within_cluster_sim: float
    per_cluster_mean_sim: Dict[int, float]
    per_cluster_silhouette_mean: Dict[int, float]
    exemplar_indices: Dict[int, List[int]]  # top-N points closest to centroid
    silhouette_overall: Optional[float]


def coherence_proxies(
    E: np.ndarray,
    labels: np.ndarray,
    exemplars_per_cluster: int = 5,
    pair_samples_per_cluster: int = 5000,
    seed: Optional[int] = 0,
    compute_silhouette: bool = True,
) -> CoherenceProxyResult:
    """
    Criterion 6 (proxy without text):
      - Within-cluster similarity (sampled pairs)
      - Mean silhouette per cluster (cosine metric), if compute_silhouette=True
      - Exemplar indices per cluster (closest to centroid), for manual inspection
    """
    E = l2_normalize(E)
    labels = np.asarray(labels)
    g = _rng(seed)

    uniq = np.unique(labels)
    uniq = uniq[uniq != -1]

    per_cluster_mean_sim: Dict[int, float] = {}
    exemplar_indices: Dict[int, List[int]] = {}

    # Exemplars and within-cluster similarity
    all_pair_sims = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        if idx.size < 2:
            per_cluster_mean_sim[int(c)] = float("nan")
            exemplar_indices[int(c)] = idx.tolist()
            continue

        X = E[idx]

        # centroid exemplar: highest cosine to centroid
        centroid = l2_normalize(X.mean(axis=0, keepdims=True))[0]
        sims_to_centroid = X @ centroid
        top = np.argsort(-sims_to_centroid)[:exemplars_per_cluster]
        exemplar_indices[int(c)] = idx[top].tolist()

        # sampled pairwise sims within cluster
        P = min(pair_samples_per_cluster, idx.size * (idx.size - 1) // 2)
        a = g.integers(0, idx.size, size=P)
        b = g.integers(0, idx.size, size=P)
        mask = a != b
        a, b = a[mask], b[mask]
        if a.size == 0:
            per_cluster_mean_sim[int(c)] = float("nan")
            continue
        sims = np.sum(X[a] * X[b], axis=1)  # cos sim since normalized
        per_cluster_mean_sim[int(c)] = float(np.mean(sims))
        all_pair_sims.append(sims)

    overall_mean = float(np.mean(np.concatenate(all_pair_sims))) if len(all_pair_sims) else float("nan")

    # Silhouette proxies (overall + per cluster)
    per_cluster_sil_mean: Dict[int, float] = {}
    sil_overall: Optional[float] = None
    if compute_silhouette and uniq.size >= 2:
        try:
            sil_samples = silhouette_samples(E, labels, metric="cosine")
            sil_overall = float(np.mean(sil_samples[labels != -1]))
            for c in uniq:
                per_cluster_sil_mean[int(c)] = float(np.mean(sil_samples[labels == c]))
        except Exception:
            sil_overall = None
            for c in uniq:
                per_cluster_sil_mean[int(c)] = float("nan")
    else:
        for c in uniq:
            per_cluster_sil_mean[int(c)] = float("nan")

    return CoherenceProxyResult(
        overall_mean_within_cluster_sim=overall_mean,
        per_cluster_mean_sim=per_cluster_mean_sim,
        per_cluster_silhouette_mean=per_cluster_sil_mean,
        exemplar_indices=exemplar_indices,
        silhouette_overall=sil_overall,
    )


# -----------------------------
# Example usage
# -----------------------------
def _load_embeddings(path: str, npz_key: Optional[str] = None) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if npz_key is None:
            if len(arr.files) != 1:
                raise ValueError(f"NPZ has multiple arrays; pass --embeddings-key. Keys: {arr.files}")
            npz_key = arr.files[0]
        arr = arr[npz_key]
    return np.asarray(arr, dtype=np.float32)


def _infer_label_key(rows: List[Dict[str, Any]], label_key: Optional[str] = None) -> Optional[str]:
    if label_key:
        return label_key
    common_label_keys = ["intent", "label", "class", "domain"]
    common_text_keys = ["text", "utterance", "sentence", "query", "input"]

    sample = rows[0] if rows else {}
    label_key = next((k for k in common_label_keys if k in sample), None)
    if label_key is None:
        for k, v in sample.items():
            if k in common_text_keys:
                continue
            if isinstance(v, str) and v.strip():
                label_key = k
                break
    return label_key


def _load_labels_from_jsonl(path: Path, label_key: Optional[str] = None) -> Optional[List[str]]:
    rows = load_jsonl(path)
    if not rows:
        return None
    label_key = _infer_label_key(rows, label_key=label_key)
    if label_key is None:
        return None
    labels = [str(r.get(label_key, "")).strip() for r in rows if str(r.get(label_key, "")).strip()]
    return labels


def main() -> None:
    def _fmt(val: Optional[float]) -> str:
        if val is None:
            return "None"
        try:
            if np.isnan(val):
                return "nan"
        except Exception:
            pass
        return f"{val:.4f}"

    parser = argparse.ArgumentParser(description="Clusterability / suitability checks for text datasets.")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH), help="JSONL dataset path.")
    parser.add_argument("--data-name", type=str, default=DATA_NAME, help="Dataset name (for embeddings pipeline).")
    parser.add_argument("--data-choice", type=str, default=DATA_CHOICE, help="Dataset choice (for embeddings pipeline).")
    parser.add_argument("--model-name", type=str, default=EMBED_MODEL_NAME1, help="Embedding model name.")
    parser.add_argument("--embeddings", type=str, default=None, help="Optional .npy/.npz embeddings to load.")
    parser.add_argument("--embeddings-key", type=str, default=None, help="Key for .npz embeddings.")
    parser.add_argument("--second-embeddings", type=str, default=None, help="Optional second embeddings file for C5.")
    parser.add_argument("--second-embeddings-key", type=str, default=None, help="Key for second .npz embeddings.")
    parser.add_argument("--second-model-name", type=str, default=EMBED_MODEL_NAME2, help="Second embedding model for C5.")
    parser.add_argument("--label-key", type=str, default=None, help="Override label key for JSONL (used for C7).")
    parser.add_argument("--k", type=int, default=None, help="Neighbor size (default: sqrt(N) rule).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bootstrap", type=int, default=3, help="Number of bootstrap runs.")
    parser.add_argument("--sample-frac", type=float, default=0.85, help="Bootstrap sample fraction.")
    parser.add_argument("--coassign-pairs", type=int, default=20000, help="Pair samples per bootstrap.")
    parser.add_argument("--exemplars", type=int, default=5, help="Exemplars per cluster for C6.")
    parser.add_argument("--hopkins-sample", type=int, default=None, help="Sample size for C8 (Hopkins statistic).")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    label_list: Optional[List[str]] = None
    if data_path.exists():
        label_list = _load_labels_from_jsonl(data_path, label_key=args.label_key)
        if label_list is None:
            print(f"[WARN] label key not found in {data_path}. C7 will be skipped.")
    else:
        print(f"[WARN] data_path not found: {data_path}. C7 will be skipped.")

    if args.embeddings:
        E = _load_embeddings(args.embeddings, npz_key=args.embeddings_key)
    else:
        if not data_path.exists():
            raise FileNotFoundError("data_path not found and no --embeddings provided.")
        E, _ = run_and_save_embeddings(
            data_name=args.data_name,
            data_choice=args.data_choice,
            data_path=str(data_path),
            model_name=args.model_name,
        )

    # ---- Criterion 1
    c1 = neighborhood_signal_vs_null(E, k=args.k, seed=args.seed)
    c1_score = _score_sigmoid(c1.effect_size)
    print("[C1] mean_knn_sim:", _fmt(c1.mean_knn_sim),
          "mean_rand_sim:", _fmt(c1.mean_rand_sim),
          "delta:", _fmt(c1.delta),
          "effect_size:", _fmt(c1.effect_size),
          "score:", _fmt(c1_score))

    # ---- Criterion 2
    c2 = graph_community_structure(E, k=args.k, seed=args.seed)
    c2_score = _score_avg([
        _score_unit(c2.modularity),
        _score_unit(1.0 - c2.conductance_mean),
    ])
    print("[C2] edges:", c2.n_edges, "communities:", len(c2.communities),
          "modularity:", _fmt(c2.modularity), "cond_mean:", _fmt(c2.conductance_mean),
          "score:", _fmt(c2_score))

    # ---- Criterion 3 (stability) + get labels from a chosen clustering
    c3 = bootstrap_stability(
        E,
        k=None,
        n_bootstrap=args.bootstrap,
        sample_frac=args.sample_frac,
        seed=args.seed,
        pair_samples_per_bootstrap=args.coassign_pairs,
    )
    c3_score = _score_avg([
        _score_minus1_to1(c3.ari_kmeans),
        _score_minus1_to1(c3.ari_agglomerative),
        _score_unit(c3.coassign_mean),
    ])
    print("[C3] inferred k:", c3.k,
          "ARI(KMeans):", _fmt(c3.ari_kmeans),
          "ARI(Agg):", _fmt(c3.ari_agglomerative),
          "coassign_mean:", _fmt(c3.coassign_mean),
          "score:", _fmt(c3_score))

    # Produce one clustering to feed Criteria 4/6 (use k from stability)
    cluster_labels = _cluster_kmeans(l2_normalize(E), k=c3.k, seed=args.seed)

    # ---- Criterion 4
    c4 = separation_ratio(E, cluster_labels)
    c4_score = _score_ratio_high(c4.ratio_inter_over_intra)
    print("[C4] intra:", _fmt(c4.intra_mean),
          "inter:", _fmt(c4.inter_mean),
          "ratio:", _fmt(c4.ratio_inter_over_intra),
          "score:", _fmt(c4_score))

    # ---- Criterion 5 (requires second embedding set)
    E2 = None
    if args.second_embeddings:
        E2 = _load_embeddings(args.second_embeddings, npz_key=args.second_embeddings_key)
    elif args.second_model_name:
        E2, _ = run_and_save_embeddings(
            data_name=args.data_name,
            data_choice=args.data_choice,
            data_path=str(data_path),
            model_name=args.second_model_name,
        )

    if E2 is not None:
        c5 = cross_embedding_agreement(E, E2, k=args.k, compute_cluster_ari=True, cluster_k=c3.k, seed=args.seed)
        c5_score = _score_avg([
            _score_unit(c5.knn_jaccard_mean),
            _score_minus1_to1(c5.ari_between_clusterings),
        ])
        print("[C5] knn_jaccard_mean:", _fmt(c5.knn_jaccard_mean),
              "knn_jaccard_median:", _fmt(c5.knn_jaccard_median),
              "ARI_between_clusterings:", _fmt(c5.ari_between_clusterings),
              "score:", _fmt(c5_score))
    else:
        print("[C5] skipped (no second embedding set).")

    # ---- Criterion 6
    c6 = coherence_proxies(E, cluster_labels, exemplars_per_cluster=args.exemplars, seed=args.seed)
    c6_score = _score_avg([
        _score_minus1_to1(c6.overall_mean_within_cluster_sim),
        _score_minus1_to1(c6.silhouette_overall),
    ])
    print("[C6] overall_mean_within_cluster_sim:", _fmt(c6.overall_mean_within_cluster_sim),
          "silhouette_overall:", _fmt(c6.silhouette_overall),
          "score:", _fmt(c6_score))

    # ---- Criterion 7
    if label_list is not None:
        c7 = class_imbalance_ratio(label_list)
        c7_score = _score_ratio_low(c7.ratio_max_over_min)
        print("[C7] n_labels:", c7.n,
              "classes:", c7.num_classes,
              "max:", c7.max_count,
              "min:", c7.min_count,
              "ratio:", _fmt(c7.ratio_max_over_min),
              "score:", _fmt(c7_score))
    else:
        print("[C7] skipped (no labels).")

    # ---- Criterion 8
    c8 = hopkins_statistic(E, sample_size=args.hopkins_sample, seed=args.seed)
    c8_score = _score_unit(c8.hopkins)
    print("[C8] hopkins:", _fmt(c8.hopkins),
          "sample:", c8.sample_size,
          "mean_u:", _fmt(c8.mean_u),
          "mean_w:", _fmt(c8.mean_w),
          "score:", _fmt(c8_score))


if __name__ == "__main__":
    main()
