"""
Compute FSD metrics from on-disk graph data (no hardcoded values).

This script loads graph datasets stored as either:
  - PyTorch Geometric (PyG) objects serialized with torch.save (.pt)
  - Pickled Python objects (.pkl / .pickle)

It computes three metrics directly from the graph:
  - delta_agg: aggregation dilution
  - rho_FS: feature-structure alignment
  - h: homophily

Output: a JSON file `fsd_metrics_computed.json` with computed metrics per dataset file.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GraphData:
    edge_index: torch.Tensor  # [2, E] long
    features: torch.Tensor  # [N, F] float
    labels: Optional[torch.Tensor]  # [N] long (optional)


def _as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _extract_from_mapping(obj: Dict[str, Any]) -> GraphData:
    edge_index = obj.get("edge_index", None)
    x = obj.get("x", obj.get("features", None))
    y = obj.get("y", obj.get("labels", None))
    if edge_index is None or x is None:
        raise ValueError("Missing required keys: edge_index and x/features")
    edge_index_t = _as_tensor(edge_index).long()
    x_t = _as_tensor(x).float()
    y_t = None if y is None else _as_tensor(y).long()
    return GraphData(edge_index=edge_index_t, features=x_t, labels=y_t)


def _extract_from_pyg_data(data_obj: Any) -> GraphData:
    edge_index = getattr(data_obj, "edge_index", None)
    x = getattr(data_obj, "x", None)
    y = getattr(data_obj, "y", None)
    if edge_index is None or x is None:
        raise ValueError("PyG Data is missing edge_index or x")
    edge_index_t = _as_tensor(edge_index).long()
    x_t = _as_tensor(x).float()
    y_t = None if y is None else _as_tensor(y).long()
    return GraphData(edge_index=edge_index_t, features=x_t, labels=y_t)


def load_graph(path: str | os.PathLike[str]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Load a graph from a .pt or pickle file.

    Returns:
        (edge_index, features, labels)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    if suffix == ".pt":
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(p, map_location="cpu")
    elif suffix in {".pkl", ".pickle"}:
        with open(p, "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    if isinstance(obj, (tuple, list)):
        # Common PyG InMemoryDataset storage format: (data_dict, slices, data_cls)
        if len(obj) in {2, 3} and isinstance(obj[0], dict) and "edge_index" in obj[0]:
            gd = _extract_from_mapping(obj[0])
        elif len(obj) == 1:
            gd = _extract_from_pyg_data(obj[0])
        elif len(obj) == 3:
            edge_index, x, y = obj
            gd = GraphData(_as_tensor(edge_index).long(), _as_tensor(x).float(), _as_tensor(y).long())
        elif len(obj) == 2:
            edge_index, x = obj
            gd = GraphData(_as_tensor(edge_index).long(), _as_tensor(x).float(), None)
        else:
            raise ValueError(f"Unsupported tuple/list length: {len(obj)}")
    elif isinstance(obj, dict):
        # Some pipelines store {"data": ..., "slices": ...}
        if "data" in obj:
            d = obj["data"]
            if isinstance(d, dict):
                gd = _extract_from_mapping(d)
            elif hasattr(d, "edge_index") and hasattr(d, "x"):
                # Handle PyG Data objects nested in dict
                gd = _extract_from_pyg_data(d)
            else:
                raise ValueError(f"Cannot extract graph from obj['data']: {type(d)}")
        elif "edge_index" in obj:
            gd = _extract_from_mapping(obj)
        else:
            # Try to find edge_index in nested structures
            raise ValueError(f"Dict missing 'data' or 'edge_index' key. Available keys: {list(obj.keys())}")
    else:
        gd = _extract_from_pyg_data(obj)

    if gd.edge_index.ndim != 2 or gd.edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(gd.edge_index.shape)}")
    if gd.features.ndim != 2:
        raise ValueError(f"features must have shape [N, F], got {tuple(gd.features.shape)}")
    if gd.labels is not None and gd.labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {tuple(gd.labels.shape)}")

    return gd.edge_index, gd.features, gd.labels


def _coalesce_and_sanitize_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = edge_index.long()
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0)

    src = edge_index[0].clamp(min=0, max=num_nodes - 1)
    dst = edge_index[1].clamp(min=0, max=num_nodes - 1)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def compute_delta_agg(edge_index: torch.Tensor, labels: Optional[torch.Tensor]) -> float:
    """
    Compute aggregation dilution (delta_agg).

    Label-based definition (fits the requested signature):
        delta_agg = E_i[ d_i * (1 - p_same(i)) ],
    where p_same(i) is the fraction of i's neighbors that share i's label.

    If labels are unavailable, use `compute_delta_agg_from_features(edge_index, features)` instead.
    """
    if labels is None:
        return float("nan")

    num_nodes = labels.size(0)
    edge_index = _coalesce_and_sanitize_edge_index(edge_index, num_nodes)
    if edge_index.numel() == 0:
        return float("nan")

    src, dst = edge_index[0], edge_index[1]
    same_edge = (labels[src] == labels[dst]).to(torch.float32)

    deg = torch.zeros(num_nodes, device=labels.device, dtype=torch.float32)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))

    same_sum = torch.zeros(num_nodes, device=labels.device, dtype=torch.float32)
    same_sum.scatter_add_(0, src, same_edge)

    mask = deg > 0
    p_same = torch.zeros(num_nodes, device=labels.device, dtype=torch.float32)
    p_same[mask] = same_sum[mask] / deg[mask]
    delta_per_node = deg * (1.0 - p_same)

    return delta_per_node[mask].mean().item()


def compute_delta_agg_from_features(edge_index: torch.Tensor, features: torch.Tensor) -> float:
    edge_index = _coalesce_and_sanitize_edge_index(edge_index, features.size(0))
    if edge_index.numel() == 0:
        return float("nan")

    x_norm = F.normalize(features.float(), p=2, dim=1)
    src, dst = edge_index[0], edge_index[1]

    deg = torch.zeros(features.size(0), device=x_norm.device, dtype=torch.float32)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))

    neigh_sum = torch.zeros_like(x_norm)
    neigh_sum.scatter_add_(0, src.view(-1, 1).expand(-1, x_norm.size(1)), x_norm[dst])

    mask = deg > 0
    neigh_mean = torch.zeros_like(x_norm)
    neigh_mean[mask] = neigh_sum[mask] / deg[mask].unsqueeze(1)
    neigh_mean_norm = F.normalize(neigh_mean, p=2, dim=1)

    sim_to_agg = (x_norm * neigh_mean_norm).sum(dim=1).clamp(min=-1.0, max=1.0)
    delta_per_node = deg * (1.0 - sim_to_agg)

    return delta_per_node[mask].mean().item()


def _sorted_edge_hashes(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    hashes = src * np.int64(num_nodes) + dst
    hashes = np.unique(hashes)
    hashes.sort()
    return hashes


def _sample_non_edges(
    edge_hashes_sorted: np.ndarray,
    num_nodes: int,
    n_samples: int,
    rng: np.random.Generator,
    oversample_factor: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized non-edge sampling by rejection against a sorted edge-hash array.

    Returns:
        (src_idx, dst_idx) arrays of length n_samples.
    """
    if n_samples <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    collected_src: list[np.ndarray] = []
    collected_dst: list[np.ndarray] = []
    remaining = n_samples

    while remaining > 0:
        k = max(remaining * oversample_factor, 1024)
        src = rng.integers(0, num_nodes, size=k, dtype=np.int64)
        dst = rng.integers(0, num_nodes, size=k, dtype=np.int64)
        not_self = src != dst
        src = src[not_self]
        dst = dst[not_self]

        cand_hash = src * np.int64(num_nodes) + dst
        pos = np.searchsorted(edge_hashes_sorted, cand_hash)
        # Clip pos to valid range before indexing
        pos_clipped = np.clip(pos, 0, edge_hashes_sorted.size - 1)
        is_edge = (pos < edge_hashes_sorted.size) & (edge_hashes_sorted[pos_clipped] == cand_hash)
        keep = ~is_edge

        src = src[keep]
        dst = dst[keep]

        if src.size == 0:
            continue

        take = min(remaining, src.size)
        collected_src.append(src[:take])
        collected_dst.append(dst[:take])
        remaining -= take

    return np.concatenate(collected_src), np.concatenate(collected_dst)


def compute_rho_fs(edge_index: torch.Tensor, features: torch.Tensor) -> float:
    """
    Compute feature-structure alignment (rho_FS).

    Definition (unified): rho_FS = E[cos(x_i, x_j) | (i,j) in E] - E[cos(x_i, x_j) | (i,j) not in E]
    Non-edges are estimated via random sampling (vectorized, rejection against existing edges).
    """
    num_nodes = features.size(0)
    edge_index = _coalesce_and_sanitize_edge_index(edge_index, num_nodes)
    if edge_index.numel() == 0:
        return float("nan")

    x_norm = F.normalize(features.float(), p=2, dim=1)
    src, dst = edge_index[0], edge_index[1]

    edge_sims = (x_norm[src] * x_norm[dst]).sum(dim=1)
    mean_edge_sim = edge_sims.mean().item()

    # Sample non-edges: cap to keep runtime bounded for large graphs.
    n_samples = int(min(max(edge_sims.numel(), 1) * 2, 200_000, num_nodes * (num_nodes - 1)))
    rng = np.random.default_rng(0)
    edge_hashes_sorted = _sorted_edge_hashes(edge_index, num_nodes)
    ne_src, ne_dst = _sample_non_edges(edge_hashes_sorted, num_nodes, n_samples, rng=rng)

    if ne_src.size == 0:
        return float("nan")

    ne_src_t = torch.from_numpy(ne_src).to(x_norm.device)
    ne_dst_t = torch.from_numpy(ne_dst).to(x_norm.device)
    non_edge_sims = (x_norm[ne_src_t] * x_norm[ne_dst_t]).sum(dim=1)
    mean_non_edge_sim = non_edge_sims.mean().item()

    return mean_edge_sim - mean_non_edge_sim


def compute_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute homophily ratio: fraction of edges whose endpoints share the same label.
    """
    num_nodes = labels.size(0)
    edge_index = _coalesce_and_sanitize_edge_index(edge_index, num_nodes)
    if edge_index.numel() == 0:
        return float("nan")
    same = (labels[edge_index[0]] == labels[edge_index[1]]).to(torch.float32)
    return same.mean().item()


def _default_search_roots() -> Sequence[Path]:
    return [Path("./data"), Path("./processed"), Path("./ieee_cis_data")]


def _iter_dataset_files(roots: Sequence[Path]) -> Iterable[Path]:
    exts = {".pt", ".pkl", ".pickle"}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def _dataset_key(path: Path, base: Optional[Path]) -> str:
    if base is None:
        return str(path.as_posix())
    try:
        return str(path.relative_to(base).as_posix())
    except Exception:
        return str(path.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FSD metrics from saved graph data files.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[str(p) for p in _default_search_roots()],
        help="Directories to search recursively for .pt/.pkl/.pickle graph files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fsd_metrics_computed.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for computation (e.g., cpu, cuda).",
    )
    args = parser.parse_args()

    roots = [Path(r) for r in args.roots]
    output_path = Path(args.output)
    device = torch.device(args.device)

    files = sorted(set(_iter_dataset_files(roots)))
    if not files:
        raise SystemExit(f"No dataset files found under: {', '.join(str(r) for r in roots)}")

    results: Dict[str, Dict[str, Any]] = {}
    for f in files:
        try:
            edge_index, features, labels = load_graph(f)
            edge_index = edge_index.to(device)
            features = features.to(device)
            if labels is not None:
                labels = labels.to(device)

            rho_fs = compute_rho_fs(edge_index, features)
            if labels is not None:
                delta_agg = compute_delta_agg(edge_index, labels)
                delta_agg_source = "labels"
            else:
                delta_agg = compute_delta_agg_from_features(edge_index, features)
                delta_agg_source = "features"

            homophily = None if labels is None else compute_homophily(edge_index, labels)

            results[_dataset_key(f, base=Path.cwd())] = {
                "path": str(f.as_posix()),
                "num_nodes": int(features.size(0)),
                "num_edges": int(edge_index.size(1)),
                "num_features": int(features.size(1)),
                "has_labels": labels is not None,
                "rho_fs": float(rho_fs),
                "delta_agg": float(delta_agg),
                "delta_agg_source": delta_agg_source,
                "homophily": None if homophily is None else float(homophily),
            }
        except Exception as e:
            results[_dataset_key(f, base=Path.cwd())] = {
                "path": str(f.as_posix()),
                "error": f"{type(e).__name__}: {e}",
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} entries to {output_path.as_posix()}")


if __name__ == "__main__":
    main()
