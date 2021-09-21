from typing import Tuple

import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack, to_dlpack

try:
    import cupy as cp

    _cupy_available = True
except ImportError:
    import numpy as cp  # Dummy for mypy annotation.

    _cupy_available = False

try:
    import pytorch_pfn_extras as ppe

    _ppe_available = True
except ImportError:
    _ppe_available = False


if _ppe_available and _cupy_available:
    ppe.cuda.use_torch_mempool_in_cupy()


def _torch2cupy(tensor: Tensor) -> cp.ndarray:
    return cp.fromDlpack(to_dlpack(tensor))


def _cupy2torch(array: cp.ndarray) -> Tensor:
    return from_dlpack(array.toDlpack())


if _cupy_available:
    _calc_triplets_core_gpu_kernel = cp.ElementwiseKernel(
        "raw int64 counts, raw int64 unique, raw int64 dst, raw int64 edge_indices, raw int64 batch_edge, raw int64 counts_cumsum",
        "raw int64 triplet_node_index, raw T multiplicity, raw int64 edge_jk, raw int64 batch_triplets",
        """
        long long n_unique = unique.size();
        long long a = 0;
        // a, b, c corresponds to i, j, k in the original function.
        long long current_counts = 0;
        long long _i = 0;
        for (a = 0; a < n_unique; a++) {
            current_counts += counts[a] * (counts[a] - 1) / 2;
            if (i < current_counts) {
                _i = i - (current_counts - counts[a] * (counts[a] - 1) / 2);
                break;
            }
        }
    
        long long _src = unique[a];
        long long _n_edges = counts[a];
        long long _offset = counts_cumsum[a];
        long long _batch_index = batch_edge[_offset];
    
        long long b, c;
        for (b = 1; b < _n_edges; b++) {
            if (_i < (2 * _n_edges - b - 1) * b / 2) {
                b -= 1;
                c = _i - (2 * _n_edges - b - 1) * b / 2 + b + 1;
                break;
            }
        }
        long long _dst0 = dst[_offset + b];
        long long _dst1 = dst[_offset + c];
        if (_dst0 > _dst1) {
            // Swap _dst0 & _dst1, b & c.
            long long tmp = _dst0;
            _dst0 = _dst1;
            _dst1 = tmp;
            tmp = b;
            b = c;
            c = tmp;
        }
    
        // --- triplet_node_index ---
        triplet_node_index[3 * i] = _src;   // idx_i
        triplet_node_index[3 * i + 1] = _dst0;  // idx_j
        triplet_node_index[3 * i + 2] = _dst1;  // idx_k
    
        // --- multiplicity ---
        if (_dst0 == _dst1) {
            if (_src == _dst0) {
                // Case 0: _src == _dst0 == _dst1
                multiplicity[i] = 3.0;
            } else {
                // Case 1: _src < _dst0 == _dst1
                multiplicity[i] = 1.0;
            }
        } else {
            if (_src == _dst0) {
                // Case 2: _src == _dst0 < _dst1
                multiplicity[i] = 2.0;
            } else {
                // Case 3: i < _dst0 < _dst1
                multiplicity[i] = 1.0;
            }
        }
    
        // --- edge_jk ---
        edge_jk[2 * i] = edge_indices[_offset + b];
        edge_jk[2 * i + 1] = edge_indices[_offset + c];
    
        // --- batch_triplets ---
        batch_triplets[i] = _batch_index;
        """,
        "_calc_triplets_core_gpu_kernel",
    )
else:
    _calc_triplets_core_gpu_kernel = None


def _calc_triplets_core_gpu(
    counts: Tensor,
    unique: Tensor,
    dst: Tensor,
    edge_indices: Tensor,
    batch_edge: Tensor,
    counts_cumsum: Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if not _ppe_available:
        raise ImportError("Please install pytorch_pfn_extras to use `_calc_triplets_core_gpu`!")
    if not _cupy_available:
        raise ImportError("Please install cupy to use `_calc_triplets_core_gpu`!")
    device = unique.device
    n_triplets = int(torch.sum(counts * (counts - 1) / 2).item())

    # (n_triplet_edges, 3)
    triplet_node_index = torch.zeros((n_triplets, 3), dtype=torch.long, device=device)
    # (n_triplet_edges)
    multiplicity = torch.zeros((n_triplets,), dtype=dtype, device=device)
    # (n_triplet_edges, 2=(j, k))
    edge_jk = torch.zeros((n_triplets, 2), dtype=torch.long, device=device)
    # (n_triplet_edges)
    batch_triplets = torch.zeros((n_triplets,), dtype=torch.long, device=device)
    if n_triplets == 0:
        return triplet_node_index, multiplicity, edge_jk, batch_triplets

    _calc_triplets_core_gpu_kernel(
        _torch2cupy(counts),
        _torch2cupy(unique),
        _torch2cupy(dst),
        _torch2cupy(edge_indices),
        _torch2cupy(batch_edge),
        _torch2cupy(counts_cumsum),
        # n_triplets,
        _torch2cupy(triplet_node_index),
        _torch2cupy(multiplicity),
        _torch2cupy(edge_jk),
        _torch2cupy(batch_triplets),
        size=n_triplets,
    )
    # torch tensor buffer is already modified in above cupy functions.
    return triplet_node_index, multiplicity, edge_jk, batch_triplets
