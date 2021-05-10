from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_dftd.functions.triplets_kernel import _calc_triplets_core_gpu


def calc_triplets(
    edge_index: Tensor,
    shift: Optional[Tensor] = None,
    dtype=torch.float32,
    batch_edge: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate triplet edge index.

    Args:
        edge_index (Tensor): (2, n_edges) edge_index for graph. It must be bidirectional edge.
        shift (Tensor or None): (n_edges, 3) used to calculate unique atoms when pbc=True.
        dtype: dtype for `multiplicity`
        batch_edge (Tensor or None): Specify batch indices for `edge_index`.

    Returns:
        triplet_node_index (Tensor): (3, n_triplets) index for node `i`, `j`, `k` respectively.
           i.e.: idx_i, idx_j, idx_k = triplet_node_index
        multiplicity (Tensor): (n_triplets,) multiplicity indicates duplication of same triplet pair.
            It only takes 1 in non-pbc, but it takes 2 or 3 in pbc case. dtype is specified in the argument.
        triplet_shift (Tensor): (n_triplets, 3=(ij, ik, jk), 3=(xyz)) shift for edge `i->j`, `i->k`, `j->k`.
           i.e.: idx_ij, idx_ik, idx_jk = triplet_shift
        batch_triplets (Tensor): (n_triplets,) batch indices for each triplets.
    """
    dst, src = edge_index
    is_larger = dst >= src
    dst = dst[is_larger]
    src = src[is_larger]
    # sort `src`
    sort_inds = torch.argsort(src)
    src = src[sort_inds]
    dst = dst[sort_inds]

    if shift is None:
        shift = torch.zeros((src.shape[0], 3), dtype=dtype, device=edge_index.device)
    else:
        shift = shift[is_larger][sort_inds]

    if batch_edge is None:
        batch_edge = torch.zeros(src.shape[0], dtype=torch.long, device=edge_index.device)
    else:
        batch_edge = batch_edge[is_larger][sort_inds]

    unique, counts = torch.unique_consecutive(src, return_counts=True)
    counts_cumsum = torch.cumsum(counts, dim=0)
    counts_cumsum = torch.cat(
        [torch.zeros((1,), device=counts.device, dtype=torch.long), counts_cumsum], dim=0
    )

    if str(unique.device) == "cpu":
        return _calc_triplets_core(
            counts, unique, dst, shift, batch_edge, counts_cumsum, dtype=dtype
        )
    else:
        return _calc_triplets_core_gpu(
            counts, unique, dst, shift, batch_edge, counts_cumsum, dtype=dtype
        )


def _calc_triplets_core(counts, unique, dst, shift, batch_edge, counts_cumsum, dtype):
    device = unique.device
    n_triplets = torch.sum(counts * (counts - 1) / 2)
    if n_triplets == 0:
        # (n_triplet_edges, 3)
        triplet_node_index = torch.zeros((0, 3), dtype=torch.long, device=device)
        # (n_triplet_edges)
        multiplicity = torch.zeros((0,), dtype=dtype, device=device)
        # (n_triplet_edges, 3=(ij, ik, jk), 3=(xyz) )
        triplet_shift = torch.zeros((0, 3, 3), dtype=dtype, device=device)
        # (n_triplet_edges)
        batch_triplets = torch.zeros((0,), dtype=torch.long, device=device)
        return triplet_node_index, multiplicity, triplet_shift, batch_triplets

    triplet_node_index_list = []  # (n_triplet_edges, 3)
    shift_list = []  # (n_triplet_edges, 3, 3) represents shift vector
    multiplicity_list = []  # (n_triplet_edges) represents multiplicity
    batch_triplets_list = []  # (n_triplet_edges) represents batch index for triplets
    for i in range(len(unique)):
        _src = unique[i].item()
        _n_edges = counts[i].item()
        _dst = dst[counts_cumsum[i] : counts_cumsum[i + 1]]
        _shift = shift[counts_cumsum[i] : counts_cumsum[i + 1]]
        _batch_index = batch_edge[counts_cumsum[i]].item()
        for j in range(_n_edges - 1):
            for k in range(j + 1, _n_edges):
                _dst0 = _dst[j].item()  # _dst0 maybe swapped with _dst1, need to reset here.
                _dst1 = _dst[k].item()
                batch_triplets_list.append(_batch_index)
                # --- triplet_node_index_list & shift_list in sorted way... ---
                # sort order to be _src <= _dst0 <= _dst1, and i <= _j <= _k
                if _dst0 <= _dst1:
                    _j, _k = j, k
                else:
                    _dst0, _dst1 = _dst1, _dst0
                    _j, _k = k, j

                triplet_node_index_list.append([_src, _dst0, _dst1])
                shift_list.append(
                    torch.stack([-_shift[_j], -_shift[_k], _shift[_j] - _shift[_k]], dim=0)
                )
                # --- multiplicity ---
                if _dst0 == _dst1:
                    if _src == _dst0:
                        # Case 0: _src == _dst0 == _dst1
                        multiplicity_list.append(3.0)
                    else:
                        # Case 1: _src < _dst0 == _dst1
                        multiplicity_list.append(1.0)
                else:
                    if _src == _dst0:
                        # Case 2: _src == _dst0 < _dst1
                        multiplicity_list.append(2.0)
                    else:
                        assert i < _dst0
                        assert i < _dst1
                        # Case 3: i < _dst0 < _dst1
                        multiplicity_list.append(1.0)

    # (n_triplet_edges, 3)
    triplet_node_index = torch.as_tensor(triplet_node_index_list, device=device)
    # (n_triplet_edges)
    multiplicity = torch.as_tensor(multiplicity_list, dtype=dtype, device=device)
    # (n_triplet_edges, 3=(ij, ik, jk), 3=(xyz) )
    triplet_shift = torch.stack(shift_list, dim=0)
    batch_triplets = torch.as_tensor(batch_triplets_list, dtype=torch.long, device=device)
    return triplet_node_index, multiplicity, triplet_shift, batch_triplets
