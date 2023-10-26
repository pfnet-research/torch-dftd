"""pytorch implementation of Grimme's D2 method"""  # NOQA
from typing import Dict, Optional

import torch
from torch import Tensor
from torch_dftd.functions.smoothing import poly_smoothing


def edisp_d2(
    Z: Tensor,
    r: Tensor,
    edge_index: Tensor,
    r0ab: Tensor,
    c6ab: Tensor,
    params: Dict[str, float],
    damping: str = "zero",
    bidirectional: bool = False,
    cutoff: Optional[float] = None,
    batch: Optional[Tensor] = None,
    batch_edge: Optional[Tensor] = None,
    cutoff_smoothing: str = "none",
):
    """compute d3 dispersion energy in Hartree

    Args:
        Z (Tensor): (n_atoms,) atomic numbers
        r (Tensor): (n_edges,) distance in **bohr**
        edge_index (Tensor): (2, n_edges)
        r0ab (Tensor): (n_atom_types, n_atom_types) Pre-computed R0AB parameter.
        c6ab (Tensor): (n_atom_types, n_atom_types) Pre-computed C6AB parameter.
        params (dict): xc-dependent parameters. alp6, s6, rs6.
        damping (str): damping method, only "zero" is supported.
        bidirectional (bool): calculated `edge_index` is bidirectional or not.
        cutoff (float or None): cutoff distance in **bohr**
        batch (Tensor or None): (n_atoms,)
        batch_edge (Tensor or None): (n_edges,)
        cutoff_smoothing (str): cutoff smoothing makes gradient smooth at `cutoff` distance

    Returns:
        energy: (n_graphs,) Energy in Hartree unit.
    """
    # compute all necessary powers of the distance
    # square of distances
    r2 = r**2
    r6 = r2**3

    idx_i, idx_j = edge_index
    # compute all necessary quantities
    Zi = Z[idx_i]  # (n_edges,)
    Zj = Z[idx_j]

    if damping != "zero":
        raise ValueError(
            f"Only zero-damping can be used with the D2 dispersion correction method!"
        )
    alp6 = params["alp"]
    s6 = params["s6"]
    rs6 = params["rs6"]

    r0ab = r0ab.to(r.device)
    c6ab = c6ab.to(r.device)
    c6 = c6ab[Zi, Zj]  # (n_edges,)
    damp6 = 1.0 / (1.0 + torch.exp(-alp6 * (r / (rs6 * r0ab[Zi, Zj]) - 1.0)))
    e6 = damp6 / r6
    e6 = -0.5 * s6 * c6 * e6  # (n_edges,)

    if cutoff is not None and cutoff_smoothing == "poly":
        e6 *= poly_smoothing(r, cutoff)

    if batch_edge is None:
        # (1,)
        g = e6.sum()[None]
    else:
        # (n_graphs,)
        if batch.size()[0] == 0:
            n_graphs = 1
        else:
            n_graphs = int(batch[-1]) + 1
        g = e6.new_zeros((n_graphs,))
        g.scatter_add_(0, batch_edge, e6)

    if not bidirectional:
        g *= 2.0
    return g  # (n_graphs,)
