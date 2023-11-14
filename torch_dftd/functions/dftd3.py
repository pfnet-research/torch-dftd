"""pytorch implementation of Grimme's D3 method"""  # NOQA
from typing import Dict, Optional

import torch
from torch import Tensor
from torch_dftd.functions.distance import calc_distances
from torch_dftd.functions.smoothing import poly_smoothing
from torch_dftd.functions.triplets import calc_triplets

# conversion factors used in grimme d3 code

d3_autoang = 0.52917726  # for converting distance from bohr to angstrom
d3_autoev = 27.21138505  # for converting a.u. to eV

d3_k1 = 16.000
d3_k2 = 4 / 3
d3_k3 = -4.000
d3_maxc = 5  # maximum number of coordination complexes


def _ncoord(
    Z: Tensor,
    r: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    rcov: Tensor,
    cutoff: Optional[float] = None,
    k1: float = d3_k1,
    cutoff_smoothing: str = "none",
    bidirectional: bool = False,
) -> Tensor:
    """Compute coordination numbers by adding an inverse damping function

    Args:
        Z: (n_atoms,)
        r: (n_edges,)
        idx_i: (n_edges,)
        cutoff:
        k1:
        rcov:

    Returns:
        g (Tensor): (n_atoms,) coordination number for each atom
    """
    if cutoff is not None:
        # Calculate _ncoord only for r < cutoff
        indices = torch.nonzero(r <= cutoff).reshape(-1)
        r = r[indices]
        # Zi = Zi[indices]
        # Zj = Zj[indices]
        idx_i = idx_i[indices]
        idx_j = idx_j[indices]
    Zi = Z[idx_i]
    Zj = Z[idx_j]
    rco = rcov[Zi] + rcov[Zj]  # (n_edges,)
    rr = rco.type(r.dtype) / r
    damp = 1.0 / (1.0 + torch.exp(-k1 * (rr - 1.0)))
    if cutoff is not None and cutoff_smoothing == "poly":
        damp *= poly_smoothing(r, cutoff)

    n_atoms = Z.shape[0]
    g = damp.new_zeros((n_atoms,))
    g = g.scatter_add_(0, idx_i, damp)
    if not bidirectional:
        g = g.scatter_add_(0, idx_j, damp)
    return g  # (n_atoms,)


def _getc6(
    Zi: Tensor,
    Zj: Tensor,
    nci: Tensor,
    ncj: Tensor,
    c6ab: Tensor,
    k3: float = d3_k3,
    n_chunks: Optional[int] = None,
) -> Tensor:
    """interpolate c6

    Args:
        Zi: (n_edges,)
        Zj: (n_edges,)
        nci: (n_edges,)
        ncj: (n_edges,)
        c6ab:
        k3:
        n_chunks:

    Returns:
        c6 (Tensor): (n_edges,)
    """
    if n_chunks is None:
        return _getc6_impl(Zi, Zj, nci, ncj, c6ab, k3=k3)

    # TODO(takagi) More balanced split like torch.tensor_split as, for example,
    # trying to split 13 elements into 6 chunks currently gives 5 chunks:
    # ([0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12])
    n_chunks_t = torch.tensor(n_chunks)
    chunk_size = torch.ceil(Zi.shape[0] / n_chunks_t).to(torch.int64)
    c6s = []
    for i in range(0, n_chunks):
        chunk_start = i * chunk_size
        slc = slice(chunk_start, chunk_start + chunk_size)
        c6s.append(_getc6_impl(Zi[slc], Zj[slc], nci[slc], ncj[slc], c6ab, k3=k3))
    return torch.cat(c6s, 0)


def _getc6_impl(
    Zi: Tensor, Zj: Tensor, nci: Tensor, ncj: Tensor, c6ab: Tensor, k3: float = d3_k3
) -> Tensor:
    # gather the relevant entries from the table
    # c6ab (95, 95, 5, 5, 3) --> cni (9025, 5, 5, 1)
    cn0, cn1, cn2 = c6ab.reshape(-1, 5, 5, 3).split(1, dim=3)
    index = Zi * c6ab.size(1) + Zj

    # cni (9025, 5, 5, 1) --> cni (n_edges, 5, 5)
    cn0 = cn0.squeeze(dim=3)[index].type(nci.dtype)
    cn1 = cn1.squeeze(dim=3)[index].type(nci.dtype)
    cn2 = cn2.squeeze(dim=3)[index].type(nci.dtype)

    r = (cn1 - nci[:, None, None]) ** 2 + (cn2 - ncj[:, None, None]) ** 2

    n_edges = r.shape[0]
    n_c6ab = r.shape[1] * r.shape[2]
    if cn0.size(0) == 0:
        k3_rnc = (k3 * r).view(n_edges, n_c6ab)
    else:
        k3_rnc = torch.where(cn0 > 0.0, k3 * r, -1.0e20).view(n_edges, n_c6ab)
    r_ratio = torch.softmax(k3_rnc, dim=1)
    c6 = (r_ratio * cn0.view(n_edges, n_c6ab)).sum(dim=1)
    return c6


def edisp(
    Z: Tensor,
    r: Tensor,
    edge_index: Tensor,
    c6ab: Tensor,
    r0ab: Tensor,
    rcov: Tensor,
    r2r4: Tensor,
    params: Dict[str, float],
    cutoff: Optional[float] = None,
    cnthr: Optional[float] = None,
    batch: Optional[Tensor] = None,
    batch_edge: Optional[Tensor] = None,
    shift_pos: Optional[Tensor] = None,
    pos: Optional[Tensor] = None,
    cell: Optional[Tensor] = None,
    r2=None,
    r6=None,
    r8=None,
    k1=d3_k1,
    k2=d3_k2,
    k3=d3_k3,
    cutoff_smoothing: str = "none",
    damping: str = "zero",
    bidirectional: bool = False,
    abc: bool = False,
    n_chunks: Optional[int] = None,
):
    """compute d3 dispersion energy in Hartree

    Args:
        Z (Tensor): (n_atoms,) atomic numbers
        r (Tensor): (n_edges,) distance in **bohr**
        edge_index (Tensor): (2, n_edges)
        c6ab (Tensor): (n_atom_types, n_atom_types, n_cn=5, n_cn=5, 3) Pre-computed C6AB parameter
        r0ab (Tensor): (n_atom_types, n_atom_types) Pre-computed R0AB parameter
        rcov (Tensor): (n_atom_types,) Pre-computed Rcov parameter
        r2r4 (Tensor): (n_atom_types,) Pre-computed R2R4 parameter
        params (dict): xc-dependent parameters. alp, s6, rs6, s18, rs18.
        cutoff (float or None): cutoff distance in **bohr**
        cnthr (float or None): cutoff distance for coordination number calculation in **bohr**
        batch (Tensor or None): (n_atoms,)
        batch_edge (Tensor or None): (n_edges,)
        shift_pos (Tensor or None): (n_atoms,) used to calculate 3-body term when abc=True
        pos (Tensor): (n_atoms, 3) position in **bohr**
        cell (Tensor): (3, 3) cell size in **bohr**
        r2 (Tensor or None):
        r6 (Tensor or None):
        r8 (Tensor or None):
        k1 (float):
        k2 (float):
        k3 (float):
        cutoff_smoothing (str): cutoff smoothing makes gradient smooth at `cutoff` distance
        damping (str): damping method, only "zero" is supported.
        bidirectional (bool): calculated `edge_index` is bidirectional or not.
        abc (bool): ATM 3-body interaction
        n_chunks (int or None): number of times to split c6 computation to reduce peak memory

    Returns:
        energy: (n_graphs,) Energy in Hartree unit.
    """
    # compute all necessary powers of the distance
    if r2 is None:
        r2 = r**2  # square of distances
    if r6 is None:
        r6 = r2**3
    if r8 is None:
        r8 = r6 * r2

    idx_i, idx_j = edge_index
    # compute all necessary quantities
    Zi = Z[idx_i]  # (n_edges,)
    Zj = Z[idx_j]

    nc = _ncoord(
        Z,
        r,
        idx_i,
        idx_j,
        rcov=rcov,
        cutoff=cnthr,
        cutoff_smoothing=cutoff_smoothing,
        k1=k1,
        bidirectional=bidirectional,
    )  # coordination numbers (n_atoms,)

    nci = nc[idx_i]
    ncj = nc[idx_j]
    c6 = _getc6(Zi, Zj, nci, ncj, c6ab=c6ab, k3=k3, n_chunks=n_chunks)  # c6 coefficients

    c8 = 3 * c6 * r2r4[Zi].type(c6.dtype) * r2r4[Zj].type(c6.dtype)  # c8 coefficient

    s6 = params["s6"]
    s8 = params["s18"]
    if damping in ["bj", "bjm"]:
        a1 = params["rs6"]
        a2 = params["rs18"]

        # Becke-Johnson damping, zero-damping introduces spurious repulsion
        # and is therefore not supported/implemented
        tmp = a1 * torch.sqrt(c8 / c6) + a2
        tmp2 = tmp**2
        tmp6 = tmp2**3
        tmp8 = tmp6 * tmp2
        e6 = 1 / (r6 + tmp6)
        e8 = 1 / (r8 + tmp8)
    elif damping == "zero":
        rs6 = params["rs6"]
        rs8 = params["rs18"]
        alp = params["alp"]
        alp6 = alp
        alp8 = alp + 2.0
        tmp2 = r0ab[Zi, Zj]
        rr = tmp2 / r
        damp6 = 1.0 / (1.0 + 6.0 * (rs6 * rr) ** alp6)
        damp8 = 1.0 / (1.0 + 6.0 * (rs8 * rr) ** alp8)
        e6 = damp6 / r6
        e8 = damp8 / r8
    elif damping == "zerom":
        rs6 = params["rs6"]
        rs8 = params["rs18"]
        alp = params["alp"]
        alp6 = alp
        alp8 = alp + 2.0
        tmp2 = r0ab[Zi, Zj]
        r0_beta = rs8 * tmp2
        rr = r / tmp2
        tmp = rr / rs6 + r0_beta
        damp6 = 1.0 / (1.0 + 6.0 * tmp ** (-alp6))
        tmp = rr + r0_beta
        damp8 = 1.0 / (1.0 + 6.0 * tmp ** (-alp8))
        e6 = damp6 / r6
        e8 = damp8 / r8
    else:
        raise ValueError(f"[ERROR] Unexpected value damping={damping}")

    e6 = -0.5 * s6 * c6 * e6  # (n_edges,)
    e8 = -0.5 * s8 * c8 * e8  # (n_edges,)
    e68 = e6 + e8

    if cutoff is not None and cutoff_smoothing == "poly":
        e68 *= poly_smoothing(r, cutoff)

    if batch_edge is None:
        # (1,)
        g = e68.to(torch.float64).sum()[None]
    else:
        # (n_graphs,)
        if batch.size()[0] == 0:
            n_graphs = 1
        else:
            n_graphs = cell.size(0)
        g = e68.new_zeros((n_graphs,), dtype=torch.float64)
        g.scatter_add_(0, batch_edge, e68.to(torch.float64))

    if not bidirectional:
        g *= 2.0

    if abc:
        within_cutoff = r <= cnthr
        # r_abc = r[within_cutoff]
        # r2_abc = r2[within_cutoff]
        edge_index_abc = edge_index[:, within_cutoff]
        batch_edge_abc = None if batch_edge is None else batch_edge[within_cutoff]
        # c6_abc = c6[within_cutoff]
        shift_abc = None if shift_pos is None else shift_pos[within_cutoff]

        n_atoms = Z.shape[0]
        if not bidirectional:
            # (2, n_edges) -> (2, n_edges * 2)
            edge_index_abc = torch.cat([edge_index_abc, edge_index_abc.flip(dims=[0])], dim=1)
            # (n_edges, ) -> (n_edges * 2, )
            batch_edge_abc = (
                None
                if batch_edge_abc is None
                else torch.cat([batch_edge_abc, batch_edge_abc], dim=0)
            )
            # (n_edges, ) -> (n_edges * 2, )
            shift_abc = None if shift_abc is None else torch.cat([shift_abc, -shift_abc], dim=0)
        with torch.no_grad():
            # triplet_node_index, triplet_edge_index = calc_triplets_cycle(edge_index_abc, n_atoms, shift=shift_abc)
            # Type hinting
            triplet_node_index: Tensor
            multiplicity: Tensor
            edge_jk: Tensor
            batch_triplets: Optional[Tensor]
            triplet_node_index, multiplicity, edge_jk, batch_triplets = calc_triplets(
                edge_index_abc,
                shift_pos=shift_abc,
                dtype=pos.dtype,
                batch_edge=batch_edge_abc,
            )
            batch_triplets = None if batch_edge is None else batch_triplets

        # Apply `cnthr` cutoff threshold for r_kj
        idx_j, idx_k = triplet_node_index[:, 1], triplet_node_index[:, 2]
        shift_jk = (
            None if shift_abc is None else shift_abc[edge_jk[:, 0]] - shift_abc[edge_jk[:, 1]]
        )
        r_jk = calc_distances(pos, torch.stack([idx_j, idx_k], dim=0), cell, shift_jk)
        kj_within_cutoff = r_jk <= cnthr
        del shift_jk

        triplet_node_index = triplet_node_index[kj_within_cutoff]
        multiplicity, edge_jk, batch_triplets = (
            multiplicity[kj_within_cutoff],
            edge_jk[kj_within_cutoff],
            None if batch_triplets is None else batch_triplets[kj_within_cutoff],
        )

        idx_i, idx_j, idx_k = (
            triplet_node_index[:, 0],
            triplet_node_index[:, 1],
            triplet_node_index[:, 2],
        )
        shift_ij = None if shift_abc is None else -shift_abc[edge_jk[:, 0]]
        shift_ik = None if shift_abc is None else -shift_abc[edge_jk[:, 1]]

        r_ij = calc_distances(pos, torch.stack([idx_i, idx_j], dim=0), cell, shift_ij)
        r_ik = calc_distances(pos, torch.stack([idx_i, idx_k], dim=0), cell, shift_ik)
        r_jk = r_jk[kj_within_cutoff]

        Zti, Ztj, Ztk = Z[idx_i], Z[idx_j], Z[idx_k]
        rrjk, rrij, rrik = r0ab[Ztk, Ztj] / r_jk, r0ab[Ztj, Zti] / r_ij, r0ab[Zti, Ztk] / r_ik
        rr3_jk, rr3_ij, rr3_ik = (
            (1.0 / rrjk) ** (1.0 / 3.0),
            (1.0 / rrij) ** (1.0 / 3.0),
            (1.0 / rrik) ** (1.0 / 3.0),
        )
        rav = (4.0 / 3.0) / (rr3_jk * rr3_ij * rr3_ik)
        alp = params["alp"]
        alp8 = alp + 2.0
        damp = 1.0 / (1.0 + 6.0 * rav**alp8)

        c6_mem = torch.zeros((n_atoms, n_atoms), dtype=c6.dtype, device=c6.device)
        c6_mem[edge_index[0], edge_index[1]] = c6
        c6_mem[edge_index[1], edge_index[0]] = c6

        c9 = torch.sqrt(c6_mem[idx_k, idx_j] * c6_mem[idx_j, idx_i] * c6_mem[idx_i, idx_k])
        r2ik, r2jk, r2ij = r_ik**2, r_jk**2, r_ij**2
        t1 = r2jk + r2ij - r2ik
        t2 = r2ij + r2ik - r2jk
        t3 = r2ik + r2jk - r2ij
        tmp2 = r2ik * r2jk * r2ij
        ang = (0.375 * t1 * t2 * t3 / tmp2 + 1.0) / (tmp2**1.5)
        e3 = damp * c9 * ang / multiplicity

        # ---------------------------------------------------------------
        # TODO: support cutoff_smoothing
        if batch_edge is None:
            e6abc = e3.to(torch.float64).sum()
            g += e6abc
        else:
            g.scatter_add_(0, batch_triplets, e3.to(torch.float64))
    return g  # (n_graphs,)
