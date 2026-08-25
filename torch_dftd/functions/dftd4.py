"""DFT-D4 dispersion energy (Caldeweyher et al., J. Chem. Phys. 150, 154122 (2019)) in torch-dftd's
edge-list formulation, with periodic EEQ charges evaluated by Ewald summation.

Differences from ``dftd3.py``: D4 needs (i) an electronegativity-weighted erf coordination number,
(ii) EEQ partial charges (a linear solve; Ewald for PBC), (iii) C6 from charge- and CN-weighted
reference polarizabilities. Damping is Becke-Johnson only. All distances/energies here are in
Bohr/Hartree; the caller converts.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from torch_dftd.functions.distance import calc_distances
from torch_dftd.functions.smoothing import poly_smoothing
from torch_dftd.functions.triplets import calc_triplets

# --- constants (dftd4 / tad-dftd4 defaults)
D4_KCN = 7.5
D4_K4 = 4.10451
D4_K5 = 19.08857
D4_K6 = 2 * 11.28174**2
EEQ_CN_MAX = 8.0
GA_DEFAULT = 3.0
GC_DEFAULT = 2.0
WF_DEFAULT = 6.0
SQRT2PI = math.sqrt(2.0 / math.pi)


def _scatter_pairs(
    vals: Tensor, idx_i: Tensor, idx_j: Tensor, n: int, bidirectional: bool
) -> Tensor:
    """Sum per-edge values onto atoms. For unidirectional edge lists each pair contributes to both ends."""
    out = vals.new_zeros(n)
    out.index_add_(0, idx_i, vals)
    if not bidirectional:
        out.index_add_(0, idx_j, vals)
    return out


def _erf_count(r: Tensor, rc: Tensor, kcn: float = D4_KCN) -> Tensor:
    return 0.5 * (1.0 + torch.erf(-kcn * (r / rc - 1.0)))


def ncoord_d4(Z, r, idx_i, idx_j, rcov, en, cutoff, bidirectional, n_atoms) -> Tensor:
    """Electronegativity-weighted D4 coordination number."""
    m = r <= cutoff
    Zi, Zj = Z[idx_i], Z[idx_j]
    rc = rcov[Zi] + rcov[Zj]
    endiff = torch.abs(en[Zi] - en[Zj])
    w = D4_K4 * torch.exp(-((endiff + D4_K5) ** 2) / D4_K6)
    cnt = torch.where(m, _erf_count(r, rc) * w, torch.zeros_like(r))
    return _scatter_pairs(cnt, idx_i, idx_j, n_atoms, bidirectional)


def ncoord_eeq(
    Z, r, idx_i, idx_j, rcov, cutoff, bidirectional, n_atoms, cn_max=EEQ_CN_MAX
) -> Tensor:
    """EEQ coordination number (erf counting, no EN weight, smooth log cutoff at cn_max)."""
    m = r <= cutoff
    rc = rcov[Z[idx_i]] + rcov[Z[idx_j]]
    cnt = torch.where(m, _erf_count(r, rc), torch.zeros_like(r))
    cn = _scatter_pairs(cnt, idx_i, idx_j, n_atoms, bidirectional)
    cn_max_t = torch.tensor(cn_max, dtype=cn.dtype, device=cn.device)
    return torch.log1p(torch.exp(cn_max_t)) - torch.log1p(torch.exp(cn_max_t - cn))


def _reciprocal_vectors(cell: Tensor, gcut: float) -> Tuple[Tensor, Tensor]:
    """Half-space set of reciprocal lattice vectors with |G| <= gcut (G != 0) and their multiplicity (2)."""
    rec = 2.0 * math.pi * torch.linalg.inv(cell).transpose(0, 1)  # rows = b1, b2, b3
    # number of images needed along each reciprocal axis
    lengths = torch.linalg.norm(cell, dim=1)
    nmax = [int(math.ceil(gcut * float(l) / (2.0 * math.pi))) + 1 for l in lengths]
    dev = cell.device
    grids = torch.meshgrid(
        *[torch.arange(-n, n + 1, device=dev, dtype=cell.dtype) for n in nmax], indexing="ij"
    )
    nvec = torch.stack([g.reshape(-1) for g in grids], dim=1)  # (M, 3)
    # keep lexicographically positive half (G and -G give identical cos terms)
    key = nvec[:, 0] * 1e6 + nvec[:, 1] * 1e3 + nvec[:, 2]
    nvec = nvec[key > 0]
    G = nvec @ rec  # (M, 3)
    g2 = torch.sum(G * G, dim=1)
    keep = g2 <= gcut * gcut
    return G[keep], g2[keep]


def eeq_charges(
    Z: Tensor,
    pos: Tensor,
    r: Tensor,
    edge_index: Tensor,
    cn_eeq: Tensor,
    chi: Tensor,
    eta: Tensor,
    kcn: Tensor,
    rad: Tensor,
    cell: Optional[Tensor],
    pbc: Optional[Tensor],
    batch: Optional[Tensor],
    batch_edge: Optional[Tensor],
    n_graphs: int,
    total_charge: Tensor,
    eeq_cutoff: float,
    ewald_tol: float,
    bidirectional: bool,
) -> Tensor:
    """EEQ partial charges for every graph. Periodic graphs use Ewald summation of the
    erf(gamma r)/r Coulomb kernel; molecular graphs use the dense pairwise kernel.
    Linear algebra is done in float64. Returns q (n_atoms,) in the dtype of ``pos``.
    """
    n_atoms = Z.shape[0]
    dev = pos.device
    d64 = torch.float64
    q_out = pos.new_zeros(n_atoms, dtype=d64)
    idx_i, idx_j = edge_index
    batch_a = torch.zeros(n_atoms, dtype=torch.long, device=dev) if batch is None else batch
    batch_e = (
        torch.zeros(idx_i.shape[0], dtype=torch.long, device=dev)
        if batch_edge is None
        else batch_edge
    )
    r64 = r.to(d64)
    pos64 = pos.to(d64)

    for g in range(n_graphs):
        amask = batch_a == g
        aidx = torch.nonzero(amask, as_tuple=False).reshape(-1)
        n = int(aidx.shape[0])
        if n == 0:
            continue
        local = torch.full((n_atoms,), -1, dtype=torch.long, device=dev)
        local[aidx] = torch.arange(n, device=dev)
        Zg = Z[aidx]
        radg = rad[Zg].to(d64)
        etag = eta[Zg].to(d64)
        rhs = -chi[Zg].to(d64) + kcn[Zg].to(d64) * torch.sqrt(
            torch.clamp(cn_eeq[aidx].to(d64), min=0.0)
        )
        periodic = (
            cell is not None
            and pbc is not None
            and bool(torch.any(pbc[g] if pbc.dim() == 2 else pbc))
        )
        if periodic:
            pb = pbc[g] if pbc.dim() == 2 else pbc
            if not bool(torch.all(pb)):
                raise NotImplementedError(
                    "DFT-D4 EEQ: only fully periodic or non-periodic systems are supported"
                )
            cellg = (cell[g] if cell.dim() == 3 else cell).to(d64)
            vol = torch.abs(torch.det(cellg))
            # Ewald splitting: real-space part decays as erfc(alpha r) -> tol at eeq_cutoff
            sqlog = math.sqrt(-math.log(ewald_tol))
            alpha = sqlog / eeq_cutoff
            gcut = 2.0 * alpha * sqlog
            # real space over edges: [erf(gamma r) - erf(alpha r)] / r
            emask = (batch_e == g) & (r64 <= eeq_cutoff)
            ei, ej, er = local[idx_i[emask]], local[idx_j[emask]], r64[emask]
            gam = 1.0 / torch.sqrt(radg[ei] ** 2 + radg[ej] ** 2)
            v = (torch.erf(gam * er) - torch.erf(alpha * er)) / er
            A = torch.zeros((n, n), dtype=d64, device=dev)
            A.index_put_((ei, ej), v, accumulate=True)
            if not bidirectional:
                A.index_put_((ej, ei), v, accumulate=True)
            # reciprocal space
            G, g2 = _reciprocal_vectors(cellg, gcut)
            wG = (
                2.0 * (4.0 * math.pi / vol) * torch.exp(-g2 / (4.0 * alpha * alpha)) / g2
            )  # x2: half space
            phase = pos64[aidx] @ G.transpose(0, 1)  # (n, M)
            C, S = torch.cos(phase), torch.sin(phase)
            A = A + (C * wG) @ C.transpose(0, 1) + (S * wG) @ S.transpose(0, 1)
            # neutralizing background and self terms
            A = A - math.pi / (vol * alpha * alpha)
            diag = etag + SQRT2PI / radg - 2.0 * alpha / math.sqrt(math.pi)
            A = A + torch.diag(diag)
        else:
            p = pos64[aidx]
            d = torch.cdist(p, p)
            eye = torch.eye(n, dtype=torch.bool, device=dev)
            d_safe = torch.where(eye, torch.ones_like(d), d)
            gam = 1.0 / torch.sqrt(radg.unsqueeze(0) ** 2 + radg.unsqueeze(1) ** 2)
            A = torch.where(eye, torch.zeros_like(d), torch.erf(gam * d_safe) / d_safe)
            A = A + torch.diag(etag + SQRT2PI / radg)
        # bordered system with charge constraint
        ones = torch.ones((n, 1), dtype=d64, device=dev)
        M = torch.cat(
            [
                torch.cat([A, ones], dim=1),
                torch.cat(
                    [ones.transpose(0, 1), torch.zeros((1, 1), dtype=d64, device=dev)], dim=1
                ),
            ],
            dim=0,
        )
        b = torch.cat([rhs, total_charge[g].to(d64).reshape(1)])
        x = torch.linalg.solve(M, b)
        q_out = q_out.index_copy(0, aidx, x[:n])
    return q_out.to(pos.dtype)


def weight_references(Z, cn, q, refc, refcovcn, refq, zeff, gam, ga, gc, wf) -> Tensor:
    """Gaussian CN weights times charge-scaling zeta for the reference systems, shape (n_atoms, 7)."""
    d64 = torch.float64
    rc = refc[Z]  # (n, 7) int
    mask = rc > 0
    refcn = refcovcn[Z].to(d64)
    dcn = cn.to(d64).unsqueeze(-1) - refcn
    tmp = torch.exp(-dcn * dcn)
    expw = torch.zeros_like(tmp)
    for k in range(1, 4):  # refc is 1 or 3
        expw = expw + torch.where(rc >= k, torch.pow(tmp, k * wf), torch.zeros_like(tmp))
    expw = torch.where(mask, expw, torch.zeros_like(expw))
    norm = torch.sum(expw, dim=-1, keepdim=True)
    gw = expw / torch.where(norm > 0, norm, torch.ones_like(norm))
    # underflow fallback: put all weight on the reference with the largest CN
    maxcn = torch.max(
        torch.where(mask, refcn, torch.full_like(refcn, -1.0)), dim=-1, keepdim=True
    )[0]
    bad = (norm <= 0) | ~torch.isfinite(norm)
    gw = torch.where(bad.expand_as(gw), (refcn == maxcn).to(d64) * mask.to(d64), gw)
    zf = zeff[Z].to(d64).unsqueeze(-1)
    gm = gam[Z].to(d64).unsqueeze(-1) * gc
    qref = refq[Z].to(d64) + zf
    qmod = q.to(d64).unsqueeze(-1) + zf
    scale = torch.exp(gm * (1.0 - qref / torch.where(qmod > 0, qmod, torch.ones_like(qmod))))
    zeta = torch.where(
        qmod > 0.0, torch.exp(ga * (1.0 - scale)), torch.full_like(scale, math.exp(ga))
    )
    zeta = torch.where(mask, zeta, torch.zeros_like(zeta))
    return (zeta * gw).to(cn.dtype)


def _edge_c6(Zi, Zj, gwi, gwj, rc6, chunk: int = 500000) -> Tensor:
    """C6 per edge from the reference-C6 table (nz, nz, 7, 7); chunked to bound the (E, 7, 7) gather."""
    n = Zi.shape[0]
    if n <= chunk:
        return torch.einsum("eab,ea,eb->e", rc6[Zi, Zj].to(gwi.dtype), gwi, gwj)
    return torch.cat(
        [
            torch.einsum(
                "eab,ea,eb->e",
                rc6[Zi[s : s + chunk], Zj[s : s + chunk]].to(gwi.dtype),
                gwi[s : s + chunk],
                gwj[s : s + chunk],
            )
            for s in range(0, n, chunk)
        ]
    )


def edisp_d4(
    Z: Tensor,
    r: Tensor,
    edge_index: Tensor,
    pos: Tensor,
    cell: Optional[Tensor],
    pbc: Optional[Tensor],
    shift_pos: Optional[Tensor],
    batch: Optional[Tensor],
    batch_edge: Optional[Tensor],
    params: Dict[str, float],
    tables: Dict[str, Tensor],
    cutoff: float,
    cnthr: float,
    cn_eeq_thr: float,
    eeq_cutoff: float,
    abc_cutoff: float,
    total_charge: Optional[Tensor] = None,
    abc: bool = True,
    bidirectional: bool = False,
    cutoff_smoothing: str = "none",
    ewald_tol: float = 1e-8,
    ga: float = GA_DEFAULT,
    gc: float = GC_DEFAULT,
    wf: float = WF_DEFAULT,
    return_charges: bool = False,
):
    """Compute the DFT-D4 dispersion energy in Hartree, per graph (n_graphs,).

    Args:
        Z: (n_atoms,) atomic numbers; r: (n_edges,) distances in Bohr; edge_index (2, n_edges);
        pos (n_atoms, 3) and cell (bs, 3, 3) in Bohr; params: s6, s8, s9, a1, a2, alp;
        tables: rcov, r4r2, en, zeff, gam, refc, refcovcn, refq, rc6, eeq_chi/eta/kcn/rad;
        cutoff / cnthr / cn_eeq_thr / eeq_cutoff / abc_cutoff in Bohr.
    """
    n_atoms = Z.shape[0]
    idx_i, idx_j = edge_index
    Zi, Zj = Z[idx_i], Z[idx_j]
    if batch is None or batch.numel() == 0:
        n_graphs = 1
    else:
        n_graphs = int(batch[-1]) + 1
    if total_charge is None:
        total_charge = torch.zeros(n_graphs, dtype=pos.dtype, device=pos.device)

    cn = ncoord_d4(Z, r, idx_i, idx_j, tables["rcov"], tables["en"], cnthr, bidirectional, n_atoms)
    cn_eeq = ncoord_eeq(Z, r, idx_i, idx_j, tables["rcov"], cn_eeq_thr, bidirectional, n_atoms)
    q = eeq_charges(
        Z,
        pos,
        r,
        edge_index,
        cn_eeq,
        tables["eeq_chi"],
        tables["eeq_eta"],
        tables["eeq_kcn"],
        tables["eeq_rad"],
        cell,
        pbc,
        batch,
        batch_edge,
        n_graphs,
        total_charge,
        eeq_cutoff,
        ewald_tol,
        bidirectional,
    )
    gw = weight_references(
        Z,
        cn,
        q,
        tables["refc"],
        tables["refcovcn"],
        tables["refq"],
        tables["zeff"],
        tables["gam"],
        ga,
        gc,
        wf,
    )
    within = r <= cutoff
    c6 = _edge_c6(Zi, Zj, gw[idx_i], gw[idx_j], tables["rc6"])
    r4r2 = tables["r4r2"].to(c6.dtype)
    qq = 3.0 * r4r2[Zi] * r4r2[Zj]
    c8 = c6 * qq
    s6, s8, a1, a2 = params["s6"], params["s8"], params["a1"], params["a2"]
    r0 = a1 * torch.sqrt(qq) + a2
    r2 = r * r
    r6 = r2 * r2 * r2
    r8 = r6 * r2
    r0_2 = r0 * r0
    r0_6 = r0_2 * r0_2 * r0_2
    r0_8 = r0_6 * r0_2
    e68 = -0.5 * (s6 * c6 / (r6 + r0_6) + s8 * c8 / (r8 + r0_8))
    e68 = torch.where(within, e68, torch.zeros_like(e68))
    if cutoff_smoothing == "poly":
        e68 = e68 * poly_smoothing(r, cutoff)
    if batch_edge is None:
        g = e68.to(torch.float64).sum()[None]
    else:
        g = e68.new_zeros((n_graphs,), dtype=torch.float64)
        g.scatter_add_(0, batch_edge, e68.to(torch.float64))
    if not bidirectional:
        g = g * 2.0

    if abc and params.get("s9", 1.0) != 0.0:
        # ATM three-body term: C9 from charge-independent C6 (q = 0), BJ radii, zero damping (alp/3 on the triple product)
        gw0 = weight_references(
            Z,
            cn,
            torch.zeros_like(q),
            tables["refc"],
            tables["refcovcn"],
            tables["refq"],
            tables["zeff"],
            tables["gam"],
            ga,
            gc,
            wf,
        )
        within_abc = r <= abc_cutoff
        edge_abc = edge_index[:, within_abc]
        batch_edge_abc = None if batch_edge is None else batch_edge[within_abc]
        shift_abc = None if shift_pos is None else shift_pos[within_abc]
        if not bidirectional:
            edge_abc = torch.cat([edge_abc, edge_abc.flip(dims=[0])], dim=1)
            batch_edge_abc = (
                None
                if batch_edge_abc is None
                else torch.cat([batch_edge_abc, batch_edge_abc], dim=0)
            )
            shift_abc = None if shift_abc is None else torch.cat([shift_abc, -shift_abc], dim=0)
        with torch.no_grad():
            triplet_node_index, multiplicity, edge_jk, batch_triplets = calc_triplets(
                edge_abc, shift_pos=shift_abc, dtype=pos.dtype, batch_edge=batch_edge_abc
            )
            batch_triplets = None if batch_edge is None else batch_triplets
        tj, tk = triplet_node_index[:, 1], triplet_node_index[:, 2]
        shift_jk = (
            None if shift_abc is None else shift_abc[edge_jk[:, 0]] - shift_abc[edge_jk[:, 1]]
        )
        r_jk = calc_distances(pos, torch.stack([tj, tk], dim=0), cell, shift_jk)
        ok = r_jk <= abc_cutoff
        triplet_node_index, multiplicity, edge_jk = (
            triplet_node_index[ok],
            multiplicity[ok],
            edge_jk[ok],
        )
        batch_triplets = None if batch_triplets is None else batch_triplets[ok]
        r_jk = r_jk[ok]
        ti, tj, tk = triplet_node_index[:, 0], triplet_node_index[:, 1], triplet_node_index[:, 2]
        shift_ij = None if shift_abc is None else -shift_abc[edge_jk[:, 0]]
        shift_ik = None if shift_abc is None else -shift_abc[edge_jk[:, 1]]
        r_ij = calc_distances(pos, torch.stack([ti, tj], dim=0), cell, shift_ij)
        r_ik = calc_distances(pos, torch.stack([ti, tk], dim=0), cell, shift_ik)
        # C6 lookup table for the triplets. NOTE: fill it from *unique* atom pairs. Writing the same
        # (i,j) position once per periodic image (as dftd3.py does) leaves the energy unchanged but
        # makes index_put_'s backward hand the full dE/dC6 to every duplicate -> wrong forces under PBC.
        lo = torch.minimum(edge_abc[0], edge_abc[1])
        hi = torch.maximum(edge_abc[0], edge_abc[1])
        upair = torch.unique(lo * n_atoms + hi)
        ua, ub = upair // n_atoms, upair % n_atoms
        c6_u = _edge_c6(Z[ua], Z[ub], gw0[ua], gw0[ub], tables["rc6"])
        c6_mem = torch.zeros((n_atoms, n_atoms), dtype=c6_u.dtype, device=c6_u.device)
        c6_mem = c6_mem.index_put((ua, ub), c6_u)
        offd = ua != ub
        c6_mem = c6_mem.index_put((ub[offd], ua[offd]), c6_u[offd])
        c9 = torch.sqrt(torch.clamp(c6_mem[tk, tj] * c6_mem[tj, ti] * c6_mem[ti, tk], min=0.0))
        r0m = lambda a, b: a1 * torch.sqrt(3.0 * r4r2[Z[a]] * r4r2[Z[b]]) + a2
        rav = (r_ij * r_ik * r_jk) / (r0m(ti, tj) * r0m(ti, tk) * r0m(tj, tk))  # (r/r0)^3 product
        alp = params.get("alp", 16.0)
        damp = 1.0 / (1.0 + 6.0 * rav ** (-alp / 3.0))
        r2ik, r2jk, r2ij = r_ik**2, r_jk**2, r_ij**2
        t1 = r2jk + r2ij - r2ik
        t2 = r2ij + r2ik - r2jk
        t3 = r2ik + r2jk - r2ij
        tmp2 = r2ik * r2jk * r2ij
        ang = (0.375 * t1 * t2 * t3 / tmp2 + 1.0) / (tmp2**1.5)
        e3 = params.get("s9", 1.0) * damp * c9 * ang / multiplicity
        if batch_edge is None:
            g = g + e3.to(torch.float64).sum()
        else:
            g.scatter_add_(0, batch_triplets, e3.to(torch.float64))
    if return_charges:
        return g, q, cn, cn_eeq
    return g
