import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from ase.units import Bohr
from torch import Tensor

from torch_dftd.functions.dftd3 import d3_autoang, d3_autoev
from torch_dftd.functions.dftd4 import edisp_d4
from torch_dftd.functions.distance import calc_distances
from torch_dftd.nn.base_dftd_module import BaseDFTDModule


class DFTD4Module(BaseDFTDModule):
    """DFT-D4 (BJ damping, EEQ charges, optional ATM three-body term).

    Args:
        params (dict): damping parameters s6, s8, s9, a1, a2, alp (see ``get_dftd4_default_params``).
        cutoff (float): two-body real-space cutoff in angstrom (dftd4 default 60 bohr).
        cnthr (float): D4 coordination-number cutoff in angstrom (dftd4 default 30 bohr).
        cn_eeq_thr (float): EEQ coordination-number cutoff in angstrom (dftd4 default 25 bohr).
        eeq_cutoff (float): real-space cutoff of the Ewald sum for the EEQ Coulomb matrix, angstrom.
        abc (bool): include the ATM three-body term (standard D4: True).
        abc_cutoff (float): three-body cutoff in angstrom (dftd4 default 40 bohr; 20 bohr is ~0.1 kJ/mol accurate and much cheaper).
        ewald_tol (float): Ewald convergence tolerance.
        dtype: dtype for the pairwise part (EEQ linear algebra is always float64).
    """

    def __init__(
        self,
        params: Dict[str, float],
        cutoff: float = 60.0 * Bohr,
        cnthr: float = 30.0 * Bohr,
        cn_eeq_thr: float = 25.0 * Bohr,
        eeq_cutoff: float = 25.0 * Bohr,
        abc: bool = True,
        abc_cutoff: float = 40.0 * Bohr,
        ewald_tol: float = 1e-8,
        dtype=torch.float32,
        bidirectional: bool = False,
        cutoff_smoothing: str = "none",
    ):
        super(DFTD4Module, self).__init__()
        d4_filepath = str(Path(os.path.abspath(__file__)).parent / "params" / "dftd4_params.npz")
        d4 = np.load(d4_filepath)
        for k in [
            "rc6",
            "refcovcn",
            "refq",
            "rcov",
            "r4r2",
            "en",
            "zeff",
            "gam",
            "eeq_chi",
            "eeq_eta",
            "eeq_kcn",
            "eeq_rad",
        ]:
            self.register_buffer(
                k,
                torch.tensor(
                    d4[k],
                    dtype=(
                        torch.float64
                        if k
                        in (
                            "refcovcn",
                            "refq",
                            "zeff",
                            "gam",
                            "eeq_chi",
                            "eeq_eta",
                            "eeq_kcn",
                            "eeq_rad",
                        )
                        else dtype
                    ),
                ),
            )
        self.register_buffer("refc", torch.tensor(d4["refc"], dtype=torch.int64))
        for k in ("cnthr", "cn_eeq_thr", "abc_cutoff", "eeq_cutoff"):
            if locals()[k] > cutoff:
                print(
                    f"WARNING: {k} {locals()[k]} is larger than cutoff {cutoff}; cutoff distance is used"
                )
        self.params = params
        self.cutoff = cutoff
        self.cnthr = min(cnthr, cutoff)
        self.cn_eeq_thr = min(cn_eeq_thr, cutoff)
        self.eeq_cutoff = min(eeq_cutoff, cutoff)
        self.abc = abc
        self.abc_cutoff = min(abc_cutoff, cutoff)
        self.ewald_tol = ewald_tol
        self.dtype = dtype
        self.bidirectional = bidirectional
        self.cutoff_smoothing = cutoff_smoothing
        self.last_charges: Optional[Tensor] = None
        self.total_charge: Optional[Tensor] = None  # (n_graphs,) optional, set by the calculator

    def _tables(self) -> Dict[str, Tensor]:
        return {
            k: getattr(self, k)
            for k in [
                "rc6",
                "refc",
                "refcovcn",
                "refq",
                "rcov",
                "r4r2",
                "en",
                "zeff",
                "gam",
                "eeq_chi",
                "eeq_eta",
                "eeq_kcn",
                "eeq_rad",
            ]
        }

    def calc_energy_batch(
        self,
        Z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
        shift_pos: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_edge: Optional[Tensor] = None,
        damping: str = "bj",
        total_charge: Optional[Tensor] = None,
    ) -> Tensor:
        if damping not in ("bj", "zero"):  # "zero" is torch-dftd's default arg; D4 is BJ-only
            raise ValueError(f"DFT-D4 supports only BJ (rational) damping, got damping={damping}")
        shift_pos = pos.new_zeros((edge_index.size()[1], 3)) if shift_pos is None else shift_pos
        pos_bohr = pos / d3_autoang
        cell_bohr = None if cell is None else cell / d3_autoang
        shift_bohr = shift_pos / d3_autoang
        r = calc_distances(pos_bohr, edge_index, cell_bohr, shift_bohr)
        E, q, _, _ = edisp_d4(
            Z,
            r,
            edge_index,
            pos_bohr,
            cell_bohr,
            pbc,
            shift_bohr,
            batch,
            batch_edge,
            params=self.params,
            tables=self._tables(),
            cutoff=self.cutoff / Bohr,
            cnthr=self.cnthr / Bohr,
            cn_eeq_thr=self.cn_eeq_thr / Bohr,
            eeq_cutoff=self.eeq_cutoff / Bohr,
            abc_cutoff=self.abc_cutoff / Bohr,
            total_charge=self.total_charge if total_charge is None else total_charge,
            abc=self.abc,
            bidirectional=self.bidirectional,
            cutoff_smoothing=self.cutoff_smoothing,
            ewald_tol=self.ewald_tol,
            return_charges=True,
        )
        self.last_charges = q.detach()
        return d3_autoev * E
