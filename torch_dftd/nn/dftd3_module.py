import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from ase.units import Bohr
from torch import Tensor
from torch_dftd.functions.dftd3 import d3_autoang, d3_autoev, edisp
from torch_dftd.functions.distance import calc_distances
from torch_dftd.nn.base_dftd_module import BaseDFTDModule


class DFTD3Module(BaseDFTDModule):
    """DFTD3Module

    Args:
        params (dict): xc-dependent parameters. alp, s6, rs6, s18, rs18.
        cutoff (float): cutoff distance in angstrom. Default value is 95bohr := 50 angstrom.
        cnthr (float): coordination number cutoff distance in angstrom.
            Default value is 40bohr := 21 angstrom.
        abc (bool): ATM 3-body interaction
        dtype (dtype): internal calculation is done in this precision.
        bidirectional (bool): calculated `edge_index` is bidirectional or not.
    """

    c6ab: Tensor
    r0ab: Tensor
    rcov: Tensor
    r2r4: Tensor

    def __init__(
        self,
        params: Dict[str, float],
        cutoff: float = 95.0 * Bohr,
        cnthr: float = 40.0 * Bohr,
        abc: bool = False,
        dtype=torch.float32,
        bidirectional: bool = False,
        cutoff_smoothing: str = "none",
    ):
        super(DFTD3Module, self).__init__()

        # relative filepath to package folder
        d3_filepath = str(Path(os.path.abspath(__file__)).parent / "params" / "dftd3_params.npz")
        d3_params = np.load(d3_filepath)
        c6ab = torch.tensor(d3_params["c6ab"], dtype=dtype)
        r0ab = torch.tensor(d3_params["r0ab"], dtype=dtype)
        rcov = torch.tensor(d3_params["rcov"], dtype=dtype)
        r2r4 = torch.tensor(d3_params["r2r4"], dtype=dtype)
        # (95, 95, 5, 5, 3) c0, c1, c2 for coordination number dependent c6ab term.
        self.register_buffer("c6ab", c6ab)
        self.register_buffer("r0ab", r0ab)  # atom pair distance (95, 95)
        self.register_buffer("rcov", rcov)  # atom covalent distance (95)
        self.register_buffer("r2r4", r2r4)  # (95,)

        if cnthr > cutoff:
            print(
                f"WARNING: cnthr {cnthr} is larger than cutoff {cutoff}. "
                f"cutoff distance is used for cnthr"
            )
            cnthr = cutoff
        self.params = params
        self.cutoff = cutoff
        self.cnthr = cnthr
        self.abc = abc
        self.dtype = dtype
        self.bidirectional = bidirectional
        self.cutoff_smoothing = cutoff_smoothing
        self._bohr = Bohr  # For torch.jit, `Bohr` must be local variable.

    @torch.jit.export
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
        damping: str = "zero",
        autoang: float = d3_autoang,
        autoev: float = d3_autoev,
    ) -> Tensor:
        """Forward computation to calculate atomic wise dispersion energy"""
        shift_pos = pos.new_zeros((edge_index.size()[1], 3)) if shift_pos is None else shift_pos
        pos_bohr = pos / autoang  # angstrom -> bohr
        if cell is None:
            cell_bohr: Optional[Tensor] = None
        else:
            cell_bohr = cell / autoang  # angstrom -> bohr
        shift_bohr = shift_pos / autoang  # angstrom -> bohr
        r = calc_distances(pos_bohr, edge_index, cell_bohr, shift_bohr)
        # E_disp (n_graphs,): Energy in eV unit
        E_disp = autoev * edisp(
            Z,
            r,
            edge_index,
            c6ab=self.c6ab,
            r0ab=self.r0ab,
            rcov=self.rcov,
            r2r4=self.r2r4,
            params=self.params,
            cutoff=self.cutoff / self._bohr,
            cnthr=self.cnthr / self._bohr,
            batch=batch,
            batch_edge=batch_edge,
            shift_pos=shift_bohr,
            damping=damping,
            cutoff_smoothing=self.cutoff_smoothing,
            bidirectional=self.bidirectional,
            abc=self.abc,
            pos=pos_bohr,
            cell=cell_bohr,
        )
        return E_disp
