from typing import Dict, Optional

import torch
from ase.units import Bohr
from torch import Tensor
from torch_dftd.functions.dftd2 import edisp_d2
from torch_dftd.functions.dftd3 import d3_autoang, d3_autoev
from torch_dftd.functions.distance import calc_distances
from torch_dftd.nn.base_dftd_module import BaseDFTDModule
from torch_dftd.nn.params.dftd2_params import get_dftd2_params


class DFTD2Module(BaseDFTDModule):
    """DFTD2Module

    Args:
        params (dict): xc-dependent parameters. alp6, s6, rs6.
        cutoff (float): cutoff distance in angstrom. Default value is 95bohr := 50 angstrom.
        dtype (dtype): internal calculation is done in this precision.
        bidirectional (bool): calculated `edge_index` is bidirectional or not.
    """

    def __init__(
        self,
        params: Dict[str, float],
        cutoff: float = 95.0 * Bohr,
        dtype=torch.float32,
        bidirectional: bool = False,
        cutoff_smoothing: str = "none",
    ):
        super(DFTD2Module, self).__init__()

        self.params = params
        self.cutoff = cutoff
        self.dtype = dtype
        self.bidirectional = bidirectional
        self.cutoff_smoothing = cutoff_smoothing
        r0ab, c6ab = get_dftd2_params()
        # atom pair coefficient (87, 87)
        self.register_buffer("c6ab", c6ab)
        # atom pair distance (95, 95)
        self.register_buffer("r0ab", r0ab)

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
    ) -> Tensor:
        """Forward computation to calculate atomic wise dispersion energy"""
        shift_pos = pos.new_zeros((edge_index.size()[1], 3, 3)) if shift_pos is None else shift_pos
        pos_bohr = pos / d3_autoang  # angstrom -> bohr
        if cell is None:
            cell_bohr: Optional[Tensor] = None
        else:
            cell_bohr = cell / d3_autoang  # angstrom -> bohr
        shift_bohr = shift_pos / d3_autoang  # angstrom -> bohr
        r = calc_distances(pos_bohr, edge_index, cell_bohr, shift_bohr)

        # E_disp (n_graphs,): Energy in eV unit
        E_disp = d3_autoev * edisp_d2(
            Z,
            r,
            edge_index,
            c6ab=self.c6ab,  # type:ignore
            r0ab=self.r0ab,  # type:ignore
            params=self.params,
            damping=damping,
            bidirectional=self.bidirectional,
            cutoff=self.cutoff / Bohr,
            batch=batch,
            batch_edge=batch_edge,
            cutoff_smoothing=self.cutoff_smoothing,
        )
        return E_disp
