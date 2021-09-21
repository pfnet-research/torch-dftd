from typing import Dict, Optional, Tuple

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.units import Bohr
from torch import Tensor
from torch_dftd.dftd3_xc_params import get_dftd3_default_params
from torch_dftd.functions.edge_extraction import calc_edge_index
from torch_dftd.nn.dftd2_module import DFTD2Module
from torch_dftd.nn.dftd3_module import DFTD3Module


class TorchDFTD3Calculator(Calculator):
    """ase compatible DFTD3 calculator using pytorch

    Args:
        dft (Calculator or None): base dft calculator can be set here
        atoms (Atoms):
        damping (str): damping method. "zero", "bj", "zerom", "bjm"
        xc (str): exchange correlation functional
        old (bool): Use DFTD2 method when `True`, DFTD3 method is used when `False`
        device (str): torch device. Ex. "cuda:0" to use GPU ID 0
        cutoff (float): cutoff distance in angstrom. Default value is 95bohr := 50 angstrom.
        cnthr (float): coordination number cutoff distance in angstrom.
            Default value is 40bohr := 21 angstrom.
        abc (bool): ATM 3-body interaction
        dtype (dtype): internal calculation is done in this precision.
        bidirectional (bool): calculated `edge_index` is bidirectional or not.
        cutoff_smoothing (str): cutoff smoothing makes gradient smooth at `cutoff` distance
        **kwargs:
    """

    name = "TorchDFTD3Calculator"
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        dft: Optional[Calculator] = None,
        atoms: Atoms = None,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = False,
        device: str = "cpu",
        cutoff: float = 95.0 * Bohr,
        cnthr: float = 40.0 * Bohr,
        abc: bool = False,
        # --- torch dftd3 specific params ---
        dtype: torch.dtype = torch.float32,
        bidirectional: bool = True,
        cutoff_smoothing: str = "none",
        **kwargs,
    ):
        self.dft = dft
        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.damping = damping
        self.abc = abc
        self.old = old
        self.device = torch.device(device)
        if old:
            self.dftd_module: torch.nn.Module = DFTD2Module(
                self.params,
                cutoff=cutoff,
                dtype=dtype,
                bidirectional=bidirectional,
                cutoff_smoothing=cutoff_smoothing,
            )
        else:
            self.dftd_module = DFTD3Module(
                self.params,
                cutoff=cutoff,
                cnthr=cnthr,
                abc=abc,
                dtype=dtype,
                bidirectional=bidirectional,
                cutoff_smoothing=cutoff_smoothing,
            )
        self.dftd_module.to(device)
        self.dtype = dtype
        self.cutoff = cutoff
        self.bidirectional = bidirectional
        super(TorchDFTD3Calculator, self).__init__(atoms=atoms, **kwargs)

    def _calc_edge_index(
        self,
        pos: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return calc_edge_index(
            pos, cell, pbc, cutoff=self.cutoff, bidirectional=self.bidirectional
        )

    def _preprocess_atoms(self, atoms: Atoms) -> Dict[str, Optional[Tensor]]:
        pos = torch.tensor(atoms.get_positions(), device=self.device, dtype=self.dtype)
        Z = torch.tensor(atoms.get_atomic_numbers(), device=self.device)
        if any(atoms.pbc):
            cell: Optional[Tensor] = torch.tensor(
                atoms.get_cell(), device=self.device, dtype=self.dtype
            )
        else:
            cell = None
        pbc = torch.tensor(atoms.pbc, device=self.device)
        edge_index, S = self._calc_edge_index(pos, cell, pbc)
        if cell is None:
            shift_pos = S
        else:
            shift_pos = torch.mm(S, cell.detach())
        input_dicts = dict(
            pos=pos, Z=Z, cell=cell, pbc=pbc, edge_index=edge_index, shift_pos=shift_pos
        )
        return input_dicts

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        input_dicts = self._preprocess_atoms(atoms)

        if "forces" in properties or "stress" in properties:
            results = self.dftd_module.calc_energy_and_forces(**input_dicts, damping=self.damping)[
                0
            ]
        else:
            results = self.dftd_module.calc_energy(**input_dicts, damping=self.damping)[0]
        self.results["energy"] = results["energy"]
        self.results["free_energy"] = self.results["energy"]

        # Referenced DFTD3 impl.
        if self.dft is not None:
            try:
                efree = self.dft.get_potential_energy(force_consistent=True)
                self.results["free_energy"] += efree
            except PropertyNotImplementedError:
                pass

        if "forces" in results:
            self.results["forces"] = results["forces"]
        if "stress" in results:
            self.results["stress"] = results["stress"]

    def get_property(self, name, atoms=None, allow_calculation=True):
        dft_result = None
        if self.dft is not None:
            dft_result = self.dft.get_property(name, atoms, allow_calculation)

        dftd3_result = Calculator.get_property(self, name, atoms, allow_calculation)

        if dft_result is None and dftd3_result is None:
            return None
        elif dft_result is None:
            return dftd3_result
        elif dftd3_result is None:
            return dft_result
        else:
            return dft_result + dftd3_result

    def batch_calculate(self, atoms_list=None, properties=["energy"], system_changes=all_changes):
        # Calculator.calculate(self, atoms, properties, system_changes)
        input_dicts_list = [self._preprocess_atoms(atoms) for atoms in atoms_list]
        # --- Make batch ---
        n_nodes_list = [d["Z"].shape[0] for d in input_dicts_list]
        shift_index_array = torch.cumsum(torch.tensor([0] + n_nodes_list), dim=0)
        cell_batch = torch.stack(
            [
                torch.eye(3, device=self.device, dtype=self.dtype)
                if d["cell"] is None
                else d["cell"]
                for d in input_dicts_list
            ]
        )
        batch_dicts = dict(
            Z=torch.cat([d["Z"] for d in input_dicts_list], dim=0),  # (n_nodes,)
            pos=torch.cat([d["pos"] for d in input_dicts_list], dim=0),  # (n_nodes,)
            cell=cell_batch,  # (bs, 3, 3)
            pbc=torch.stack([d["pbc"] for d in input_dicts_list]),  # (bs, 3)
            shift_pos=torch.cat([d["shift_pos"] for d in input_dicts_list], dim=0),  # (n_nodes,)
        )

        batch_dicts["edge_index"] = torch.cat(
            [d["edge_index"] + shift_index_array[i] for i, d in enumerate(input_dicts_list)],
            dim=1,
        )
        batch_dicts["batch"] = torch.cat(
            [
                torch.full((n_nodes,), i, dtype=torch.long, device=self.device)
                for i, n_nodes in enumerate(n_nodes_list)
            ],
            dim=0,
        )
        batch_dicts["batch_edge"] = torch.cat(
            [
                torch.full((d["edge_index"].shape[1],), i, dtype=torch.long, device=self.device)
                for i, d in enumerate(input_dicts_list)
            ],
            dim=0,
        )

        if "forces" in properties or "stress" in properties:
            results_list = self.dftd_module.calc_energy_and_forces(
                **batch_dicts, damping=self.damping
            )
        else:
            results_list = self.dftd_module.calc_energy(**batch_dicts, damping=self.damping)
        return results_list
