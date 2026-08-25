from typing import Optional

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr

from torch_dftd.dftd4_xc_params import get_dftd4_default_params
from torch_dftd.nn.dftd4_module import DFTD4Module
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator


class TorchDFTD4Calculator(TorchDFTD3Calculator):
    """ase compatible DFT-D4 calculator using pytorch (BJ damping, Ewald-EEQ charges, ATM).

    Args:
        xc (str): exchange correlation functional (dftd4 parameter set ``d4.bj-eeq-atm``).
        device (str): torch device.
        cutoff (float): two-body cutoff in angstrom (dftd4 default 60 bohr).
        cnthr (float): D4 coordination-number cutoff in angstrom (default 30 bohr).
        abc (bool): ATM three-body term (default True = standard D4).
        abc_cutoff (float): three-body cutoff in angstrom (dftd4 default 40 bohr).
        eeq_cutoff (float): real-space Ewald cutoff for the EEQ matrix in angstrom.
        charge (float): total charge of the system.
    """

    name = "TorchDFTD4Calculator"
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        dft: Optional[Calculator] = None,
        atoms: Atoms = None,
        xc: str = "pbe",
        device: str = "cpu",
        cutoff: float = 60.0 * Bohr,
        cnthr: float = 30.0 * Bohr,
        cn_eeq_thr: float = 25.0 * Bohr,
        eeq_cutoff: float = 25.0 * Bohr,
        abc: bool = True,
        abc_cutoff: float = 40.0 * Bohr,
        ewald_tol: float = 1e-8,
        charge: float = 0.0,
        dtype: torch.dtype = torch.float32,
        bidirectional: bool = True,
        cutoff_smoothing: str = "none",
        **kwargs,
    ):
        self.dft = dft
        self.params = get_dftd4_default_params(xc)
        self.damping = "bj"
        self.abc = abc
        self.old = False
        self.charge = charge
        self.device = torch.device(device)
        self.dftd_module = DFTD4Module(
            self.params, cutoff=cutoff, cnthr=cnthr, cn_eeq_thr=cn_eeq_thr, eeq_cutoff=eeq_cutoff, abc=abc,
            abc_cutoff=abc_cutoff, ewald_tol=ewald_tol, dtype=dtype, bidirectional=bidirectional,
            cutoff_smoothing=cutoff_smoothing,
        )
        self.dftd_module.to(device)
        self.dtype = dtype
        self.cutoff = cutoff
        self.bidirectional = bidirectional
        Calculator.__init__(self, atoms=atoms, **kwargs)

    def _preprocess_atoms(self, atoms: Atoms):
        d = super()._preprocess_atoms(atoms)
        self.dftd_module.total_charge = torch.tensor([self.charge], device=self.device, dtype=self.dtype)
        return d
