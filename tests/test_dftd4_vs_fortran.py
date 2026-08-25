"""DFT-D4 regression test against the reference `dftd4` Fortran library (skipped if not installed).
NOTE: `dftd4` must be imported before torch in the same process (LAPACK/OpenMP symbol clash otherwise)."""
import numpy as np
import pytest

dftd4 = pytest.importorskip("dftd4")
from dftd4.interface import DampingParam, DispersionModel  # noqa: E402

import torch  # noqa: E402
from ase.build import bulk, molecule  # noqa: E402
from ase.units import Bohr, Hartree  # noqa: E402

from torch_dftd.torch_dftd4_calculator import TorchDFTD4Calculator  # noqa: E402


def _fortran(atoms, atm, cut, charge=0.0):
    per = bool(atoms.pbc.any())
    m = DispersionModel(atoms.numbers, atoms.positions / Bohr, charge=charge,
                        lattice=atoms.cell.array / Bohr if per else None,
                        periodic=np.array([True] * 3) if per else None)
    m.set_realspace_cutoff(*cut)
    res = m.get_dispersion(DampingParam(method="pbe", atm=atm), grad=True)
    return res["energy"] * Hartree, -res["gradient"] * Hartree / Bohr, m.get_properties()["partial charges"]


@pytest.mark.parametrize("atm", [False, True])
@pytest.mark.parametrize("system", ["C6H6", "C60", "NaCl", "Cu"])
def test_dftd4_matches_fortran(system, atm):
    if system in ("NaCl", "Cu"):
        atoms = bulk(system, cubic=True) * (2, 2, 2)
        cut = (60.0, 20.0, 30.0)
    else:
        atoms = molecule(system)
        atoms.pbc = False
        cut = (60.0, 40.0, 30.0)
    E_ref, F_ref, q_ref = _fortran(atoms, atm, cut)
    calc = TorchDFTD4Calculator(xc="pbe", dtype=torch.float64, abc=atm, cutoff=cut[0] * Bohr,
                                abc_cutoff=cut[1] * Bohr, cnthr=cut[2] * Bohr)
    atoms.calc = calc
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    q = calc.dftd_module.last_charges.cpu().numpy()
    assert abs(E - E_ref) < 2e-5 * max(1.0, len(atoms) / 100)
    assert np.abs(q - q_ref).max() < 1e-5
    assert np.abs(F - F_ref).max() < 1e-4


def test_dftd4_charged_molecule():
    atoms = molecule("NH3")
    atoms.pbc = False
    cut = (60.0, 40.0, 30.0)
    E_ref, _, q_ref = _fortran(atoms, True, cut, charge=1.0)
    calc = TorchDFTD4Calculator(xc="pbe", dtype=torch.float64, charge=1.0)
    atoms.calc = calc
    assert abs(atoms.get_potential_energy() - E_ref) < 1e-6
    assert np.abs(calc.dftd_module.last_charges.cpu().numpy() - q_ref).max() < 1e-5
