"""DFT-D4 regression test against the reference `dftd4` Fortran library (skipped if not installed).

The reference is evaluated in a subprocess: `dftd4` must be imported before torch in a process
(LAPACK/OpenMP symbol clash otherwise), which cannot be guaranteed inside pytest.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("dftd4")

import torch  # noqa: E402
from ase.build import bulk, molecule  # noqa: E402
from ase.units import Bohr  # noqa: E402

from torch_dftd.torch_dftd4_calculator import TorchDFTD4Calculator  # noqa: E402

_REF_SCRIPT = r"""
import json, sys
import numpy as np
from dftd4.interface import DampingParam, DispersionModel
from ase.units import Bohr, Hartree
d = json.loads(sys.stdin.read())
per = d["lattice"] is not None
m = DispersionModel(np.array(d["numbers"]), np.array(d["positions"]) / Bohr, charge=d["charge"],
                    lattice=np.array(d["lattice"]) / Bohr if per else None,
                    periodic=np.array([True] * 3) if per else None)
m.set_realspace_cutoff(*d["cut"])
res = m.get_dispersion(DampingParam(method="pbe", atm=d["atm"]), grad=True)
print(json.dumps({"E": res["energy"] * Hartree, "F": (-res["gradient"] * Hartree / Bohr).tolist(),
                  "q": m.get_properties()["partial charges"].tolist()}))
"""


def _fortran(atoms, atm, cut, charge=0.0):
    payload = dict(
        numbers=atoms.numbers.tolist(),
        positions=atoms.positions.tolist(),
        lattice=atoms.cell.array.tolist() if bool(atoms.pbc.any()) else None,
        charge=charge,
        cut=list(cut),
        atm=atm,
    )
    out = subprocess.run(
        [sys.executable, "-c", _REF_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    res = json.loads(out.stdout.strip().splitlines()[-1])
    return res["E"], np.array(res["F"]), np.array(res["q"])


@pytest.mark.parametrize("atm", [False, True])
@pytest.mark.parametrize("system", ["C6H6", "C60", "NaCl", "Cu"])
def test_dftd4_matches_fortran(system, atm):
    if system in ("NaCl", "Cu"):
        if system == "NaCl":
            atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)
        else:
            atoms = bulk(system, cubic=True) * (2, 2, 2)
        cut = (60.0, 20.0, 30.0)
    else:
        atoms = molecule(system)
        atoms.pbc = False
        cut = (60.0, 40.0, 30.0)
    E_ref, F_ref, q_ref = _fortran(atoms, atm, cut)
    calc = TorchDFTD4Calculator(
        xc="pbe",
        dtype=torch.float64,
        abc=atm,
        cutoff=cut[0] * Bohr,
        abc_cutoff=cut[1] * Bohr,
        cnthr=cut[2] * Bohr,
    )
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
