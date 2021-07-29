"""
DFTD3 program need to be installed to test this method.
"""
import tempfile
from typing import List

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from torch_dftd.testing.damping import damping_method_list, damping_xc_combination_list
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator


def _create_atoms() -> List[Atoms]:
    """Initialization"""
    H = molecule("H")
    H_pbc = molecule("H", vacuum=100.0, pbc=True)
    null = Atoms()
    null_pbc = Atoms(cell=np.eye(3) * 100, pbc=True)
    return [H, H_pbc, null, null_pbc]


def _assert_energy_equal(calc, atoms: Atoms):
    calc.reset()
    atoms.calc = calc
    e1 = atoms.get_potential_energy()

    e2 = 0.0
    assert np.allclose(e1, e2, atol=1e-4, rtol=1e-4)


def _test_calc_energy(damping, xc, old, atoms, device="cpu", dtype=torch.float64, abc=False):
    cutoff = 25.0  # Make test faster
    torch_dftd3_calc = TorchDFTD3Calculator(
        damping=damping, xc=xc, device=device, dtype=dtype, old=old, cutoff=cutoff, abc=abc
    )
    _assert_energy_equal(torch_dftd3_calc, atoms)


def _assert_energy_force_stress_equal(calc, atoms: Atoms):
    calc.reset()
    atoms.calc = calc
    f1 = atoms.get_forces()
    e1 = atoms.get_potential_energy()

    if calc.dft is not None:
        calc2 = calc.dft
        calc2.reset()
        atoms.calc = calc2
        e2 = atoms.get_potential_energy()
        f2 = atoms.get_forces()
    else:
        f2 = np.zeros_like(atoms.get_positions())
        e2 = 0.0
    assert np.allclose(e1, e2, atol=1e-4, rtol=1e-4), (e1, e2)
    assert np.allclose(f1, f2, atol=1e-5, rtol=1e-5)
    if np.all(atoms.pbc == np.array([True, True, True])):
        s1 = atoms.get_stress()
        s2 = np.zeros([6])
        assert np.allclose(s1, s2, atol=1e-5, rtol=1e-5)


def _test_calc_energy_force_stress(
    damping, xc, old, atoms, device="cpu", dtype=torch.float64, abc=False, cnthr=15.0
):
    cutoff = 22.0  # Make test faster
    torch_dftd3_calc = TorchDFTD3Calculator(
        damping=damping,
        xc=xc,
        device=device,
        dtype=dtype,
        old=old,
        cutoff=cutoff,
        cnthr=cnthr,
        abc=abc,
    )
    _assert_energy_force_stress_equal(torch_dftd3_calc, atoms)


@pytest.mark.parametrize("damping,xc,old", damping_xc_combination_list)
@pytest.mark.parametrize("atoms", _create_atoms())
def test_calc_energy(damping, xc, old, atoms):
    """Test1-1: check damping,xc,old combination works for energy"""
    _test_calc_energy(damping, xc, old, atoms, device="cpu")


@pytest.mark.parametrize("damping,xc,old", damping_xc_combination_list)
@pytest.mark.parametrize("atoms", _create_atoms())
def test_calc_energy_force_stress(damping, xc, old, atoms):
    """Test1-2: check damping,xc,old combination works for energy, force & stress"""
    _test_calc_energy_force_stress(damping, xc, old, atoms, device="cpu")


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("atoms", _create_atoms())
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calc_energy_device(damping, old, atoms, device, dtype):
    """Test2-1: check device, dtype dependency. with only various damping method."""
    xc = "pbe"
    _test_calc_energy(damping, xc, old, atoms, device=device, dtype=dtype)


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("atoms", _create_atoms())
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calc_energy_force_stress_device(damping, old, atoms, device, dtype):
    """Test2-2: check device, dtype dependency. with only various damping method."""
    xc = "pbe"
    _test_calc_energy_force_stress(damping, xc, old, atoms, device=device, dtype=dtype)


@pytest.mark.parametrize("atoms", _create_atoms())
@pytest.mark.parametrize("damping,old", damping_method_list)
def test_calc_energy_force_stress_bidirectional(atoms, damping, old):
    """Test with bidirectional=False"""
    device = "cpu"
    xc = "pbe"
    torch_dftd3_calc = TorchDFTD3Calculator(
        damping=damping, xc=xc, device=device, old=old, bidirectional=False
    )
    if np.all(atoms.pbc):
        # TODO: bidirectional=False is not implemented for pbc now.
        with pytest.raises(NotImplementedError):
            _assert_energy_force_stress_equal(torch_dftd3_calc, atoms)
    else:
        _assert_energy_force_stress_equal(torch_dftd3_calc, atoms)


@pytest.mark.parametrize("atoms", _create_atoms())
@pytest.mark.parametrize("damping,old", damping_method_list)
def test_calc_energy_force_stress_cutoff_smoothing(atoms, damping, old):
    """Test wit cutoff_smoothing."""
    device = "cpu"
    xc = "pbe"
    cutoff_smoothing = "poly"
    torch_dftd3_calc = TorchDFTD3Calculator(
        damping=damping,
        xc=xc,
        device=device,
        old=old,
        bidirectional=False,
        cutoff_smoothing=cutoff_smoothing,
    )
    try:
        _assert_energy_force_stress_equal(torch_dftd3_calc, atoms)
    except NotImplementedError:
        print("NotImplementedError with atoms", atoms)
        # sometimes, bidirectional=False is not implemented.
        pass


def test_calc_energy_force_stress_with_dft():
    """Test with `dft` argument"""
    atoms = molecule("H")
    # Set calculator. EMT supports H & C just for fun, which is enough for the test!
    # https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html#module-ase.calculators.emt
    dft = EMT()
    damping = "bj"
    old = False
    device = "cpu"
    xc = "pbe"
    torch_dftd3_calc = TorchDFTD3Calculator(
        damping=damping, xc=xc, device=device, old=old, bidirectional=False, dft=dft
    )
    _assert_energy_force_stress_equal(torch_dftd3_calc, atoms)


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("atoms", _create_atoms())
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("abc", [True])
def test_calc_energy_force_stress_device_abc(damping, old, atoms, device, dtype, abc):
    """Test: check tri-partite calc with device, dtype dependency."""
    xc = "pbe"
    _test_calc_energy_force_stress(
        damping, xc, old, atoms, device=device, dtype=dtype, abc=abc, cnthr=7.0
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
