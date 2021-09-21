"""
DFTD3 program need to be installed to test this method.
"""
import tempfile
from typing import List

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.calculators.dftd3 import DFTD3
from ase.calculators.emt import EMT
from torch_dftd.testing.damping import damping_method_list, damping_xc_combination_list
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator


@pytest.fixture(
    params=[
        pytest.param("mol", id="mol"),
        pytest.param("slab", id="slab"),
        pytest.param("large", marks=[pytest.mark.slow], id="large"),
    ]
)
def atoms(request) -> Atoms:
    """Initialization"""
    mol = molecule("CH3CH2OCH3")

    slab = fcc111("Au", size=(2, 1, 3), vacuum=80.0)
    slab.set_cell(
        slab.get_cell().array @ np.array([[1.0, 0.1, 0.2], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]])
    )
    slab.pbc = np.array([True, True, True])

    large_bulk = bulk("Pt", "fcc") * (4, 4, 4)

    atoms_dict = {"mol": mol, "slab": slab, "large": large_bulk}

    return atoms_dict[request.param]


def _assert_energy_equal(calc1, calc2, atoms: Atoms):
    calc1.reset()
    atoms.calc = calc1
    e1 = atoms.get_potential_energy()

    calc2.reset()
    atoms.calc = calc2
    e2 = atoms.get_potential_energy()
    assert np.allclose(e1, e2, atol=1e-4, rtol=1e-4)


def _test_calc_energy(damping, xc, old, atoms, device="cpu", dtype=torch.float64, abc=False):
    cutoff = 25.0  # Make test faster
    with tempfile.TemporaryDirectory() as tmpdirname:
        dftd3_calc = DFTD3(
            damping=damping,
            xc=xc,
            grad=True,
            old=old,
            cutoff=cutoff,
            directory=tmpdirname,
            abc=abc,
        )
        torch_dftd3_calc = TorchDFTD3Calculator(
            damping=damping, xc=xc, device=device, dtype=dtype, old=old, cutoff=cutoff, abc=abc
        )
        _assert_energy_equal(dftd3_calc, torch_dftd3_calc, atoms)


def _assert_energy_force_stress_equal(calc1, calc2, atoms: Atoms, force_tol: float = 1e-5):
    calc1.reset()
    atoms.calc = calc1
    f1 = atoms.get_forces()
    e1 = atoms.get_potential_energy()
    if np.all(atoms.pbc == np.array([True, True, True])):
        s1 = atoms.get_stress()

    calc2.reset()
    atoms.calc = calc2
    f2 = atoms.get_forces()
    e2 = atoms.get_potential_energy()
    assert np.allclose(e1, e2, atol=1e-4, rtol=1e-4)
    assert np.allclose(f1, f2, atol=force_tol, rtol=force_tol)
    if np.all(atoms.pbc == np.array([True, True, True])):
        s2 = atoms.get_stress()
        assert np.allclose(s1, s2, atol=1e-5, rtol=1e-5)


def _test_calc_energy_force_stress(
    damping,
    xc,
    old,
    atoms,
    device="cpu",
    dtype=torch.float64,
    bidirectional=True,
    abc=False,
    cnthr=15.0,
):
    cutoff = 22.0  # Make test faster
    force_tol = 1e-5
    if dtype == torch.float32:
        force_tol = 1.0e-4
    with tempfile.TemporaryDirectory() as tmpdirname:
        dftd3_calc = DFTD3(
            damping=damping,
            xc=xc,
            grad=True,
            old=old,
            cutoff=cutoff,
            cnthr=cnthr,
            directory=tmpdirname,
            abc=abc,
        )
        torch_dftd3_calc = TorchDFTD3Calculator(
            damping=damping,
            xc=xc,
            device=device,
            dtype=dtype,
            old=old,
            cutoff=cutoff,
            cnthr=cnthr,
            abc=abc,
            bidirectional=bidirectional,
        )
        _assert_energy_force_stress_equal(dftd3_calc, torch_dftd3_calc, atoms, force_tol=force_tol)


@pytest.mark.parametrize("damping,xc,old", damping_xc_combination_list)
def test_calc_energy(damping, xc, old, atoms):
    """Test1-1: check damping,xc,old combination works for energy"""
    _test_calc_energy(damping, xc, old, atoms, device="cpu")


@pytest.mark.parametrize("damping,xc,old", damping_xc_combination_list)
def test_calc_energy_force_stress(damping, xc, old, atoms):
    """Test1-2: check damping,xc,old combination works for energy, force & stress"""
    _test_calc_energy_force_stress(damping, xc, old, atoms, device="cpu")


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calc_energy_device(damping, old, atoms, device, dtype):
    """Test2-1: check device, dtype dependency. with only various damping method."""
    xc = "pbe"
    _test_calc_energy(damping, xc, old, atoms, device=device, dtype=dtype)


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calc_energy_force_stress_device(damping, old, atoms, device, dtype):
    """Test2-2: check device, dtype dependency. with only various damping method."""
    xc = "pbe"
    _test_calc_energy_force_stress(damping, xc, old, atoms, device=device, dtype=dtype)


@pytest.mark.parametrize("damping,old", damping_method_list)
def test_calc_energy_force_stress_bidirectional(atoms, damping, old):
    """Test with bidirectional=False"""
    device = "cpu"
    xc = "pbe"
    with tempfile.TemporaryDirectory() as tmpdirname:
        dftd3_calc = DFTD3(damping=damping, xc=xc, grad=True, old=old, directory=tmpdirname)
        torch_dftd3_calc = TorchDFTD3Calculator(
            damping=damping, xc=xc, device=device, old=old, bidirectional=False
        )
        if np.all(atoms.pbc):
            # TODO: bidirectional=False is not implemented for pbc now.
            with pytest.raises(NotImplementedError):
                _assert_energy_force_stress_equal(dftd3_calc, torch_dftd3_calc, atoms)
        else:
            _assert_energy_force_stress_equal(dftd3_calc, torch_dftd3_calc, atoms)


@pytest.mark.parametrize("damping,old", damping_method_list)
def test_calc_energy_force_stress_cutoff_smoothing(atoms, damping, old):
    """Test wit cutoff_smoothing."""
    device = "cpu"
    xc = "pbe"
    cutoff_smoothing = "poly"
    with tempfile.TemporaryDirectory() as tmpdirname:
        dftd3_calc = DFTD3(damping=damping, xc=xc, grad=True, old=old, directory=tmpdirname)
        torch_dftd3_calc = TorchDFTD3Calculator(
            damping=damping,
            xc=xc,
            device=device,
            old=old,
            bidirectional=False,
            cutoff_smoothing=cutoff_smoothing,
        )
        try:
            _assert_energy_force_stress_equal(dftd3_calc, torch_dftd3_calc, atoms)
        except NotImplementedError:
            print("NotImplementedError with atoms", atoms)
            # sometimes, bidirectional=False is not implemented.
            pass


def test_calc_energy_force_stress_with_dft():
    """Test with `dft` argument"""
    atoms = molecule("CH3CH2OCH3")
    # Set calculator. EMT supports H & C just for fun, which is enough for the test!
    # https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html#module-ase.calculators.emt
    dft = EMT()
    damping = "bj"
    old = False
    device = "cpu"
    xc = "pbe"
    with tempfile.TemporaryDirectory() as tmpdirname:
        dftd3_calc = DFTD3(
            damping=damping, xc=xc, grad=True, old=old, directory=tmpdirname, dft=dft
        )
        torch_dftd3_calc = TorchDFTD3Calculator(
            damping=damping, xc=xc, device=device, old=old, bidirectional=False, dft=dft
        )
        _assert_energy_force_stress_equal(dftd3_calc, torch_dftd3_calc, atoms)


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("abc", [True])
def test_calc_energy_force_stress_device_abc(
    damping, old, atoms, device, dtype, bidirectional, abc
):
    """Test: check tri-partite calc with device, dtype dependency."""
    xc = "pbe"
    if np.all(atoms.pbc) and bidirectional == False:
        # TODO: bidirectional=False is not implemented for pbc now.
        with pytest.raises(NotImplementedError):
            _test_calc_energy_force_stress(
                damping,
                xc,
                old,
                atoms,
                device=device,
                dtype=dtype,
                bidirectional=bidirectional,
                abc=abc,
                cnthr=7.0,
            )
    else:
        _test_calc_energy_force_stress(
            damping,
            xc,
            old,
            atoms,
            device=device,
            dtype=dtype,
            bidirectional=bidirectional,
            abc=abc,
            cnthr=7.0,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
