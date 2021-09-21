"""
DFTD3 program need to be installed to test this method.
"""
from copy import deepcopy
from typing import List

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from torch_dftd.testing.damping import damping_method_list
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator


@pytest.fixture(
    params=[
        pytest.param("case1", id="mol+slab"),
        pytest.param("case2", id="mol+slab(wo_pbc)"),
        pytest.param("case3", id="null"),
        pytest.param("case4", marks=[pytest.mark.slow], id="large"),
    ]
)
def atoms_list(request) -> List[Atoms]:
    """Initialization"""
    mol = molecule("CH3CH2OCH3")

    slab = fcc111("Au", size=(2, 1, 3), vacuum=80.0)
    slab.pbc = np.array([True, True, True])

    slab_wo_pbc = slab.copy()
    slab_wo_pbc.pbc = np.array([False, False, False])

    null = Atoms()

    large_bulk = bulk("Pt", "fcc") * (8, 8, 8)

    atoms_dict = {
        "case1": [mol, slab],
        "case2": [mol, slab_wo_pbc],
        "case3": [null],
        "case4": [large_bulk],
    }

    return atoms_dict[request.param]


def _assert_energy_equal_batch(calc1, atoms_list: List[Atoms]):
    expected_results_list = []
    for atoms in atoms_list:
        calc1.reset()
        atoms.calc = calc1
        calc1.calculate(atoms, properties=["energy"])
        expected_results_list.append(deepcopy(calc1.results))

    results_list = calc1.batch_calculate(atoms_list, properties=["energy"])
    for exp, actual in zip(expected_results_list, results_list):
        assert np.allclose(exp["energy"], actual["energy"], atol=1e-4, rtol=1e-4)


def _test_calc_energy(damping, xc, old, atoms_list, device="cpu", dtype=torch.float64):
    cutoff = 25.0  # Make test faster
    torch_dftd3_calc = TorchDFTD3Calculator(
        damping=damping, xc=xc, device=device, dtype=dtype, old=old, cutoff=cutoff
    )
    _assert_energy_equal_batch(torch_dftd3_calc, atoms_list)


def _assert_energy_force_stress_equal_batch(calc1, atoms_list: List[Atoms]):
    expected_results_list = []
    for atoms in atoms_list:
        calc1.reset()
        atoms.calc = calc1
        calc1.calculate(atoms, properties=["energy", "forces", "stress"])
        expected_results_list.append(deepcopy(calc1.results))

    results_list = calc1.batch_calculate(atoms_list, properties=["energy", "forces", "stress"])
    for exp, actual in zip(expected_results_list, results_list):
        assert np.allclose(exp["energy"], actual["energy"], atol=1e-4, rtol=1e-4)
        assert np.allclose(exp["forces"], actual["forces"], atol=1e-5, rtol=1e-5)
        if hasattr(exp, "stress"):
            assert np.allclose(exp["stress"], actual["stress"], atol=1e-5, rtol=1e-5)


def _test_calc_energy_force_stress(
    damping,
    xc,
    old,
    atoms_list,
    device="cpu",
    dtype=torch.float64,
    bidirectional=True,
    abc=False,
    cnthr=15.0,
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
        bidirectional=bidirectional,
    )
    _assert_energy_force_stress_equal_batch(torch_dftd3_calc, atoms_list)


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calc_energy_device_batch(damping, old, atoms_list, device, dtype):
    """Test2-1: check device, dtype dependency. with only various damping method."""
    xc = "pbe"
    _test_calc_energy(damping, xc, old, atoms_list, device=device, dtype=dtype)


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calc_energy_force_stress_device_batch(damping, old, atoms_list, device, dtype):
    """Test2-2: check device, dtype dependency. with only various damping method."""
    xc = "pbe"
    _test_calc_energy_force_stress(damping, xc, old, atoms_list, device=device, dtype=dtype)


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_calc_energy_force_stress_device_batch_abc(
    damping, old, atoms_list, device, bidirectional, dtype
):
    """Test2-3: check device, dtype dependency. with only various damping method."""
    xc = "pbe"
    abc = True
    if any([np.all(atoms.pbc) for atoms in atoms_list]) and bidirectional == False:
        # TODO: bidirectional=False is not implemented for pbc now.
        with pytest.raises(NotImplementedError):
            _test_calc_energy_force_stress(
                damping,
                xc,
                old,
                atoms_list,
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
            atoms_list,
            device=device,
            dtype=dtype,
            bidirectional=bidirectional,
            abc=abc,
            cnthr=7.0,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
