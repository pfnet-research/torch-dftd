import pytest
import torch
from torch_dftd.nn.dftd3_module import DFTD3Module


def test_dftd3_module_init():
    params = dict(s6=1.0, alp=14.0, rs6=0.486434, s18=0.672820, rs18=3.656466)
    module = DFTD3Module(params)
    assert module.c6ab.shape == (95, 95, 5, 5, 3)
    assert module.r0ab.shape == (95, 95)
    assert module.rcov.shape == (95,)
    assert module.r2r4.shape == (95,)


def test_dftd3_module_calc():
    params = dict(s6=1.0, alp=14.0, rs6=0.486434, s18=0.672820, rs18=3.656466)
    dtype = torch.float32
    module = DFTD3Module(params, bidirectional=False, dtype=dtype)
    Z = torch.tensor([1, 2, 3], dtype=torch.int64)
    pos = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]], dtype=dtype)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.int64)
    results = module.calc_energy(Z, pos, edge_index)
    assert results[0]["energy"] == pytest.approx(-0.6810069680213928)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
