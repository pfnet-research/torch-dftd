import pytest
import torch
from torch_dftd.nn.dftd2_module import DFTD2Module


def test_dftd2_module_init():
    params = dict(s6=1.2, alp=20.0, rs6=1.1, s18=0.0)  # rs18=None
    module = DFTD2Module(params)
    assert module.c6ab.shape == (87, 87)
    assert module.r0ab.shape == (87, 87)


def test_dftd2_module_calc():
    params = dict(s6=1.2, alp=20.0, rs6=1.1, s18=0.0)  # rs18=None
    dtype = torch.float32
    module = DFTD2Module(params, bidirectional=False, dtype=dtype)
    Z = torch.tensor([1, 2, 3], dtype=torch.int64)
    pos = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]], dtype=dtype)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.int64)
    results = module.calc_energy(Z, pos, edge_index)
    assert results[0]["energy"] == pytest.approx(-0.0808654052663793)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
