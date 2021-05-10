import pkg_resources
import pytest
import torch_dftd


def test_version():
    expect = pkg_resources.get_distribution("torch_dftd").version
    actual = torch_dftd.__version__
    assert expect == actual


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
