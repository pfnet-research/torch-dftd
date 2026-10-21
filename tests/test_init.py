from importlib.metadata import version

import pytest

import torch_dftd


def test_version():
    expect = version("torch_dftd")
    actual = torch_dftd.__version__
    assert expect == actual


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
