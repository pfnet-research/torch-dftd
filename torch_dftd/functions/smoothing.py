import torch
from torch import Tensor


def poly_smoothing(r: Tensor, cutoff: float) -> Tensor:
    """Computes a smooth step from 1 to 0 starting at 1 bohr before the cutoff

    Args:
        r (Tensor): (n_edges,)
        cutoff (float): ()

    Returns:
        r (Tensor): Smoothed `r`
    """
    cuton = cutoff - 1
    x = (cutoff - r) / (cutoff - cuton)
    x2 = x**2
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    return torch.where(
        r <= cuton,
        torch.ones_like(x),
        torch.where(r >= cutoff, torch.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3),
    )
