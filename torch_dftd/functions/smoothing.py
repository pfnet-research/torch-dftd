import torch
from torch import Tensor


@torch.jit.script
def poly_smoothing(r: Tensor, cutoff: float) -> Tensor:
    """Computes a smooth step from 1 to 0 starting at 1 bohr before the cutoff

    Args:
        r (Tensor): (n_edges,)
        cutoff (float): cutoff length

    Returns:
        r (Tensor): Smoothed `r`
    """
    cuton = cutoff - 1
    x = (cutoff - r) / (cutoff - cuton)
    x2 = x ** 2
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    return torch.where(
        r <= cuton,
        torch.ones_like(x),
        torch.where(r >= cutoff, torch.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3),
    )


if __name__ == "__main__":
    from time import perf_counter

    num_runs = 50
    old_prof_exec_state = torch._C._jit_set_profiling_executor(False)
    old_prof_mode_state = torch._C._jit_set_profiling_mode(False)
    old_num_prof_runs = torch._C._jit_set_num_profiled_runs(num_runs)
    print(
        "old_prof_exec_state",
        old_prof_exec_state,
        "old_prof_mode_state",
        old_prof_mode_state,
        "old_num_prof_runs",
        old_num_prof_runs,
    )
    print("profiled runs: ", torch._C._jit_get_num_profiled_runs())

    device = "cpu"
    n_edges = 10000
    r = torch.rand(n_edges).to(device) * 10.0

    time_list = []
    for i in range(200):
        torch.cuda.synchronize()
        s0 = perf_counter()
        r2 = poly_smoothing(r, 5.0)
        torch.cuda.synchronize()
        e0 = perf_counter()
        time_list.append(e0 - s0)
    print("Time: ", time_list)
    import numpy as np

    time_array = np.array(time_list)
    print(
        "time:",
        np.mean(time_array),
        np.mean(time_array[10:]),
        np.mean(time_array[50:]),
        np.mean(time_array[100:]),
    )
    import IPython

    IPython.embed()
