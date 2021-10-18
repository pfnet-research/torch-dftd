from typing import Optional

import torch
from torch import Tensor


@torch.jit.script
def calc_distances(
    pos: Tensor,
    edge_index: Tensor,
    cell: Optional[Tensor] = None,
    shift_pos: Optional[Tensor] = None,
    eps: float = 1e-20,
) -> Tensor:
    """Distance calculation function.

    Args:
        pos (Tensor): (n_atoms, 3) atom positions.
        edge_index (Tensor): (2, n_edges) edge_index for graph.
        cell (Tensor): cell size, None for non periodic system.
            This it NOT USED now, it is left for backward compatibility.
        shift_pos (Tensor): (n_edges, 3) position shift vectors of edges owing to the periodic boundary. It should be length unit.
        eps (float): Small float value to avoid NaN in backward when the distance is 0.

    Returns:
        Dij (Tensor): (n_edges, ) distances of edges

    """

    idx_i, idx_j = edge_index[0], edge_index[1]
    # calculate interatomic distances
    Ri = pos[idx_i]
    Rj = pos[idx_j]
    if shift_pos is not None:
        Rj += shift_pos
    # eps is to avoid Nan in backward when Dij = 0 with sqrt.
    Dij = torch.sqrt(torch.sum((Ri - Rj) ** 2, dim=-1) + eps)
    return Dij


if __name__ == "__main__":
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

    device = "cuda:0"
    # device = "cpu"
    n_atoms = 1000
    pos = torch.randn(n_atoms, 3, device=device) * n_atoms
    edge_index = torch.stack([torch.arange(n_atoms), torch.arange(n_atoms - 1, -1, -1)], dim=0).to(
        device
    )
    print("edge_index", edge_index.shape)
    calc_distances(pos, edge_index)
    from time import perf_counter

    time_list = []
    for i in range(100):
        torch.cuda.synchronize()
        s0 = perf_counter()
        calc_distances(pos, edge_index)
        # if i < 10:
        #     print(f"----- {i} ------------------------")
        #     print(torch.jit.last_executed_optimized_graph())

        torch.cuda.synchronize()
        e0 = perf_counter()
        time_list.append(e0 - s0)
    print("Time: ", time_list)
    import IPython

    IPython.embed()
