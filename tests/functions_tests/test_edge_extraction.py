import pytest
import torch
from torch_dftd.functions.edge_extraction import calc_neighbor_by_ase, calc_neighbor_by_pymatgen


@pytest.mark.parametrize("pbc", [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (False, False, True),
    (False, True, False),
    (True, False, False),
    (False, False, False),
])
def test_calc_neighbor_equivalent(pbc):
    torch.manual_seed(42)
    n_nodes = 5
    pbc = torch.tensor(pbc)
    cell = torch.randn((3, 3))
    rel_pos = torch.rand((n_nodes, 3))  # relative position inside cell
    pos = torch.matmul(rel_pos, cell)
    cutoff = torch.rand(1).item() * 5.0

    edge_index1, S1 = calc_neighbor_by_ase(pos, cell, pbc, cutoff)
    edge_index2, S2 = calc_neighbor_by_pymatgen(pos, cell, pbc, cutoff)

    n_edges = edge_index1.shape[1]
    assert (
        edge_index1.shape == edge_index2.shape
    ), f"{edge_index1.shape} != {edge_index2.shape}, edge shape does not match!"
    assert S1.shape == S2.shape, f"{S1.shape} != {S2.shape}, Shift tensor shape does not match!"
    edge_shift_list1 = []
    edge_shift_list2 = []
    S1_int = S1.type(torch.long)
    S2_int = S2.type(torch.long)
    for i in range(n_edges):
        edge_shift_list1.append(
            (
                edge_index1[0, i].item(),
                edge_index1[1, i].item(),
                S1_int[i, 0].item(),
                S1_int[i, 1].item(),
                S1_int[i, 2].item(),
            )
        )
        edge_shift_list2.append(
            (
                edge_index2[0, i].item(),
                edge_index2[1, i].item(),
                S2_int[i, 0].item(),
                S2_int[i, 1].item(),
                S2_int[i, 2].item(),
            )
        )
    assert set(edge_shift_list1) == set(edge_shift_list2)


def test_not_all_periodic():
    pbc = torch.tensor([True, True, False])
    cell = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]])
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.9, 1.9, 0.0],
        [0.0, 0.0, 9.9],
    ])
    cutoff = torch.ones(1)

    edge_index, _ = calc_neighbor_by_ase(positions, cell, pbc, cutoff.item())
    assert edge_index.shape == (2, 2)

    edge_index, _ = calc_neighbor_by_pymatgen(positions, cell, pbc, cutoff.item())
    assert edge_index.shape == (2, 2)

    pbc = torch.tensor([False, False, True])
    edge_index, _ = calc_neighbor_by_ase(positions, cell, pbc, cutoff.item())
    assert edge_index.shape == (2, 2)
    assert edge_index[0].tolist() == [0, 2]

    edge_index, _ = calc_neighbor_by_pymatgen(positions, cell, pbc, cutoff.item())
    assert edge_index.shape == (2, 2)
    assert edge_index[0].tolist() == [0, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
