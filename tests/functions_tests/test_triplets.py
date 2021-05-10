import pytest
import torch
from torch_dftd.functions.triplets import calc_triplets


def test_calc_triplets():
    # TODO: Currently returned value has different order due to torch.argsort,
    # and expected value is not set correctly for GPU.
    # device = "cuda:0"
    device = "cpu"
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 4, 3, 1, 2, 4, 3], [4, 3, 1, 2, 4, 3, 0, 0, 0, 1, 1, 1]],
        dtype=torch.long,
        device=device,
    )
    shift = torch.zeros((edge_index.shape[1], 3), dtype=torch.float32, device=device)
    shift[:, 0] = torch.tensor(
        [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6], dtype=torch.float32, device=device
    )
    # print("shift", shift.shape)
    triplet_node_index, multiplicity, triplet_shift, batch_triplets = calc_triplets(
        edge_index, shift
    )
    # print("triplet_node_index", triplet_node_index.shape, triplet_node_index)
    # print("multiplicity", multiplicity.shape, multiplicity)
    # print("triplet_shift", triplet_shift.shape, triplet_shift)
    # print("triplet_shift[:, :, 0]", triplet_shift.shape, triplet_shift[:, :, 0])

    # 6 triplets exist.
    n_triplets = 6
    # idx_i, idx_j, idx_k = triplet_node_index
    assert triplet_node_index.shape == (n_triplets, 3)
    assert torch.all(
        triplet_node_index.cpu()
        == torch.tensor(
            [[0, 3, 4], [0, 1, 4], [0, 1, 3], [1, 2, 4], [1, 2, 3], [1, 3, 4]], dtype=torch.long
        )
    )
    assert multiplicity.shape == (n_triplets,)
    assert torch.all(multiplicity.cpu() == torch.ones((n_triplets,), dtype=torch.float32))
    assert torch.allclose(
        triplet_shift.cpu()[:, :, 0],
        torch.tensor(
            [
                [2.0, 1.0, -1.0],
                [3.0, 1.0, -2.0],
                [3.0, 2.0, -1.0],
                [4.0, 5.0, 1.0],
                [4.0, 6.0, 2.0],
                [6.0, 5.0, -1.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.all(batch_triplets.cpu() == torch.zeros((n_triplets,), dtype=torch.long))


def test_calc_triplets_noshift():
    # device = "cuda:0"
    device = "cpu"
    edge_index = torch.tensor(
        [[0, 1, 1, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 1, 3]], dtype=torch.long, device=device
    )
    triplet_node_index, multiplicity, triplet_shift, batch_triplets = calc_triplets(
        edge_index, dtype=torch.float64
    )
    # print("triplet_node_index", triplet_node_index.shape, triplet_node_index)
    # print("multiplicity", multiplicity.shape, multiplicity)
    # print("triplet_shift", triplet_shift.shape, triplet_shift)
    # print("batch_triplets", batch_triplets.shape, batch_triplets)

    # 2 triplets exist
    n_triplets = 2
    assert triplet_node_index.shape == (n_triplets, 3)
    assert torch.all(
        triplet_node_index.cpu() == torch.tensor([[0, 1, 3], [1, 2, 3]], dtype=torch.long)
    )
    assert multiplicity.shape == (n_triplets,)
    assert multiplicity.dtype == torch.float64
    assert torch.all(multiplicity.cpu() == torch.ones((n_triplets,), dtype=torch.float64))
    assert torch.all(
        triplet_shift.cpu()
        == torch.zeros(
            (n_triplets, 3, 3),
            dtype=torch.float32,
        )
    )
    assert torch.all(batch_triplets.cpu() == torch.zeros((n_triplets,), dtype=torch.long))


@pytest.mark.parametrize(
    "edge_index",
    [torch.zeros((2, 0), dtype=torch.long), torch.tensor([[0, 0], [1, 2]], dtype=torch.long)],
)
def test_calc_triplets_no_triplets(edge_index):
    # edge_index = edge_index.to("cuda:0")
    # No triplet exist in this graph. Case1: No edge, Case 2 No triplets in this edge.
    triplet_node_index, multiplicity, triplet_shift, batch_triplets = calc_triplets(edge_index)
    # print("triplet_node_index", triplet_node_index.shape, triplet_node_index)
    # print("multiplicity", multiplicity.shape, multiplicity)
    # print("triplet_shift", triplet_shift.shape, triplet_shift)
    # print("batch_triplets", batch_triplets.shape, batch_triplets)

    # 0 triplets exist.
    assert triplet_node_index.shape == (0, 3)
    assert multiplicity.shape == (0,)
    assert triplet_shift.shape == (0, 3, 3)
    assert batch_triplets.shape == (0,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
