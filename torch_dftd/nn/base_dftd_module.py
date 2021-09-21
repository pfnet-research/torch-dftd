from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from ase.units import Bohr
from torch import Tensor, nn


class BaseDFTDModule(nn.Module):
    """BaseDFTDModule"""

    def calc_energy_batch(
        self,
        Z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
        shift_pos: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_edge: Optional[Tensor] = None,
        damping: str = "zero",
    ) -> Tensor:
        """Forward computation to calculate atomic wise dispersion energy.

        Each subclass should override this method

        Args:
            Z (Tensor): (n_atoms,) atomic numbers.
            pos (Tensor): (n_toms, 3) atom positions in angstrom
            edge_index (Tensor): (2, n_edges) edge index within cutoff
            cell (Tensor): (n_atoms, 3) cell size in angstrom, None for non periodic system.
            pbc (Tensor): (bs, 3) pbc condition, None for non periodic system.
            shift_pos (Tensor): (n_atoms, 3) shift vector (length unit).
            batch (Tensor): (n_atoms,) Specify which graph this atom belongs to
            batch_edge (Tensor): (n_edges, 3) Specify which graph this edge belongs to
            damping (str):

        Returns:
            energy (Tensor): (n_atoms,)
        """
        raise NotImplementedError()

    def calc_energy(
        self,
        Z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
        shift_pos: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_edge: Optional[Tensor] = None,
        damping: str = "zero",
    ) -> List[Dict[str, Any]]:
        """Forward computation of dispersion energy

        Backward computation is skipped for fast computation of only energy.

        Args:
            Z (Tensor): (n_atoms,) atomic numbers.
            pos (Tensor): atom positions in angstrom
            edge_index (Tensor):
            cell (Tensor): cell size in angstrom, None for non periodic system.
            pbc (Tensor): pbc condition, None for non periodic system.
            shift_pos (Tensor):  (n_atoms, 3) shift vector (length unit).
            batch (Tensor):
            batch_edge (Tensor):
            damping (str): damping method. "zero", "bj", "zerom", "bjm"

        Returns:
            results_list (list): calculated result. It contains calculate energy in "energy" key.
        """
        with torch.no_grad():
            E_disp = self.calc_energy_batch(
                Z, pos, edge_index, cell, pbc, shift_pos, batch, batch_edge, damping=damping
            )
        if batch is None:
            return [{"energy": E_disp.item()}]
        else:
            if batch.size()[0] == 0:
                n_graphs = 1
            else:
                n_graphs = int(batch[-1]) + 1
            return [{"energy": E_disp[i].item()} for i in range(n_graphs)]

    def calc_energy_and_forces(
        self,
        Z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
        shift_pos: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_edge: Optional[Tensor] = None,
        damping: str = "zero",
    ) -> List[Dict[str, Any]]:
        """Forward computation of dispersion energy, force and stress

        Args:
            Z (Tensor): (n_atoms,) atomic numbers.
            pos (Tensor): atom positions in angstrom
            cell (Tensor): cell size in angstrom, None for non periodic system.
            pbc (Tensor): pbc condition, None for non periodic system.
            shift_pos (Tensor):  (n_atoms, 3) shift vector (length unit).
            damping (str): damping method. "zero", "bj", "zerom", "bjm"

        Returns:
            results (list): calculated results. Contains following:
                "energy": ()
                "forces": (n_atoms, 3)
                "stress": (6,)
        """
        pos.requires_grad_(True)
        if cell is not None:
            # pos is depending on `cell` size
            # We need to explicitly include this dependency to calculate cell gradient
            # for stress computation.
            # pos is assumed to be inside "cell", so relative position `rel_pos` lies between 0~1.
            assert isinstance(shift_pos, Tensor)
            shift_pos.requires_grad_(True)

        E_disp = self.calc_energy_batch(
            Z, pos, edge_index, cell, pbc, shift_pos, batch, batch_edge, damping=damping
        )

        E_disp.sum().backward()
        forces = -pos.grad  # [eV/angstrom]
        if batch is None:
            results_list = [{"energy": E_disp.item(), "forces": forces.cpu().numpy()}]
        else:
            if batch.size()[0] == 0:
                n_graphs = 1
            else:
                n_graphs = int(batch[-1]) + 1
            results_list = [{"energy": E_disp[i].item()} for i in range(n_graphs)]
            for i in range(n_graphs):
                results_list[i]["forces"] = forces[batch == i].cpu().numpy()

        if cell is not None:
            # stress = torch.mm(cell_grad, cell.T) / cell_volume
            # Get stress in Voigt notation (xx, yy, zz, yz, xz, xy)
            assert isinstance(shift_pos, Tensor)
            voigt_left = [0, 1, 2, 1, 2, 0]
            voigt_right = [0, 1, 2, 2, 0, 1]
            if batch is None:
                cell_volume = torch.det(cell).abs()
                cell_grad = torch.sum(
                    (pos[:, voigt_left] * pos.grad[:, voigt_right]).to(torch.float64), dim=0
                )
                cell_grad += torch.sum(
                    (shift_pos[:, voigt_left] * shift_pos.grad[:, voigt_right]).to(torch.float64),
                    dim=0,
                )
                stress = cell_grad.to(cell.dtype) / cell_volume
                results_list[0]["stress"] = stress.detach().cpu().numpy()
            else:
                assert isinstance(batch_edge, Tensor)
                # cell (bs, 3, 3)
                cell_volume = torch.det(cell).abs()
                cell_grad = pos.new_zeros((n_graphs, 6), dtype=torch.float64)
                cell_grad.scatter_add_(
                    0,
                    batch.view(batch.size()[0], 1).expand(batch.size()[0], 6),
                    (pos[:, voigt_left] * pos.grad[:, voigt_right]).to(torch.float64),
                )
                cell_grad.scatter_add_(
                    0,
                    batch_edge.view(batch_edge.size()[0], 1).expand(batch_edge.size()[0], 6),
                    (shift_pos[:, voigt_left] * shift_pos.grad[:, voigt_right]).to(torch.float64),
                )
                stress = cell_grad.to(cell.dtype) / cell_volume[:, None]
                stress = stress.detach().cpu().numpy()
                for i in range(n_graphs):
                    results_list[i]["stress"] = stress[i]
        return results_list
