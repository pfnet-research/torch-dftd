import os
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from ase import Atoms
from ase.calculators.dftd3 import DFTD3
from ase.cluster.cubic import FaceCenteredCubic
from ase.io import write
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator


def compare_forces(atoms: Atoms, calc1, calc2):
    print(f"atoms # {len(atoms.numbers)}")
    calc1.reset()
    atoms.calc = calc1
    start = perf_counter()
    try:
        F1 = atoms.get_forces()
        t1 = perf_counter() - start
    except:
        print("Calculation failed")
        F1 = np.array([np.nan])
        t1 = np.nan

    print(f"F1 {F1.shape} took {t1} sec")

    calc2.reset()
    atoms.calc = calc2
    start = perf_counter()
    F2 = atoms.get_forces()
    t2 = perf_counter() - start
    print(f"F2 {F2.shape} took {t2} sec")
    # print(F2)
    print(f"diff {np.max(np.abs(F1 - F2))}, calc1/calc2 -> {t1 / t2} times faster")
    return t1, t2, F1, F2


def create_fcc_cluster_atoms(layers):
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    # layers = [4, 4, 4]
    lc = 3.61000
    cluster = FaceCenteredCubic("Cu", surfaces, layers, latticeconstant=lc)
    return Atoms(cluster.symbols, cluster.positions, cell=cluster.cell)


if __name__ == "__main__":
    os.makedirs(str("results"), exist_ok=True)

    damping = "bj"
    xc = "pbe"
    device = "cuda:0"
    old = False
    print("Initializing calculators...")
    print(f"xc = {xc}, damping = {damping}, old = {old}")
    torch_dftd3_calc = TorchDFTD3Calculator(
        damping=damping, xc=xc, device=device, dtype=torch.float64, old=old, bidirectional=True
    )
    dftd3_calc = DFTD3(damping=damping, xc=xc, grad=True, old=old, directory=".")

    F1_F2_list = []
    t1_list = []
    t2_list = []
    name_list = []

    # Dry-run once.
    atoms = create_fcc_cluster_atoms([3, 3, 3])
    t1, t2, F1, F2 = compare_forces(atoms, dftd3_calc, torch_dftd3_calc)

    n_repeat = 10
    for i in [3, 5, 7, 9]:
        print(f"Calculate Cu cluster with size ({i}, {i}, {i})")
        atoms = create_fcc_cluster_atoms([i, i, i])

        _t1_list = []
        _t2_list = []
        for j in range(n_repeat):
            t1, t2, F1, F2 = compare_forces(atoms, dftd3_calc, torch_dftd3_calc)
            _t1_list.append(t1)
            _t2_list.append(t2)

        if np.sum(np.isnan(F1)) == 0:
            F1_F2_list.append([F1, F2])  # Only add successful results
        t1_list.append(np.mean(_t1_list) * 1000)  # take average in ms order
        t2_list.append(np.mean(_t2_list) * 1000)  # take average in ms order
        name_list.append(f"cluster{i}{i}{i}: {atoms.get_number_of_atoms()} atoms")

        write(f"results/cluster{i}{i}{i}_v1.png", atoms)
        write(f"results/cluster{i}{i}{i}_v2.png", atoms, rotation="225z, -60x")

    # --- Check time ---
    df = pd.DataFrame(
        {
            "name": name_list,
            "DFTD3": t1_list,
            "TorchDFTD3": t2_list,
        }
    )
    melt_df = pd.melt(df, id_vars="name", value_vars=["DFTD3", "TorchDFTD3"])
    melt_df["value1"] = melt_df["value"].round(0)
    melt_df = melt_df.rename(
        {"name": "Atoms", "variable": "Calculator", "value": "time (ms)", "value1": "time_round"},
        axis=1,
    )
    fig = px.bar(
        melt_df,
        x="Atoms",
        y="time (ms)",
        color="Calculator",
        barmode="group",
        title=f"Execution time comparison",
        text="time_round",
        height=600,
        width=1200,
        orientation="v",
    )
    # fig.show()
    fig.write_image("results/exe_time.png")

    print("Saved to exe_time.png")
    print("Execution time list:")
    print(t1_list)
    print(t2_list)

    # --- Check calculated result is same ---
    # (n_total_atoms, 3)
    F1 = np.concatenate([f1 for f1, f2 in F1_F2_list], axis=0)
    F2 = np.concatenate([f2 for f1, f2 in F1_F2_list], axis=0)
    mae = np.mean(np.abs(F1 - F2))
    max_ae = np.max(np.abs(F1 - F2))

    F1_F2 = np.array([F1, F2])
    fig, ax = plt.subplots()
    E_max = np.max(F1_F2)
    E_min = np.min(F1_F2)

    ax.plot([E_min, E_max], [E_min, E_max])
    for i in range(3):
        # Fx, Fy, Fz scatter plot
        ax.scatter(F1[:, i], F2[:, i], label=["x", "y", "z"][i], marker="x")
    ax.set_xlabel("ase DFTD3")
    ax.set_ylabel("pytorch DFTD3")
    ax.set_title(f"DFTD3 Force difference MAE: {mae:.3} eV")
    ax.legend()
    fig.savefig("results/F1-F2.png")

    print("Saved to F1-F2.png")
    print("MAE", mae)
    print("Max AE", max_ae)
