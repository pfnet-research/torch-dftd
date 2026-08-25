"""Speed benchmark: torch-dftd DFT-D4 (CPU / GPU) vs dftd4 Fortran (CPU). Energy-only and energy+forces."""

import sys, time, json, os, numpy as np
from dftd4.interface import DampingParam, DispersionModel  # before torch
import torch
from ase.units import Bohr, Hartree
from ase.io import read
from torch_dftd.torch_dftd4_calculator import TorchDFTD4Calculator

VAL3 = "/development-pvc/work/wenwenli/aramoc-sampling/pipeline/v2/val3_exp/cifs"
systems = [
    ("CALF-20 1x1x1", read(f"{VAL3}/CALF-20_111_DFT_OPT.cif")),
    ("SIFSIX-3-Ni 2x2x1", read(f"{VAL3}/SIFSIX_3_Ni_GEO_OPT.cif") * (2, 2, 1)),
    ("CALF-20 2x2x2", read(f"{VAL3}/CALF-20_111_DFT_OPT.cif") * (2, 2, 2)),
]
cutsets = {
    "d4-2body-default(60,20,30)": (60.0, 20.0, 30.0),
    "pfp-like(26.5,20,26.5)": (26.5, 20.0, 26.5),
}  # ATM cutoff fixed at 20 Bohr: explicit triplet enumeration at 40 Bohr needs ~1e9 triplets


def med(f, n=3):
    f()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


rows = []
for name, a in systems:
    for cname, cut in cutsets.items():
        for atm in (False, True):
            row = {"system": name, "N": len(a), "cutoffs": cname, "atm": atm}

            # Fortran
            def fort(grad):
                m = DispersionModel(
                    a.numbers,
                    a.positions / Bohr,
                    charge=0.0,
                    lattice=a.cell.array / Bohr,
                    periodic=np.array([True] * 3),
                )
                m.set_realspace_cutoff(*cut)
                return (
                    m.get_dispersion(DampingParam(method="pbe", atm=atm), grad=grad)["energy"]
                    * Hartree
                )

            if True:
                row["fortran_E_ms"] = round(1e3 * med(lambda: fort(False)), 1)
                row["fortran_EF_ms"] = round(1e3 * med(lambda: fort(True), 1), 1)
                row["E_fortran"] = round(fort(False), 6)
            for dev in ("cpu", "cuda"):
                if dev == "cpu" and len(a) > 1000 and cut[0] == 60.0:
                    continue
                try:
                    calc = TorchDFTD4Calculator(
                        xc="pbe",
                        device=dev,
                        dtype=torch.float32,
                        abc=atm,
                        cutoff=cut[0] * Bohr,
                        abc_cutoff=cut[1] * Bohr,
                        cnthr=cut[2] * Bohr,
                    )
                    b = a.copy()
                    b.calc = calc

                    def E():
                        b.rattle(1e-6, seed=int(time.time() * 1e6) % 2**31)
                        e = b.get_potential_energy()
                        if dev == "cuda":
                            torch.cuda.synchronize()
                        return e

                    def EF():
                        b.rattle(1e-6, seed=int(time.time() * 1e6) % 2**31)
                        f = b.get_forces()
                        if dev == "cuda":
                            torch.cuda.synchronize()

                    row[f"torch_{dev}_E_ms"] = round(1e3 * med(E), 1)
                    row[f"torch_{dev}_EF_ms"] = round(1e3 * med(EF, 2), 1)
                    t = time.perf_counter()
                    d = calc._preprocess_atoms(b)
                    row[f"torch_{dev}_edges_ms"] = round(1e3 * (time.perf_counter() - t), 1)
                    row["n_edges"] = int(d["edge_index"].shape[1])
                    row[f"E_torch_{dev}"] = round(float(b.get_potential_energy()), 6)
                    if dev == "cuda":
                        row["gpu_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                        torch.cuda.reset_peak_memory_stats()
                except Exception as e:
                    row[f"torch_{dev}_E_ms"] = "FAIL: " + str(e).splitlines()[0][:80]
            print(json.dumps(row), flush=True)
            rows.append(row)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
