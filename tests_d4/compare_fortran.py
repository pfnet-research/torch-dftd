"""Compare torch-dftd DFT-D4 against the reference dftd4 (Fortran) library: CN, EEQ charges, energy, forces."""
import sys, time, numpy as np
from dftd4.interface import DampingParam, DispersionModel   # import BEFORE torch (symbol clash otherwise)
import torch
from ase.units import Bohr, Hartree
from ase.build import molecule
from ase.io import read
from torch_dftd.torch_dftd4_calculator import TorchDFTD4Calculator

def fortran(atoms, atm, cutoffs=None, charge=0.0):
    lat = atoms.cell.array / Bohr if atoms.pbc.any() else None
    per = np.array(atoms.pbc, dtype=bool) if atoms.pbc.any() else None
    m = DispersionModel(atoms.numbers, atoms.positions / Bohr, charge=charge, lattice=lat, periodic=per)
    if cutoffs: m.set_realspace_cutoff(*cutoffs)
    props = m.get_properties()
    res = m.get_dispersion(DampingParam(method="pbe", atm=atm), grad=True)
    return dict(E=res["energy"] * Hartree, F=-res["gradient"] * Hartree / Bohr, q=props["partial charges"], cn=props["coordination numbers"])

def run(name, atoms, atm, cut_bohr=(60, 40, 30), device="cpu", dtype=torch.float64, charge=0.0):
    ref = fortran(atoms, atm, cut_bohr, charge)
    calc = TorchDFTD4Calculator(xc="pbe", device=device, dtype=dtype, abc=atm, cutoff=cut_bohr[0]*Bohr, abc_cutoff=cut_bohr[1]*Bohr, cnthr=cut_bohr[2]*Bohr, charge=charge)
    a = atoms.copy(); a.calc = calc
    t=time.perf_counter(); E = a.get_potential_energy(); F = a.get_forces(); dt=time.perf_counter()-t
    q = calc.dftd_module.last_charges.cpu().numpy()
    print(f"{name:28s} N={len(a):4d} atm={int(atm)} E_torch={E:12.6f} E_fortran={ref['E']:12.6f} dE={E-ref['E']:+.2e} eV | max|dq|={np.abs(q-ref['q']).max():.2e} | max|dF|={np.abs(F-ref['F']).max():.2e} eV/A | {dt*1e3:.0f} ms", flush=True)
    return E, ref

if __name__ == "__main__":
    mols = [("CO2", molecule("CO2")), ("H2O", molecule("H2O")), ("benzene", molecule("C6H6")), ("C60", molecule("C60"))]
    for n, m in mols[:4]:
        m.center(vacuum=0); m.pbc=False
        for atm in (False, True): run(n, m, atm)
    # charged molecule
    m = molecule("NH3"); m.pbc=False; run("NH4+ (NH3 geom, q=+1)", m, True, charge=1.0)
    VAL3="/development-pvc/work/wenwenli/aramoc-sampling/pipeline/v2/val3_exp/cifs"
    for n, sc in [("CALF-20", (1,1,1)), ("SIFSIX_3_Ni", None)]:
        f = {"CALF-20": "CALF-20_111_DFT_OPT.cif", "SIFSIX_3_Ni": "SIFSIX_3_Ni_GEO_OPT.cif"}[n]
        a = read(f"{VAL3}/{f}")
        for atm in (False, True): run(n, a, atm, cut_bohr=(60, 20, 30))
