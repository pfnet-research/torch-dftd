"""Finite-difference check of ATM forces (periodic) against Fortran analytic gradient and Fortran energies."""
import numpy as np
from dftd4.interface import DampingParam, DispersionModel
import torch
from ase.units import Bohr, Hartree
from ase.io import read
from torch_dftd.torch_dftd4_calculator import TorchDFTD4Calculator
a = read("/development-pvc/work/wenwenli/aramoc-sampling/pipeline/v2/val3_exp/cifs/SIFSIX_3_Ni_GEO_OPT.cif")
cut=(60,20,30)
def fE(atoms):
    m=DispersionModel(atoms.numbers, atoms.positions/Bohr, charge=0.0, lattice=atoms.cell.array/Bohr, periodic=np.array([True]*3)); m.set_realspace_cutoff(*cut)
    r=m.get_dispersion(DampingParam(method="pbe", atm=True), grad=True); return r["energy"]*Hartree, -r["gradient"]*Hartree/Bohr
E0,Ff=fE(a)
calc=TorchDFTD4Calculator(xc="pbe", dtype=torch.float64, abc=True, cutoff=cut[0]*Bohr, abc_cutoff=cut[1]*Bohr, cnthr=cut[2]*Bohr); b=a.copy(); b.calc=calc; Ft=b.get_forces()
d=np.abs(Ft-Ff); i,k=np.unravel_index(d.argmax(), d.shape); print("max |F_torch-F_fortran| at atom",i,"comp",k,":",d.max())
for (ii,kk) in [(i,k),(0,0),(5,2)]:
    h=1e-3; ap=a.copy(); ap.positions[ii,kk]+=h; am=a.copy(); am.positions[ii,kk]-=h
    fd=-(fE(ap)[0]-fE(am)[0])/(2*h)
    print(f"atom {ii} comp {kk}: FD(fortran E)={fd:+.6f}  F_torch={Ft[ii,kk]:+.6f}  F_fortran={Ff[ii,kk]:+.6f}")
