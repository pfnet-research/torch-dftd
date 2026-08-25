import sys, time, json, numpy as np
from dftd4.interface import DampingParam, DispersionModel
import torch
from ase.units import Bohr, Hartree
from ase.io import read
from torch_dftd.torch_dftd4_calculator import TorchDFTD4Calculator
VAL3="/development-pvc/work/wenwenli/aramoc-sampling/pipeline/v2/val3_exp/cifs"
def med(f,n=3):
    f(); ts=[]
    for _ in range(n): t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return float(np.median(ts))
for name,a in [("CALF-20 1x1x1",read(f"{VAL3}/CALF-20_111_DFT_OPT.cif")),("SIFSIX-3-Ni 2x2x1",read(f"{VAL3}/SIFSIX_3_Ni_GEO_OPT.cif")*(2,2,1)),("CALF-20 2x2x2",read(f"{VAL3}/CALF-20_111_DFT_OPT.cif")*(2,2,2))]:
    cut=(26.5,20.,26.5)
    m=DispersionModel(a.numbers,a.positions/Bohr,charge=0.0,lattice=a.cell.array/Bohr,periodic=np.array([True]*3)); m.set_realspace_cutoff(*cut)
    Ef=m.get_dispersion(DampingParam(method="pbe",atm=True),grad=False)["energy"]*Hartree
    for atm in (False,True):
        try:
            torch.cuda.reset_peak_memory_stats()
            calc=TorchDFTD4Calculator(xc="pbe",device="cuda",dtype=torch.float32,abc=atm,cutoff=cut[0]*Bohr,abc_cutoff=cut[1]*Bohr,cnthr=cut[2]*Bohr); b=a.copy(); b.calc=calc
            def E(): b.rattle(1e-6,seed=int(time.time()*1e6)%2**31); e=b.get_potential_energy(); torch.cuda.synchronize(); return e
            def EF(): b.rattle(1e-6,seed=int(time.time()*1e6)%2**31); b.get_forces(); torch.cuda.synchronize()
            tE=med(E); tEF=med(EF,2); e=b.get_potential_energy()
            print(f"{name:18s} N={len(a):5d} atm={int(atm)} GPU E {1e3*tE:7.1f} ms  E+F {1e3*tEF:7.1f} ms  peak {torch.cuda.max_memory_allocated()/2**30:5.2f} GB  " + (f"dE vs Fortran(atm) {e-Ef:+.1e} eV" if atm else ""),flush=True)
        except Exception as ex:
            print(f"{name:18s} N={len(a):5d} atm={int(atm)} FAIL {str(ex).splitlines()[0][:100]}",flush=True); torch.cuda.empty_cache()
