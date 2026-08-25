import sys, time, numpy as np, torch
from ase.units import Bohr
from ase.io import read
import torch_dftd.functions.dftd4 as F
from torch_dftd.torch_dftd4_calculator import TorchDFTD4Calculator
dev=sys.argv[1]; abc=sys.argv[2]=="1"; cut=tuple(float(x) for x in sys.argv[3].split(","))
a=read("/development-pvc/work/wenwenli/aramoc-sampling/pipeline/v2/val3_exp/cifs/CALF-20_111_DFT_OPT.cif")
calc=TorchDFTD4Calculator(xc="pbe", device=dev, dtype=torch.float32, abc=abc, cutoff=cut[0]*Bohr, abc_cutoff=cut[1]*Bohr, cnthr=cut[2]*Bohr)
# wrap functions with timers
timers={}
def wrap(name):
    f=getattr(F,name)
    def g(*args,**kw):
        if dev.startswith("cuda"): torch.cuda.synchronize()
        t=time.perf_counter(); out=f(*args,**kw)
        if dev.startswith("cuda"): torch.cuda.synchronize()
        timers[name]=timers.get(name,0)+time.perf_counter()-t; return out
    setattr(F,name,g)
for n in ["ncoord_d4","ncoord_eeq","eeq_charges","weight_references","_edge_c6","calc_triplets"]: wrap(n)
a.calc=calc
t=time.perf_counter(); d=calc._preprocess_atoms(a); 
if dev.startswith("cuda"): torch.cuda.synchronize()
t_edges=time.perf_counter()-t; n_edges=d["edge_index"].shape[1]
t=time.perf_counter(); E=a.get_potential_energy(); 
if dev.startswith("cuda"): torch.cuda.synchronize()
t_tot=time.perf_counter()-t
timers={}
t=time.perf_counter(); a.rattle(1e-5); E=a.get_potential_energy()
if dev.startswith("cuda"): torch.cuda.synchronize()
t_tot2=time.perf_counter()-t
print(f"dev={dev} abc={abc} cut={cut} N={len(a)} edges={n_edges} | edge-list {t_edges:.2f}s | energy(1st) {t_tot:.2f}s | energy(2nd) {t_tot2:.2f}s | parts: " + " ".join(f"{k}={v:.2f}" for k,v in timers.items()))
