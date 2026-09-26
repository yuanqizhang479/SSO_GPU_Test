"""Numerical illustration of existing supplement T6, not a new theorem.
Float64 checks are not interval-validated certificates or training results.
The polynomial coefficients are the eight-step sequence from Xie et al.,
Controlled LLM Training on Spectral Sphere, arXiv:2601.08393v3; the evaluation
here is an independent float64 diagnostic, not the released BF16 operator.
"""
if not __debug__:
    raise RuntimeError("Verification requires assertions: rerun without -O / PYTHONOPTIMIZE.")
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
G=np.array([[0.,1.],[1.,2.]])
Theta=np.diag([1.,0.])
def nuclear(A):return float(np.linalg.svd(A,compute_uv=False).sum())
def compact(A):
 u,s,vh=np.linalg.svd(A,full_matrices=False); keep=s>1e-12*max(1.,s[0]);return u[:,keep]@vh[keep,:]
coeffs=[(8.2051,-22.9019,16.4607),(4.0664,-2.8612,.5184),(3.9096,-2.8234,.5250),(3.2856,-2.4153,.4853),(2.2779,-1.6198,.3985),(1.8726,-1.2307,.3585),(1.8564,-1.2132,.3568),(1.875,-1.25,.375)]
def ns64(A):
 X=A/np.linalg.norm(A)
 for a,b,c in coeffs:
  B=X@X.T;X=a*X+(b*B+c*(B@B))@X
 return X
lam=brentq(lambda t:float(np.sum(Theta*ns64(G+t*Theta))),.499,.5,xtol=1e-15)
Phi=np.array([[0.,.5],[.5,.75]])
cases=[('unconstrained_compact_at_zero',0.,compact(G)),('compact_at_dual_minimizer',.5,compact(G+.5*Theta)),('completed_exact_optimizer',.5,Phi),('eight_step_NS_float64',lam,ns64(G+lam*Theta))]
rows=[]
for name,t,Z in cases:
 Zbar=Z/max(1.,float(np.linalg.norm(Z,2)));c=float(np.sum(Theta*Zbar));repair=(Zbar-c*Theta)/(1+abs(c));A=G+t*Theta
 rpol=nuclear(A)-float(np.sum(A*Zbar)); gap=nuclear(A)-float(np.sum(G*repair));upper=rpol+abs(t)*abs(c)+2*nuclear(G)*abs(c)/(1+abs(c))
 d={'case':name,'lambda':t,'actual_direction_tangent_residual':abs(c),'substituted_compact_residual':abs(float(np.sum(Theta*compact(A)))),'naive_dual_minus_unrepaired_objective':nuclear(A)-float(np.sum(G*Zbar)),'repaired_gap_float64':gap,'repaired_true_suboptimality_float64':2.5-float(np.sum(G*repair)),'T6_decomposition_upper_float64':upper,'repaired_spectral_norm':float(np.linalg.norm(repair,2)),'repaired_tangent_residual':abs(float(np.sum(Theta*repair)))}
 assert d['repaired_spectral_norm']<=1+1e-12
 assert d['repaired_tangent_residual']<=1e-12
 assert gap>=d['repaired_true_suboptimality_float64']-1e-12
 assert gap<=upper+1e-12
 rows.append(d)
# Counterexample to the manuscript's unqualified relative-interior equivalence.
G0=np.diag([1.,0.]);T0=np.array([[0.,1.],[1.,0.]])
assert all(abs(nuclear(G0+t*T0)-np.sqrt(1+4*t*t))<1e-12 for t in [-1.,-.1,0.,.1,1.])
res={'scope':'Existing-T6 numerical illustration; NumPy float64, not directed-rounding certification; NS is a float64 polynomial comparison, not production BF16.','cases':rows,'ri_counterexample':{'G':G0.tolist(),'Theta':T0.tolist(),'a':0.,'beta':0.,'unique_minimizer':0.,'phi':'sqrt(1+4 lambda^2)','conclusion':'0 belongs to ri{0}, although abs(a)<beta is false.'}}
p=Path(__file__).resolve().parent/'results'/'certificate_example_results.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
