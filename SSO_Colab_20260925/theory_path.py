"""New Torch path implementation vs independent NumPy/Brent and analytic centers.
Not a re-proof of the manuscript. Runs CPU and CUDA on the same fixed examples.
"""
import argparse, math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import torch
from common import save_json, environment, seed_all
from oracles import solve_ph, smooth_polar, assess, repair


def center(a, theta):
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    r = int((s > 1e-12).sum())
    p = u[:, :r] @ vh[:r]
    if r == min(a.shape): return p
    c = u[:, r:].T @ theta @ vh[r:].T
    # Used only for these corank-one square examples, not a general center rule.
    return p - np.sum(theta*p)/float(c[0,0])*np.outer(u[:,-1], vh[-1])


def independent_root(g, theta, delta):
    def h(x):
        u,s,vh = np.linalg.svd(g+x*theta, full_matrices=False)
        return np.sum(theta*((u*(s/np.hypot(s,delta)))@vh))
    return brentq(h, -4, 4, xtol=5e-15, rtol=1e-15)


def run(device, out):
    seed_all(719); torch.set_num_threads(1)
    theta = np.diag([1.,0.])
    cases = [
        ('regular', np.array([[0.,1.],[1.,.5]]), .5, [1e-2,3e-3,1e-3,3e-4], 2.),
        ('strict', np.array([[0.,.5],[.5,1.]]), .25, [1e-3,3e-4,1e-4,3e-5], 1.),
        ('boundary', np.array([[0.,1.],[1.,1.]]), 1., [1e-4,3e-5,1e-5,3e-6], 2/3)]
    rows=[]; slopes={}; checks=[]
    t=torch.tensor(theta,device=device,dtype=torch.float64)
    for name,g,star,ds,expected in cases:
        gg=torch.tensor(g,device=device,dtype=torch.float64)
        target=torch.tensor(center(g+star*theta,theta),device=device)
        errors=[]
        for delta in ds:
            lam,z,info=solve_ph(gg,t,delta)
            ref=independent_root(g,theta,delta)
            err=float((z-target).norm()); errors.append(err)
            row=dict(case=name,delta=delta,lambda_value=lam,lambda_star=star,
                     lambda_error=abs(lam-star),direction_error=err,
                     independent_lambda_difference=abs(ref-lam),solver=info,
                     **assess(gg,t,z,lam))
            rows.append(row)
            checks.append(abs(ref-lam)<1e-9 and row['tangency']<2e-10 and row['spectral_norm']<=1+1e-12)
        slope=float(np.polyfit(np.log(ds), np.log(errors),1)[0])
        slopes[name]=dict(observed=slope,expected=expected,tolerance=.04)
        checks.append(abs(slope-expected)<.04)
    # Uniform critical window in the actual manuscript's explicit analytic family.
    crossover=[]
    for zeta in [0.,1.,3.,10.]:
        y=brentq(lambda v:.5*v**3+zeta*v*v-1,1e-12,3)
        errors=[]
        for delta in [1e-4,1e-5,1e-6]:
            gap=zeta*delta**(2/3); c=math.sqrt((1+gap)/(1-gap)); star=1/c
            g=torch.tensor([[0.,1.],[1.,c]],device=device,dtype=torch.float64)
            lam,z,info=solve_ph(g,t,delta)
            observed=(star-lam)/delta**(2/3)
            errors.append(abs(observed-y))
            crossover.append(dict(zeta=zeta,delta=delta,gap=gap,observed=observed,target=y,solver=info))
        checks.append(errors[-1]<.02 and errors[-1]<errors[0])
    # Degenerate contact: violates kappa>0; selected endpoint is not the path limit.
    g=torch.diag(torch.tensor([1.,0.],dtype=torch.float64,device=device)); tt=torch.eye(2,dtype=torch.float64,device=device)
    lam,z,info=solve_ph(g,tt,.1)
    checks.append(abs(lam+.5)<1e-10)
    # Rectangular, zero matrix, general-normal repair and transpose equivariance.
    for shape in [(3,7),(7,3),(4,4)]:
        g=torch.randn(*shape,device=device,dtype=torch.float64)
        th=torch.randn(*shape,device=device,dtype=torch.float64)
        l,z,inf=solve_ph(g,th,.1)
        lr,zr,_=solve_ph(g.T,th.T,.1)
        a=assess(g,th,z,l)
        checks += [abs(l-lr)<1e-9,float((zr.T-z).norm())<1e-9,
                   a['repaired_tangency']<1e-10,a['repaired_spectral_norm']<=1+1e-10,
                   a['repaired_gap']>=-1e-9]
    z0=smooth_polar(torch.zeros(3,5,device=device),.1)
    checks.append(float(z0.norm())==0)
    result=dict(environment=environment(),device=str(device),gpu_executed=str(device).startswith('cuda'),
                all_passed=all(checks),check_count=len(checks),failed_check_indices=[i for i,v in enumerate(checks) if not v],slopes=slopes,rows=rows,crossover=crossover,
                degenerate_contact_lambda=lam,
                meaning='Numerical implementation/asymptotic check; not a new theorem or interval proof.')
    save_json(out,result)
    print({k:v for k,v in result.items() if k in ['all_passed','check_count','slopes']})
    if not all(checks): raise RuntimeError('Theory-path validation failed; inspect JSON before training')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--out',default='results/theory_path.json');a=p.parse_args()
    run(a.device,a.out)
