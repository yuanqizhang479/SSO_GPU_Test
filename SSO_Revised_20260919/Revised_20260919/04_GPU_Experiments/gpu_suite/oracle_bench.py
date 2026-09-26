"""E0: explicitly synthetic rank-deficient counterexample and random matrix diagnostics."""
import argparse,json,time
from pathlib import Path
import torch
from common import environment,save_json,seed_all,sync
from optimizers import finite_ns,solve_ns,feasible_repair,diagnostics,power_normal,compact_polar

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--out',default='results/e0.json')
    p.add_argument('--sizes',default='64x64,64x128,128x64');p.add_argument('--seeds',default='11,22,33');p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads);device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    g=torch.tensor([[0.,1.],[1.,2.]],device=device,dtype=torch.float64)
    theta=torch.diag(torch.tensor([1.,0.],device=device,dtype=torch.float64))
    phi=torch.tensor([[0.,.5],[.5,.75]],device=device,dtype=torch.float64)
    exact={'lambda_star':.5,'dual_value':float(torch.linalg.svdvals(g+.5*theta).sum()),
           'primal_value':float((g*phi).sum()),'completed_tangency':float((theta*phi).sum()),
           'completed_spectral_norm':float(torch.linalg.matrix_norm(phi,2)),
           'compact_tangency_at_minimizer':float((theta*compact_polar(g+.5*theta)).sum()),
           'left_limit_h':-.6,'right_limit_h':1.,'interpretation':'single-valued compact polar has no zero; feasible completion is exact'}
    if abs(exact['dual_value']-2.5)>1e-12 or abs(exact['primal_value']-2.5)>1e-12: raise AssertionError(exact)
    rows=[]
    for seed in map(int,a.seeds.split(',')):
        seed_all(seed)
        for spec in a.sizes.split(','):
            m,n=map(int,spec.split('x'));w=torch.randn(m,n,device=device);g=torch.randn_like(w);g/=g.norm()
            _,theta_est=power_normal(w,100)
            for dtype in ['float32','bfloat16']:
                # One warmup, then synchronized measured solve; result measures this implementation.
                finite_ns(g,dtype=dtype);sync(device);start=time.perf_counter()
                lam,z,info=solve_ns(g,theta_est,dtype=dtype);sync(device);elapsed=time.perf_counter()-start
                row={'seed':seed,'shape':[m,n],'ns_dtype':dtype,'lambda':lam,'solve_seconds':elapsed,**info}
                row.update(diagnostics(w,g,z,theta_est,lam));rows.append(row)
    result={'environment':environment(),'gpu_executed':device.type=='cuda','counterexample':exact,'random_cases':rows,
            'scope':'Numerical diagnostics, no claim about training incidence; SVD float64 is not interval certification.'}
    save_json(a.out,result);print(json.dumps({'out':a.out,'cases':len(rows),'gpu_executed':result['gpu_executed']}))
if __name__=='__main__':main()
