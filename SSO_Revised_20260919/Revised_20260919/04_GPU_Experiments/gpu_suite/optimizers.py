"""Independent, instrumented implementations; not Xie et al.'s Megatron code.
The finite Polar-Express coefficients match their standalone public sso.py.
Finite NS is not assumed to equal the polar factor or any Fenchel smoothing.
"""
import math, torch
from torch.optim import Optimizer
COEFFICIENTS = [(8.2051,-22.9019,16.4607),(4.0664,-2.8612,.5184),(3.9096,-2.8234,.5250),
 (3.2856,-2.4153,.4853),(2.2779,-1.6198,.3985),(1.8726,-1.2307,.3585),
 (1.8564,-1.2132,.3568),(1.875,-1.25,.375)]

@torch.no_grad()
def finite_ns(g, steps=8, dtype='float32'):
    x = g.to(getattr(torch, dtype)); transposed = x.shape[0] > x.shape[1]
    if transposed: x = x.mT
    x = x / x.norm().clamp_min(1e-7)
    for i in range(steps):
        a,b,c = COEFFICIENTS[min(i,7)]; aa = x @ x.mT
        x = a*x + (b*aa+c*(aa@aa))@x
    return (x.mT if transposed else x).to(g.dtype)

@torch.no_grad()
def power_normal(w, steps=100):
    w = w.float(); v = torch.ones(w.shape[1], device=w.device, dtype=w.dtype)
    for _ in range(steps):
        u = w @ v; u = u/u.norm().clamp_min(1e-20)
        v = w.mT @ u; v = v/v.norm().clamp_min(1e-20)
    sigma = torch.dot(u, w@v)
    if float(sigma) <= 0 or not torch.isfinite(sigma):
        raise RuntimeError('Power iteration produced zero/nonfinite sigma; recorded failure, no silent retraction')
    return sigma, torch.outer(u,v)

@torch.no_grad()
def exact_normal(w):
    u,s,vh = torch.linalg.svd(w.double(), full_matrices=False)
    gap = float((s[0]-s[1])/s[0]) if len(s)>1 else 1.
    return float(s[0]), torch.outer(u[:,0],vh[0]), gap

@torch.no_grad()
def compact_polar(a):
    u,s,vh = torch.linalg.svd(a.double(), full_matrices=False)
    tol = max(a.shape)*torch.finfo(torch.float64).eps*s[0].clamp_min(1)
    return ((u*(s>tol))@vh).to(a.dtype)

@torch.no_grad()
def solve_ns(g, theta, initial=0., steps=8, dtype='float32', tol=1e-4, max_iter=20, max_expand=10):
    def evaluate(lam):
        z=finite_ns(g+lam*theta,steps,dtype); return float((theta*z).sum()),z
    f0,z0 = evaluate(initial); calls=1
    if abs(f0)<=tol: return initial,z0,{'status':'initial_ok','calls':calls,'estimated_residual':abs(f0)}
    step = .001 if f0<0 else -.001; a=initial; fa=f0
    best=(abs(f0),initial,z0)
    for _ in range(max_expand):
        b=a+step; fb,zb=evaluate(b); calls+=1
        if abs(fb)<best[0]: best=(abs(fb),b,zb)
        if (fa<=0)!=(fb<=0): break
        a,fa=b,fb; step*=2
    else:
        # Preserve original standalone behavior, but expose failure explicitly.
        return initial,z0,{'status':'bracket_failed_fallback','calls':calls,'estimated_residual':abs(f0)}
    if fa>0: a,b,fa,fb=b,a,fb,fa
    for _ in range(max_iter):
        mid=(a+b)/2; fm,zm=evaluate(mid); calls+=1
        if abs(fm)<best[0]: best=(abs(fm),mid,zm)
        if abs(fm)<=tol: break
        if fm<0: a,fa=mid,fm
        else: b,fb=mid,fm
    residual,lam,z=best
    return lam,z,{'status':'converged' if residual<=tol else 'residual_failed',
                 'calls':calls,'estimated_residual':residual}

@torch.no_grad()
def feasible_repair(z, theta):
    """SVD diagnostic repair: exact-real feasible for unit rank-one theta; float64 not interval proof."""
    z=z.double(); theta=theta.double(); z=z/torch.linalg.matrix_norm(z,2).clamp_min(1)
    c=(theta*z).sum(); return (z-c*theta)/(1+c.abs())

@torch.no_grad()
def diagnostics(w, g, z, theta_est, lam, after=None, radius=None):
    sigma,theta,gap=exact_normal(w); theta=theta.double(); z=z.double(); g=g.double()
    repair=feasible_repair(z,theta); a=g+lam*theta
    d={'sigma_before':sigma,'relative_top_gap':gap,
       'normal_unique_at_numeric_tolerance':gap>1e-10,
       'actual_tangency_exact_normal':float((theta*z).sum().abs()),
       'actual_tangency_estimated_normal':float((theta_est.double()*z).sum().abs()),
       'substituted_compact_tangency_same_estimated_problem':float((theta*compact_polar(g+lam*theta_est.double())).sum().abs()),
       'substituted_compact_tangency_true_problem':float((theta*compact_polar(a)).sum().abs()),
       'direction_spectral_norm':float(torch.linalg.matrix_norm(z,2)),
       'repaired_primal_dual_gap_float64':float(torch.linalg.svdvals(a).sum()-(g*repair).sum()),
       'normal_frobenius_error':float(torch.linalg.vector_norm(theta-theta_est.double()))}
    if after is not None:
        d['sigma_after']=float(torch.linalg.matrix_norm(after.double(),2))
        d['radius_relative_error_after']=abs(d['sigma_after']/radius-1)
        d['actual_displacement_normal_component']=float((theta*(after.double()-w.double())).sum())
    return d

class MatrixOptimizer(Optimizer):
    def __init__(self, named_params, method, lr=.01, momentum=.9, ns_dtype='float32', ns_steps=8,
                 pi_steps=100, solver_tol=1e-4, max_expand=10, weight_decay=0.):
        self.names=[n for n,p in named_params]; params=[p for n,p in named_params]
        super().__init__(params,dict(lr=lr,momentum=momentum)); self.method=method
        self.ns_dtype=ns_dtype; self.ns_steps=ns_steps; self.pi_steps=pi_steps
        self.solver_tol=solver_tol; self.max_expand=max_expand; self.weight_decay=weight_decay
        self.diagnostic_rows=[]; self.solver_counts={}

    @torch.no_grad()
    def step(self, diagnostic=False):
        self.diagnostic_rows=[]; self.solver_counts={}; idx=0
        for group in self.param_groups:
            for p in group['params']:
                name=self.names[idx]; idx+=1
                if p.grad is None: continue
                st=self.state[p]; beta=group['momentum']; g=p.grad.float()
                if 'momentum' not in st: st['momentum']=torch.zeros_like(p)
                st['momentum'].mul_(beta).add_(g,alpha=1-beta)
                direction=g*(1-beta)+st['momentum']*beta
                direction=direction/direction.norm().clamp_min(1e-7)
                radius=math.sqrt(p.shape[0]/p.shape[1]); lam=0.
                if self.method!='muon':
                    # Match the pre-update retraction convention: direction is evaluated at this state.
                    sigma,theta=power_normal(p,self.pi_steps); p.mul_(radius/sigma)
                elif diagnostic:
                    _,theta,_=exact_normal(p)
                else:
                    theta=None
                before=p.detach().clone() if diagnostic else None
                if self.method.startswith('sso_ns'):
                    lam,z,info=solve_ns(direction,theta,st.get('lambda',0.),self.ns_steps,self.ns_dtype,
                                       self.solver_tol,max_expand=self.max_expand)
                    st['lambda']=lam; self.solver_counts[info['status']]=self.solver_counts.get(info['status'],0)+1
                else: z=finite_ns(direction,self.ns_steps,self.ns_dtype)
                if self.method=='sso_ns_repair_svd':
                    _,theta_true,_=exact_normal(p)
                    z=feasible_repair(z,theta_true).to(p.dtype)
                if self.weight_decay: p.mul_(1-group['lr']*self.weight_decay)
                p.add_(z,alpha=-group['lr']*radius)
                if diagnostic:
                    row=diagnostics(before,direction,z,theta,lam,p,radius); row['parameter']=name
                    self.diagnostic_rows.append(row)
