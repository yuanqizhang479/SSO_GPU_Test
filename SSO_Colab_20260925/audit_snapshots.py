"""Paired oracles on saved FULL matrices; never manufacture rank deficiency.

The finite NS bracket uses sign changes, not a claimed monotonicity theorem.
BF16 is deliberately skipped on GPUs without native BF16 support.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import save_json, sha256, environment, sync, json_hash, seed_all
from optimizers import solve_ns, finite_ns, compact_polar, exact_normal
from oracles import solve_ph, assess


def run_one(path, device):
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']))
        arrays={k:torch.tensor(f[k],device=device) for k in ['W','G','theta_est','Z','W_after']}
    w,g,te,zactual=[arrays[k] for k in ['W','G','theta_est','Z']]
    _,tt,gap=exact_normal(w)
    tn_error=float((tt-te.double()).norm())
    records=[]

    def record(label,z,lam,info,normal,seconds):
        r=dict(label=label,lambda_value=lam,solver=info,normal_used=normal,
               diagnostic_solve_seconds=seconds,**assess(g,tt,z,lam))
        r['tangency_estimated_normal']=float((te.double()*z.double()).sum().abs())
        records.append(r)
        return r

    record('actual_saved_update',zactual,meta['lambda_value'],meta['solver'],'estimated',None)
    ph_ref=None; ref_lam=None; ph_status=[]
    for normal,th in [('true_svd_reference',tt),('estimated_pi',te)]:
        for delta in ([1e-2,1e-3,1e-4] if normal=='true_svd_reference' else [1e-3]):
            sync(device); start=time.perf_counter()
            lam,z,info=solve_ph(g,th,delta)
            sync(device); elapsed=time.perf_counter()-start
            # assess(g,tt,...) evaluates the SAME true-normal optimization problem for every candidate.
            row=record(f'ph_{normal}_delta{delta}',z,lam,info,normal,elapsed)
            row['delta']=delta; row['solver_residual_own_normal']=info['residual']
            ph_status.append(info['residual']<=1e-8)
            if normal=='true_svd_reference' and delta==1e-4:
                ph_ref=z; ref_lam=lam
    dtypes=['float32']
    bf16_available=torch.device(device).type=='cpu' or torch.cuda.is_bf16_supported()
    if bf16_available: dtypes.append('bfloat16')
    for dtype in dtypes:
        for steps in [5,8,12]:
            sync(device); start=time.perf_counter()
            lam,z,info=solve_ns(g,te,initial=meta['lambda_initial'],steps=steps,dtype=dtype)
            sync(device); elapsed=time.perf_counter()-start
            row=record(f'ns_k{steps}_{dtype}',z,lam,info,'estimated_pi',elapsed)
            row['distance_to_ph_delta1e-4']=float((z.double()-ph_ref).norm())
    # Separate normal approximation from polynomial/solver error.
    for name,th,tol,expand in [('true_normal',tt.float(),1e-4,10),
                              ('tighter_residual',te,1e-6,10),
                              ('larger_bracket',te,1e-4,20)]:
        sync(device); start=time.perf_counter()
        lam,z,info=solve_ns(g,th,initial=meta['lambda_initial'],tol=tol,max_expand=expand)
        sync(device)
        record('ns_'+name,z,lam,info,name,time.perf_counter()-start)
    # A sampled monotonicity diagnostic is not a global proof of monotonicity/nonmonotonicity.
    grid=np.linspace(ref_lam-.25,ref_lam+.25,17)
    h=[float((te*finite_ns(g+float(l)*te,8,'float32')).sum()) for l in grid]
    ss=torch.linalg.svdvals(g.double()+ref_lam*tt)
    rel=ss/ss[0].clamp_min(1e-30)
    substituted=compact_polar(g.double()+ref_lam*tt)
    record('compact_at_ph_lambda_not_an_exact_minimizer',substituted,ref_lam,
           {'status':'substitution_only'},'true_svd_reference',None)
    best_upper=min(r['primal_upper'] for r in records)
    best_lower=max(r['repaired_objective'] for r in records)
    for r in records:
        r['common_upper_minus_repaired_objective']=best_upper-r['repaired_objective']
    actual=records[0]
    return dict(snapshot=str(path),snapshot_sha256=sha256(path),metadata=meta,
                shape=list(g.shape),full_matrix=True,relative_weight_top_gap=gap,
                normal_nonunique_or_illconditioned_flag=gap<=1e-6,
                normal_frobenius_error=tn_error,all_ph_residuals_ok=all(ph_status),
                bf16_skipped=not bf16_available,
                numerical_small_singular_value_counts={str(e):int((rel<e).sum()) for e in [1e-2,1e-4,1e-6]},
                thresholds_are_not_exact_rank=True,
                weight_radius_relative_error_after=abs(float(torch.linalg.matrix_norm(arrays['W_after'].double(),2))/meta['radius']-1),
                reference_lower=best_lower,reference_upper=best_upper,reference_interval_width=best_upper-best_lower,
                actual_material_residual_flag=actual['tangency']>1e-3 or actual['spectral_norm']>1.001,
                scan=dict(lambdas=grid.tolist(),values=h,negative_increments_below_minus_1e6=int((np.diff(h)<-1e-6).sum()),
                          note='Local 17-point diagnostic, not a root uniqueness or global monotonicity proof'),
                records=records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(1);seed_all(719)
    paths=sorted(Path(a.root).rglob('snapshots/*.npz'))
    if not paths: raise FileNotFoundError('No full-matrix snapshots found')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    code_hash=json_hash({p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    identity=dict(code_hash=code_hash,device=a.device,torch=torch.__version__,numpy=np.__version__)
    rows=[]
    for i,path in enumerate(paths):
        digest=sha256(path);key=json_hash({'snapshot':digest,**identity})
        target=out/(key+'.json')
        if target.exists(): result=json.loads(target.read_text())
        else:
            try: result=run_one(path,a.device);result['audit_identity']=identity
            except Exception as e:
                save_json(out/(key+'.failure.json'),dict(snapshot=str(path),error=repr(e),audit_identity=identity))
                raise
            save_json(target,result)
        rows.append(result)
        print(f'audit {i+1}/{len(paths)}: {path.name}',flush=True)
    counted=[r for r in rows if r['metadata']['optimizer']=='sso_ns']
    result=dict(environment=environment(),audit_identity=identity,snapshot_count=len(rows),
        sso_ns_snapshot_count=len(counted),sso_ns_material_residual_count=sum(r['actual_material_residual_flag'] for r in counted),
        normal_illconditioned_count=sum(r['normal_nonunique_or_illconditioned_flag'] for r in rows),
        ph_reference_failures=sum(not r['all_ph_residuals_ok'] for r in rows),
        snapshots=[dict(file=r['snapshot'],sha256=r['snapshot_sha256'],
           optimizer=r['metadata']['optimizer'],step=r['metadata']['step'],parameter=r['metadata']['parameter'],
           material_residual=r['actual_material_residual_flag'],reference_interval_width=r['reference_interval_width']) for r in rows],
        interpretation='Selected dependent snapshots, not an estimate of population failure probability. All bounds are float64 diagnostics, not interval-certified.')
    save_json(out/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k.endswith('count') or k.endswith('failures')},indent=2))
    if result['ph_reference_failures']: raise RuntimeError('Reference residual too large: inspect results before interpreting NS errors')


if __name__=='__main__':main()
