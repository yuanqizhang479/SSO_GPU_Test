"""OPTIONAL equal-budget LR pilots and paired-seed training. No default auto-launch.
Plans isolate (repair), NS iteration count (precision), or expensive PH reference (ph).
All three LR candidates get equal steps; held-out seeds=11,22,33. Never select on test.
"""
import argparse, json, math, statistics
from pathlib import Path
from common import save_json, sha256, json_hash
from run_stage import ROOT, invoke

PLANS={
    'repair': [('adamw','adamw',8),('muon','muon',8),('muon_sphere','muon_sphere',8),
               ('sso_ns','sso_ns',8),('sso_ns_repair_svd','sso_ns_repair_svd',8)],
    'precision': [('ns5','sso_ns',5),('ns8','sso_ns',8),('ns12','sso_ns',12)],
    'ph': [('sso_ns','sso_ns',8),('sso_ph_svd','sso_ph_svd',8)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','select','main','summarize'],required=True)
    p.add_argument('--plan',choices=list(PLANS),default='repair');p.add_argument('--work',required=True)
    p.add_argument('--dataset',default='fineweb_edu',choices=['fineweb_edu','olmo_mix','tinystories'])
    p.add_argument('--device',default='cuda');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--base-config',help='Optional PREDECLARED full-budget config; freeze before pilot.')
    p.add_argument('--pilot-steps',type=int,default=200);a=p.parse_args()
    work=Path(a.work).resolve();data=work/'data'/a.dataset
    if not (data/'manifest.json').exists():raise FileNotFoundError('Prepare the chosen dataset first using run_stage')
    root=work/'studies'/f'{a.dataset}_{a.plan}';root.mkdir(parents=True,exist_ok=True)
    if a.base_config:c=json.loads(Path(a.base_config).read_text())
    else:
        c=json.loads((ROOT/'configs'/f'lm_{a.dataset}.json').read_text())
        c.update(width=128,layers=4,heads=4,steps=1000,accum_steps=8,warmup_steps=100,
                 checkpoint_interval=50,eval_interval=100,eval_batches=32)
        # Keep the SAME forward precision chosen at the exploratory stage, including T4 FP32.
        saved=work/'configs'/f'lm_{a.dataset}.json'
        c['forward_dtype']=json.loads(saved.read_text())['forward_dtype'] if saved.exists() else 'float32'
    c.update(snapshot_steps=[],snapshot_names=[],diagnostic_interval=0)
    if not 0<a.pilot_steps<=c['steps']:raise ValueError('pilot steps must be positive and at most main budget')
    protocol=dict(config=c,pilot_steps=a.pilot_steps,plan=a.plan,
                  manifest_sha256=sha256(data/'manifest.json'),device=a.device,
                  code_sha256={p.name:sha256(p) for p in ROOT.glob('*.py')},
                  lr_adamw=[.0001,.0003,.001],lr_other=[.003,.01,.03],seeds=[11,22,33])
    frozen=root/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=protocol:raise ValueError('Study protocol changed; use a new work directory')
    if not frozen.exists():save_json(frozen,protocol)
    defs=PLANS[a.plan]

    def config_for(label,method,k,phase,lr,seed):
        cc=dict(c);cc['ns_steps']=k
        if phase=='pilot':cc.update(steps=a.pilot_steps,warmup_steps=min(20,max(1,a.pilot_steps//10)),eval_interval=max(1,a.pilot_steps//4),checkpoint_interval=max(1,a.pilot_steps//2))
        cc.update(optimizer=method,seed=seed,matrix_lr=lr)
        return cc

    def read_valid(path,expected,phase):
        r=json.loads((path/'result.json').read_text());ident=json.loads((path/'run_identity.json').read_text())['identity']
        if ident['config']!=expected or ident['manifest_sha256']!=protocol['manifest_sha256'] or ident['device_type']!=a.device.split(':')[0]:
            raise ValueError(f'Run protocol mismatch: {path}')
        if r['completed_steps']!=expected['steps'] or not math.isfinite(r['validation']['loss']):raise ValueError(f'Incomplete/nonfinite run: {path}')
        if phase=='pilot' and 'test' in r:raise ValueError('Pilot test leakage')
        return r

    if a.phase=='select':
        selected={}
        for label,method,k in defs:
            candidates=[]
            for lr in protocol['lr_adamw' if method=='adamw' else 'lr_other']:
                path=root/'pilot'/f'{label}_lr{lr}_seed7'
                r=read_valid(path,config_for(label,method,k,'pilot',lr,7),'pilot')
                candidates.append((r['validation']['loss'],lr,sha256(path/'result.json')))
            loss,lr,digest=min(candidates)
            selected[label]=dict(lr=lr,validation_loss=loss,pilot_result_sha256=digest)
        value=dict(protocol_sha256=sha256(frozen),methods=selected)
        if (root/'selection.json').exists() and json.loads((root/'selection.json').read_text())!=value:raise ValueError('Frozen selection changed')
        save_json(root/'selection.json',value);print(json.dumps(value,indent=2));return
    selected=None
    if a.phase in ['main','summarize']:
        selected=json.loads((root/'selection.json').read_text())
        if selected['protocol_sha256']!=sha256(frozen):raise ValueError('Selection protocol mismatch')
    if a.phase=='summarize':
        per={};raw={}
        for label,method,k in defs:
            values=[];timings=[];rows=[]
            for seed in protocol['seeds']:
                lr=selected['methods'][label]['lr'];path=root/'main'/f'{label}_seed{seed}'
                r=read_valid(path,config_for(label,method,k,'main',lr,seed),'main')
                values.append(r['test']['loss']);timings.append(r['completed_sessions_wall_seconds']);rows.append(r)
            raw[label]=rows
            per[label]=dict(test_losses=values,mean=statistics.mean(values),sample_sd=statistics.stdev(values),
                mean_completed_wall_seconds=statistics.mean(timings),
                max_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rows),
                unfinished_sessions=sum(x['unfinished_session_count'] for x in rows))
        baseline='sso_ns' if 'sso_ns' in per else 'ns8'
        paired={k:[x-y for x,y in zip(v['test_losses'],per[baseline]['test_losses'])] for k,v in per.items() if k!=baseline}
        save_json(root/'summary.json',dict(methods=per,paired_test_loss_differences_vs=baseline,paired_differences=paired,
            seeds=protocol['seeds'],raw_results=raw,
            interpretation='Three paired training seeds; no automatic significance or general superiority claim. Full wall includes SVD repairs.'))
        print(json.dumps(per,indent=2));return
    for label,method,k in defs:
        lrs=protocol['lr_adamw' if method=='adamw' else 'lr_other'] if a.phase=='pilot' else [selected['methods'][label]['lr']]
        for lr in lrs:
            for seed in ([7] if a.phase=='pilot' else protocol['seeds']):
                cc=config_for(label,method,k,a.phase,lr,seed);cfg=root/'configs'/f'{a.phase}_{label}_{lr}_{seed}.json';save_json(cfg,cc)
                path=root/a.phase/(f'{label}_lr{lr}_seed7' if a.phase=='pilot' else f'{label}_seed{seed}')
                cmd=[ROOT/'train.py','--config',cfg,'--data',data,'--optimizer',method,'--seed',seed,'--device',a.device,'--out',path]
                if a.phase=='main':cmd.append('--evaluate-test')
                if (path/'result.json').exists():
                    r=json.loads((path/'result.json').read_text())
                    if r['completed_steps']==cc['steps'] and (a.phase=='pilot' or 'test' in r):
                        read_valid(path,cc,a.phase);print('Already completed:',path);continue
                if (path/'checkpoint.pt').exists():cmd.append('--resume')
                if a.dry_run:print(' '.join(map(str,cmd)))
                else:invoke(work,f'{a.plan}_{a.phase}_{label}_{seed}',cmd)


if __name__=='__main__':main()
