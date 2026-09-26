"""Equal-count LR pilot (seed 7), freeze selection, then paired held-out seeds 11/22/33.
Use --shard 0/1 --num-shards 2 on separate GPUs; no DDP or shared run directories.
"""
import argparse,json,subprocess,sys
from pathlib import Path
from common import save_json,sha256

METHODS=['adamw','muon','muon_sphere','sso_ns']
LR={'adamw':[.0001,.0003,.001],'muon':[.003,.01,.03],
    'muon_sphere':[.003,.01,.03],'sso_ns':[.003,.01,.03]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','select','main'],required=True)
    p.add_argument('--dataset',choices=['tinystories','fineweb_edu','olmo_mix'],required=True);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--shard',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not 0<=a.shard<a.num_shards:raise ValueError('Invalid shard')
    root=Path(__file__).resolve().parent;out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    c=json.loads((root/'configs'/f'lm_{a.dataset}.json').read_text());data=str(Path(a.data).resolve())
    if a.phase=='pilot':
        c['steps']=200;c['warmup_steps']=20;c['diagnostic_interval']=0;c['eval_interval']=50;c['checkpoint_interval']=100
        config=out/'pilot_config.json';save_json(config,c)
        jobs=[(m,lr,7,out/'pilot'/f'{m}_lr{lr}_seed7') for m in METHODS for lr in LR[m]]
    elif a.phase=='select':
        selection={}
        for m in METHODS:
            candidates=[]
            for lr in LR[m]:
                path=out/'pilot'/f'{m}_lr{lr}_seed7'/'result.json'
                if not path.exists():raise FileNotFoundError(f'Complete all pilot jobs before selection: {path}')
                r=json.loads(path.read_text())
                identity=json.loads((path.parent/'run_identity.json').read_text())['identity']
                expected=dict(c);expected.update(steps=200,warmup_steps=20,diagnostic_interval=0,eval_interval=50,checkpoint_interval=100,optimizer=m,seed=7,matrix_lr=lr)
                if identity['manifest_sha256']!=sha256(Path(data)/'manifest.json') or identity['config']!=expected:
                    raise ValueError(f'Pilot configuration or data mismatch: {path}')
                if r['completed_steps']!=200 or 'test' in r:raise ValueError('Invalid pilot budget or test leakage')
                candidates.append((r['validation']['loss'],lr,str(path),sha256(path)))
            loss,lr,path,digest=min(candidates)
            selection[m]={'matrix_lr':lr,'pilot_validation_loss':loss,'pilot_result':path,'pilot_sha256':digest}
        save_json(out/'selection.json',{'dataset':a.dataset,'data_manifest_sha256':sha256(Path(data)/'manifest.json'),
          'rule':'Lowest pilot validation loss; three LR candidates each, seed 7, 200 steps; test never opened.',
          'aux_lr':c['aux_lr'],'base_lm_config':c,'methods':selection});print(json.dumps(selection,indent=2));return
    else:
        selected=json.loads((out/'selection.json').read_text())
        if selected['dataset']!=a.dataset or selected['data_manifest_sha256']!=sha256(Path(data)/'manifest.json'):raise ValueError('Selection belongs to different dataset')
        if selected['base_lm_config']!=c:raise ValueError('Model/training config changed after pilot selection; repeat the declared protocol')
        config=out/'main_config.json';save_json(config,c)
        jobs=[(m,selected['methods'][m]['matrix_lr'],seed,out/'main'/f'{m}_seed{seed}') for m in METHODS for seed in [11,22,33]]
    for i,(method,lr,seed,run_dir) in enumerate(jobs):
        if i%a.num_shards!=a.shard:continue
        cmd=[sys.executable,str(root/'train.py'),'--config',str(config),'--data',data,'--optimizer',method,
             '--lr',str(lr),'--seed',str(seed),'--device',a.device,'--out',str(run_dir)]
        if a.phase=='main':cmd.append('--evaluate-test')
        if (run_dir/'checkpoint.pt').exists():cmd.append('--resume')
        print(' '.join(cmd),flush=True)
        if not a.dry_run:subprocess.run(cmd,check=True,cwd=root)
if __name__=='__main__':main()
