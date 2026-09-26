"""Colab stage driver. Every stage is restartable; checkpoints live under --work.
Stages do not implicitly start the optional training-comparison grid.
"""
import argparse, json, os, subprocess, sys, time, zipfile
from pathlib import Path
from common import save_json, environment, sha256, json_hash

ROOT=Path(__file__).resolve().parent


def invoke(work, label, arguments):
    log=work/'logs'/f'{label}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,*map(str,arguments)]
    print('RUN:', ' '.join(command),flush=True)
    with log.open('w') as f:
        f.write(json.dumps({'command':command})+'\n');f.flush()
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout: print(line,end='',flush=True);f.write(line);f.flush()
        returncode=proc.wait()
    if returncode: raise RuntimeError(f'{label} failed ({returncode}). Keep log: {log}')


def frozen_config(work,dataset,device):
    import torch
    path=work/'configs'/f'lm_{dataset}.json'
    c=json.loads((ROOT/'configs'/f'lm_{dataset}.json').read_text())
    if path.exists():
        existing=json.loads(path.read_text())
        candidate=dict(c);candidate['forward_dtype']=existing['forward_dtype']
        if candidate!=existing: raise ValueError('Source config changed after this experiment was frozen; use a new work path')
        return path
    c['forward_dtype']='bfloat16' if device.startswith('cuda') and torch.cuda.is_bf16_supported() else 'float32'
    save_json(path,c)
    return path


def prepare(work,dataset):
    data=work/'data'/dataset
    if not (data/'manifest.json').exists():
        invoke(work,'prepare_'+dataset,[ROOT/'prepare_data.py','--config',ROOT/'configs'/f'data_{dataset}.json','--out',data,'--cache',work/'hf_cache'])
    save_json(work/'data_manifests'/f'{dataset}.json',json.loads((data/'manifest.json').read_text()))
    return data


def train(work,dataset,method,device,stop=None):
    config=frozen_config(work,dataset,device); data=prepare(work,dataset)
    out=work/'runs'/dataset/f'{method}_seed11'
    cmd=[ROOT/'train.py','--config',config,'--data',data,'--optimizer',method,'--device',device,
         '--seed','11','--out',out]
    if stop:cmd+=['--stop-after',str(stop)]
    if (out/'checkpoint.pt').exists():
        # Skip a completed run (or an already passed intermediate stop) without duplicating evaluations.
        if (out/'result.json').exists():
            r=json.loads((out/'result.json').read_text())
            goal=stop or json.loads(config.read_text())['steps']
            if r['completed_steps']>=goal:
                print(f'Already completed: {out} ({r["completed_steps"]} steps)',flush=True);return
        cmd.append('--resume')
    invoke(work,f'train_{dataset}_{method}',cmd)


def export(work):
    # Small, complete research return bundle: logs, configs, manifests, metrics and FULL snapshots.
    # Large corpus/cache/checkpoints stay in Drive for continuation, not in this return zip.
    target=work/'SSO_RETURN_RESULTS.zip'
    include=[]
    for p in sorted(work.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(work)
        if any(x in ['hf_cache','data','__pycache__'] for x in rel.parts):continue
        if p.suffix in ['.pt','.tmp','.zip','.bin']:continue
        if p.suffix in ['.json','.jsonl','.npz','.csv','.txt','.log','.md','.png']:
            include.append(p)
    index={str(p.relative_to(work)):sha256(p) for p in include}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in include:z.write(p,p.relative_to(work))
        z.writestr('RETURN_MANIFEST_SHA256.json',json.dumps(index,indent=2))
    print('RETURN:',target,'bytes=',target.stat().st_size,flush=True)
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['preflight','tiny','capture','audit','export'])
    p.add_argument('--work',required=True);p.add_argument('--dataset',default='fineweb_edu',choices=['tinystories','fineweb_edu','olmo_mix'])
    p.add_argument('--device',default='cuda');a=p.parse_args();work=Path(a.work).resolve();work.mkdir(parents=True,exist_ok=True)
    if a.stage=='export':export(work);return
    # Refuse to mix source versions even when previously completed jobs are skipped.
    source_current={p.name:sha256(p) for p in sorted(ROOT.glob('*.py'))}
    previous=work/'source_hashes.json'
    if previous.exists() and json.loads(previous.read_text())!=source_current:
        raise ValueError('Code changed in an existing work directory. Preserve old results and choose a new --work.')
    status=work/'stage_status';status.mkdir(exist_ok=True)
    label=a.stage+'_'+a.dataset
    identity={p.name:sha256(p) for p in sorted(ROOT.glob('*.py'))}
    save_json(work/'environment.json',environment())
    save_json(work/'source_hashes.json',identity)
    try:
        if a.stage=='preflight':
            import torch
            if a.device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('Select a GPU runtime in Colab. No silent CPU fallback.')
            invoke(work,'theory_cpu',[ROOT/'theory_path.py','--device','cpu','--out',work/'checks/theory_cpu.json'])
            if a.device!='cpu':invoke(work,'theory_cuda',[ROOT/'theory_path.py','--device',a.device,'--out',work/'checks/theory_cuda.json'])
            invoke(work,'smoke',[ROOT/'test_smoke.py','--device',a.device,'--out',work/'checks/smoke'])
        elif a.stage=='tiny':
            train(work,'tinystories','sso_ns',a.device,stop=15)
            train(work,'tinystories','sso_ns',a.device)
        elif a.stage=='capture':
            for method in ['muon_sphere','sso_ns']:train(work,a.dataset,method,a.device)
        elif a.stage=='audit':
            invoke(work,'audit_'+a.dataset,[ROOT/'audit_snapshots.py','--root',work/'runs'/a.dataset,
                                          '--out',work/'audits'/a.dataset,'--device',a.device])
        save_json(status/(label+'.json'),dict(status='completed',stage=a.stage,dataset=a.dataset,source_hash=json_hash(identity)))
    except Exception as e:
        save_json(status/(label+'.json'),dict(status='failed',stage=a.stage,dataset=a.dataset,error=repr(e),source_hash=json_hash(identity)))
        raise
    finally:export(work)


if __name__=='__main__':main()
