"""Offline tests: real LM forward/backward for every method, exact CPU restart and data integrity.
The fixture is synthetic byte tokens: it is NOT downloaded TinyStories or a GPU result.
"""
import argparse,json,shutil,subprocess,sys,tempfile
from pathlib import Path
import torch
from common import save_json,environment
from prepare_data import write_data
from train import parser,run

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/cpu_smoke');p.add_argument('--device',default='cpu');a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='sso_cpu_',dir=out))
    config={'dataset_label':'OFFLINE SYNTHETIC BYTE FIXTURE (NOT REAL DATA)', 'sources':[{'name':'fixture','weight':1}],
            'train_tokens':4096,'val_tokens':1024,'test_tokens':1024,'min_val_source_tokens':1,'validation_permyriad':2000}
    stream=({'text':f'Unique document {i}. The cat {i%13} reads a story about number {i*i}.','id':i} for i in range(2000))
    manifest=write_data(config,work/'data',[stream],lambda x:list(x.encode()),257,256,{'fixture':True})
    rows=[json.loads(x) for x in (work/'data/fixture_documents.jsonl').read_text().splitlines()]
    bysplit={s:{x['content_sha256'] for x in rows if x['split']==s} for s in ['train','val','test']}
    assert all(not bysplit[a]&bysplit[b] for a,b in [('train','val'),('train','test'),('val','test')])
    c={'width':16,'layers':1,'heads':2,'seq_len':8,'batch_size':2,'accum_steps':2,'steps':4,'warmup_steps':1,
       'matrix_lr':.003,'aux_lr':.001,'matrix_weight_decay':0.,'aux_weight_decay':.1,'momentum':.9,
       'ns_steps':8,'ns_dtype':'float32','forward_dtype':'float32','pi_steps':20,'solver_tol':1e-4,'max_expand':10,
       'grad_clip':1.,'eval_batches':2,'eval_interval':2,'checkpoint_interval':2,'diagnostic_interval':2,
       'print_interval':4,'deterministic':True,'snapshot_steps':[1,2,4],
       'snapshot_names':['blocks.0.attn.q.weight','blocks.0.ff1.weight','blocks.0.ff2.weight']}
    save_json(work/'config.json',c)
    def args(method,name,extra=()):
        return parser().parse_args(['--config',str(work/'config.json'),'--data',str(work/'data'),'--out',str(work/name),
          '--optimizer',method,'--seed','11','--device',a.device,'--threads','1',*extra])
    results={}
    for method in ['adamw','muon','muon_sphere','sso_ns','sso_ns_repair_svd','sso_ph_svd']:
        results[method]=run(args(method,method,['--evaluate-test']))
    run(args('sso_ns','resumed',['--stop-after','2']))
    resumed=run(args('sso_ns','resumed',['--resume','--evaluate-test']))
    full=torch.load(work/'sso_ns/checkpoint.pt',weights_only=False,map_location='cpu')
    split=torch.load(work/'resumed/checkpoint.pt',weights_only=False,map_location='cpu')
    maximum=max(float((full['model'][k]-split['model'][k]).abs().max()) for k in full['model'])
    assert maximum==0.,maximum
    assert full['sampler']==split['sampler']
    assert results['sso_ns']['validation']==resumed['validation']
    completed=run(args('sso_ns','resumed',['--resume','--evaluate-test']))
    assert completed['validation']==resumed['validation']
    backwards_refused=False
    try:run(args('sso_ns','resumed',['--resume','--stop-after','2']))
    except ValueError as e:backwards_refused='smaller' in str(e)
    assert backwards_refused
    dirty=work/'dirty';dirty.mkdir();(dirty/'metrics.jsonl').write_text('stale')
    dirty_refused=False
    try:run(args('sso_ns','dirty'))
    except FileExistsError:dirty_refused=True
    assert dirty_refused
    ids=[json.loads((work/m/'run_identity.json').read_text())['init_parameters_sha256'] for m in results]
    assert len(set(ids))==1
    refused=False
    try:run(args('sso_ns','resumed',['--resume','--lr','.04']))
    except ValueError as e:refused='Resume refused' in str(e)
    assert refused
    data=work/'data/fixture_train.bin';raw=data.read_bytes();data.write_bytes(b'X'+raw[1:]);corruption_refused=False
    try:run(args('sso_ns','resumed',['--resume']))
    except ValueError as e:corruption_refused='checksum' in str(e)
    assert corruption_refused;data.write_bytes(raw)
    import numpy as np
    snapshots=list((work/'sso_ns/snapshots').glob('*.npz'))
    assert len(snapshots)==9
    with np.load(snapshots[0],allow_pickle=False) as snap:
        meta=json.loads(str(snap['metadata']))
        assert np.allclose(snap['W_after'],snap['W']-meta['matrix_lr']*meta['radius']*snap['Z'],atol=1e-6)
    summary={'environment':environment(),'gpu_executed':a.device.startswith('cuda'),'six_optimizers_real_lm_steps':True,
             'resume_parameter_max_abs_difference':maximum,'identical_initialization_all_optimizers':True,
             'full_matrix_snapshot_actual_update_verified':True,
             'split_document_hashes_disjoint':True,'changed_config_resume_refused':refused,'completed_resume_preserves_validation':True,
             'dirty_output_directory_refused':dirty_refused,'backward_resume_stop_refused':backwards_refused,
             'corrupt_data_refused':corruption_refused,'run_directory':str(work),
             'validation_loss':{k:v['validation']['loss'] for k,v in results.items()},
             'scope':'Engineering smoke test on synthetic bytes; not evidence of optimizer superiority or real-corpus/GPU replication.'}
    # Exercise the reporting CLI on only the five distinct full runs.
    for method in results:
        d=work/'summary_input'/method;d.mkdir(parents=True)
        for f in ['result.json','run_identity.json','metrics.jsonl','diagnostics.jsonl']:
            if (work/method/f).exists():shutil.copy2(work/method/f,d/f)
    subprocess.run([sys.executable,str(ROOT/'summarize.py'),'--root',str(work/'summary_input'),'--out',str(out/'summary.json')],check=True)
    summary['summary_cli_passed']=True
    save_json(out/'smoke_result.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
