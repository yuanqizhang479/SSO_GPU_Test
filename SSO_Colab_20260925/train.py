"""Single-GPU/CPU decoder-only LM study; paired seeds, exact checkpoint state, explicit timings."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, contextlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import sha256,json_hash,save_json,seed_all,sync,environment,rng_state,set_rng_state
from optimizers import MatrixOptimizer,power_normal

class Attention(nn.Module):
    def __init__(self,width,heads):
        super().__init__(); self.heads=heads
        self.q=nn.Linear(width,width,bias=False); self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False); self.out=nn.Linear(width,width,bias=False)
    def forward(self,x):
        b,t,c=x.shape
        q,k,v=[layer(x).view(b,t,self.heads,c//self.heads).transpose(1,2) for layer in [self.q,self.k,self.v]]
        # Small controlled experiments prioritize reproducible resume across GPU models.
        # Avoid depending on a device-specific fused attention backward implementation.
        if torch.are_deterministic_algorithms_enabled():
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                x=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        else:
            x=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(x.transpose(1,2).reshape(b,t,c))

class Block(nn.Module):
    def __init__(self,width,heads):
        super().__init__(); self.norm1=nn.LayerNorm(width); self.norm2=nn.LayerNorm(width)
        self.attn=Attention(width,heads)
        self.ff1=nn.Linear(width,4*width,bias=False); self.ff2=nn.Linear(4*width,width,bias=False)
    def forward(self,x):
        x=x+self.attn(self.norm1(x)); return x+self.ff2(F.gelu(self.ff1(self.norm2(x))))

class Decoder(nn.Module):
    def __init__(self,vocab,width,layers,heads,seq_len):
        super().__init__(); self.token=nn.Embedding(vocab,width); self.position=nn.Embedding(seq_len,width)
        self.blocks=nn.ModuleList([Block(width,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(width); self.head=nn.Linear(width,vocab,bias=False)
        self.head.weight=self.token.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,std=.02)
    def forward(self,x):
        h=self.token(x)+self.position(torch.arange(x.shape[1],device=x.device))
        for b in self.blocks: h=b(h)
        return self.head(self.norm(h))
    def hidden_matrices(self):
        return [(n,p) for n,p in self.named_parameters() if n.startswith('blocks.') and p.ndim==2]

class TokenSampler:
    def __init__(self,directory,manifest,split,seq_len,seed):
        self.arrays=[np.memmap(Path(directory)/(s['name']+'_'+split+'.bin'),mode='r',dtype='<u4') for s in manifest['sources']]
        if any(len(a)<=seq_len for a in self.arrays): raise ValueError('Each source/split needs at least seq_len+1 tokens')
        self.weights=np.array([s['weight'] for s in manifest['sources']]); self.weights/=self.weights.sum()
        self.rng=np.random.default_rng(seed); self.seq_len=seq_len; self.draws=0; self.source_draws=[0]*len(self.arrays)
    def batch(self,n,device):
        data=[]
        for _ in range(n):
            i=int(self.rng.choice(len(self.arrays),p=self.weights)); a=self.arrays[i]
            start=int(self.rng.integers(0,len(a)-self.seq_len))
            data.append(np.array(a[start:start+self.seq_len+1],dtype=np.int64)); self.draws+=1; self.source_draws[i]+=1
        batch=torch.from_numpy(np.stack(data)).to(device)
        return batch[:,:-1],batch[:,1:]
    def state_dict(self): return {'rng':self.rng.bit_generator.state,'draws':self.draws,'source_draws':self.source_draws}
    def load_state_dict(self,state):
        self.rng.bit_generator.state=state['rng']; self.draws=state['draws']; self.source_draws=state['source_draws']

def amp_context(device,dtype):
    return torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16) if dtype=='bfloat16' else contextlib.nullcontext()

@torch.no_grad()
def evaluate(model,data,manifest,c,device,split):
    model.eval(); sampler=TokenSampler(data,manifest,split,c['seq_len'],812733 if split=='val' else 982113)
    loss=0.; tokens=0
    for _ in range(c['eval_batches']):
        x,y=sampler.batch(c['batch_size'],device)
        with amp_context(device,c['forward_dtype']): out=model(x)
        loss+=float(F.cross_entropy(out.float().flatten(0,1),y.flatten(),reduction='sum')); tokens+=y.numel()
    model.train(); value=loss/tokens
    return {'loss':value,'perplexity':math.exp(value) if value<700 else None,'tokens':tokens,
            'sampling':'fixed independent windows with replacement; identical across all runs', 'source_sequence_counts':sampler.source_draws}

def run(args):
    c=json.loads(Path(args.config).read_text()); c.update({'optimizer':args.optimizer,'seed':args.seed})
    if args.lr is not None: c['matrix_lr']=args.lr
    if args.aux_lr is not None: c['aux_lr']=args.aux_lr
    if args.ns_dtype: c['ns_dtype']=args.ns_dtype
    if args.forward_dtype: c['forward_dtype']=args.forward_dtype
    if args.ns_steps is not None: c['ns_steps']=args.ns_steps
    if args.solver_tol is not None: c['solver_tol']=args.solver_tol
    if args.ph_delta is not None: c['ph_delta']=args.ph_delta
    if args.threads: torch.set_num_threads(args.threads)
    for key in ['width','layers','heads','seq_len','batch_size','accum_steps','steps','pi_steps','ns_steps','eval_batches','eval_interval','checkpoint_interval']:
        if c[key]<=0: raise ValueError(f'{key} must be positive')
    if c['width']%c['heads']: raise ValueError('width must be divisible by heads')
    if args.stop_after is not None and args.stop_after<=0: raise ValueError('stop-after must be positive')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    if c['forward_dtype']=='bfloat16' and device.type=='cuda' and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unsupported')
    seed_all(args.seed); torch.use_deterministic_algorithms(c.get('deterministic',True))
    manifest_path=Path(args.data)/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    for name,info in manifest['files'].items():
        if sha256(Path(args.data)/name)!=info['sha256']: raise ValueError(f'Data checksum mismatch: {name}')
    identity={'config':c,'manifest_sha256':sha256(manifest_path),'device_type':device.type,
              'runtime':{'torch':torch.__version__,'numpy':np.__version__,'cuda_runtime':torch.version.cuda},
              'code_sha256':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    identity_hash=json_hash(identity); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()) and not args.resume: raise FileExistsError('Nonempty output directory: use --resume or a fresh path')
    if args.resume and not (out/'checkpoint.pt').exists(): raise FileNotFoundError('Resume requires checkpoint.pt')
    model=Decoder(manifest['vocab_size'],c['width'],c['layers'],c['heads'],c['seq_len']).to(device)
    matrix=model.hidden_matrices(); mids={id(p) for n,p in matrix}
    # Identical initialization for every optimizer; use the same PI-normalized hidden matrices.
    with torch.no_grad():
        for n,p in matrix:
            sigma,_=power_normal(p,c['pi_steps']); p.mul_(math.sqrt(p.shape[0]/p.shape[1])/sigma)
    init_hash=__import__('hashlib').sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    aux=[p for p in model.parameters() if id(p) not in mids]
    opt_aux=torch.optim.AdamW(aux,lr=c['aux_lr'],betas=(.9,.95),weight_decay=c['aux_weight_decay'])
    if args.optimizer=='adamw': opt_matrix=torch.optim.AdamW([p for n,p in matrix],lr=c['matrix_lr'],betas=(.9,.95),weight_decay=c['matrix_weight_decay'])
    else: opt_matrix=MatrixOptimizer(matrix,args.optimizer,lr=c['matrix_lr'],momentum=c['momentum'],ns_dtype=c['ns_dtype'],
             ns_steps=c['ns_steps'],pi_steps=c['pi_steps'],solver_tol=c['solver_tol'],max_expand=c['max_expand'],weight_decay=c['matrix_weight_decay'],ph_delta=c.get('ph_delta',1e-3))
    sampler=TokenSampler(args.data,manifest,'train',c['seq_len'],args.seed+70341)
    start=0; elapsed_train=0.; elapsed_diagnostics=0.; elapsed_eval=0.; elapsed_checkpoint=0.; elapsed_other=0.
    peak_memory_prior=0; full_start=time.perf_counter(); val=None
    if args.resume:
        # Load only this suite's trusted checkpoint. torch.load uses pickle for RNG state.
        ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if ck['identity_hash']!=identity_hash: raise ValueError('Resume refused: config, data, code, seed, optimizer or device type changed')
        model.load_state_dict(ck['model']); opt_matrix.load_state_dict(ck['matrix_optimizer']); opt_aux.load_state_dict(ck['aux_optimizer'])
        sampler.load_state_dict(ck['sampler']); set_rng_state(ck['rng']); start=ck['step']
        elapsed_train=ck['elapsed_train']; elapsed_diagnostics=ck['elapsed_diagnostics']; elapsed_eval=ck['elapsed_eval']
        elapsed_checkpoint=ck.get('elapsed_checkpoint',0.); elapsed_other=ck.get('elapsed_other',0.); peak_memory_prior=ck['peak_memory']
        # Drop log rows newer than the last durable checkpoint after a crash.
        for filename in ['metrics.jsonl','diagnostics.jsonl']:
            p=out/filename
            if p.exists(): p.write_text(''.join(line for line in p.read_text().splitlines(True) if json.loads(line)['step']<=start))
        for p in (out/'snapshots').glob('step*.npz'):
            if int(p.name.split('_')[0][4:])>start: p.unlink()
    stop=min(c['steps'],args.stop_after if args.stop_after else c['steps'])
    if stop<start: raise ValueError('stop-after is smaller than the resumed checkpoint step')
    session_id=__import__('uuid').uuid4().hex
    with open(out/'sessions.jsonl','a') as f: f.write(json.dumps({'session_id':session_id,'event':'start','unix_time':time.time(),'start_step':start})+'\n')
    save_json(out/'run_identity.json',{'identity':identity,'identity_hash':identity_hash,'environment':environment(),
        'init_parameters_sha256':init_hash,'parameter_count':sum(p.numel() for p in model.parameters()),
        'not_claimed':'Not the Xie 100B-token/1.8B-model experiment; independent implementation and architecture.'})
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    for step in range(start,stop):
        sync(device); tick=time.perf_counter()
        factor=min(1.,(step+1)/max(1,c['warmup_steps']))*(.1+.9*.5*(1+math.cos(math.pi*step/c['steps'])))
        for g in opt_matrix.param_groups: g['lr']=c['matrix_lr']*factor
        for g in opt_aux.param_groups: g['lr']=c['aux_lr']*factor
        opt_matrix.zero_grad(set_to_none=True); opt_aux.zero_grad(set_to_none=True); train_loss=0.
        for _ in range(c['accum_steps']):
            x,y=sampler.batch(c['batch_size'],device)
            with amp_context(device,c['forward_dtype']): logits=model(x)
            loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten())/c['accum_steps']
            loss.backward(); train_loss+=float(loss.detach())
        grad_norm=float(nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip']))
        if not math.isfinite(train_loss+grad_norm): raise FloatingPointError('Nonfinite loss/gradient; no silent skipped step')
        do_diag=c['diagnostic_interval']>0 and (step+1)%c['diagnostic_interval']==0 and args.optimizer!='adamw'
        # Diagnostic calls are excluded by measuring their explicit time inside the optimizer below.
        if isinstance(opt_matrix,MatrixOptimizer):
            capture=(step+1) in c.get('snapshot_steps',[])
            def sink(name,arrays,metadata):
                folder=out/'snapshots'; folder.mkdir(exist_ok=True)
                target=folder/f'step{step+1:06d}_{name}.npz'
                meta=dict(step=step+1,parameter=name,optimizer=args.optimizer,seed=args.seed,
                    identity_hash=identity_hash,manifest_sha256=identity['manifest_sha256'],
                    dataset_label=manifest['dataset_label'],normal_convention='PI at pre-update retracted W',
                    gradient_convention='normalized Nesterov EMA of clipped gradient, full matrix',**metadata)
                with open(target.with_suffix('.tmp'),'wb') as f:
                    np.savez_compressed(f,**{k:v.detach().cpu().numpy() for k,v in arrays.items()},metadata=json.dumps(meta))
                os.replace(target.with_suffix('.tmp'),target)
            opt_matrix.snapshot_sink=sink if capture else None
            opt_matrix.snapshot_names=set(c.get('snapshot_names',[]))
            opt_matrix.step(diagnostic=do_diag)
        else: opt_matrix.step()
        opt_aux.step(); sync(device); duration=time.perf_counter()-tick
        # On diagnostic steps the entire step is flagged; performance summary excludes these steps.
        if do_diag: elapsed_diagnostics+=duration
        else: elapsed_train+=duration
        tokens_per_step=c['batch_size']*c['accum_steps']*c['seq_len']
        row={'step':step+1,'train_loss':train_loss,'lr':c['matrix_lr']*factor,'grad_norm':grad_norm,
             'tokens_seen':(step+1)*tokens_per_step,'step_seconds':duration,'diagnostic_step':do_diag,
             'snapshot_step':(step+1) in c.get('snapshot_steps',[]),
             'elapsed_training_without_diagnostic_steps':elapsed_train,'elapsed_diagnostic_steps':elapsed_diagnostics,
             'solver_status_counts':opt_matrix.solver_counts if isinstance(opt_matrix,MatrixOptimizer) else {}}
        if do_diag:
            with open(out/'diagnostics.jsonl','a') as f:
                for d in opt_matrix.diagnostic_rows: f.write(json.dumps({'step':step+1,**d})+'\n')
        if (step+1)%c['eval_interval']==0 or step+1==stop:
            sync(device); et=time.perf_counter(); val=evaluate(model,args.data,manifest,c,device,'val'); sync(device)
            elapsed_eval+=time.perf_counter()-et; row['validation']=val
        with open(out/'metrics.jsonl','a') as f: f.write(json.dumps(row)+'\n')
        if (step+1)%c['checkpoint_interval']==0 or step+1==stop:
            peak=max(peak_memory_prior,torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
            ct=time.perf_counter()
            ck={'identity_hash':identity_hash,'model':model.state_dict(),'matrix_optimizer':opt_matrix.state_dict(),
                'aux_optimizer':opt_aux.state_dict(),'sampler':sampler.state_dict(),'rng':rng_state(),'step':step+1,
                'elapsed_train':elapsed_train,'elapsed_diagnostics':elapsed_diagnostics,'elapsed_eval':elapsed_eval,
                'elapsed_checkpoint':elapsed_checkpoint,'elapsed_other':elapsed_other,'peak_memory':peak}
            torch.save(ck,out/'checkpoint.tmp'); os.replace(out/'checkpoint.tmp',out/'checkpoint.pt')
            elapsed_checkpoint+=time.perf_counter()-ct
        if (step+1)%c.get('print_interval',20)==0 or step+1==stop: print(json.dumps(row),flush=True)
    if val is None:
        et=time.perf_counter();val=evaluate(model,args.data,manifest,c,device,'val');sync(device);elapsed_eval+=time.perf_counter()-et
    result={'completed_steps':stop,'planned_steps':c['steps'],'validation':val,'optimizer':args.optimizer,'seed':args.seed,
         'dataset':manifest['dataset_label'],'identity_hash':identity_hash,'matrix_lr':c['matrix_lr'],'aux_lr':c['aux_lr'],
         'train_tokens':stop*c['batch_size']*c['accum_steps']*c['seq_len'],'elapsed_training_without_diagnostic_steps':elapsed_train,
         'elapsed_diagnostic_steps':elapsed_diagnostics,'elapsed_evaluation':elapsed_eval,'elapsed_checkpoint':elapsed_checkpoint,
         'invocation_wall_seconds':time.perf_counter()-full_start,'peak_allocated_bytes':max(peak_memory_prior,torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0),
         'data_reuse':'Random windows sampled with replacement; train_tokens includes repeated exposures.',
         'gpu_executed':device.type=='cuda'}
    if args.evaluate_test:
        if stop!=c['steps']: raise ValueError('Test evaluation allowed only at the predeclared full budget')
        tt=time.perf_counter();result['test']=evaluate(model,args.data,manifest,c,device,'test');sync(device)
        result['test_evaluation_seconds']=time.perf_counter()-tt
    result['invocation_wall_seconds']=time.perf_counter()-full_start
    with open(out/'sessions.jsonl','a') as f:
        f.write(json.dumps({'session_id':session_id,'event':'complete','unix_time':time.time(),'end_step':stop,
          'wall_seconds_through_test':result['invocation_wall_seconds']})+'\n')
    events=[json.loads(x) for x in (out/'sessions.jsonl').read_text().splitlines()]
    ended={x['session_id'] for x in events if x['event']=='complete'}
    result['completed_sessions_wall_seconds']=sum(x['wall_seconds_through_test'] for x in events if x['event']=='complete')
    result['unfinished_session_count']=sum(x['event']=='start' and x['session_id'] not in ended for x in events)
    result['timing_note']='Session wall includes training, diagnostics, evaluation and test through final bookkeeping; startup before model construction and abrupt unclosed sessions are not recovered. Detailed checkpoint counter omits last write on resumed checkpoint; use completed_sessions_wall_seconds for cumulative wall.'
    save_json(out/'result.json',result)
    return result

def parser():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--optimizer',choices=['adamw','muon','muon_sphere','sso_ns','sso_ns_repair_svd','sso_ph_svd'],required=True)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--device',default='cuda');p.add_argument('--lr',type=float);p.add_argument('--aux-lr',type=float)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--ns-dtype',choices=['float32','bfloat16']);p.add_argument('--forward-dtype',choices=['float32','bfloat16']);p.add_argument('--threads',type=int)
    p.add_argument('--ns-steps',type=int);p.add_argument('--solver-tol',type=float);p.add_argument('--ph-delta',type=float)
    return p
if __name__=='__main__': run(parser().parse_args())
