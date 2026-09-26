"""Aggregate completed runs without mixing datasets, hyperparameters, budgets or device types."""
import argparse,json,math,statistics
from pathlib import Path
from common import save_json,json_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    groups={};incomplete=[]
    for f in sorted(Path(a.root).rglob('result.json')):
        r=json.loads(f.read_text());identity=json.loads((f.parent/'run_identity.json').read_text())
        if r['completed_steps']!=r['planned_steps']:incomplete.append(str(f));continue
        config=dict(identity['identity']['config']);config.pop('seed',None)
        protocol={'config':config,'manifest':identity['identity']['manifest_sha256'],'device':identity['identity']['device_type']}
        key=json_hash(protocol);g=groups.setdefault(key,{'protocol':protocol,'runs':[]})
        metrics=[json.loads(l) for l in (f.parent/'metrics.jsonl').read_text().splitlines()]
        perf=[m for m in metrics if not m['diagnostic_step'] and m['step']>config['warmup_steps']]
        d=[json.loads(l) for l in (f.parent/'diagnostics.jsonl').read_text().splitlines()] if (f.parent/'diagnostics.jsonl').exists() else []
        failures=sum(sum(v for k,v in m['solver_status_counts'].items() if 'failed' in k) for m in metrics)
        perstep=config['batch_size']*config['seq_len']*config['accum_steps']
        g['runs'].append({'path':str(f.parent),'seed':r['seed'],'validation_loss':r['validation']['loss'],
          'test_loss':r.get('test',{}).get('loss'),'steady_tokens_per_second':sum(perstep for m in perf)/sum(m['step_seconds'] for m in perf) if perf else None,
          'mean_step_seconds':statistics.mean(m['step_seconds'] for m in perf) if perf else None,
          'peak_allocated_bytes':r['peak_allocated_bytes'],'solver_failure_count':failures,
          'completed_sessions_wall_seconds':r.get('completed_sessions_wall_seconds'),
          'unfinished_session_count':r.get('unfinished_session_count'),
          'actual_tangency_max':max((x['actual_tangency_exact_normal'] for x in d),default=None),
          'mean_radius_relative_error':statistics.mean(x['radius_relative_error_after'] for x in d) if d else None})
    for g in groups.values():
        if len({r['seed'] for r in g['runs']})!=len(g['runs']):raise ValueError('Duplicate seed within identical protocol; remove duplicate run directories')
        v=[r['validation_loss'] for r in g['runs']];g['validation_mean']=statistics.mean(v);g['validation_sample_sd']=statistics.stdev(v) if len(v)>1 else None
        g['seed_count']=len(v);g['statistical_note']='Three seeds describe variability; no automatic claim of significance or universality.'
    save_json(a.out,{'groups':list(groups.values()),'incomplete_runs_excluded':incomplete,
        'comparison_rule':'Compare paired seeds only when initialization, data, model, total tokens and hyperparameter-selection budgets match; throughput excludes diagnostic steps and warmup.'})
    print(json.dumps({'groups':len(groups),'incomplete':len(incomplete),'output':a.out}))
if __name__=='__main__':main()
