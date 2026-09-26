#!/usr/bin/env python3
"""Run deterministic verification; retain environment, script hashes and logs."""
if not __debug__:
    raise RuntimeError('Verification requires assertions: rerun without -O / PYTHONOPTIMIZE.')
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

SCRIPTS=(
    ('verify_rate_constants_exact.py',),
    ('verify_unified_crossover.py',),
    ('verify_partial_isometry.py',),
    ('verify_direction_rates.py',),
    ('verify_theorems.py',),
    ('verify_corank2_strict_center.py',),
    ('verify_arbitrary_corank_strict_law.py',),
    ('verify_direction_rate_general_corank.py',),
    ('verify_intrinsic_constants.py',),
    ('verify_smoothing_family.py',),
    ('verify_publication_tables.py','--check'),
    ('verify_certificate_example.py',),
    ('verify_finite_cancellation.py',),
    ('verify_boundary_noncancellation.py',),
)

def main():
    root=Path(__file__).resolve().parent
    out=root/'results';logs=out/'verification_logs';logs.mkdir(parents=True,exist_ok=True)
    report={
        'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','scipy','mpmath')},
        'optimize_flag':sys.flags.optimize,
        'thread_environment':{name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'scope':'CPU mathematical regression checks; no GPU training and no interval certification.',
        'scripts':[],
    }
    for command in SCRIPTS:
        name=command[0];print(f'\n=== {name} ===',flush=True);start=time.monotonic()
        run=subprocess.run([sys.executable,str(root/name),*command[1:]],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(run.stdout,end='',flush=True);log=logs/(Path(name).stem+'.log');log.write_text(run.stdout)
        row={'script':name,'arguments':list(command[1:]),'exit_code':run.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'source_sha256':hashlib.sha256((root/name).read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(run.stdout.encode()).hexdigest(),'log':str(log.relative_to(root))}
        report['scripts'].append(row);report['all_passed']=all(x['exit_code']==0 for x in report['scripts']) and len(report['scripts'])==len(SCRIPTS)
        (out/'verification_latest.json').write_text(json.dumps(report,indent=2)+'\n')
        if run.returncode:raise SystemExit(f'FAILED: {name}; see {log}')
    print(f'\nALL {len(SCRIPTS)} VERIFICATION SCRIPTS PASSED.')
if __name__=='__main__':main()
