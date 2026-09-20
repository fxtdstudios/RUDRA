"""Run full scene-balanced validation and highlights-only comparison sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rudra.batch import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args()
    a.out=a.out.resolve(); a.out.mkdir(parents=True,exist_ok=False)
    commands=[('baseline', [sys.executable,'-u','training/quality_benchmark.py',
        '--manifest',str(a.manifest.resolve()),'--checkpoint',str(a.checkpoint.resolve()),
        '--out',str(a.out/'baseline'),'--scenes','100000']),
        ('highlights', [sys.executable,'-u','training/recovery_ablation.py',
        '--benchmark',str(a.out/'baseline'),'--out',str(a.out/'highlights'),'--modes','highlights'])]
    status=dict(pid=os.getpid(),state='running',completed=[],test_set_used=False)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    try:
        for stage,command in commands:
            status.update(stage=stage,command=command); save(a.out/'status.json',status)
            with (a.out/f'{stage}.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            status['completed'].append(stage)
        status['state']='complete'
    except Exception as exc:
        status.update(state='failed',error=str(exc))
        raise
    finally:
        save(a.out/'status.json',status)


if __name__=='__main__': main()
