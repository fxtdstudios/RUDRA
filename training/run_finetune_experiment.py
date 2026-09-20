"""Run three sequential image/gate seeds with durable status and fail-fast logs.

No checkpoint is promoted to the shipped registry. Keep this experiment's
frozen test set untouched until candidates have been selected on validation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--image-steps', type=int, default=2000)
    p.add_argument('--gate-steps', type=int, default=3000)
    a = p.parse_args()
    run = a.run_dir.resolve()
    manifest = run/'data'/'manifest.jsonl'
    provenance = json.loads((run/'data'/'provenance.json').read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != provenance['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    status_file = run/'status.json'
    if status_file.exists():
        raise ValueError('Existing experiment: inspect status and resume explicitly; never overwrite')
    status = dict(pid=os.getpid(), state='running', completed=[], manifest_sha256=provenance['manifest_sha256'],
                  license='Non-commercial v5 derivative', commands=[])
    tracked = [ROOT/'rudra/sdr2hdr.py', *sorted((ROOT/'training').glob('*.py')),
               ROOT/'checkpoints/sdr2hdr_image_v5.pt']
    status['source_sha256'] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in tracked}
    def save():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        temp = status_file.with_suffix('.tmp')
        temp.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temp.replace(status_file)
    def execute(name, args):
        status['stage'] = name
        command = [sys.executable, '-u', *map(str,args)]
        status['commands'].append(command)
        save()
        with (run/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status['completed'].append(name)
        save()
    try:
        for seed in (20260920, 2, 3):
            image = run/f'image_seed{seed}'
            gate = run/f'gate_seed{seed}'
            if image.exists() or gate.exists():
                raise ValueError('Checkpoint directory already exists')
            execute(f'image_seed{seed}', ['training/train_sdr2hdr.py','--mode','image',
                '--manifest',manifest,'--init-checkpoint',ROOT/'checkpoints/sdr2hdr_image_v5.pt',
                '--output-dir',image,'--steps',a.image_steps,'--lr','2e-5',
                '--batch-size','4','--crop-size','256','--workers','2','--seed',seed,
                '--best-metric','preserved_composite_gain','--best-smoothing','5',
                '--eval-every','200','--eval-batches','32','--save-every','1000',
                '--log-every','20','--device','cuda'])
            execute(f'gate_seed{seed}', ['training/train_shadow_gate.py',
                '--manifest',manifest,'--init-checkpoint',image/'best.pt','--output-dir',gate,
                '--steps',a.gate_steps,'--seed',seed,'--eval-seed','20260920',
                '--eval-every','200','--eval-batches','128','--workers','2','--device','cuda'])
        execute('assessment', ['training/assess_finetune.py','--run-dir',run])
        status['state'] = 'complete_review_assessment'
        status['stage'] = 'complete'
        save()
    except BaseException as exc:
        status['state'] = 'failed'
        status['error'] = repr(exc)
        save()
        raise


if __name__ == '__main__':
    main()
