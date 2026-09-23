"""Sequential video queues with durable job-level restart and artifact verification."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import video


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    fd, name = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


@contextmanager
def locked(path):
    # Keep the lock file: unlinking it creates an inode race on POSIX.
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This queue is already running') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt': msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream, fcntl.LOCK_UN)


def load_jobs(path):
    spec = json.loads(path.read_text(encoding='utf-8'))
    if spec.get('version') != 1 or not spec.get('jobs'):
        raise ValueError('Queue requires version 1 and a nonempty jobs list')
    parser = argparse.ArgumentParser(exit_on_error=False)
    video.add_arguments(parser)
    jobs = []
    for job in spec['jobs']:
        options = dict(spec.get('defaults', {}))
        options.update(job.get('options', {}))
        if 'input' in options or 'output' in options:
            raise ValueError('Set input/output on the job, not in options')
        tokens = [job['input'], '--output', job['output']]
        for key, value in options.items():
            tokens.extend(['--' + key.replace('_', '-'), str(value)])
        try:
            args = parser.parse_args(tokens)
        except (SystemExit, argparse.ArgumentError) as exc:
            raise ValueError('Invalid video options in queue') from exc
        for key in ('input', 'output', 'checkpoint', 'work_dir'):
            value = getattr(args, key)
            if value is not None:
                setattr(args, key, (path.parent / value).resolve())
        jobs.append(args)
    reserved = {path, Path(str(path)+'.state.json'), Path(str(path)+'.lock')}
    sources = {p for a in jobs for p in (a.input, a.checkpoint)}
    targets = set()
    for a in jobs:
        for p in (a.output, a.output.with_suffix(a.output.suffix+'.json')):
            if p in reserved | sources | targets:
                raise ValueError(f'Queue path collision: {p}')
            targets.add(p)
    return jobs


def run_queue(path, retry_failed=False):
    path = Path(path).resolve()
    state_path = Path(str(path)+'.state.json')
    with locked(Path(str(path)+'.lock')):
        jobs = load_jobs(path)
        signature = digest(path)
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state['queue_sha256'] != signature:
                raise ValueError('Queue changed; use a new queue filename for changed jobs')
            if len(state['jobs']) != len(jobs):
                raise ValueError('Queue state has an invalid job count')
        else:
            state = dict(version=1, queue_sha256=signature,
                         jobs=[dict(status='pending') for _ in jobs])
        save(state_path, state)
        for index, (args, record) in enumerate(zip(jobs, state['jobs'])):
            if record['status'] == 'failed' and not retry_failed:
                continue
            output = args.output
            sidecar = output.with_suffix(output.suffix+'.json')
            try:
                identity = dict(source=digest(args.input), checkpoint=digest(args.checkpoint))
                if record.get('identity', identity) != identity:
                    raise ValueError('Source or checkpoint changed since this job started')
                record['identity'] = identity
                if record['status'] == 'complete':
                    if record['artifacts'] != [digest(output), digest(sidecar)]:
                        raise ValueError('Completed export changed; refusing to skip or overwrite it')
                    print(f'Job {index+1}/{len(jobs)}: verified, skipped', flush=True)
                    continue
                if output.exists() or sidecar.exists():
                    raise ValueError('Existing output needs review; refusing overwrite (including interrupted publication)')
                record.update(status='running', progress=dict(phase='starting'))
                record.pop('error', None)
                save(state_path, state)
                print(f'Job {index+1}/{len(jobs)}: {args.input.name}', flush=True)

                def progress(value):
                    record['progress'] = value
                    save(state_path, state)

                video.convert_video(args, progress=progress)
                report = json.loads(sidecar.read_text(encoding='utf-8'))
                if report.get('qc', {}).get('passed') is not True:
                    raise RuntimeError('Export report does not confirm passing QC')
                if identity != dict(source=digest(args.input), checkpoint=digest(args.checkpoint)):
                    raise RuntimeError('Source or checkpoint changed during export')
                record.update(status='complete', artifacts=[digest(output), digest(sidecar)],
                              progress=dict(phase='complete'))
            except KeyboardInterrupt:
                record.update(status='interrupted', error='Interrupted; restart this clip on next run')
                save(state_path, state)
                raise
            except Exception as exc:
                record.update(status='failed', error=str(exc))
                print(f'Job {index+1} failed: {exc}', flush=True)
            save(state_path, state)
        return 0 if all(j['status']=='complete' for j in state['jobs']) else 1


def add_arguments(parser):
    commands = parser.add_subparsers(dest='batch_command', required=True)
    run = commands.add_parser('run', help='Run or resume a JSON queue')
    run.add_argument('queue', type=Path)
    run.add_argument('--retry-failed', action='store_true')
    run.set_defaults(fn=lambda a: run_queue(a.queue, a.retry_failed))
    status = commands.add_parser('status', help='Read saved progress without running jobs')
    status.add_argument('queue', type=Path)
    status.set_defaults(fn=show_status)


def show_status(args):
    path = Path(str(args.queue.resolve())+'.state.json')
    print(path.read_text(encoding='utf-8') if path.exists() else 'Queue has not run yet.')
    return 0
