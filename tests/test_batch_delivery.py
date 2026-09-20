import json
from pathlib import Path

import pytest

from rudra import batch


@pytest.fixture
def queue(tmp_path, monkeypatch):
    (tmp_path/'model.pt').write_bytes(b'model')
    for name in ('a', 'b'):
        (tmp_path/f'{name}.mp4').write_bytes(b'source')
    path = tmp_path/'queue.json'
    path.write_text(json.dumps(dict(version=1, defaults=dict(checkpoint='model.pt'),
        jobs=[dict(input=f'{n}.mp4', output=f'{n}-hdr.mp4') for n in ('a', 'b')])))
    calls = []
    def convert(args, progress=None):
        calls.append(args.input.name)
        progress(dict(phase='inference', frames_done=1, frames_total=1))
        args.output.write_bytes(b'master')
        args.output.with_suffix('.mp4.json').write_text(json.dumps(dict(qc=dict(passed=True))))
    monkeypatch.setattr(batch.video, 'convert_video', convert)
    return path, calls, convert


def state(path):
    return json.loads(Path(str(path)+'.state.json').read_text())


def test_complete_resume_and_tampering(queue):
    path, calls, _ = queue
    assert batch.run_queue(path) == 0
    assert batch.run_queue(path) == 0
    assert calls == ['a.mp4', 'b.mp4']
    (path.parent/'a-hdr.mp4').write_bytes(b'changed')
    assert batch.run_queue(path) == 1
    assert state(path)['jobs'][0]['status'] == 'failed'
    assert len(calls) == 2


def test_failure_continues_and_explicit_retry(queue, monkeypatch):
    path, calls, convert = queue
    def fail(args, progress=None):
        if args.input.name == 'a.mp4': raise RuntimeError('decoder failed')
        return convert(args, progress)
    monkeypatch.setattr(batch.video, 'convert_video', fail)
    assert batch.run_queue(path) == 1
    assert state(path)['jobs'][1]['status'] == 'complete'
    monkeypatch.setattr(batch.video, 'convert_video', convert)
    assert batch.run_queue(path) == 1
    assert batch.run_queue(path, retry_failed=True) == 0
    assert calls == ['b.mp4', 'a.mp4']


def test_interrupt_restart(queue, monkeypatch):
    path, calls, convert = queue
    def stop(args, progress=None):
        progress(dict(phase='inference', frames_done=2, frames_total=10))
        raise KeyboardInterrupt()
    monkeypatch.setattr(batch.video, 'convert_video', stop)
    with pytest.raises(KeyboardInterrupt): batch.run_queue(path)
    assert state(path)['jobs'][0]['status'] == 'interrupted'
    monkeypatch.setattr(batch.video, 'convert_video', convert)
    assert batch.run_queue(path) == 0


def test_changed_queue_and_inputs(queue):
    path, _, _ = queue
    batch.run_queue(path)
    (path.parent/'model.pt').write_bytes(b'new weights')
    assert batch.run_queue(path) == 1
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='Queue changed'): batch.run_queue(path)


def test_collisions(queue):
    path, _, _ = queue
    spec = json.loads(path.read_text())
    spec['jobs'][1]['output'] = 'a-hdr.mp4.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='collision'): batch.run_queue(path)


def test_lock_excludes_second_runner(tmp_path):
    path = tmp_path/'queue.lock'
    with batch.locked(path):
        with pytest.raises(RuntimeError, match='already running'):
            with batch.locked(path): pass
    with batch.locked(path): pass


def test_partial_publication_is_not_overwritten(queue):
    path, calls, _ = queue
    target = path.parent/'a-hdr.mp4'
    target.write_bytes(b'partial publication')
    assert batch.run_queue(path) == 1
    assert target.read_bytes() == b'partial publication'
    assert calls == ['b.mp4']
