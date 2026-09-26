import json
import pytest
from rudra import batch


@pytest.mark.parametrize('permanent', [False, True])
def test_replace_retries_preserve_previous_status(tmp_path, monkeypatch, permanent):
    path = tmp_path / 'status.json'
    path.write_text('{"state":"old"}')
    original = batch.os.replace
    calls = []
    def replace(source, target):
        calls.append(1)
        assert json.loads(path.read_text()) == {'state': 'old'}
        if permanent or len(calls) < 3:
            raise PermissionError('file temporarily open')
        original(source, target)
    monkeypatch.setattr(batch.os, 'replace', replace)
    monkeypatch.setattr(batch.time, 'sleep', lambda _: None)
    if permanent:
        with pytest.raises(PermissionError):
            batch.save(path, {'state': 'new'})
        assert len(calls) == 8
        assert json.loads(path.read_text()) == {'state': 'old'}
    else:
        batch.save(path, {'state': 'new'})
        assert len(calls) == 3
        assert json.loads(path.read_text()) == {'state': 'new'}
    assert list(tmp_path.iterdir()) == [path]
