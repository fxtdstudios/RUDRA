from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ui'))
import server


def test_cuda_request_falls_back_when_unavailable():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                             device=lambda name: name)
    device, warning = server.resolve_device('cuda', torch)
    assert device == 'cpu'
    assert 'unavailable' in warning


def test_cpu_request_is_preserved():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                             device=lambda name: name)
    device, warning = server.resolve_device('cpu', torch)
    assert device == 'cpu'
    assert warning is None
