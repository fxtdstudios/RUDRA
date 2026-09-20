import numpy as np
import torch

from rudra.delivery.bench import load_frame
from rudra.delivery.cli import _frames
from rudra.sdr2hdr import SDR2HDRNet
from training.infer_sdr2hdr import predict_image, write_outputs, infer_video


def test_delivery_exr_keeps_absolute_luminance(tmp_path):
    write_outputs(np.full((16, 16, 3), 0.0203, np.float32), tmp_path/'frame', False)
    frames = _frames(tmp_path)
    assert len(frames) == 1
    np.testing.assert_allclose(load_frame(frames[0], 203), 203, rtol=1e-6)


def test_every_tile_receives_one_global_shadow_weight():
    torch.manual_seed(1)
    model = SDR2HDRNet(base_channels=8, shadow_conditioning=True).eval()
    frame = torch.rand(1, 3, 48, 80)
    seen = []
    original = model.forward
    def record(*args, **kwargs):
        seen.append(kwargs.get('shadow_weight'))
        assert kwargs['recovery_mode'] == 'all'
        return original(*args, **kwargs)
    model.forward = record
    predict_image(model, frame, True, 32, 8)
    assert len(seen) > 1
    assert seen[0] is not None
    assert all(w is seen[0] for w in seen)


def test_video_propagates_preservation(tmp_path, monkeypatch):
    import training.infer_sdr2hdr as inference
    class Capture:
        def isOpened(self): return True
        def get(self, _): return 24
        def read(self):
            if hasattr(self, 'done'): return False, None
            self.done = True
            return True, np.zeros((16,16,3), np.uint8)
        def release(self): pass
    monkeypatch.setattr(inference.cv2, 'VideoCapture', lambda _: Capture())
    seen = []
    def predict(model, frame, preserve, *args):
        seen.append(preserve)
        return frame
    monkeypatch.setattr(inference, 'predict_image', predict)
    monkeypatch.setattr(inference, 'write_outputs', lambda *args: None)
    for preserve in (True, False):
        infer_video(tmp_path/'input.mp4', tmp_path, None, None, torch.device('cpu'),
                    9, False, 'srgb', 'full', 0, 0, 'all', 1, preserve)
    assert seen == [True, False]
