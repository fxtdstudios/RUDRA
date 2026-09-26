import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from ui.server import scopes, SCOPE_LO_NITS, SCOPE_HI_NITS


def test_pq_vectorscope_reference_values_and_sample_accounting():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    module = str(Path('ui/scope_math.js').resolve())
    code = ('const s=require('+json.dumps(module)+');'
            'const v=s.vectorDensity([1,0,0,1,0,0,1,1,.1,.1,.1,1,2,-1,0,1,NaN,0,0,1],10000,64);'
            'console.log(JSON.stringify({pq:[100,1000,10000].map(s.pqSignal),'
            'neutral:s.signalChroma(.5,.5,.5),targets:s.vectorTargets(),'
            'sum:Array.from(v.counts).reduce((a,b)=>a+b,0),invalid:v.invalid,clipped:v.clipped}));')
    result = json.loads(subprocess.check_output([node, '-e', code], text=True))
    assert result['pq'] == pytest.approx([.5080784215,.7518270962,1], abs=1e-9)
    assert result['neutral'] == pytest.approx(dict(cb=0,cr=0), abs=1e-12)
    targets = {t['label']:t for t in result['targets']}
    assert targets['R']['cr'] == pytest.approx(.375)
    assert targets['R']['cb'] == pytest.approx(-.75*.2627/1.8814)
    assert targets['B']['cb'] == pytest.approx(.375)
    for a,b in [('R','C'),('G','M'),('B','Y')]:
        for axis in ('cb','cr'):
            assert targets[a][axis] == pytest.approx(-targets[b][axis], abs=1e-12)
    # Full red/blue lie beyond radius .5: neither may be silently discarded.
    assert (result['sum'],result['invalid'],result['clipped']) == (4,1,1)


def test_waveform_preserves_bimodal_gaps_and_horizontal_position():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to execute browser scope calculations')
    module = str(Path('ui/scope_math.js').resolve())
    code = ('const s=require('+json.dumps(module)+');'
            'const w=s.waveformDensity([1,100,1,100,100,1,100,NaN],2,4,2,3,1,100);'
            'console.log(JSON.stringify({counts:Array.from(w.counts),peak:w.peak,invalid:w.invalid}));')
    result = json.loads(subprocess.check_output([node,'-e',code],text=True))
    assert result == dict(counts=[2,0,2,1,0,2],peak=2,invalid=1)


def test_browser_luminance_matches_server_on_primaries_and_white():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to execute browser scope calculations')
    module = str(Path('ui/scope_math.js').resolve())
    code = ('const s=require('+json.dumps(module)+');'
            'console.log(JSON.stringify(Array.from(s.luminanceSamples('
            '[1,0,0,1, 0,1,0,1, 0,0,1,1, .1,.1,.1,1, 0,0,0,1],10000))));')
    values = json.loads(subprocess.check_output([node, '-e', code], text=True))
    assert values == pytest.approx([2627, 6780, 593, 1000, 0], abs=.001)
    for rgb, value in zip([[1,0,0],[0,1,0],[0,0,1],[.1,.1,.1],[0,0,0]], values):
        hdr = np.asarray(rgb, dtype=np.float32).reshape(3,1,1)
        result = scopes(hdr, columns=1)
        expected = np.log10(np.clip(value,SCOPE_LO_NITS,SCOPE_HI_NITS)/SCOPE_LO_NITS)/np.log10(SCOPE_HI_NITS/SCOPE_LO_NITS)
        assert result['mid'][0] == pytest.approx(expected, abs=1e-4)
