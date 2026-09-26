from pathlib import Path


def test_modern_scope_panel_has_waveform_histogram_and_vectorscope():
    html=Path('ui/index.html').read_text(encoding='utf-8')
    js=Path('ui/app.js').read_text(encoding='utf-8')
    assert 'id="vector"' in html
    assert 'PQ / Rec.2020' in html
    assert 'RudraScopeMath.vectorDensity' in js
    assert 'targetMarks' in js
    assert 'sampled luminance density' in html
