from pathlib import Path


def test_hdr_scope_labels_and_markers_are_present():
    html = Path('ui/index.html').read_text(encoding='utf-8')
    js = Path('ui/app.js').read_text(encoding='utf-8')
    assert 'luma percentile' in html
    assert 'absolute nits' in html
    assert 'Diffuse white 203' in js
    assert '>DW</text>' in js
    assert ' nits</text>' in js
