"""Assemble the Windows Studio preview from a verified wheel and shipped model."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    out=Path('outputs/windows_preview_release'); out.mkdir(exist_ok=False)
    package=out/'RUDRA-Studio-0.3.3rc1-windows-x64'; package.mkdir()
    wheel=Path('outputs/windows_preview_build/rudra_hdr-0.3.3rc1-py3-none-any.whl')
    shutil.copy2(wheel,package/wheel.name)
    (package/'checkpoints').mkdir()
    for name in ('sdr2hdr_shadow_v1.pt','sdr2hdr_shadow_v1.config.json','LICENSE'):
        shutil.copy2(Path('checkpoints')/name,package/'checkpoints'/name)
    shutil.copy2('LICENSE',package/'LICENSE-code')
    shutil.copy2('scripts/windows_preview_install.cmd',package/'Install Studio.cmd')
    shutil.copy2('scripts/windows_preview_start.cmd',package/'Start Studio.cmd')
    shutil.copy2('docs/WINDOWS_STUDIO_PREVIEW_033RC1.md',package/'README.md')
    python=Path('outputs/windows_preview_clean/Scripts/python.exe')
    installed=json.loads(subprocess.check_output([str(python),'-m','pip','list','--format=json'],text=True))
    pins=sorted(f"{p['name']}=={p['version']}" for p in installed if p['name'].lower() not in ('pip','rudra-hdr','torch'))
    (package/'requirements.lock').write_text('\n'.join(pins)+'\n')
    hashes={str(p.relative_to(package)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob('*') if p.is_file()}
    assert hashes['checkpoints/sdr2hdr_shadow_v1.pt']=='6b7f73f0b44a6edc3e117373f7397ae3a8df12cb398ec4815597b6e2ebeb63f9'
    (package/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(hashes.items())))
    archive=out/(package.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in package.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(out))
    (out/(archive.name+'.sha256')).write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    print(archive)


if __name__=='__main__': main()
