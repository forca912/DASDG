"""Mechanical manifest/ZIP build, including UTF-8 filename support. No training."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name != 'package_manifest.json')
    if any('__pycache__' in p.parts or p.suffix in ('.pt', '.pth', '.pyc') or p.name.startswith('.') for p in files):
        raise ValueError('Unexpected generated/cache/hidden files in delivery tree')
    for p in files:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = ROOT / 'package_manifest.json'
    manifest.write_text(json.dumps(hashes, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    archive = ROOT.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError('Delivery ZIP already exists; refusing overwrite')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in files + [manifest]:
            z.write(p, arcname=ROOT.name + '/' + p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        assert sum(n.endswith('/LICENSE') for n in z.namelist()) == 2
    print('ZIP_VERIFIED', archive, 'bytes=', archive.stat().st_size, 'files=', len(files) + 1)
    print('SHA256', hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
