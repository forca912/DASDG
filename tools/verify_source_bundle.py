"""Verify source archive integrity; excludes generated runtime artifacts."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    entries = (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    checked = []
    for line in entries:
        digest, name = line.split('  ',1)
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name:
            raise ValueError('Unsafe manifest path')
        p = root.joinpath(*rel.parts)
        if not p.resolve().is_relative_to(root) or p.is_symlink() or not p.is_file():
            raise ValueError('Missing or linked source: '+name)
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed: '+name)
        checked.append(name)
    print(json.dumps({'status':'PASS','checked_files':len(checked),'scope':'source_manifest_not_model_performance'}))


if __name__ == '__main__':
    main()
