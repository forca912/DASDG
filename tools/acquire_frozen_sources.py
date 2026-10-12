"""Explicit user-triggered acquisition of pinned external sources; no autorun."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description='Obtain separate pinned upstream files; preserves both conflicting DGCRN notices')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'frozen/dgcrn_md/source_manifest.json')
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.output.exists():
        raise FileExistsError('Choose a new upstream directory; do not replace prior source evidence')
    args.output.mkdir(parents=True)
    checked = []
    for name, spec in source['repositories'].items():
        for relative, record in spec['files'].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or '\\' in relative:
                raise ValueError('Unsafe manifest path')
            url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/{relative}"
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(2_000_000)
            digest = hashlib.sha256(raw).hexdigest()
            blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if digest != record['sha256'] or len(raw) != record['size'] or blob != record['git_blob']:
                raise ValueError('SOURCE_CHANGED: '+name+'/'+relative)
            dest = args.output/name/path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            checked.append({'file': name+'/'+relative, 'sha256': digest})
            print('SOURCE_VERIFIED', name+'/'+relative, flush=True)
    (args.output/'acquisition.json').write_text(json.dumps({'status': 'PASS', 'checked': checked,
        'dgcrn_license_status': 'MIT_file_vs_CC_BY_NC_SA_README_conflict_unresolved',
        'redistribution_permission_not_asserted': True}, indent=2))
    print('ACQUISITION_PASS; external source remains separate from the MIT core')


if __name__ == '__main__':
    main()
