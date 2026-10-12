"""Prepare externally supplied DGCRN; no author model source is bundled here."""
import ast
import difflib
import hashlib
import json
from importlib.resources import files
from pathlib import Path


def spec():
    return json.loads(files('dasdg').joinpath('dgcrn_sources.json').read_text(encoding='utf-8'))


def verify(upstream):
    upstream = Path(upstream)
    source = spec()
    checked = {}
    for name in ('LICENSE', 'README.md', 'methods/DGCRN/net.py', 'methods/DGCRN/layer.py'):
        p = upstream / name
        if p.is_symlink() or not p.is_file():
            raise ValueError(f'Missing or linked upstream source: {name}')
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        expected = source['files'][name]
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if digest != expected['sha256'] or len(raw) != expected['size'] or blob != expected['git_blob']:
            raise ValueError(f'Pinned upstream source changed: {name}')
        checked[name] = digest
    return checked


def prepare(upstream, output):
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    if output == upstream or upstream in output.parents or output in upstream.parents:
        raise ValueError('Upstream and compatibility output must be separate directories')
    checked = verify(upstream)
    if output.exists():
        raise FileExistsError(f'Refusing compatibility output overwrite: {output}')
    texts = {}
    records, patches = [], []
    for name in ('net.py', 'layer.py'):
        original = (upstream / 'methods/DGCRN' / name).read_text(encoding='utf-8')
        result = original
        if name == 'net.py':
            for old, new in [('from layer import *', 'from pf_dgcrn.layer import *'),
                             ('torch.squeeze(x[..., i])', 'x[..., i]')]:
                if result.count(old) != 1:
                    raise ValueError('Frozen compatibility replacement no longer matches')
                result = result.replace(old, new, 1)
        ast.parse(result)
        texts[name] = result
        records.append({'source': 'methods/DGCRN/' + name, 'output': 'pf_dgcrn/' + name,
                        'source_sha256': checked['methods/DGCRN/' + name],
                        'target_sha256': hashlib.sha256(result.encode('utf-8')).hexdigest()})
        patches.extend(difflib.unified_diff(original.splitlines(True), result.splitlines(True),
                       fromfile='upstream/methods/DGCRN/' + name, tofile='pf_dgcrn/' + name))
    output.mkdir(parents=True)
    package = output / 'pf_dgcrn'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    for name, text in texts.items():
        with (package / name).open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    for name in ('LICENSE', 'README.md'):
        (output / ('UPSTREAM_' + name)).write_bytes((upstream / name).read_bytes())
    with (output / 'compatibility.diff').open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(''.join(patches))
    result = {'status': 'PREPARED', 'repository': spec()['repository'], 'commit': spec()['commit'],
              'checked_files': checked, 'compatibility_files': records,
              'changes': ['namespace import', 'preserve encoder batch dimension'],
              'same_transform_as_formal_v3': True,
              'license_status': 'CONFLICT_NOT_CLARIFIED_BY_RIGHTSHOLDER',
              'upstream_modified': False, 'formal_training_rerun': False}
    (output / 'compatibility_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output / 'NOTICE.txt').write_text(
        'External DGCRN: tsinghua-fib-lab/Traffic-Benchmark.\n'
        'Pinned repository LICENSE says MIT; README badge says CC BY-NC-SA.\n'
        'This preparation preserves both documents and does not resolve their conflict.\n'
        'The generated upstream compatibility copy is not licensed by DASDG.\n'
        'Follow applicable upstream terms; obtain clarification before redistributing it.\n', encoding='utf-8')
    return result
