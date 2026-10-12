"""Build an isolated, auditable compatibility copy; never edit upstream."""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root):
    manifest = json.loads((root / 'source_manifest.json').read_text(encoding='utf-8'))
    checked = []
    for name, spec in manifest['repositories'].items():
        for rel, record in spec['files'].items():
            path = root / 'upstream' / name / rel
            raw = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            if (hashlib.sha256(raw).hexdigest() != record['sha256'] or
                    len(raw) != record['size'] or blob != record['git_blob']):
                raise ValueError(f'SOURCE_CHANGED: {name}/{rel}')
            checked.append(f'{name}/{rel}')
    return {'verified_files': checked, 'repositories': {
        name: {'repository': spec['repository'], 'commit': spec['commit']}
        for name, spec in manifest['repositories'].items()}}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Pinned compatibility replacement no longer matches source')
    return text.replace(old, new, 1)


def transform_model(text):
    tree = ast.parse(text)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    removed = []
    lines = text.splitlines(keepends=True)
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            module = getattr(n, 'module', '') or ''
            if module in {'utils.masking', 'models.encoder', 'models.decoder', 'models.attn',
                          'models.embed'} or (isinstance(n, ast.Import) and
                          any(a.name == 'data.dcrnn_utils' for a in n.names)):
                names = {a.asname or a.name.split('.')[0] for a in n.names}
                if used & names:
                    raise ValueError(f'Cannot remove used import: {names & used}')
                removed.extend(names)
                for i in range(n.lineno - 1, n.end_lineno):
                    lines[i] = ''
    text = ''.join(lines).replace('from models.gnn import ', 'from pf_gcnm.gnn import ')
    text = text.replace('from models.memoryModule import ', 'from pf_gcnm.memoryModule import ')
    if text.count('nn.Conv1d(') != 6:
        raise ValueError('Unexpected GCN-M convolution source')
    text = text.replace('nn.Conv1d(', 'nn.Conv2d(')
    ast.parse(text)
    return text, {'removed_unused_imports_ast_checked': sorted(removed),
                  'Conv1d_to_Conv2d': 6, 'package_import_rename': 'pf_gcnm'}


def transform_memory(text):
    text = replace_once(text, '''        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
''', '''        self.register_buffer('filter_square_matrix', filter_square_matrix.detach().clone())
''')
    text = replace_once(text, '''        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))
''', '''        self.register_buffer('identity', torch.eye(input_size))
        self.register_buffer('zeros', torch.zeros(input_size))
''')
    text, outer = re.subn(r'torch.squeeze\((input\[:, [0-7], :, :\])\)', r'\1', text)
    text, inner = re.subn(r'torch.squeeze\((\w+\[:, i:i \+ 1, :\])\)', r'\1.squeeze(1)', text)
    if (outer, inner) != (8, 8):
        raise ValueError('Unexpected GCN-M squeeze source')
    ast.parse(text)
    return text, {'register_buffers': ['identity', 'zeros', 'filter_square_matrix'],
                  'preserve_batch_dimension': {'outer_squeeze_removed': outer, 'inner_dim1': inner}}


def build_compat(root, destination):
    verify_sources(root)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Compatibility destination already exists: {destination}')
    destination.mkdir(parents=True)
    mapping = [
        ('gcnm/models/model.py', 'pf_gcnm/model.py', transform_model),
        ('gcnm/models/memoryModule.py', 'pf_gcnm/memoryModule.py', transform_memory),
        ('gcnm/models/gnn.py', 'pf_gcnm/gnn.py', None),
        ('dgcrn/methods/DGCRN/net.py', 'pf_dgcrn/net.py', 'dgcrn_import'),
        ('dgcrn/methods/DGCRN/layer.py', 'pf_dgcrn/layer.py', None)]
    records, patches = [], []
    for source, target, transform in mapping:
        src = root / 'upstream' / source
        original = src.read_text(encoding='utf-8')
        changes = {}
        if callable(transform):
            result, changes = transform(original)
        elif transform == 'dgcrn_import':
            result = replace_once(original, 'from layer import *', 'from pf_dgcrn.layer import *')
            result = replace_once(result, 'torch.squeeze(x[..., i])', 'x[..., i]')
            changes = {'package_import_rename': 'pf_dgcrn',
                       'preserve_batch_dimension': 'encoder time slice already has B,C,N dimensions'}
        else:
            result = original
        ast.parse(result)
        dest = destination / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(result)
        records.append({'source': source, 'target': target, 'source_sha256': sha256(src),
                        'target_sha256': sha256(dest), 'changes': changes})
        patches.extend(difflib.unified_diff(original.splitlines(True), result.splitlines(True),
                                           fromfile='upstream/' + source, tofile=target))
    for package in ('pf_gcnm', 'pf_dgcrn'):
        (destination / package / '__init__.py').write_text('', encoding='utf-8')
    (destination / 'compatibility.diff').write_text(''.join(patches), encoding='utf-8')
    (destination / 'compatibility_manifest.json').write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return records
