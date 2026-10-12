"""Check the release's declared runtime, separately from inherited packages."""
import importlib.metadata
import json
import re
import sys


def pair(value):
    match = re.match(r'^(\d+)\.(\d+)', value)
    if match is None:
        raise ValueError('Unrecognized runtime version: '+value)
    return tuple(map(int, match.groups()))


if __name__ == '__main__':
    versions = {name: importlib.metadata.version(name) for name in ('dasdg', 'numpy', 'torch')}
    assert (3,10) <= sys.version_info[:2] < (3,13), 'Unsupported Python'
    assert versions['dasdg']=='1.1.0', 'Wrong candidate distribution'
    assert (1,26) <= pair(versions['numpy']) < (3,0), 'Unsupported NumPy'
    assert (2,3) <= pair(versions['torch']) < (3,0), 'Unsupported PyTorch'
    print(json.dumps({'status': 'PASS', 'versions': versions,
                     'scope': 'candidate_declared_dependencies_only'}, indent=2))
