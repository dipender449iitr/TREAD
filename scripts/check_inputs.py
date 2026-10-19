#!/usr/bin/env python3
"""Compare the supplied 19 original CSVs with the recorded SHA-256 checksums."""
from pathlib import Path
import json, hashlib, argparse, sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--data-root',default=str(ROOT/'data/Full Dataset-MSR'))
args=parser.parse_args()
base=Path(args.data_root)
expected=json.loads((ROOT/'data/expected_input_sha256.json').read_text())
issues=[]
for key, digest in expected.items():
    family, project=key.split('/',1)
    candidates=[p for p in base.rglob('*.csv') if p.stem.lower()==project.lower() and p.parent.name.lower()==family.lower()]
    if len(candidates)!=1:
        issues.append(f'{key}: expected exactly one CSV in matching family folder; got {len(candidates)}')
        continue
    actual=hashlib.sha256(candidates[0].read_bytes()).hexdigest()
    if actual!=digest:
        issues.append(f'{key}: SHA256 mismatch (expected {digest}, observed {actual})')
    else:
        print('OK',key)
print('Compared',len(expected),'recorded project hashes')
if issues:
    for msg in issues: print('ERROR',msg,file=sys.stderr)
    sys.exit(1)
print('PASS: all input hashes match the original recorded run.')
