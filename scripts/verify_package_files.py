#!/usr/bin/env python3
"""Validate all archived file checksums. Use after extracting the ZIP."""
from pathlib import Path
import hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'SHA256SUMS.txt'
missing=[];broken=[];count=0
for line in manifest.read_text().splitlines():
    if not line.strip():continue
    digest,rel=line.split('  ',1)
    f=ROOT/rel
    if not f.is_file():missing.append(rel);continue
    actual=hashlib.sha256(f.read_bytes()).hexdigest()
    if actual!=digest:broken.append(rel)
    count+=1
print(f'Checked {count} files; missing {len(missing)}; modified {len(broken)}')
for x in missing:print('MISSING:',x)
for x in broken:print('MODIFIED:',x)
sys.exit(1 if missing or broken else 0)
