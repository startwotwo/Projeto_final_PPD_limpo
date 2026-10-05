#!/usr/bin/env python
import json
import re

for f in ['data/horizons_cache.json', 'data/horizons_cache_20260928.json']:
    try:
        d = json.load(open(f, encoding='utf-8'))
    except Exception as e:
        print(f, 'ERR', e)
        continue
    b = d.get('bodies', {}).get('-170', {})
    print('===', f)
    print('keys:', list(b.keys()))
    print('center:', b.get('center'))
    raw = b.get('raw', '')
    print('raw len:', len(raw))
    m = re.search(r'\$\$SOE(.*?)\$\$EOE', raw, re.S)
    print('SOE:', repr(m.group(1).strip()[:800]) if m else None)
    # header: referencia e unidades
    for line in raw.splitlines():
        if 'Reference frame' in line or 'Output units' in line or 'Center' in line or 'ECHELLE' in line:
            print('HDR:', line.strip())
