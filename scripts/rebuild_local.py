"""Reexecuta notebooks 02/03 com dados locais e registra proveniência.

Execute da raiz: .venv/Scripts/python scripts/rebuild_local.py
Não baixa dados. Arquiva os relatórios anteriores antes da reconstrução.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = ROOT / 'reports' / 'archive' / stamp
    for directory in ('reports/tables', 'data/processed'):
        for path in (ROOT / directory).iterdir():
            if path.is_file() and path.name != '.gitkeep':
                target = archive / directory / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    manifest = {'started_utc': stamp, 'status': 'running',
                'futures_policy': 'unadjusted_yahoo_proxy_pending_verified_roll_factors',
                'scope': 'exploratory_in_sample', 'inputs_sha256': {}}
    for path in sorted((ROOT / 'data/raw').glob('*')):
        if path.is_file() and path.suffix in ('.parquet', '.xlsx'):
            manifest['inputs_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    out = ROOT / 'reports/tables/rebuild_manifest.json'
    out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'requirements-lock.txt').write_text(environment, encoding='utf-8')
    try:
        for name in ('02_feature_engineering', '03_hmm_regimes'):
            path = ROOT / 'notebooks' / (name + '.ipynb')
            nb = nbformat.read(path, as_version=4)
            client = NotebookClient(nb, timeout=3600, resources={'metadata': {'path': str(ROOT / 'notebooks')}})
            def progress(cell, cell_index, **kwargs):
                if cell.cell_type == 'code':
                    print(f'{name}: célula {cell_index}', flush=True)
            client.on_cell_start = progress
            print(f'Executando {name}', flush=True)
            try:
                client.execute()
            finally:
                nbformat.write(nb, path)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    else:
        manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    finally:
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
