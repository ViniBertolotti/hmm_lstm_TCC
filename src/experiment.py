"""Identificação de execuções e conferência dos artefatos antes da retomada."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

MODELS = ('lstm', 'hmm_lstm')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str), encoding='utf-8')
    temporary.replace(path)


def snapshot(root, config, inputs, *, options, versions=None):
    root = Path(root).resolve()
    sources = sorted(p for folder in ('src', 'scripts', 'r')
                     for p in (root / folder).rglob('*') if p.suffix in ('.py', '.R'))
    sources += [root / name for name in ('requirements.txt', 'requirements-lock.txt')
                if (root / name).exists()]
    if versions is None:
        versions = {'python': platform.python_version()}
        for name in ('numpy', 'pandas', 'scipy', 'torch', 'hmmlearn', 'scikit-learn',
                     'arch', 'pandas_market_calendars', 'threadpoolctl', 'pyarrow'):
            versions[name] = metadata.version(name)
    value = {'schema_version': 1, 'configuration': config, 'options': options,
             'versions': versions,
             'inputs_sha256': {Path(p).resolve().relative_to(root).as_posix(): file_hash(p) for p in inputs},
             'sources_sha256': {p.relative_to(root).as_posix(): file_hash(p) for p in sources}}
    value['fingerprint'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()
    return value


def prepare_output(out, experiment, *, resume):
    out = Path(out)
    path = out / 'experiment.json'
    existing = out.exists() and any(out.iterdir())
    if existing:
        if not path.exists():
            raise ValueError('Saídas antigas sem experiment.json. Use outra pasta; a origem do treino não pode ser reconstruída depois.')
        recorded = json.loads(path.read_text(encoding='utf-8'))
        if recorded.get('fingerprint') != experiment['fingerprint']:
            raise ValueError('Execução incompatível: dados, código, ambiente ou configuração mudaram. Use outra pasta.')
        if not resume:
            raise ValueError('A pasta já contém uma execução. Use --resume ou escolha outra pasta.')
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(path, experiment)


def fold_artifacts(fold):
    return [f'forecasts_fold{fold:02d}.parquet', f'calibration_fold{fold:02d}.parquet',
            *[f'checkpoint_{model}_fold{fold:02d}.pt' for model in MODELS]]


def completed_folds(out, manifest, candidates):
    out = Path(out)
    complete = set()
    for fold in candidates:
        rows = [m for m in manifest if m['fold'] == fold]
        if len(rows) != 2 or {m['model'] for m in rows} != set(MODELS):
            continue
        names = fold_artifacts(fold)
        if not all((out / name).is_file() for name in names):
            continue
        expected = rows[0].get('artifacts_sha256', {})
        if rows[1].get('artifacts_sha256') != expected or set(expected) != set(names):
            raise ValueError(f'Janela {fold}: registro SHA256 incompleto ou divergente.')
        if any(file_hash(out / name) != expected[name] for name in names):
            raise ValueError(f'Janela {fold}: SHA256 de artefato divergente; confira a execução antes de retomar.')
        complete.add(fold)
    return complete
