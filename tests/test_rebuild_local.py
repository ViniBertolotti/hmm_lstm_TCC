import json

import nbformat
import pytest

from scripts import rebuild_local


@pytest.mark.parametrize('failure', [None, RuntimeError('failed cell'), KeyboardInterrupt()])
def test_manifest_records_completion_and_interruption(tmp_path, monkeypatch, failure):
    for directory in ('reports/tables', 'data/processed', 'data/raw', 'notebooks'):
        (tmp_path / directory).mkdir(parents=True)
    for name in ('02_feature_engineering', '03_hmm_regimes'):
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('pass')]),
                       tmp_path / 'notebooks' / (name + '.ipynb'))
    class Client:
        def __init__(self, nb, **kwargs):
            self.nb = nb
        def execute(self):
            self.on_cell_start(self.nb.cells[0], 0)
            if failure is not None:
                raise failure
    monkeypatch.setattr(rebuild_local, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rebuild_local, 'NotebookClient', Client)
    monkeypatch.setattr(rebuild_local.subprocess, 'check_output', lambda *a, **kw: 'numpy==test\n')
    if failure is None:
        rebuild_local.main()
    else:
        with pytest.raises(type(failure)):
            rebuild_local.main()
    manifest = json.loads((tmp_path / 'reports/tables/rebuild_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == ('complete' if failure is None else 'failed')
    assert manifest['finished_utc']
    assert manifest['current_cell'] == 0
    assert len(manifest['completed_notebooks']) == (2 if failure is None else 0)
