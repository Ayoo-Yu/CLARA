"""Portable paths for chronological evaluation; datasets are release assets."""
from pathlib import Path
import os,sys,json
CODE_ROOT=Path(__file__).resolve().parent
REPOSITORY=CODE_ROOT.parents[1]
EVALUATED=REPOSITORY/'src/evaluated'
DATA_ROOT=Path(os.environ.get('CLARA_CHRONO_ROOT',str(REPOSITORY/'assets/gefcom_chronological_v1_1_0'))).resolve()
for record in json.loads((REPOSITORY/'configs/source_provenance.json').read_text(encoding='utf8')):
    if record['path'].endswith('.py'):
        folder=str((REPOSITORY/record['path']).parent)
        if folder not in sys.path:sys.path.insert(0,folder)
