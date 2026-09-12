"""Locate the evaluated source snapshot without machine-specific paths."""
from pathlib import Path
import json
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
EVALUATED = REPOSITORY / "src" / "evaluated"


def bootstrap():
    records = json.loads((REPOSITORY / "configs" / "source_provenance.json").read_text(encoding="utf-8"))
    directories = sorted({str((REPOSITORY / r["path"]).parent) for r in records if r["path"].endswith(".py")})
    for directory in directories:
        if directory not in sys.path:
            sys.path.insert(0, directory)
    return EVALUATED


def source_file(module):
    matches = list(EVALUATED.rglob(module + ".py"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one evaluated module {module!r}; found {len(matches)}")
    return matches[0]


bootstrap()
