"""The documentation stays true to the code: every input is in the parameter reference with its real default, and the
worked example's numbers are the ones examples/regional_dc.py produces."""
import re
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from supplyflex.network import NetworkParams
from supplyflex.site import SiteParams

ROOT = Path(__file__).resolve().parents[1]
PARAMS_MD = (ROOT / "docs" / "parameters.md").read_text(encoding="utf-8")
EXAMPLE_MD = (ROOT / "docs" / "worked-example.md").read_text(encoding="utf-8")


def _documented(section: str) -> dict:
    """{input: default as written} from the tables of one section of parameters.md."""
    rows = re.findall(r"^\| `(\w+)` \| ([^|]+?) \|", section, flags=re.M)
    return {name: default.strip() for name, default in rows}


def _same(written: str, value) -> bool:
    if value is None:
        return written == "None"
    if isinstance(value, bool):
        return written == str(value)
    if isinstance(value, str):
        return written == f'`"{value}"`'
    return float(written) == float(value)


def test_every_site_input_is_documented_with_its_default():
    section = PARAMS_MD.split("## One site")[1].split("## Two regions")[0]
    doc = _documented(section)
    for f in fields(SiteParams):
        assert f.name in doc, f"{f.name} is missing from docs/parameters.md"
        assert _same(doc[f.name], f.default), f"{f.name}: documented {doc[f.name]}, default {f.default!r}"


def test_every_network_input_is_documented_with_its_default():
    section = PARAMS_MD.split("## Two regions")[1].split("## Calibration")[0]
    doc = _documented(section)
    shared = set(re.findall(r"`(\w+)`", section.split("\n\n")[1]))   # the paragraph listing inputs shared with SiteParams
    site_defaults = {f.name: f.default for f in fields(SiteParams)}
    for f in fields(NetworkParams):
        if f.name in doc:
            assert _same(doc[f.name], f.default), f"{f.name}: documented {doc[f.name]}, default {f.default!r}"
        else:
            assert f.name in shared, f"{f.name} is missing from docs/parameters.md"
            assert site_defaults[f.name] == f.default, f"{f.name} is documented as shared but its default differs"


@pytest.fixture(scope="module")
def example():
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        import regional_dc
        return regional_dc.run(verbose=False)
    finally:
        sys.path.pop(0)


def test_worked_example_numbers_match_the_script(example):
    assert example["recommended"] == "stage"
    for k, v in example["npv"].items():
        assert f"{v:.2f}" in EXAMPLE_MD, f"NPV of {k} ({v:.2f}) is not in docs/worked-example.md"
    for k in ("stage", "lease", "break_clauses", "sell", "wait"):
        assert f"{example['options'][k]:+.2f}".replace("-", "−") in EXAMPLE_MD, k
    for k, r in example["risk"].items():
        assert f"{r['mean']:.2f}" in EXAMPLE_MD and f"{r['p_loss']:.0%}" in EXAMPLE_MD, k
    c = example["calibration"]
    assert f"{c['raw_sigma']:.3f}" in EXAMPLE_MD and f"{c['sigma']:.3f}" in EXAMPLE_MD


def test_worked_example_growth_table_matches_the_script(example):
    row = re.search(r"^\| Highest value \|(.+)\|$", EXAMPLE_MD, flags=re.M).group(1)
    written = [cell.strip().strip("*") for cell in row.split("|")]
    assert written == [best for best, _ in example["growth_sweep"].values()]
    assert all(max(n, key=n.get) == "stage" for n in example["sigma_sweep"].values())
