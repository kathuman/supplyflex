"""The input template runs, on its sample history and on a history read from a CSV file, and converts everyday units
into the model's units correctly."""
import math
import sys
from pathlib import Path

import pytest

from supplyflex import calibrate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def template():
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        import inputs_template
        return inputs_template
    finally:
        sys.path.pop(0)


def test_template_converts_units(template):
    est = calibrate.estimate(template.load_history(None), 12, deseason=True, scale=1000)
    p = template.to_params(est)
    assert math.isclose(p.big_factor * p.d0 * 1000, template.FULL_DESIGN)
    assert math.isclose(p.margin, template.MARGIN_PER_UNIT / 1000)
    assert math.isclose(p.fixed_cost, template.FIXED_COST / 1e6)
    assert math.isclose(p.build_lead, template.BUILD_LEAD_MONTHS / 12)
    assert math.isclose(p.lease_fit * p.fixed_cost, template.FIT_OUT_COST / 1e6)


def test_template_runs_on_sample_and_on_csv(template, tmp_path):
    out = template.main(None, verbose=False)
    assert out["recommended"] in out["solution"].npvs()
    csv = tmp_path / "history.csv"
    csv.write_text(calibrate.sample_text(**calibrate.SAMPLES["Volatile industrial (monthly)"]), encoding="utf-8")
    out = template.main(str(csv), verbose=False)
    assert out["estimate"].n == 61 and math.isfinite(out["solution"].stage)
