"""Calibration behaves as it should (ported from the web app's tests/calibrate.test.js).

Parsing the formats people paste; frequency detection; estimates on synthetic histories with known parameters
(unbiased on average, 90% intervals that cover the truth about 90% of the time); removing the season; demand today;
the warnings; the samples.
"""
import math

import pytest

from supplyflex import calibrate as C


def near(a, b, tol):
    return abs(a - b) <= tol


def test_parsing_formats():
    a = C.parse("date,demand\n2024-01,100\n2024-02,110\n2024-03,105\n")
    assert a.values == [100, 110, 105] and len(a.dates) == 3 and C.detect_frequency(a.dates) == 12
    b = C.parse("2024-01-01;10\n2024-01-08;11\n2024-01-15;12\n2024-01-22;13")
    assert len(b.values) == 4 and C.detect_frequency(b.dates) == 52
    c = C.parse("2019\t500\n2020\t520\n2021\t560\n2022\t540")
    assert len(c.values) == 4 and C.detect_frequency(c.dates) == 1
    d = C.parse("2023 Q1,7\n2023 Q2,8\n2023 Q3,9\n2023 Q4,9.5")
    assert len(d.values) == 4 and C.detect_frequency(d.dates) == 4
    e = C.parse("120\n130\n\n125\n140\n")
    assert e.values == [120, 130, 125, 140] and e.dates is None
    g = C.parse("100\nabc\n110\n120")
    assert len(g.values) == 3 and "skipped" in g.warnings[0]


def test_estimates_unbiased_with_honest_intervals():
    truth = dict(f=12, years=8, start=1000, growth=0.05, sigma=0.25)
    R, sg, gg, cov_s, cov_g = 400, 0.0, 0.0, 0, 0
    for s in range(1, R + 1):
        est = C.estimate(C.synthetic(seed=s, **truth), 12)
        sg += est.sigma; gg += est.growth
        cov_s += est.ci_sigma[0] <= truth["sigma"] <= est.ci_sigma[1]
        cov_g += est.ci_growth[0] <= truth["growth"] <= est.ci_growth[1]
    assert near(sg / R, truth["sigma"], 0.005) and near(gg / R, truth["growth"], 0.015)
    assert 0.85 < cov_s / R < 0.95 and 0.85 < cov_g / R < 0.95


def test_growth_uncertainty_depends_on_span_not_sampling():
    short = C.estimate(C.synthetic(52, 2, 100, 0.05, 0.25, seed=3), 52)
    long = C.estimate(C.synthetic(1, 30, 100, 0.05, 0.25, seed=3), 1)
    assert short.se_growth > 2 * long.se_growth and short.se_sigma < long.se_sigma
    exact = C.estimate([100, 110, 121, 133.1], 1)
    assert near(exact.growth, math.log(1.1), 1e-9) and near(exact.sigma, 0, 1e-9)


def test_seasons():
    v = C.synthetic(12, 6, 1000, 0.03, 0.1, season=0.15, seed=9)
    raw, ds = C.estimate(v, 12), C.estimate(v, 12, deseason=True)
    assert raw.sigma > 1.5 * ds.sigma and near(ds.sigma, 0.1, 0.03)
    assert any("seasonal" in w for w in raw.warnings) and near(ds.growth, raw.growth, 0.03)
    assert any("two years" in w for w in C.estimate(v[:18], 12, deseason=True).warnings)


def test_demand_today_errors_and_warnings():
    m = C.estimate([10] * 11 + [12] * 13, 12, units="period", scale=1000)
    assert near(m.d0, 12 * 12 / 1000, 1e-12)
    assert near(C.estimate([100, 120, 130, 150], 1, units="year").d0, 150, 1e-12)
    with pytest.raises(ValueError):
        C.estimate([1, 0, 2], 12)
    with pytest.raises(ValueError):
        C.estimate([1, 2], 12)
    rev, x = [], 100.0
    for k in range(120):
        rev.append(x)
        x = x * 1.2 if k % 2 else x / 1.2
    assert any("autocorrelated" in w for w in C.estimate(rev, 12).warnings)
    assert any("three years" in w for w in C.estimate([100, 105, 103, 110, 112], 4).warnings)


@pytest.mark.parametrize("name", list(C.SAMPLES))
def test_samples_recover_their_parameters(name):
    o = C.SAMPLES[name]
    p = C.parse(C.sample_text(**o))
    f = C.detect_frequency(p.dates)
    est = C.estimate(p.values, f, deseason=bool(o.get("season")))
    assert f == o["f"]
    assert est.ci_sigma[0] <= o["sigma"] <= est.ci_sigma[1]
    assert est.ci_growth[0] - 0.02 <= o["growth"] <= est.ci_growth[1] + 0.02


def test_estimate_feeds_the_site_model():
    from supplyflex import site
    est = C.estimate(C.synthetic(**C.SAMPLES["Volatile industrial (monthly)"]), 12, scale=1000)
    sol = site.solve(**est.site_inputs())
    assert sol.params.sigma == est.sigma and sol.params.d0 == est.d0 and math.isfinite(sol.stage)
