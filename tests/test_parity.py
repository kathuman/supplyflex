"""supplyflex gives the same numbers as the JavaScript engine of Real Options for Network Design.

tests/fixtures/parity.json is written by tools/make_fixtures.js from the web app's engine; every value here must
match it. Floating-point sums can be ordered differently in NumPy, so the tolerance is relative, 1e-9.
"""
import json
import math
from pathlib import Path

import pytest

from supplyflex import calibrate, network, site

FIX = json.loads((Path(__file__).parent / "fixtures" / "parity.json").read_text(encoding="utf-8"))

SITE_NAMES = {"D0": "d0", "sigma": "sigma", "g": "growth", "ceiling": "ceiling", "life": "life", "window": "window",
              "margin": "margin", "opcost": "op_cost", "penalty": "penalty", "outMargin": "out_margin", "outCap": "out_cap",
              "basis": "basis", "wacc": "wacc", "rf": "risk_free", "lambda": "risk_price", "smallF": "small_factor",
              "bigF": "big_factor", "stepF": "step_factor", "premium": "premium", "escal": "escalation", "fFix": "fixed_cost",
              "fVar": "var_cost", "buildLead": "build_lead", "expandLead": "expand_lead", "salvF": "salvage_factor",
              "rent": "rent", "leaseLead": "lease_lead", "leaseFit": "lease_fit", "minTerm": "min_term",
              "breakEvery": "break_every", "breakFee": "break_fee", "moth": "mothball", "mothF": "mothball_factor",
              "mothFix": "mothball_cost", "reactFix": "reactivate_cost", "contract": "contract", "stepYears": "step_years"}
NET_NAMES = {"DA": "demand_a", "DB": "demand_b", "sigA": "sigma_a", "sigB": "sigma_b", "gA": "growth_a", "gB": "growth_b",
             "rho": "rho", "ceiling": "ceiling", "life": "life", "window": "window", "buildLead": "build_lead", "margin": "margin",
             "opcost": "op_cost", "outMargin": "out_margin", "outCap": "out_cap", "penalty": "penalty", "xcost": "cross_cost",
             "basis": "basis", "wacc": "wacc", "rf": "risk_free", "lambda": "risk_price", "fFix": "fixed_cost", "fVar": "var_cost",
             "escal": "escalation", "learn": "learning", "sizeA": "size_a", "sizeB": "size_b"}
BOOLS = {"moth", "contract"}


def site_params(js):
    return site.SiteParams(**{SITE_NAMES[k]: (bool(v) if k in BOOLS else v) for k, v in js.items()})


def net_params(js):
    return network.NetworkParams(**{NET_NAMES[k]: v for k, v in js.items()})


def close(a, b, rel=1e-9, abs_=1e-9):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


@pytest.mark.parametrize("case", FIX["site"], ids=lambda c: json.dumps(c["params"]))
def test_site_strategies_match(case):
    s = site.solve(site_params(case["params"]))
    L = s.lattice
    assert (L.dt, L.n, len(L.warnings)) == (case["dt"], case["N"], case["warnings"])
    assert close(L.p, case["p"], 1e-12)
    assert len(s.levels) == len(case["levels"]) and all(close(a, b) for a, b in zip(s.levels, case["levels"]))
    got = {"full": s.full.npv, "best": s.best.npv, "bestK": s.best.capacity, "noExit": s.best.no_exit, "noMoth": s.best.no_mothball,
           "defer": s.defer, "stage": s.stage, "lease": s.lease.npv, "leaseK": s.lease.capacity, "noBreak": s.lease.no_break}
    for k, v in got.items():
        assert close(v, case[k]), f"{k}: {v} vs {case[k]}"
    assert int(sum(int(r.sum()) for r in s.defer_build)) == case["deferBuilds"]


@pytest.mark.parametrize("case", FIX["fixed"], ids=lambda c: json.dumps(c["params"]) + f" K={c['K']}")
def test_fixed_npv_matches(case):
    assert close(site.fixed_npv(site_params(case["params"]), case["K"]), case["npv"])


@pytest.mark.parametrize("case", FIX["lease"], ids=lambda c: json.dumps(c["params"]) + f" breaks={c['breaks']}")
def test_lease_npv_matches(case):
    assert close(site.lease_npv(site_params(case["params"]), case["K"], case["breaks"]), case["npv"])


def test_european_call_matches():
    assert close(site.european_call(*FIX["call"]["args"]), FIX["call"]["value"], 1e-12)


def test_simulated_paths_are_the_same_paths():
    c = FIX["simulation"]
    sim = site.simulate(site.solve(site_params(c["params"])), c["n"], c["seed"])
    for got, want in zip(sim, c["paths"]):
        for k in ("full", "best", "defer", "stage", "lease"):
            assert close(got[k], want[k]), k
        assert close(got["end_demand"], want["endDemand"])
    for k, v in c["mean"].items():
        assert close(sum(r[k] for r in sim) / len(sim), v)


@pytest.mark.parametrize("case", FIX["network"], ids=lambda c: json.dumps(c["params"]))
def test_network_strategies_match(case):
    s = network.solve(net_params(case["params"]))
    assert close(s.lattice.rho, case["rho"]) and len(s.lattice.warnings) == case["warnings"]
    assert all(close(a, b, 1e-12) for a, b in zip(s.lattice.q, case["q"]))
    got = {"central": s.central.npv, "centralK": s.central.capacity_a, "two": s.two.npv, "twoKA": s.two.capacity_a,
           "twoKB": s.two.capacity_b, "both": s.both.npv, "pilot": s.pilot.npv, "pilotOther": s.pilot.other, "wait": s.wait}
    for k, v in got.items():
        assert close(v, case[k], 1e-8), f"{k}: {v} vs {case[k]}"
    assert s.pilot.region == case["pilotRegion"]


@pytest.mark.parametrize("case", FIX["netFixed"], ids=lambda c: json.dumps(c["params"]))
def test_network_fixed_designs_match(case):
    p = net_params(case["params"])
    assert close(network.fixed_npv(p, case["KA"], case["KB"]), case["sum"])
    assert close(network.fixed_npv_backward(p, case["KA"], case["KB"]), case["backward"])


def test_network_simulated_paths_are_the_same_paths():
    c = FIX["netSimulation"]
    sim = network.simulate(network.solve(), c["n"], c["seed"])
    for got, want in zip(sim, c["paths"]):
        for k, v in want.items():
            assert close(got[k], v, 1e-8), k


@pytest.mark.parametrize("case", FIX["calibrate"], ids=lambda c: c["name"])
def test_calibration_matches(case):
    o = case["sample"]
    v = calibrate.synthetic(o["f"], o["years"], o["start"], o["g"], o["sigma"], o.get("season", 0.0), o["seed"])
    assert len(v) == case["n"] and all(close(a, b, 1e-12) for a, b in zip(v[:5], case["first"])) and close(sum(v), case["sum"], 1e-10)
    for key, deseason in (("plain", False), ("deseasoned", True)):
        e, want = calibrate.estimate(v, o["f"], deseason=deseason), case[key]
        got = {"sigma": e.sigma, "mu": e.mu, "g": e.growth, "seSigma": e.se_sigma, "seG": e.se_growth, "D0": e.d0, "rho1": e.rho1}
        for k, x in got.items():
            assert close(x, want[k]), f"{key} {k}: {x} vs {want[k]}"
        assert e.seasonal == want["seasonal"] and len(e.warnings) == want["warnings"]


def test_parsing_matches():
    c = FIX["parse"]
    r = calibrate.parse(c["text"])
    assert r.values == c["values"] and r.dates == c["dates"] and len(r.warnings) == c["warnings"]
    assert calibrate.detect_frequency(r.dates) == c["f"]
