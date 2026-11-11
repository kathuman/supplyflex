"""The charts draw, and the numbers they return are the models' own numbers (needs Matplotlib)."""
import math

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from supplyflex import calibrate, network, plots, site  # noqa: E402


@pytest.fixture(scope="module")
def sol():
    return site.solve(sigma=0.35)


@pytest.fixture(scope="module")
def sim(sol):
    return sol.simulate(1500, 7)


@pytest.fixture(scope="module")
def net():
    return network.solve()


def teardown_module():
    import matplotlib.pyplot as plt
    plt.close("all")


def test_strategy_values_returns_the_npvs(sol, net):
    p = plots.strategy_values(sol)
    assert p.data["npv"] == sol.npvs() and p.data["reference"] == "best"
    assert len(p.ax.patches) == 5
    q = plots.strategy_values(net)
    assert q.data["npv"] == net.npvs() and q.data["reference"] in ("central", "two")


def test_decision_map_matches_the_policies(sol):
    p = plots.decision_map(sol)
    L = sol.lattice
    trig = {round(t["year"], 9): t["build_at_demand"] for t in sol.defer_triggers()}
    assert p.data["build_trigger"], "the wait strategy builds somewhere"
    for year, d in p.data["build_trigger"]:
        assert math.isclose(d, trig[round(year, 9)], rel_tol=1e-12)
    fan = p.data["fan"]
    assert np.all(fan["q05"] <= fan["q25"]) and np.all(fan["q25"] <= fan["q50"]) and np.all(fan["q75"] <= fan["q95"])
    assert fan["q50"][0] == sol.params.d0 and np.all(fan["q95"] <= sol.params.ceiling * sol.params.d0 + 1e-9)
    # selling happens only at low demand: below the median wherever it appears
    for year, d in p.data["sell_below"]:
        assert d < np.interp(year, p.data["years"], fan["q50"])


def test_payoff_by_outcome_bins_cover_the_simulation(sol, sim):
    p = plots.payoff_by_outcome(sol, sim=sim, bins=10)
    x = p.data["end_demand"]
    assert len(x) == 10 and np.all(np.diff(x) > 0)
    # the bin averages, weighted by bin size, are the simulated means
    sizes = [len(g) for g in np.array_split(np.arange(len(sim)), 10)]
    for k in sol.npvs():
        assert math.isclose(np.dot(p.data[k], sizes) / len(sim), np.mean([r[k] for r in sim]), rel_tol=1e-9)


def test_npv_distribution_statistics(sol, sim, net):
    p = plots.npv_distribution(sol, sim=sim)
    assert list(p.data["stats"]) == ["best", "stage"]
    for k, st in p.data["stats"].items():
        v = np.array([r[k] for r in sim])
        assert math.isclose(st["mean"], v.mean()) and st["p05"] < st["mean"] < st["p95"] and 0 <= st["p_loss"] <= 1
        assert abs(st["mean"] - sol.npvs()[k]) < 4 * v.std() / math.sqrt(len(v))
    q = plots.npv_distribution(net, n=400)
    assert set(q.data["stats"]) <= set(net.npvs())


def test_flexibility_vs_volatility_is_the_models_option_value():
    p = plots.flexibility_vs_volatility(sigmas=(0.1, 0.4))
    s = site.solve(sigma=0.4)
    assert math.isclose(p.data["stage"][1], s.stage - s.best.npv, rel_tol=1e-12)
    assert math.isclose(p.data["defer"][1], s.defer - s.best.npv, rel_tol=1e-12)


def test_tornado_rows_bracket_the_base():
    p = plots.tornado(inputs=("sigma", "rent", "margin"))
    s = site.solve()
    assert math.isclose(p.data["base"], max(s.defer, s.stage, s.lease.npv) - s.best.npv, rel_tol=1e-12)
    spans = [abs(r["high"] - r["low"]) for r in p.data["rows"]]
    assert spans == sorted(spans, reverse=True) and {r["input"] for r in p.data["rows"]} == {"sigma", "rent", "margin"}


def test_network_decision_map_is_the_wait_policy(net):
    p = plots.network_decision_map(net, year=2)
    t = int(round(2 / net.lattice.dt))
    assert np.array_equal(p.data["action"], net.wait_action[t])
    assert sum(p.data["counts"].values()) == (t + 1) ** 2


def test_pooling_reports_each_solve():
    p = plots.pooling(rhos=(-0.3, 0.6))
    s = network.solve(rho=0.6)
    assert math.isclose(p.data["central"][1], s.central.npv, rel_tol=1e-12) and len(p.data["rho_used"]) == 2


def test_calibration_fan_starts_at_the_last_observation():
    v = calibrate.synthetic(**calibrate.SAMPLES["Grocery staples (monthly, seasonal)"])
    p = plots.calibration(v, 12, deseason=True)
    est = p.data["estimate"]
    assert est.seasonal and math.isclose(p.data["median"][0], math.exp(est.logs[-1]))
    b = p.data["bands"]
    assert np.all(b["05"] <= b["25"] + 1e-9) and np.all(b["75"] <= b["95"] + 1e-9)
    lo, hi = p.data["growth_interval_paths"]
    assert lo[-1] < p.data["median"][-1] < hi[-1]


def test_themes_and_saving(tmp_path, sol):
    plots.set_theme("dark")
    try:
        p = plots.strategy_values(sol)
        out = tmp_path / "x.png"
        p.save(str(out))
        assert out.stat().st_size > 1000
    finally:
        plots.set_theme("light")
    with pytest.raises(ValueError):
        plots.set_theme("neon")
