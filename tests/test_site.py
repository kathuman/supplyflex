"""The single-site model behaves as it should (ported from the web app's tests/model.test.js).

The lattice against Black–Scholes and its own growth; a deterministic case against a closed form; the strategies'
dominance relations; option value against volatility, the decision window, lead times, penalties and escalation; the
right to sell a site and lease break clauses; mothballing and partial contraction; Monte Carlo along the optimal
policies against the lattice values; convergence as the step shrinks.
"""
import math

import numpy as np
import pytest

from supplyflex import site
from supplyflex.site import SELL, SiteParams, demand, fixed_npv, lattice, lease_npv, solve

P = SiteParams()


def near(a, b, tol):
    return abs(a - b) <= tol


def increasing(xs, tol=1e-9):
    return all(x >= xs[i - 1] - tol for i, x in enumerate(xs) if i)


@pytest.fixture(scope="module")
def base():
    return solve()


# ---- the lattice
def test_up_probability_valid_and_growth_exact_over_every_slider_combination():
    bad = n = warned = 0
    worst = 0.0
    for basis in ("wacc", "rn"):
        for s in np.arange(0.05, 0.601, 0.05):
            for g in np.arange(-0.10, 0.121, 0.02):
                L = lattice(P.with_(sigma=float(s), growth=float(g), basis=basis))
                n += 1
                bad += not (0 < L.p < 1)
                worst = max(worst, abs(math.log(L.p * L.u + (1 - L.p) * L.d) / L.dt - L.growth))
                warned += bool(L.warnings) if L.dt < 0.25 else 0
    assert bad == 0 and n > 250
    assert worst < 1e-12
    assert warned > 0 and len(lattice(P.with_(sigma=0.05, growth=0.12)).warnings) == 1


def test_risk_neutral_basis():
    L = lattice(P.with_(basis="rn", growth=0.04, sigma=0.25, risk_price=0.3, risk_free=0.03))
    assert near(L.growth, 0.04 - 0.075, 1e-12) and L.rate == 0.03


def test_lattice_prices_a_european_call_close_to_black_scholes():
    def N(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    S, X, s, r, T = 100, 110, 0.3, 0.04, 2
    d1 = (math.log(S / X) + (r + s * s / 2) * T) / (s * math.sqrt(T))
    exact = S * N(d1) - X * math.exp(-r * T) * N(d1 - s * math.sqrt(T))
    coarse, fine = site.european_call(S, X, s, r, T, 0.25), site.european_call(S, X, s, r, T, 1 / 500)
    assert abs(fine / exact - 1) < 0.005 and abs(fine - exact) < abs(coarse - exact)


def test_deterministic_demand_fixed_design_is_a_discounted_sum():
    p = P.with_(sigma=1e-4, growth=0)
    L, K = lattice(p), 600
    pv = -site.build_cost(p, K, 0)
    for t in range(L.n):
        pv += math.pow(L.disc, t) * float(site.cash_rate(p, p.d0, 0 if t < L.build_steps else K)) * L.dt
    assert near(fixed_npv(p, K), pv, 1e-3)


# ---- strategy relations
def test_strategy_relations_at_the_defaults(base):
    assert base.defer >= base.full.npv - 1e-9
    assert base.best.npv >= base.full.npv - 1e-9 and all(fixed_npv(P, K) <= base.best.npv + 1e-9 for K in (300, 400, 600, 800, 1000))
    assert base.stage >= fixed_npv(P, base.levels[0]) - 1e-9
    # flexibility is valued against the best fixed design, not a straw man
    assert max(base.defer, base.stage) - base.best.npv < max(base.defer, base.stage) - base.full.npv


@pytest.mark.parametrize("o", [dict(sigma=0.45, growth=0), dict(sigma=0.2, growth=-0.06), dict(penalty=0.03), dict(build_lead=0, expand_lead=0)])
def test_strategy_relations_hold_elsewhere(o):
    s = solve(**o)
    assert s.defer >= s.full.npv - 1e-9 and s.best.npv >= s.full.npv - 1e-9 and s.stage >= fixed_npv(P.with_(**o), s.levels[0]) - 1e-9


# ---- option value against its drivers
def test_option_value_rises_with_volatility_risk_neutral():
    sig = [solve(sigma=s, basis="rn", salvage_factor=0, mothball=False, contract=False) for s in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)]
    assert increasing([s.defer - s.full.npv for s in sig], 1e-6)
    assert increasing([s.stage - s.full.npv for s in sig], 1e-6)


def test_drivers(base):
    win = [solve(window=w) for w in (2, 4, 8, 12)]
    assert increasing([s.defer for s in win]) and increasing([s.stage for s in win])
    assert solve(build_lead=0.5).full.npv > base.full.npv > solve(build_lead=3).full.npv
    assert solve(expand_lead=0).stage > base.stage
    assert solve(penalty=0.03, out_cap=0.3).defer < solve(penalty=0.005, out_cap=0.3).defer
    nobridge = solve(out_cap=0, penalty=0.04)
    assert near(nobridge.defer, nobridge.full.npv, 1e-9) and nobridge.defer_build[0][0] == 1
    assert solve(escalation=0.12).defer < solve(escalation=0).defer
    rn0 = solve(basis="rn", risk_price=0, risk_free=0.10)
    assert near(rn0.stage, base.stage, 1e-9) and near(rn0.defer, base.defer, 1e-9)


# ---- contract and exit
def test_without_exits_the_valuation_is_the_web_apps_2_0_0():
    z = solve(salvage_factor=0, mothball=False, contract=False)
    for got, want in ((z.full.npv, 19.937568656), (z.best.npv, 22.832554721), (z.defer, 22.325083326), (z.stage, 26.069265776)):
        assert near(got, want, 1e-8)


@pytest.mark.parametrize("o", [{}, dict(sigma=0.45, growth=0), dict(sigma=0.2, growth=-0.06)])
def test_the_right_to_sell_never_lowers_value(o):
    a = solve(salvage_factor=0, mothball=False, contract=False, **o)
    b = solve(salvage_factor=0.6, mothball=False, contract=False, **o)
    assert all(x >= y - 1e-9 for x, y in ((b.full.npv, a.full.npv), (b.best.npv, a.best.npv), (b.defer, a.defer), (b.stage, a.stage)))


def test_resale_value(base):
    assert increasing([solve(salvage_factor=v, mothball=False).best.npv for v in (0, 0.2, 0.4, 0.6)])
    sell = [fixed_npv(P.with_(sigma=s, basis="rn", salvage_factor=0.5, mothball=False), 500)
            - fixed_npv(P.with_(sigma=s, basis="rn", salvage_factor=0, mothball=False), 500) for s in (0.1, 0.2, 0.3, 0.45, 0.6)]
    assert increasing(sell)
    s = solve(sigma=0.45, growth=-0.04, salvage_factor=0.6, mothball=False, contract=False)
    assert any(row is not None and (row[0] == SELL).any() for row in s.stage_policy)


def test_leases(base):
    assert base.lease.npv >= base.lease.no_break - 1e-9
    huge = solve(break_fee=1000)
    assert near(huge.lease.npv, huge.lease.no_break, 1e-9)
    assert solve(break_fee=3).lease.npv < base.lease.npv < solve(break_fee=0).lease.npv
    assert solve(break_every=1).lease.npv > base.lease.npv and solve(min_term=2).lease.npv > base.lease.npv
    assert solve(rent=0.008).lease.npv < base.lease.npv
    brk = [lease_npv(P.with_(sigma=s, basis="rn"), 500, True) - lease_npv(P.with_(sigma=s, basis="rn"), 500, False) for s in (0.1, 0.2, 0.3, 0.45, 0.6)]
    assert increasing(brk, 1e-6)
    assert near(lease_npv(P, base.lease.capacity, True), base.lease.npv, 1e-9) and near(lease_npv(P, base.lease.capacity, False), base.lease.no_break, 1e-9)
    L = lattice(P)
    assert sorted(site.break_steps(P, L, 2))[:3] == [22, 42, 62] and all(t < L.n for t in site.break_steps(P, L, 2))


def test_deterministic_demand_lease_is_a_discounted_sum():
    p = P.with_(sigma=1e-4, growth=0)
    L = lattice(p)
    Ll, K = int(math.floor(p.lease_lead / L.dt + 0.5)), 450
    pv = -p.lease_fit * p.fixed_cost
    for t in range(L.n):
        cf = float(site.cash_rate(p, p.d0, 0)) if t < Ll else float(site.cash_rate(p, p.d0, K)) - p.rent * K
        pv += math.pow(L.disc, t) * cf * L.dt
    assert near(lease_npv(p, K, False), pv, 1e-3)


# ---- switching
HI = dict(salvage_factor=0, mothball_factor=0.1, op_cost=0.01, sigma=0.4, growth=0)


def test_without_switching_the_valuation_is_the_web_apps_2_1_0():
    v = solve(mothball=False, contract=False)
    for got, want in ((v.full.npv, 21.543915116), (v.best.npv, 23.444825721), (v.defer, 22.904579583), (v.stage, 26.433358371), (v.lease.npv, 23.399606525)):
        assert near(got, want, 1e-8)


@pytest.mark.parametrize("o", [{}, HI, dict(sigma=0.2, growth=-0.06, op_cost=0.01)])
def test_switching_never_lowers_value(o):
    a, b, c = (solve(mothball=m, contract=k, **o) for m, k in ((False, False), (True, False), (False, True)))
    assert all(x >= y - 1e-9 for x, y in ((b.full.npv, a.full.npv), (b.best.npv, a.best.npv), (b.defer, a.defer), (b.stage, a.stage)))
    assert c.stage >= a.stage - 1e-9


def test_mothballing_matters_for_specialised_sites_only():
    hi = solve(contract=True, **HI)
    assert hi.best.npv - hi.best.no_mothball > 0.5
    r = solve(**{**HI, "salvage_factor": 0.3})
    assert r.best.npv - r.best.no_mothball < 0.05
    hp = P.with_(**HI)
    assert fixed_npv(hp.with_(mothball_cost=0.2, reactivate_cost=0.2), 500) > fixed_npv(hp, 500)
    assert fixed_npv(hp.with_(mothball_factor=0.02), 500) > fixed_npv(hp, 500)
    mv = [fixed_npv(hp.with_(sigma=s, basis="rn"), 500) - fixed_npv(hp.with_(sigma=s, basis="rn", mothball=False), 500) for s in (0.1, 0.2, 0.3, 0.45, 0.6)]
    assert increasing(mv)


def test_hysteresis_mothball_trigger_below_restart_trigger():
    hi = solve(contract=True, **HI)
    R, L = hi.R_best, hi.lattice
    band = crossed = 0
    for t in range(L.build_steps, L.n):
        idle = np.nonzero(R.ex[t] == 2)[0]
        restart = np.nonzero(R.exB[t] == 3)[0]
        if idle.size and restart.size:
            band += 1
            crossed += not (idle.max() < restart.min())
    assert band > 10 and crossed == 0


def test_free_switching_closed_form():
    pz = P.with_(**{**HI, "mothball_cost": 0, "reactivate_cost": 0})
    L, K = lattice(pz), 500
    pm, pv = np.array([1.0]), -site.build_cost(pz, K, 0)
    for t in range(L.n):
        D = site.demand_row(pz, L, t)
        cf = site.cash_rate(pz, D, 0) if t < L.build_steps else np.maximum(site.cash_rate(pz, D, K), site.cash_idle(pz, D, K))
        pv += math.pow(L.disc, t) * float(np.sum(pm * cf)) * L.dt
        nx = np.zeros(t + 2); nx[:-1] += pm * (1 - L.p); nx[1:] += pm * L.p; pm = nx
    assert near(fixed_npv(pz, K), pv, 1e-9)


def test_staged_site_contracts_when_demand_falls():
    cs = solve(contract=True, mothball=False, **{**HI, "growth": -0.04})
    assert any(row is not None and any((row[k] < k).any() for k in range(1, row.shape[0])) for row in cs.stage_policy)


# ---- simulated policies reproduce the lattice values
@pytest.mark.parametrize("o", [{}, dict(sigma=0.45, growth=0, salvage_factor=0.6), dict(sigma=0.4, growth=-0.02, op_cost=0.01, salvage_factor=0, mothball_factor=0.1)])
def test_simulation_matches_the_lattice(o):
    s = solve(**o)
    sim = s.simulate(4000, 3)
    if o.get("salvage_factor"):
        assert any(r["sold"]["best"] is not None for r in sim) and any(r["sold"]["stage"] is not None for r in sim) and any(r["sold"]["lease"] is not None for r in sim)
    if o.get("op_cost"):
        assert any(r["mothballs"]["best"] > 0 for r in sim) and any(r["contractions"] > 0 for r in sim)
    for k, v in s.npvs().items():
        xs = np.array([r[k] for r in sim])
        assert abs(xs.mean() - v) <= 3 * xs.std() / math.sqrt(len(xs)) + 1e-9, k
    assert all(r["end_demand"] <= s.params.ceiling * s.params.d0 + 1e-9 for r in sim)


# ---- convergence, and the policy read-outs
def test_quarterly_steps_are_close_to_fortnightly():
    q, w = solve(step_years=0.25), solve(step_years=1 / 26)
    assert all(abs(q.npvs()[k] / w.npvs()[k] - 1) < 0.04 for k in q.npvs())


def test_read_outs(base):
    assert all(isinstance(v, float) for v in base.npvs().values())
    ov = base.option_values()
    assert near(ov["stage"], base.stage - base.best.npv, 1e-12) and ov["break_clauses"] >= -1e-9 and ov["sell"] >= -1e-9
    trig = base.defer_triggers()
    assert len(trig) == base.lattice.w + 1
    # where waiting does build, it builds at the lowest demand at or above its trigger, never below
    L = base.lattice
    for t, tr in enumerate(trig):
        if tr["build_at_demand"] is not None:
            built = [demand(base.params, L, t, i) for i in np.nonzero(base.defer_build[t])[0]]
            assert near(min(built), tr["build_at_demand"], 1e-9)
