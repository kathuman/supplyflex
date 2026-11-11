"""The two-region model behaves as it should (ported from the web app's tests/network.test.js).

The joint lattice (exact growth in each region, the correlation, valid branches); fixed designs valued two ways;
reduction to the single-site model when one region is empty; separability without cross-region service; dominance
between strategies; learning, pooling, the window and volatility; Monte Carlo replay of every policy.
"""
import math

import numpy as np
import pytest

from supplyflex import network, site
from supplyflex.network import NetworkParams, fixed_npv, fixed_npv_backward, lattice, solve, weights2

P = NetworkParams()


def near(a, b, tol):
    return abs(a - b) <= tol


def test_joint_lattice_valid_exact_and_correlated():
    bad = n = clamped = 0
    worst_g = worst_r = 0.0
    for sa in (0.1, 0.2, 0.4, 0.6):
        for sb in (0.1, 0.3, 0.5):
            for ga in (-0.06, 0, 0.04):
                for gb in (-0.04, 0.03, 0.08):
                    for rho in (-0.8, -0.3, 0, 0.5, 0.9):
                        L = lattice(P.with_(sigma_a=sa, sigma_b=sb, growth_a=ga, growth_b=gb, rho=rho))
                        n += 1
                        if not all(-1e-15 <= x <= 1 for x in L.q) or abs(sum(L.q) - 1) > 1e-12:
                            bad += 1
                        pA, pB = L.q[0] + L.q[1], L.q[0] + L.q[2]
                        worst_g = max(worst_g, abs(math.log(pA * L.a.u + (1 - pA) * L.a.d) / L.dt - L.a.growth),
                                      abs(math.log(pB * L.b.u + (1 - pB) * L.b.d) / L.dt - L.b.growth))
                        corr = (L.q[0] - pA * pB) / math.sqrt(pA * (1 - pA) * pB * (1 - pB))
                        if any("correlation" in w for w in L.warnings):
                            clamped += 1
                        else:
                            worst_r = max(worst_r, abs(corr - rho))
    assert bad == 0 and worst_g < 1e-12 and worst_r < 1e-12
    assert lattice(P.with_(sigma_a=0.05, growth_a=0.12)).warnings


def test_joint_weights_sum_to_one_with_binomial_margins():
    L = lattice(P)
    w = weights2(L, 6)
    p = L.q[0] + L.q[1]
    assert near(w.sum(), 1, 1e-12)
    assert all(near(w[a].sum(), math.comb(6, a) * p ** a * (1 - p) ** (6 - a), 1e-12) for a in range(7))


def test_fixed_designs_two_ways_and_reductions():
    for KA, KB in ((700, 0), (450, 300), (0, 400), (600, 250)):
        assert near(fixed_npv(P, KA, KB), fixed_npv_backward(P, KA, KB), 1e-9)
    shared = dict(margin=0.02, op_cost=0.006, penalty=0.01, out_margin=0.006, out_cap=1, ceiling=3, life=20, build_lead=1.5, wacc=0.1, fixed_cost=8, var_cost=0.03)
    net = fixed_npv(NetworkParams(demand_a=500, demand_b=1e-9, sigma_a=0.25, growth_a=0.04, **shared), 600, 0)
    one = site.fixed_npv(site.SiteParams(d0=500, sigma=0.25, growth=0.04, salvage_factor=0, mothball=False, **shared), 600)
    assert near(net, one, 1e-6)
    sep = P.with_(cross_cost=1, learning=0)
    assert near(fixed_npv(sep, 450, 300) + fixed_npv(sep, 0, 0), fixed_npv(sep, 450, 0) + fixed_npv(sep, 0, 300), 1e-9)
    assert all(fixed_npv(P, a, b) >= fixed_npv(P.with_(cross_cost=1), a, b) - 1e-9 for a, b in ((450, 300), (700, 0), (0, 400)))
    assert fixed_npv(P, 700, 0) > fixed_npv(P.with_(cross_cost=1), 700, 0) + 1


@pytest.fixture(scope="module")
def base():
    return solve()


def test_strategy_relations(base):
    assert base.wait >= base.both.npv - 1e-9 and base.wait >= base.pilot.npv - 1e-9 and base.wait >= base.pilot.other - 1e-9
    c, t = base.central, base.two
    assert fixed_npv(P, c.capacity_a * 1.1, 0) <= c.npv + 1e-9 and fixed_npv(P, c.capacity_a * 0.9, 0) <= c.npv + 1e-9
    assert all(fixed_npv(P, t.capacity_a * ma, t.capacity_b * mb) <= t.npv + 1e-6 for ma, mb in ((1.1, 1), (0.9, 1), (1, 1.1), (1, 0.9)))
    assert base.two.npv >= base.both.npv - 1e-9


def test_learning_pooling_window_and_volatility(base):
    L0, L8 = solve(learning=0), solve(learning=0.8)
    assert L8.both.npv > base.both.npv > L0.both.npv and L8.pilot.npv > L0.pilot.npv and L8.wait > L0.wait
    assert near(L0.central.npv, L8.central.npv, 1e-9)
    pool = [fixed_npv(P.with_(rho=r), 750, 0) for r in (-0.6, -0.2, 0.2, 0.6, 0.9)]
    assert all(x <= pool[i - 1] + 1e-9 for i, x in enumerate(pool) if i)
    win = [solve(window=w).wait for w in (2, 5, 8, 12)]
    assert all(x >= win[i - 1] - 1e-9 for i, x in enumerate(win) if i)
    opt = [(lambda s: s.wait - s.both.npv)(solve(sigma_b=sb, basis="rn", risk_price=0)) for sb in (0.15, 0.25, 0.35, 0.5)]
    assert all(x >= opt[i - 1] - 1e-9 for i, x in enumerate(opt) if i)
    assert solve(cross_cost=0.012).central.npv < base.central.npv


@pytest.mark.parametrize("o", [{}, dict(sigma_b=0.5, growth_b=0.02, rho=-0.3)])
def test_simulation_matches_the_lattice(o):
    s = solve(**o)
    sim = s.simulate(3000, 5)
    for k, v in s.npvs().items():
        xs = np.array([r[k] for r in sim])
        assert abs(xs.mean() - v) <= 3 * xs.std() / math.sqrt(len(xs)) + 1e-9, k
    if o:
        assert any(r["wait_second"] is not None and r["wait_second"] > 0 for r in sim) and any(r["wait_second"] is None for r in sim)
