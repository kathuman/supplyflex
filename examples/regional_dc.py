"""Worked example: should a distributor build a regional distribution centre now, build small and expand, wait, or lease?

From five years of monthly shipments to a decision rule, a risk profile and a check of how far the answer depends on
the inputs. The narrative, with the numbers this script prints, is in docs/worked-example.md.

    python examples/regional_dc.py              # print the analysis
    python examples/regional_dc.py --charts DIR # also save its charts to DIR (needs Matplotlib)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from supplyflex import calibrate, site

# Five years of monthly shipments, in units. A seeded synthetic series stands in for your own data: replace it with a
# list, a NumPy array, a pandas Series, or calibrate.parse(open("history.csv").read()).values.
HISTORY = calibrate.synthetic(f=12, years=5, start=42000, growth=0.05, sigma=0.30, season=0.10, seed=2024)

PLANNED_GROWTH = 0.04   # the commercial plan; the history alone cannot pin growth down (step 1)


def run(charts: Optional[str] = None, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    out: dict = {}

    # ---- 1. calibrate: volatility from the history; growth from the plan, checked against the history
    raw = calibrate.estimate(HISTORY, 12, scale=1000)
    est = calibrate.estimate(HISTORY, 12, deseason=True, scale=1000)
    out["calibration"] = {"raw_sigma": raw.sigma, "raw_warnings": raw.warnings, "sigma": est.sigma, "ci_sigma": est.ci_sigma,
                          "growth": est.growth, "ci_growth": est.ci_growth, "d0": est.d0}
    say("1. Calibration")
    say(f"   raw history:  sigma {raw.sigma:.3f}   warnings: {raw.warnings}")
    say(f"   de-seasoned:  sigma {est.sigma:.3f}  (90%: {est.ci_sigma[0]:.3f} to {est.ci_sigma[1]:.3f})")
    say(f"                 growth {est.growth:+.3f} (90%: {est.ci_growth[0]:+.3f} to {est.ci_growth[1]:+.3f})")
    say(f"                 demand today {est.d0:.0f} thousand units/yr; planned growth used: {PLANNED_GROWTH:+.2f}")

    # ---- 2. value every strategy
    P = site.SiteParams(d0=est.d0, sigma=est.sigma, growth=PLANNED_GROWTH)
    sol = site.solve(P)
    npv, opt = sol.npvs(), sol.option_values()
    best = max(npv, key=npv.get)
    out.update(params=P, solution=sol, npv=npv, options=opt, recommended=best)
    say("\n2. Strategy values ($M)")
    for k, v in sorted(npv.items(), key=lambda kv: -kv[1]):
        say(f"   {k:6s} {v:6.2f}" + ("   <- highest" if k == best else ""))
    say(f"   capacities: full {sol.full.capacity:.0f}, best fixed {sol.best.capacity:.0f}, staged levels "
        f"{', '.join(f'{x:.0f}' for x in sol.levels)}, lease {sol.lease.capacity:.0f}")
    say("   option values (each against its own baseline, so they do not add up):")
    for k, v in opt.items():
        say(f"   {k:14s} {v:+6.2f}")

    # ---- 3. the decision rule behind the staged strategy
    sim = sol.simulate(3000, seed=7)
    rule = _expansion_rule(sol)
    out["expansion_rule"] = rule
    out["share_expanding"] = sum(r["expansions"] > 0 for r in sim) / len(sim)
    say("\n3. Decision rule (stage)")
    say(f"   build {sol.levels[0]:.0f} now. From the initial footprint, expand when demand reaches:")
    for k, (year, demand_at, target) in enumerate(rule):   # only where the alternating pattern changes
        if k < 2 or (round(demand_at), target) != (round(rule[k - 2][1]), rule[k - 2][2]):
            say(f"   from year {year:4.2f}: {demand_at:5.0f}  -> expand to {target:.0f}")
    say(f"   in simulation, {out['share_expanding']:.0%} of futures expand at least once")

    # ---- 4. risk: the spread of outcomes, not just the mean
    risk = {}
    for k in npv:
        v = np.array([r[k] for r in sim])
        risk[k] = {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p_loss": float((v < 0).mean())}
    out["risk"] = risk
    say("\n4. Risk (3,000 simulated futures)")
    for k, r in risk.items():
        say(f"   {k:6s} mean {r['mean']:6.2f}   5th percentile {r['p05']:6.2f}   chance of a loss {r['p_loss']:4.0%}")

    # ---- 5. robustness: does the answer survive what we do not know?
    sweep = {}
    for g in (-0.04, 0.0, 0.02, PLANNED_GROWTH, 0.06, 0.08, 0.10, 0.12):
        n = site.solve(P.with_(growth=g)).npvs()
        sweep[g] = (max(n, key=n.get), n)
    out["growth_sweep"] = sweep
    say("\n5. Robustness: highest-value strategy as planned growth changes (sigma held)")
    for g, (k, n) in sweep.items():
        say(f"   growth {g:+.2f}: {k:6s} ({n[k]:.2f})")
    sig = {s: site.solve(P.with_(sigma=s)).npvs() for s in est.ci_sigma}
    out["sigma_sweep"] = sig
    for s, n in sig.items():
        say(f"   sigma {s:.3f} (end of its 90% interval): {max(n, key=n.get)}")

    if charts:
        _charts(sol, sim, est, charts)
        say(f"\nCharts written to {charts}")
    return out


def _expansion_rule(sol: site.SiteSolution) -> list:
    """(year, lowest demand that triggers an expansion from the initial footprint, capacity expanded to) at each lattice
    step of the decision window where expanding can be optimal. Neighbouring steps sit on alternating lattice levels,
    so the trigger alternates between two values: the true threshold lies between them."""
    L, lv, rule = sol.lattice, sol.levels, []
    for t in range(min(L.w, L.n - 1) + 1):
        row = sol.stage_policy[t][0]
        idx = np.nonzero((row > 0) & (row < len(lv)))[0]
        if idx.size:
            i = int(idx.min())
            rule.append((t * L.dt, site.demand(sol.params, L, t, i), lv[int(row[i])]))
    return rule


def _charts(sol, sim, est, folder: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from supplyflex import plots

    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    plots.calibration(HISTORY, 12, deseason=True).save(str(d / "example_calibration.png"), dpi=110)
    plots.strategy_values(sol).save(str(d / "example_strategy_values.png"), dpi=110)
    plots.decision_map(sol).save(str(d / "example_decision_map.png"), dpi=110)
    plots.npv_distribution(sol, sim=sim, strategies=["best", "stage", "lease"]).save(str(d / "example_npv_distribution.png"), dpi=110)
    plots.tornado(sol.params).save(str(d / "example_tornado.png"), dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--charts", metavar="DIR", help="save the example's charts to DIR")
    run(ap.parse_args().charts)
