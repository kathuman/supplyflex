"""Input template: enter your business figures in everyday units; the script converts them to supplyflex's units,
values every strategy and prints the answers to the usual questions.

    python examples/inputs_template.py                     # runs on a sample history
    python examples/inputs_template.py history.csv         # your history: one "date,value" line per period

Edit the figures below. Every conversion to the model's units (thousand units, $M, multiples of today's demand) happens
in to_params(), so you never enter a model unit by hand. See docs/getting-started.md and docs/parameters.md.
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np

from supplyflex import calibrate, site

# ============================================================================ your figures, in everyday units
HISTORY_FREQUENCY = 12        # observations per year in the history: 52, 12, 4 or 1
HISTORY_IN_UNITS = True       # True when each value is units shipped in that period
GROWTH = 0.04                 # expected growth per year (your plan; the history only bounds it)
MARKET_CEILING = 1_500_000    # units per year the market can never exceed

MARGIN_PER_UNIT = 20.0        # $ contribution margin per unit served by the site
LOST_SALE_COST = 10.0         # $ per unit of demand nobody serves
THREEPL_MARGIN = 6.0          # $ margin per unit served by the 3PL instead
THREEPL_MAX = 400_000         # units per year the 3PL can take

FIXED_COST = 8_000_000        # $ fixed cost of a site
COST_PER_UNIT_CAPACITY = 30.0     # $ per unit per year of capacity built
HOLDING_COST = 6.0            # $ per unit of capacity per year, used or not
EXPANSION_PREMIUM = 0.25      # later capacity costs 25% more per unit
COST_ESCALATION = 0.06        # yearly escalation of a build's cost if deferred

FULL_DESIGN = 575_000         # units per year: the design you would build in one go
FIRST_PHASE = 330_000         # units per year: the smallest viable first phase
MODULE = 60_000               # units per year: one expansion step

BUILD_LEAD_MONTHS = 18
EXPAND_LEAD_MONTHS = 12
LIFE_YEARS = 20
DECISION_WINDOW_YEARS = 8

RESALE_SHARE = 0.30           # resale value of a new site, as a share of its cost
MOTHBALL_COST = 500_000       # $ one-off to idle the site
REACTIVATE_COST = 1_000_000   # $ one-off to restart it
IDLE_HOLDING_SHARE = 0.20     # share of the holding cost still paid while idle

RENT_PER_UNIT = 6.5           # $ per unit of leased capacity per year
FIT_OUT_COST = 2_000_000      # $ fit-out of a leased site, paid now
FIT_OUT_MONTHS = 6
FIRST_BREAK_YEARS = 5
BREAK_EVERY_YEARS = 5
BREAK_FEE_YEARS_OF_RENT = 1.0

WACC = 0.10


# ============================================================================ conversions (no need to edit)
def to_params(est: calibrate.Estimate, growth: float = GROWTH) -> site.SiteParams:
    """Business figures and a demand estimate -> SiteParams, in the model's units."""
    d0 = est.d0                         # thousand units per year
    k = 1 / 1000                        # $ per unit -> $M per thousand units
    return site.SiteParams(
        d0=d0, sigma=est.sigma, growth=growth, ceiling=MARKET_CEILING / 1000 / d0,
        margin=MARGIN_PER_UNIT * k, penalty=LOST_SALE_COST * k,
        out_margin=THREEPL_MARGIN * k, out_cap=THREEPL_MAX / 1000 / d0,
        fixed_cost=FIXED_COST / 1e6, var_cost=COST_PER_UNIT_CAPACITY * k, op_cost=HOLDING_COST * k,
        premium=EXPANSION_PREMIUM, escalation=COST_ESCALATION,
        big_factor=FULL_DESIGN / 1000 / d0, small_factor=FIRST_PHASE / 1000 / d0, step_factor=MODULE / 1000 / d0,
        build_lead=BUILD_LEAD_MONTHS / 12, expand_lead=EXPAND_LEAD_MONTHS / 12,
        life=LIFE_YEARS, window=DECISION_WINDOW_YEARS,
        salvage_factor=RESALE_SHARE, mothball_cost=MOTHBALL_COST / 1e6, reactivate_cost=REACTIVATE_COST / 1e6,
        mothball_factor=IDLE_HOLDING_SHARE,
        rent=RENT_PER_UNIT * k, lease_fit=FIT_OUT_COST / FIXED_COST, lease_lead=FIT_OUT_MONTHS / 12,
        min_term=FIRST_BREAK_YEARS, break_every=BREAK_EVERY_YEARS, break_fee=BREAK_FEE_YEARS_OF_RENT,
        wacc=WACC,
    )


def load_history(path: Optional[str]) -> list:
    if path is None:  # a sample: five years of monthly shipments with a season
        return calibrate.synthetic(f=12, years=5, start=42000, growth=0.05, sigma=0.30, season=0.10, seed=2024)
    with open(path, encoding="utf-8") as fh:
        parsed = calibrate.parse(fh.read())
    for w in parsed.warnings:
        print("parse warning:", w)
    return parsed.values


# ============================================================================ the analysis
def main(path: Optional[str] = None, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    history = load_history(path)
    est = calibrate.estimate(history, HISTORY_FREQUENCY, deseason=True, scale=1000 if HISTORY_IN_UNITS else 1)
    say(f"Demand today {est.d0 * 1000:,.0f} units/yr; volatility {est.sigma:.1%} "
        f"(90%: {est.ci_sigma[0]:.1%} to {est.ci_sigma[1]:.1%}); history's growth {est.growth:+.1%} "
        f"(90%: {est.ci_growth[0]:+.1%} to {est.ci_growth[1]:+.1%}); planned growth {GROWTH:+.1%}")
    for w in est.warnings:
        say("  calibration warning:", w)

    p = to_params(est)
    sol = site.solve(p)
    npv, opt = sol.npvs(), sol.option_values()
    best = max(npv, key=npv.get)
    say("\nWhich strategy? NPV, $M")
    for k, v in sorted(npv.items(), key=lambda kv: -kv[1]):
        say(f"  {k:6s} {v:8.2f}" + ("   <- highest" if k == best else ""))
    say(f"  best size to build now: {sol.best.capacity * 1000:,.0f} units/yr; best size to lease: "
        f"{sol.lease.capacity * 1000:,.0f} units/yr")

    say("\nWhat is flexibility worth? $M (each against its own baseline; they do not add up)")
    for k, v in opt.items():
        say(f"  {k:14s} {v:+8.2f}")

    sim = sol.simulate(2000, seed=7)
    say("\nHow risky? (2,000 simulated futures)")
    for k in npv:
        v = np.array([r[k] for r in sim])
        say(f"  {k:6s} mean {v.mean():8.2f}   5th percentile {np.percentile(v, 5):8.2f}   chance of a loss {(v < 0).mean():4.0%}")

    say("\nWhen to act? (wait strategy: lowest demand at which it builds, by year)")
    for row in sol.defer_triggers()[:: max(1, round(1 / sol.lattice.dt))]:
        d = row["build_at_demand"]
        say(f"  year {row['year']:4.1f}: " + ("keep waiting" if d is None else f"build at {d * 1000:,.0f} units/yr"))

    lo, hi = est.ci_growth
    sweep = {g: max((n := site.solve(to_params(est, g)).npvs()), key=n.get)
             for g in sorted({round(x, 3) for x in (max(lo, -0.2), GROWTH - 0.04, GROWTH, GROWTH + 0.04, min(hi, 0.2))})}
    say("\nHow robust? Highest-value strategy as growth changes")
    for g, k in sweep.items():
        say(f"  growth {g:+.1%}: {k}")
    say("\nNext: plots.decision_map(sol), plots.npv_distribution(sol) and plots.tornado(p) draw the same answers.")
    return {"estimate": est, "params": p, "solution": sol, "recommended": best, "growth_sweep": sweep}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
