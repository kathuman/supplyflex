"""Charts for the problems supplyflex solves (needs Matplotlib: ``pip install supplyflex[plot]``).

Every function draws on a Matplotlib axes (pass ``ax=`` to place it in your own figure, or one is created) and returns
a :class:`Plot` with the figure, the axes and ``data``: the numbers behind the chart, to tabulate or re-plot elsewhere.

One site
    :func:`strategy_values` — which strategy, and by how much
    :func:`decision_map` — the demand fan with the triggers to build, expand, sell or mothball
    :func:`payoff_by_outcome` — in which futures flexibility pays
    :func:`npv_distribution` — the spread of outcomes, best fixed against best flexible (also for networks)
Sensitivity
    :func:`flexibility_vs_volatility` — uncertainty is what makes flexibility valuable
    :func:`tornado` — which inputs move the value of flexibility most
Two regions
    :func:`network_decision_map` — what to open for each pair of regional demands
    :func:`pooling` — one hub or two as the regions' demand becomes more correlated
Calibration
    :func:`calibration` — the history, its trend, and the fitted forecast fan with its uncertainty

Colours follow the entity, never its rank: each strategy has one colour in every chart (the same as the web app's),
from a palette checked for colour-blind separation on white and on dark backgrounds. :func:`set_theme` switches
between ``"light"`` (default) and ``"dark"``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
except ImportError as e:  # pragma: no cover
    raise ImportError("supplyflex.plots needs Matplotlib: pip install supplyflex[plot]") from e

from . import calibrate as _cal
from . import network as _net
from . import site as _site

# ---- colours: the web app's categorical slots, validated (light and dark) for colour-blind separation
S1, S2, S3, S4, S5 = "#1f9fd8", "#c7820c", "#e0558f", "#4f9f15", "#9085e9"
SITE_COLORS = {"full": S4, "best": S1, "defer": S2, "stage": S3, "lease": S5}
NET_COLORS = {"central": S1, "two": S2, "both": S4, "pilot": S3, "wait": S5}
LABELS = {"full": "Full footprint now", "best": "Best fixed size now", "defer": "Wait, then build", "stage": "Build in stages",
          "lease": "Lease with breaks", "central": "One central hub", "two": "Two hubs, sized together", "both": "Two hubs, planned sizes",
          "pilot": "Pilot hub, option on the second", "wait": "Wait and open later"}

_THEMES = {
    "light": dict(bg="#ffffff", ink="#1b2430", muted="#5b6573", grid="#e3e7ec", band="#1f9fd8", neutral="#d5dbe2"),
    "dark": dict(bg="#0a2f52", ink="#e6f3ff", muted="#8fd0f2", grid="#1d4470", band="#7dd3fc", neutral="#2a5580"),
}
_T = dict(_THEMES["light"])


def set_theme(name: str = "light") -> None:
    """Use the ``"light"`` (default) or ``"dark"`` theme for the charts drawn after this call."""
    if name not in _THEMES:
        raise ValueError("theme is 'light' or 'dark'")
    _T.clear()
    _T.update(_THEMES[name])


@dataclass
class Plot:
    """A chart: the Matplotlib figure and axes, and the numbers behind it."""

    fig: object
    ax: object
    data: Dict[str, object] = field(default_factory=dict)

    def save(self, path: str, dpi: int = 150) -> None:
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=self.fig.get_facecolor())


def _axes(ax, size=(7.5, 4.2)):
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig = ax.figure
    fig.set_facecolor(_T["bg"])
    ax.set_facecolor(_T["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_T["grid"])
    ax.tick_params(colors=_T["muted"], labelsize=9)
    ax.grid(True, color=_T["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=_T["ink"], pad=20 if subtitle else 8)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", fontsize=9,
                    color=_T["muted"], va="bottom")


def _label(ax, x=None, y=None):
    if x:
        ax.set_xlabel(x, color=_T["muted"], fontsize=9)
    if y:
        ax.set_ylabel(y, color=_T["muted"], fontsize=9)


def _num(v: float, fmt: str = ".1f", sign: bool = False) -> str:
    """A number with a true minus sign (some fonts drop the hyphen-minus)."""
    return format(v, ("+" if sign else "") + fmt).replace("-", "\u2212")


def _plain_log(ax, axis="both"):
    """Plain numbers (1,000, not 10³) on log axes, major and minor ticks alike."""
    fmt = plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    axes = {"both": (ax.xaxis, ax.yaxis), "x": (ax.xaxis,), "y": (ax.yaxis,)}[axis]
    for a in axes:
        a.set_major_formatter(fmt)
        a.set_minor_formatter(fmt)
    ax.tick_params(which="both", colors=_T["muted"], labelsize=8.5)


def _legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=8.5, labelcolor=_T["ink"], **kw)
    return lg


def _is_network(sol) -> bool:
    return isinstance(sol, _net.NetworkSolution)


def _colors(sol):
    return NET_COLORS if _is_network(sol) else SITE_COLORS


# =============================================================================================== one site
def strategy_values(sol, ax=None) -> Plot:
    """NPV of every strategy (site or network), labelled, against the best design that commits now."""
    fig, ax = _axes(ax, (7.5, 3.6))
    npv = sol.npvs()
    names = list(npv)
    ref_name = "best" if not _is_network(sol) else max(("central", "two"), key=lambda k: npv[k])
    ref = npv[ref_name]
    y = np.arange(len(names))[::-1]
    cols = _colors(sol)
    ax.barh(y, [npv[k] for k in names], color=[cols[k] for k in names], height=0.62, edgecolor=_T["bg"], linewidth=2)
    ax.axvline(ref, color=_T["ink"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(ref, y.max() + 0.55, "  best design built now", color=_T["muted"], fontsize=8, va="bottom")
    span = max(abs(v) for v in npv.values()) or 1
    for yi, k in zip(y, names):
        v = npv[k]
        diff = v - ref
        txt = _num(v) + ("" if k == ref_name else f"  ({_num(diff, sign=True)})")
        ax.text(v + (0.01 * span if v >= 0 else -0.01 * span), yi, txt, va="center", ha="left" if v >= 0 else "right", color=_T["ink"], fontsize=9)
    ax.set_yticks(y, [LABELS[k] for k in names], color=_T["ink"])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0, min(npv.values()) * 1.15), max(npv.values()) * 1.25)
    _label(ax, x="NPV, $M")
    _title(ax, "Value of each strategy", "In brackets: gain over the best design built now")
    return Plot(fig, ax, {"npv": npv, "reference": ref_name, "option_values": sol.option_values()})


def _quantiles(values: np.ndarray, probs: np.ndarray, qs: Sequence[float]) -> List[float]:
    """Quantiles of a discrete distribution whose values are increasing."""
    c = np.cumsum(probs)
    return [float(values[min(len(values) - 1, int(np.searchsorted(c, q)))]) for q in qs]


def _demand_fan(sol: _site.SiteSolution, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Demand quantiles at every step, from the lattice's own node probabilities (no sampling)."""
    p, L = sol.params, sol.lattice
    pm, years, fan = np.array([1.0]), [], []
    for t in range(L.n + 1):
        D = _site.demand_row(p, L, t)
        years.append(t * L.dt)
        fan.append(_quantiles(D, pm, qs))
        nx = np.zeros(t + 2); nx[:-1] += pm * (1 - L.p); nx[1:] += pm * L.p; pm = nx
    return np.array(years), np.array(fan)


def decision_map(sol: _site.SiteSolution, ax=None) -> Plot:
    """When to act: the demand fan (5–95% and 25–75%, from the lattice) with the triggers of the optimal policies —
    build (wait strategy), expand from the initial footprint (stage strategy), and the demand below which the best
    fixed design is mothballed or sold. Under the risk-neutral basis the fan is the risk-neutral one."""
    fig, ax = _axes(ax, (8, 4.4))
    p, L = sol.params, sol.lattice
    years, fan = _demand_fan(sol)
    # a recombining lattice alternates between two sets of levels on odd and even steps; drawing every other step
    # keeps the lines from zig-zagging between them
    ev = 2 if L.n >= 8 else 1
    years, fan = years[::ev], fan[::ev]
    ax.fill_between(years, fan[:, 0], fan[:, 4], color=_T["band"], alpha=0.10, linewidth=0, label="demand, 5–95%")
    ax.fill_between(years, fan[:, 1], fan[:, 3], color=_T["band"], alpha=0.20, linewidth=0, label="demand, 25–75%")
    ax.plot(years, fan[:, 2], color=_T["muted"], linewidth=1, label="median demand")

    def lowest(mask_rows, t_max):
        xs, ys = [], []
        for t in range(0, t_max + 1, ev):
            idx = np.nonzero(mask_rows(t))[0]
            if idx.size:
                xs.append(t * L.dt); ys.append(_site.demand(p, L, t, int(idx.min())))
        return xs, ys

    def highest(mask_rows, t0, t1):
        xs, ys = [], []
        for t in range(t0 + (t0 % ev), t1, ev):
            idx = np.nonzero(mask_rows(t))[0]
            if idx.size:
                xs.append(t * L.dt); ys.append(_site.demand(p, L, t, int(idx.max())))
        return xs, ys

    W = min(L.w, L.n)
    dx, dy = lowest(lambda t: sol.defer_build[t] == 1, W)
    nL = len(sol.levels)
    sx, sy = lowest(lambda t: (sol.stage_policy[t][0] > 0) & (sol.stage_policy[t][0] < nL), min(W, L.n - 1))
    R = sol.R_best
    mx, my = highest(lambda t: R.ex[t] == 2, L.build_steps, L.n) if R.ex is not None else ([], [])
    ex, ey = highest(lambda t: R.ex[t] == 1, L.build_steps, L.n) if R.ex is not None else ([], [])
    if dx:
        ax.step(dx, dy, where="post", color=SITE_COLORS["defer"], linewidth=2, label="build the full footprint (wait strategy)")
    if sx:
        ax.step(sx, sy, where="post", color=SITE_COLORS["stage"], linewidth=2, label=f"first expansion from {sol.levels[0]:,.0f} (stage)")
    if mx:
        ax.plot(mx, my, color=SITE_COLORS["best"], linewidth=1.6, linestyle=(0, (4, 2)), label=f"mothball below (best fixed, {sol.best.capacity:,.0f})")
    if ex:
        ax.plot(ex, ey, color=SITE_COLORS["best"], linewidth=1.6, linestyle=(0, (1, 1.5)), label="sell below (best fixed)")
    ax.axvline(L.w * L.dt, color=_T["muted"], linewidth=0.8, linestyle=":")
    ax.text(L.w * L.dt, ax.get_ylim()[1], " end of decision window", color=_T["muted"], fontsize=8, va="top")
    ax.set_xlim(0, p.life)
    _label(ax, "years", "demand, thousand units / year")
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _title(ax, "Decision map: when to act", "Act when demand crosses a line; above the build line, waiting turns into building")
    return Plot(fig, ax, {"years": years, "fan": {"q05": fan[:, 0], "q25": fan[:, 1], "q50": fan[:, 2], "q75": fan[:, 3], "q95": fan[:, 4]},
                          "build_trigger": list(zip(dx, dy)), "expand_trigger": list(zip(sx, sy)),
                          "mothball_below": list(zip(mx, my)), "sell_below": list(zip(ex, ey))})


def payoff_by_outcome(sol: _site.SiteSolution, n: int = 3000, seed: int = 7, bins: int = 12, sim: Optional[list] = None, ax=None) -> Plot:
    """Average NPV of each strategy against the demand reached at the end of the asset's life (simulated along the
    optimal policies, grouped into equal-count bins): where a commitment wins and where flexibility pays."""
    fig, ax = _axes(ax)
    sim = sim if sim is not None else sol.simulate(n, seed)
    end = np.array([r["end_demand"] for r in sim])
    order = np.argsort(end)
    groups = np.array_split(order, bins)
    x = np.array([end[g].mean() for g in groups])
    data = {"end_demand": x}
    for k in sol.npvs():
        y = np.array([np.mean([sim[i][k] for i in g]) for g in groups])
        data[k] = y
        ax.plot(x, y, color=SITE_COLORS[k], linewidth=2, marker="o", markersize=4, label=LABELS[k])
    ax.axhline(0, color=_T["muted"], linewidth=0.8)
    ax.axvline(sol.params.d0, color=_T["muted"], linewidth=0.8, linestyle=":")
    ax.text(sol.params.d0, ax.get_ylim()[1], " demand today", color=_T["muted"], fontsize=8, va="top")
    _label(ax, "demand at the end of the asset's life, thousand units / year", "average NPV, $M")
    _legend(ax, loc="upper left")
    _title(ax, "Payoff by outcome", f"{len(sim):,} simulated futures, in {bins} groups of equal size")
    return Plot(fig, ax, data)


def npv_distribution(sol, n: int = 3000, seed: int = 7, strategies: Optional[Sequence[str]] = None, sim: Optional[list] = None,
                     ax=None) -> Plot:
    """The spread of NPV across simulated futures, as cumulative curves: for each NPV, the chance of doing no better.
    A curve further right is better everywhere; where curves cross, flexibility trades some upside for less downside,
    or the reverse. By default the best design built now against the best flexible strategy; works for a site or a
    network solution. Data gives the mean, 5% and 95% and the chance of a loss for each."""
    fig, ax = _axes(ax)
    npv = sol.npvs()
    if strategies is None:
        if _is_network(sol):
            strategies = [max(("central", "two"), key=lambda k: npv[k]), max(("pilot", "wait"), key=lambda k: npv[k])]
        else:
            strategies = ["best", max(("defer", "stage", "lease"), key=lambda k: npv[k])]
    sim = sim if sim is not None else sol.simulate(n, seed)
    cols, stats = _colors(sol), {}
    for k in strategies:
        v = np.sort(np.array([r[k] for r in sim]))
        cdf = np.arange(1, len(v) + 1) / len(v)
        stats[k] = {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "p_loss": float((v < 0).mean())}
        ax.step(v, cdf, where="post", color=cols[k], linewidth=2,
                label=f"{LABELS[k]}: mean {_num(stats[k]['mean'])}, 5–95% {_num(stats[k]['p05'])} to {_num(stats[k]['p95'])}")
        ax.plot([stats[k]["mean"]], [float(np.interp(stats[k]["mean"], v, cdf))], marker="o", color=cols[k], markersize=6,
                markeredgecolor=_T["bg"], markeredgewidth=1.5)
    ax.axvline(0, color=_T["muted"], linewidth=0.8)
    for q in (0.05, 0.95):
        ax.axhline(q, color=_T["muted"], linewidth=0.7, linestyle=(0, (3, 3)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1)
    _label(ax, "NPV, $M", "chance of doing no better")
    _legend(ax, loc="lower right")
    _title(ax, "Distribution of NPV", f"{len(sim):,} simulated futures; dots are the means; dashed lines at 5% and 95%")
    return Plot(fig, ax, {"stats": stats, "lattice_npv": {k: npv[k] for k in strategies}})


# =============================================================================================== sensitivity
def flexibility_vs_volatility(params: Optional[_site.SiteParams] = None, sigmas: Iterable[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6),
                              ax=None) -> Plot:
    """The value of each kind of flexibility (against the best fixed design) as demand becomes more uncertain. Each
    point is a full solve, so this takes a few seconds."""
    fig, ax = _axes(ax)
    p = params or _site.SiteParams()
    sig = list(sigmas)
    data: Dict[str, list] = {"sigma": sig, "defer": [], "stage": [], "lease": []}
    for s in sig:
        sol = _site.solve(p.with_(sigma=s))
        for k in ("defer", "stage", "lease"):
            data[k].append(sol.npvs()[k] - sol.best.npv)
    for k in ("defer", "stage", "lease"):
        ax.plot(sig, data[k], color=SITE_COLORS[k], linewidth=2, marker="o", markersize=4, label=LABELS[k])
    ax.axhline(0, color=_T["muted"], linewidth=0.8)
    ax.axvline(p.sigma, color=_T["muted"], linewidth=0.8, linestyle=":")
    ax.text(p.sigma, ax.get_ylim()[1], " your volatility", color=_T["muted"], fontsize=8, va="top")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _label(ax, "volatility of demand growth, per year", "gain over the best fixed design, $M")
    _legend(ax, loc="upper left")
    _title(ax, "Uncertainty makes flexibility valuable", "Above zero, the flexible strategy beats committing to the best size now")
    return Plot(fig, ax, data)


DEFAULT_TORNADO = ("sigma", "growth", "margin", "op_cost", "penalty", "fixed_cost", "var_cost", "build_lead", "expand_lead", "wacc",
                   "premium", "salvage_factor", "rent")


def tornado(params: Optional[_site.SiteParams] = None, inputs: Sequence[str] = DEFAULT_TORNADO, change: float = 0.2, ax=None) -> Plot:
    """What moves the value of flexibility: each input lowered and raised by ``change`` (20%), the others held, and the
    value of the best flexible strategy over the best fixed design recomputed. Sorted by how far each input moves it."""
    fig, ax = _axes(ax, (7.5, 0.42 * len(inputs) + 1.4))
    p = params or _site.SiteParams()

    def flex(q):
        s = _site.solve(q)
        return max(s.defer, s.stage, s.lease.npv) - s.best.npv

    base = flex(p)
    rows = []
    for name in inputs:
        v = getattr(p, name)
        lo, hi = flex(p.with_(**{name: v * (1 - change)})), flex(p.with_(**{name: v * (1 + change)}))
        rows.append((name, lo, hi))
    rows.sort(key=lambda r: abs(r[2] - r[1]))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] - base for r in rows], left=base, color=S2, height=0.6, edgecolor=_T["bg"], linewidth=1.5, label=f"input −{change:.0%}")
    ax.barh(y, [r[2] - base for r in rows], left=base, color=S1, height=0.6, edgecolor=_T["bg"], linewidth=1.5, label=f"input +{change:.0%}")
    ax.axvline(base, color=_T["ink"], linewidth=1)
    ax.set_yticks(y, [r[0].replace("_", " ") for r in rows], color=_T["ink"])
    ax.grid(axis="y", visible=False)
    _label(ax, x="value of flexibility (best flexible − best fixed), $M")
    _legend(ax, loc="lower right")
    _title(ax, "What moves the value of flexibility", f"Each input ±{change:.0%}, the others held; base {base:.2f}")
    return Plot(fig, ax, {"base": base, "rows": [{"input": r[0], "low": r[1], "high": r[2]} for r in rows[::-1]]})


# =============================================================================================== two regions
ACTIONS = ["wait", "open hub A", "open hub B", "open both"]


def network_decision_map(sol: _net.NetworkSolution, year: float = 0.0, ax=None) -> Plot:
    """What to open, for every pair of regional demands at a given year (with no hub open yet): wait, open A, open B or
    both. The cross marks the most likely demands at that year."""
    fig, ax = _axes(ax, (6.4, 5.4))
    L = sol.lattice
    t = min(max(0, int(round(year / L.dt))), L.n - 1)
    act = sol.wait_action[t]
    Da, Db = sol.demand[0][t], sol.demand[1][t]

    def edges(v):
        v = np.asarray(v, dtype=float)
        if v.size == 1:
            return np.array([v[0] * 0.9, v[0] * 1.1])
        mid = np.sqrt(v[1:] * v[:-1])
        return np.concatenate([[v[0] ** 2 / mid[0]], mid, [v[-1] ** 2 / mid[-1]]])

    cmap = ListedColormap([_T["neutral"], S1, S2, S3])
    ax.pcolormesh(edges(Da), edges(Db), act.T, cmap=cmap, vmin=-0.5, vmax=3.5, shading="flat", edgecolors=_T["bg"], linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log(ax)
    ax.grid(False)
    pA, pB = L.q[0] + L.q[1], L.q[0] + L.q[2]
    ia, ib = int(round(pA * t)), int(round(pB * t))
    ax.plot([Da[ia]], [Db[ib]], marker="x", color=_T["ink"], markersize=9, mew=2)
    counts = {ACTIONS[c]: int((act == c).sum()) for c in range(4)}
    ax.legend(handles=[Patch(color=c, label=a) for c, a in zip([_T["neutral"], S1, S2, S3], ACTIONS)], frameon=False, fontsize=8.5,
              labelcolor=_T["ink"], loc="upper left", bbox_to_anchor=(1.01, 1))
    can = t <= L.w and t + L.build_steps <= L.n
    _label(ax, "demand in region A, thousand units / year", "demand in region B, thousand units / year")
    _title(ax, f"What to open, year {t * L.dt:g}", "With no hub yet" + ("" if can else " — after the decision window: waiting only"))
    return Plot(fig, ax, {"year": t * L.dt, "demand_a": Da, "demand_b": Db, "action": act, "counts": counts})


def pooling(params: Optional[_net.NetworkParams] = None, rhos: Iterable[float] = (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9), ax=None) -> Plot:
    """One hub or two as the regions' demand becomes more correlated: a central hub pools uncorrelated demand, so it
    loses value as correlation rises. Each point is a full solve; a correlation the lattice cannot carry is limited
    (hollow marker) and the one used is in data."""
    fig, ax = _axes(ax)
    p = params or _net.NetworkParams()
    rr = list(rhos)
    data: Dict[str, list] = {"rho": rr, "rho_used": []}
    sols = [_net.solve(p.with_(rho=r)) for r in rr]
    data["rho_used"] = [s.lattice.rho for s in sols]
    for k in ("central", "two", "both", "wait", "pilot"):
        data[k] = [s.npvs()[k] for s in sols]
        ax.plot(rr, data[k], color=NET_COLORS[k], linewidth=2, label=LABELS[k],
                linestyle=(0, (5, 3)) if k == "pilot" else "-", zorder=3 if k == "pilot" else 2)
        for r, v, s in zip(rr, data[k], sols):
            hollow = abs(s.lattice.rho - r) > 1e-9
            ax.plot([r], [v], marker="o", markersize=5, color=NET_COLORS[k], markerfacecolor=_T["bg"] if hollow else NET_COLORS[k])
    _label(ax, "correlation of the two regions' demand", "NPV, $M")
    _legend(ax, loc="best")
    _title(ax, "Pooling and correlation", "Hollow points: correlation limited by the lattice. Pilot on wait: waiting opens the pilot at once")
    return Plot(fig, ax, data)


# =============================================================================================== calibration
def calibration(values: Sequence[float], f: int, deseason: bool = False, horizon: float = 3.0, ax=None) -> Plot:
    """The demand history (log scale) with its de-seasoned line when the season is removed, then the fitted model
    forward: the median, 50% and 90% bands of demand, and dashed lines for the 90% interval of the growth estimate
    itself, which is usually far wider than people expect."""
    fig, ax = _axes(ax, (8, 4.2))
    values = [float(v) for v in values]
    est = _cal.estimate(values, f, deseason=deseason)
    n = len(values)
    x = np.arange(n) / f
    ax.plot(x, values, color=_T["muted"], linewidth=1, label="history")
    if est.seasonal:
        ax.plot(x, np.exp(est.logs), color=S1, linewidth=1.6, label="season removed")
    x0 = math.exp(est.logs[-1])
    t = np.linspace(0, horizon, 60)
    drift = est.growth - est.sigma ** 2 / 2
    z = {"05": -1.6449, "25": -0.6745, "75": 0.6745, "95": 1.6449}
    band = {k: x0 * np.exp(drift * t + v * est.sigma * np.sqrt(t)) for k, v in z.items()}
    med = x0 * np.exp(drift * t)
    tx = x[-1] + t
    ax.fill_between(tx, band["05"], band["95"], color=_T["band"], alpha=0.10, linewidth=0, label="forecast, 5–95%")
    ax.fill_between(tx, band["25"], band["75"], color=_T["band"], alpha=0.22, linewidth=0, label="forecast, 25–75%")
    ax.plot(tx, med, color=S1, linewidth=1.8, label=f"median (growth {_num(100 * est.growth, sign=True)}%/yr)")
    g_lo, g_hi = est.ci_growth
    for g, lab in ((g_lo, "growth, 90% interval"), (g_hi, None)):
        ax.plot(tx, x0 * np.exp((g - est.sigma ** 2 / 2) * t), color=S2, linewidth=1.2, linestyle=(0, (4, 3)), label=lab)
    ax.set_yscale("log")
    _plain_log(ax, "y")
    ax.axvline(x[-1], color=_T["muted"], linewidth=0.8, linestyle=":")
    _label(ax, "years from the first observation", "demand per period")
    _legend(ax, loc="upper left")
    _title(ax, "Calibration: history and fitted model",
           f"σ {est.sigma:.1%} (90%: {est.ci_sigma[0]:.1%}–{est.ci_sigma[1]:.1%})  ·  growth {_num(100 * est.growth, sign=True)}% "
           f"(90%: {_num(100 * g_lo, sign=True)}% to {_num(100 * g_hi, sign=True)}%)")
    return Plot(fig, ax, {"estimate": est, "forecast_years": tx, "median": med, "bands": band,
                          "growth_interval_paths": (x0 * np.exp((g_lo - est.sigma ** 2 / 2) * t), x0 * np.exp((g_hi - est.sigma ** 2 / 2) * t))})
