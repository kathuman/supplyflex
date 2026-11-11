"""Real options for one site: commit, wait, stage, lease, and the exits that go with them.

Demand follows a recombining binomial lattice over the whole life of the asset. Decisions (build, wait, expand) can be
taken only inside a decision window; after it, capacity stays as it is and keeps operating to the end of the asset's
life. Building and expanding take time (lead times). Demand the site cannot serve is outsourced to a third party (3PL)
up to a limit at a lower margin, and the rest is lost with a penalty — the same rule for every strategy, including
while waiting. Two valuation bases: expected cash flows at the WACC (``"wacc"``), or risk-neutral (``"rn"``: drift
reduced by the market price of risk, discounted at the risk-free rate).

Strategies, valued against the same demand:

* ``full``  — build the full footprint now and keep it
* ``best``  — build the capacity that maximises NPV now and keep it (the fair fixed benchmark)
* ``defer`` — build the full footprint once, at the best moment inside the window (cost escalates with time)
* ``stage`` — build the initial footprint now, then add steps where they pay (each at a premium, one at a time)
* ``lease`` — lease capacity now (no capital cost, rent instead); break clauses let you hand it back at set dates

Every owned site can also be sold at any time for its resale value, which declines to zero over the asset's life;
mothballed (idled at a fraction of its holding cost) and reactivated, each switch at a one-off cost; and the staged
site can sell capacity back in steps.

Units: money in $M, demand and capacity in thousand units per year, margins in $M per thousand units (= $/unit / 1000).

This is a port of the engine of *Real Options for Network Design* (https://kathuman.github.io/claude-projects/real-options/);
for the same inputs it gives the same values (the tests compare the two).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from ._common import XorShift32, js_round

SELL, MOTHBALL = 255, 254  # stage policy codes besides a target level
_STEPS = (0.25, 1 / 12, 1 / 52)


@dataclass(frozen=True)
class SiteParams:
    """Inputs of the single-site model. Defaults are the web app's."""

    # demand
    d0: float = 500.0            # demand today, thousand units per year
    sigma: float = 0.25          # volatility of demand growth, per year
    growth: float = 0.04         # expected demand growth, per year
    ceiling: float = 3.0         # the market saturates at ceiling × today's demand
    life: float = 20.0           # asset life, years
    window: float = 8.0          # decisions (build, wait, expand) only within this many years
    # economics
    margin: float = 0.020        # margin on own sales, $M per thousand units
    op_cost: float = 0.006       # holding cost of capacity, $M per thousand units of capacity per year
    penalty: float = 0.010       # penalty on lost sales, $M per thousand units
    out_margin: float = 0.006    # margin on sales served by the 3PL
    out_cap: float = 1.0         # 3PL limit, × today's demand
    # valuation basis
    basis: str = "wacc"          # "wacc" or "rn" (risk-neutral)
    wacc: float = 0.10
    risk_free: float = 0.03
    risk_price: float = 0.3      # market price of risk (lambda), for the risk-neutral basis
    # capacity and costs
    small_factor: float = 0.8    # staged initial footprint, × d0
    big_factor: float = 1.4      # full footprint, × d0
    step_factor: float = 0.15    # expansion step, × d0
    premium: float = 0.25        # premium on the variable cost of an expansion
    escalation: float = 0.06     # yearly escalation of a deferred build's cost
    fixed_cost: float = 8.0      # fixed cost of a site, $M
    var_cost: float = 0.030      # cost per thousand units of capacity, $M
    build_lead: float = 1.5      # years from deciding to build until capacity arrives
    expand_lead: float = 1.0     # years from deciding to expand until the step arrives
    # exits
    salvage_factor: float = 0.3  # resale value of a new site, as a share of its cost (declines to 0 at end of life)
    # lease
    rent: float = 0.0065         # rent per thousand units of capacity per year, $M
    lease_lead: float = 0.5      # fit-out time, years
    lease_fit: float = 0.25      # fit-out cost, × fixed cost
    min_term: float = 5.0        # first break date, years after the lease starts
    break_every: float = 5.0     # then a break date every this many years
    break_fee: float = 1.0       # fee for a break, in years of rent
    # switching
    mothball: bool = True        # owned sites may be mothballed and reactivated
    mothball_factor: float = 0.2  # holding cost while idle, as a share
    mothball_cost: float = 0.5   # one-off cost to mothball, $M
    reactivate_cost: float = 1.0  # one-off cost to reactivate, $M
    contract: bool = True        # the staged site may sell capacity back in steps
    # numerics
    step_years: Optional[float] = None  # force a lattice step (used if valid), for convergence checks

    def with_(self, **changes) -> "SiteParams":
        """A copy with some inputs changed."""
        return replace(self, **changes)


@dataclass
class Lattice:
    """The demand lattice: step, up/down factors, up-probability, discounting and the lead times in steps."""

    dt: float
    u: float
    d: float
    p: float
    growth: float
    rate: float
    disc: float
    n: int            # steps over the asset's life
    w: int            # last step at which a decision can be taken
    build_steps: int
    expand_steps: int
    warnings: List[str] = field(default_factory=list)
    valid: bool = True


def lattice(p: SiteParams) -> Lattice:
    """The lattice for these inputs. The step is the largest of quarterly, monthly or weekly for which the
    up-probability stays inside (0, 1) with a margin (|g|·√dt ≤ 0.85·σ); a warning says when a finer one was needed."""
    g = p.growth - p.risk_price * p.sigma if p.basis == "rn" else p.growth
    r = p.risk_free if p.basis == "rn" else p.wacc
    warnings: List[str] = []
    dt = None
    if p.step_years and abs(g) * math.sqrt(p.step_years) <= 0.85 * p.sigma:
        dt = p.step_years
    for s in _STEPS:
        if dt is None and abs(g) * math.sqrt(s) <= 0.85 * p.sigma:
            dt = s
    if dt is None:
        dt = _STEPS[-1]
        warnings.append("Growth is too strong for this little volatility: even weekly steps cannot represent it "
                        "exactly. Raise the volatility or lower the growth.")
    elif dt < 0.25:
        warnings.append("The lattice uses " + ("monthly" if dt == 1 / 12 else "weekly") + " steps here, so the growth "
                        "you set can be represented exactly.")
    u = math.exp(p.sigma * math.sqrt(dt))
    d = 1 / u
    pu = (math.exp(g * dt) - d) / (u - d)
    n = js_round(p.life / dt)
    return Lattice(dt=dt, u=u, d=d, p=pu, growth=g, rate=r, disc=math.exp(-r * dt), n=n, w=min(n, js_round(p.window / dt)),
                   build_steps=js_round(p.build_lead / dt), expand_steps=js_round(p.expand_lead / dt), warnings=warnings,
                   valid=0 < pu < 1)


# ---- the economics, vectorised over a lattice row (D may be an array)
def demand_row(p: SiteParams, L: Lattice, t: int) -> np.ndarray:
    """Demand at every node of step t (i up-moves of t), capped at the market ceiling."""
    i = np.arange(t + 1)
    return np.minimum(p.d0 * np.power(L.u, 2 * i - t), p.ceiling * p.d0)


def demand(p: SiteParams, L: Lattice, t: int, i: int) -> float:
    return min(p.d0 * math.pow(L.u, 2 * i - t), p.ceiling * p.d0)


def cash_rate(p: SiteParams, D, K: float):
    """Cash flow per year with capacity K at demand D: own sales, the 3PL up to its limit, lost sales, holding cost."""
    own = np.minimum(D, K)
    short = np.maximum(D - K, 0.0)
    out = np.minimum(short, p.out_cap * p.d0)
    lost = short - out
    return p.margin * own + p.out_margin * out - p.penalty * lost - p.op_cost * K


def cash_idle(p: SiteParams, D, K: float):
    """Cash flow per year of a mothballed site: demand goes to the 3PL, a share of the holding cost remains."""
    return cash_rate(p, D, 0.0) - p.mothball_factor * p.op_cost * K


def build_cost(p: SiteParams, K: float, years: float) -> float:
    return 0.0 if K <= 0 else (p.fixed_cost + p.var_cost * K) * math.pow(1 + p.escalation, years)


def expand_cost(p: SiteParams, dK: float) -> float:
    return 0.0 if dK <= 0 else 0.25 * p.fixed_cost + p.var_cost * dK * (1 + p.premium)


def salvage(p: SiteParams, L: Lattice, K: float, t: int) -> float:
    """What selling a site of capacity K at step t brings: a share of its cost, declining linearly to 0 at end of life."""
    if K <= 0 or not p.salvage_factor > 0:
        return 0.0
    return p.salvage_factor * (p.fixed_cost + p.var_cost * K) * max(0.0, 1 - t / L.n)


def contract_proceeds(p: SiteParams, L: Lattice, dK: float, t: int) -> float:
    """Selling part of a site (dK of capacity, no share of the fixed cost)."""
    if dK <= 0 or not p.salvage_factor > 0:
        return 0.0
    return p.salvage_factor * p.var_cost * dK * max(0.0, 1 - t / L.n)


def break_steps(p: SiteParams, L: Lattice, start: int) -> set:
    """The steps at which a lease that starts at step ``start`` may be broken."""
    first = start + js_round(p.min_term / L.dt)
    if p.break_every > 0:
        every = max(1, js_round(p.break_every / L.dt))
        return set(range(first, L.n, every)) if first < L.n else set()
    return {first} if first < L.n else set()


class _Values(list):
    """Values per step (a list of row arrays), with the exercise policies when the site may be sold or idled."""

    ex: Optional[list] = None   # active: 1 sell, 2 mothball
    exB: Optional[list] = None  # mothballed: 1 sell, 3 reactivate
    B: Optional[list] = None    # values while mothballed


def _step(L: Lattice, nx: np.ndarray) -> np.ndarray:
    return L.disc * (L.p * nx[1:] + (1 - L.p) * nx[:-1])


def operate(p: SiteParams, L: Lattice, K: float, R0: Optional[list] = None, no_mothball: bool = False,
            rows: Optional[list] = None) -> _Values:
    """Value of operating capacity K from each node to the end of life. With R0 (the value of serving through the 3PL
    alone) the site may also be sold at any step, and mothballed and reactivated. Comparing continuation values first
    means no switch-and-back within one step."""
    sell = R0 is not None and K > 0 and p.salvage_factor > 0
    moth = R0 is not None and K > 0 and bool(p.mothball) and not no_mothball
    opt = sell or moth
    rows = rows or [demand_row(p, L, t) for t in range(L.n + 1)]
    R = _Values([None] * (L.n + 1))
    B: list = [None] * (L.n + 1)
    R[L.n] = np.zeros(L.n + 1)
    B[L.n] = R[L.n]
    if opt:
        R.ex = [None] * (L.n + 1)
        R.exB = [None] * (L.n + 1)
        R.ex[L.n] = np.zeros(L.n + 1, dtype=np.uint8)
        R.exB[L.n] = R.ex[L.n]
    for t in range(L.n - 1, -1, -1):
        D = rows[t]
        hold = cash_rate(p, D, K) * L.dt + _step(L, R[t + 1])
        a = hold.copy()
        c = np.zeros(t + 1, dtype=np.uint8)
        sv = salvage(p, L, K, t) if sell else 0.0
        if moth:
            idle = cash_idle(p, D, K) * L.dt + _step(L, B[t + 1])
            b = idle.copy()
            cb = np.zeros(t + 1, dtype=np.uint8)
            m = idle - p.mothball_cost > a
            a = np.where(m, idle - p.mothball_cost, a); c[m] = 2
            m = hold - p.reactivate_cost > b
            b = np.where(m, hold - p.reactivate_cost, b); cb[m] = 3
            if sell:
                m = sv + R0[t] > b
                b = np.where(m, sv + R0[t], b); cb[m] = 1
            B[t] = b
            R.exB[t] = cb
        if sell:
            m = sv + R0[t] > a
            a = np.where(m, sv + R0[t], a); c[m] = 1
        R[t] = a
        if not moth:
            B[t] = a
        if opt:
            R.ex[t] = c
            if not moth:
                R.exB[t] = np.zeros(t + 1, dtype=np.uint8)
    R.B = B
    return R


def lease_value(p: SiteParams, L: Lattice, K: float, R0: list, breaks: Optional[set], rows: Optional[list] = None) -> _Values:
    """A lease of capacity K: rent on top of operating costs; at break dates it may be handed back for a fee."""
    rows = rows or [demand_row(p, L, t) for t in range(L.n + 1)]
    fee = p.break_fee * p.rent * K
    R = _Values([None] * (L.n + 1))
    R[L.n] = np.zeros(L.n + 1)
    R.ex = [None] * (L.n + 1)
    for t in range(L.n - 1, -1, -1):
        hold = (cash_rate(p, rows[t], K) - p.rent * K) * L.dt + _step(L, R[t + 1])
        ex = np.zeros(t + 1, dtype=np.uint8)
        if breaks and t in breaks:
            m = R0[t] - fee > hold
            hold = np.where(m, R0[t] - fee, hold); ex[m] = 1
        R[t] = hold
        R.ex[t] = ex
    return R


def _weights(L: Lattice, n: int) -> List[float]:
    """Binomial weights for an n-step look-ahead (built the same way as the web app, for identical sums)."""
    w, c = [], 1.0
    for j in range(n + 1):
        w.append(c * math.pow(L.p, j) * math.pow(1 - L.p, n - j))
        c = c * (n - j) / (j + 1)
    return w


def _ahead(X: list, t: int, i: int, n: int, w: List[float], disc_n: float) -> float:
    """Discounted n-step expectation of X from node (t, i)."""
    if n == 0:
        return float(X[t][i])
    row, s = X[t + n], 0.0
    for j in range(n + 1):
        s += w[j] * row[i + j]
    return float(disc_n * s)


def _ahead_row(Xtn: np.ndarray, t: int, n: int, w: List[float], disc_n: float, Xt: Optional[np.ndarray] = None) -> np.ndarray:
    """The same for every node of step t at once (Xtn is row t+n; Xt row t, used when n = 0)."""
    if n == 0:
        return Xt.copy()
    s = np.zeros(t + 1)
    for j in range(n + 1):
        s = s + w[j] * Xtn[j:j + t + 1]
    return disc_n * s


def levels_of(p: SiteParams) -> List[float]:
    """The capacity levels of the staged site: the initial footprint, then steps up to the full footprint."""
    K0 = p.d0 * p.small_factor
    Kbig = max(K0, p.d0 * p.big_factor)
    step = max(p.d0 * p.step_factor, 1e-6)
    lv = [K0]
    while lv[-1] + step < Kbig - 1e-6 and len(lv) < 12:
        lv.append(lv[-1] + step)
    if Kbig - lv[-1] > 1e-6:
        lv.append(Kbig)
    return lv


def _grid(d0: float):
    """The capacity grid of the web app (accumulated, so the same points), then a finer one around the best."""
    K = 0.3 * d0
    while K <= 2.2 * d0 + 1e-9:
        yield K
        K += 0.02 * d0


def _refine(best_k: float, d0: float):
    K = best_k - 0.02 * d0
    while K <= best_k + 0.02 * d0:
        yield K
        K += 0.002 * d0


@dataclass
class Fixed:
    npv: float
    capacity: float
    no_exit: Optional[float] = None       # the same design without the right to sell
    no_mothball: Optional[float] = None   # the same design without the right to mothball


@dataclass
class Lease:
    npv: float
    capacity: float
    no_break: float                       # the same lease without break clauses
    lead_steps: int
    breaks: set


@dataclass
class SiteSolution:
    """The valuation of every strategy, with the policies behind the flexible ones."""

    params: SiteParams
    lattice: Lattice
    levels: List[float]
    full: Fixed
    best: Fixed
    defer: float
    stage: float
    lease: Lease
    # policies and value arrays (per step t, per node i); stage arrays are [level, node]
    defer_build: list = field(repr=False, default=None)
    stage_policy: list = field(repr=False, default=None)
    stage_policy_idle: list = field(repr=False, default=None)
    stage_values: list = field(repr=False, default=None)
    R0: list = field(repr=False, default=None)
    R_full: list = field(repr=False, default=None)
    R_best: list = field(repr=False, default=None)
    lease_values: list = field(repr=False, default=None)

    @property
    def warnings(self) -> List[str]:
        return list(self.lattice.warnings)

    def npvs(self) -> Dict[str, float]:
        """NPV of each strategy, $M."""
        return {"full": self.full.npv, "best": self.best.npv, "defer": self.defer, "stage": self.stage, "lease": self.lease.npv}

    def option_values(self) -> Dict[str, float]:
        """What each kind of flexibility is worth, $M: each flexible strategy against the best fixed design (a fair
        benchmark, not the full footprint), and the exits against the same design or lease without them."""
        out = {"wait": self.defer - self.best.npv, "stage": self.stage - self.best.npv, "lease": self.lease.npv - self.best.npv,
               "break_clauses": self.lease.npv - self.lease.no_break}
        if self.best.no_exit is not None:
            out["sell"] = self.best.npv - self.best.no_exit
        if self.best.no_mothball is not None:
            out["mothball"] = self.best.npv - self.best.no_mothball
        return out

    def defer_triggers(self) -> List[Dict[str, float]]:
        """When waiting turns into building: for each decision step, the lowest demand at which the full footprint is
        built (None where it is never built at that step). Years and demand in the model's units."""
        L, out = self.lattice, []
        for t in range(0, min(L.w, L.n) + 1):
            row = self.defer_build[t]
            idx = np.nonzero(row)[0]
            out.append({"year": t * L.dt, "build_at_demand": float(demand(self.params, L, t, int(idx.min()))) if idx.size else None})
        return out

    def simulate(self, n: int = 1000, seed: int = 7) -> List[dict]:
        """Paths through the lattice under each strategy's optimal policy (seeded, so repeatable)."""
        return simulate(self, n, seed)

    def summary(self) -> Dict[str, object]:
        return {"npv": self.npvs(), "best_capacity": self.best.capacity, "lease_capacity": self.lease.capacity,
                "options": self.option_values(), "lattice_step_years": self.lattice.dt, "warnings": self.warnings}


def solve(params: Optional[SiteParams] = None, **changes) -> SiteSolution:
    """Value every strategy for a site. ``solve()`` uses the defaults; ``solve(sigma=0.4)`` changes some inputs."""
    p = (params or SiteParams()).with_(**changes) if changes else (params or SiteParams())
    L = lattice(p)
    N, W, lv = L.n, L.w, levels_of(p)
    Kbig = lv[-1]
    rows = [demand_row(p, L, t) for t in range(N + 1)]
    wb, we = _weights(L, L.build_steps), _weights(L, L.expand_steps)
    db, de = math.pow(L.disc, L.build_steps), math.pow(L.disc, L.expand_steps)
    R0 = operate(p, L, 0.0, rows=rows)
    Rbig = operate(p, L, Kbig, R0, rows=rows)
    Lb, Le = L.build_steps, L.expand_steps

    def committed0(RK) -> float:
        return float(R0[0][0]) - _ahead(R0, 0, 0, Lb, wb, db) + _ahead(RK, 0, 0, Lb, wb, db)

    # ---- fixed capacity
    def fixed_value(K):
        return committed0(operate(p, L, K, R0, rows=rows)) - build_cost(p, K, 0)

    full = Fixed(npv=fixed_value(Kbig), capacity=Kbig)
    best_npv, best_k = -math.inf, 0.0
    for K in _grid(p.d0):
        v = fixed_value(K)
        if v > best_npv:
            best_npv, best_k = v, K
    for K in list(_refine(best_k, p.d0)):
        v = fixed_value(K)
        if v > best_npv:
            best_npv, best_k = v, K
    Rbest = operate(p, L, best_k, R0, rows=rows)
    best = Fixed(npv=best_npv, capacity=best_k,
                 no_exit=committed0(operate(p, L, best_k, rows=rows)) - build_cost(p, best_k, 0),
                 no_mothball=committed0(operate(p, L, best_k, R0, no_mothball=True, rows=rows)) - build_cost(p, best_k, 0))

    # ---- defer: build the full footprint once inside the window, arriving after the build lead time
    Wd: list = [None] * (N + 1)
    build: list = [None] * (N + 1)
    for t in range(N, -1, -1):
        wait = np.zeros(t + 1) if t == N else cash_rate(p, rows[t], 0.0) * L.dt + _step(L, Wd[t + 1])
        b = np.zeros(t + 1, dtype=np.uint8)
        if t <= W and t + Lb <= N:
            now = R0[t] - _ahead_row(R0[t + Lb], t, Lb, wb, db, R0[t]) + _ahead_row(Rbig[t + Lb], t, Lb, wb, db, Rbig[t]) \
                - build_cost(p, Kbig, t * L.dt)
            m = now >= wait
            wait = np.where(m, now, wait); b[m] = 1
        Wd[t] = wait
        build[t] = b

    # ---- stage: initial footprint now; then expansions by steps, each arriving after the expansion lead time
    nL = len(lv)
    Rk = [operate(p, L, k, rows=rows) for k in lv]
    V: list = [None] * (N + 1)
    Mb: list = [None] * (N + 1)
    pol: list = [None] * (N + 1)
    polM: list = [None] * (N + 1)
    moth_on = bool(p.mothball)
    for t in range(N, -1, -1):
        D = rows[t]
        can_act = t <= W and t + Le <= N
        Vt = np.zeros((nL, t + 1))
        Pt = np.zeros((nL, t + 1), dtype=np.uint8)
        hA = [np.zeros(t + 1) if t == N else cash_rate(p, D, lv[k]) * L.dt + _step(L, V[t + 1][k]) for k in range(nL)]
        hB = [np.zeros(t + 1) if t == N else cash_idle(p, D, lv[k]) * L.dt + _step(L, Mb[t + 1][k]) for k in range(nL)]
        arrive = None
        if can_act and Le > 0:
            arrive = [_ahead_row(V[t + Le][a], t, Le, we, de) for a in range(nL)]
        for k in range(nL):
            best_v = hA[k].copy()
            bk = np.full(t + 1, k, dtype=np.uint8)
            sv = salvage(p, L, lv[k], t)
            if sv > 0:
                m = sv + R0[t] > best_v
                best_v = np.where(m, sv + R0[t], best_v); bk[m] = SELL
            if moth_on and t < N:
                m = hB[k] - p.mothball_cost > best_v
                best_v = np.where(m, hB[k] - p.mothball_cost, best_v); bk[m] = MOTHBALL
            if can_act and Le > 0:
                meanwhile = Rk[k][t] - _ahead_row(Rk[k][t + Le], t, Le, we, de)
                for k2 in range(k + 1, nL):
                    val = meanwhile + arrive[k2] - expand_cost(p, lv[k2] - lv[k])
                    m = val > best_v
                    best_v = np.where(m, val, best_v); bk[m] = k2
            Vt[k] = best_v
            Pt[k] = bk
        # moves to another level within this step (contraction; expansion with no lead time). Every round trip loses
        # money, so repeated relaxation settles in at most nL rounds.
        contract, instant = bool(p.contract) and t < N, can_act and Le == 0
        rnd = 0
        while (contract or instant) and rnd <= nL:
            changed = False
            for k in range(nL):
                for k2 in range(nL):
                    if k2 == k:
                        continue
                    if k2 < k:
                        if not contract:
                            continue
                        cand = Vt[k2] + contract_proceeds(p, L, lv[k] - lv[k2], t)
                    else:
                        if not instant:
                            continue
                        cand = Vt[k2] - expand_cost(p, lv[k2] - lv[k])
                    m = cand > Vt[k] + 1e-12
                    if m.any():
                        Vt[k] = np.where(m, cand, Vt[k]); Pt[k][m] = k2; changed = True
            if not changed:
                break
            rnd += 1
        Mt = np.zeros((nL, t + 1))
        PMt = np.zeros((nL, t + 1), dtype=np.uint8)
        for k in range(nL):
            mb = hB[k].copy()
            mk = np.zeros(t + 1, dtype=np.uint8)
            if moth_on:
                if t < N:
                    m = Vt[k] - p.reactivate_cost > mb
                    mb = np.where(m, Vt[k] - p.reactivate_cost, mb); mk[m] = 1
                sv = salvage(p, L, lv[k], t)
                if sv > 0:
                    m = sv + R0[t] > mb
                    mb = np.where(m, sv + R0[t], mb); mk[m] = SELL
            Mt[k] = mb
            PMt[k] = mk
        V[t], Mb[t], pol[t], polM[t] = Vt, Mt, Pt, PMt
    stage_v0 = [V[t][0] for t in range(N + 1)]
    stage_npv = float(R0[0][0]) - _ahead(R0, 0, 0, Lb, wb, db) + _ahead(stage_v0, 0, 0, Lb, wb, db) - build_cost(p, lv[0], 0)

    # ---- lease: capacity chosen to maximise NPV; fit-out takes lease_lead, and its cost is paid now
    Ll = min(N, js_round(p.lease_lead / L.dt))
    wl, dl = _weights(L, Ll), math.pow(L.disc, Ll)
    brk = break_steps(p, L, Ll)

    def lease_npv(K, b):
        RL = lease_value(p, L, K, R0, b, rows=rows)
        return float(R0[0][0]) - _ahead(R0, 0, 0, Ll, wl, dl) + _ahead(RL, 0, 0, Ll, wl, dl) - p.lease_fit * p.fixed_cost, RL

    l_npv, l_k = -math.inf, 0.0
    for K in _grid(p.d0):
        v = lease_npv(K, brk)[0]
        if v > l_npv:
            l_npv, l_k = v, K
    for K in list(_refine(l_k, p.d0)):
        v = lease_npv(K, brk)[0]
        if v > l_npv:
            l_npv, l_k = v, K
    _, RL = lease_npv(l_k, brk)
    lease = Lease(npv=l_npv, capacity=l_k, no_break=lease_npv(l_k, None)[0], lead_steps=Ll, breaks=brk)

    return SiteSolution(params=p, lattice=L, levels=lv, full=full, best=best, defer=float(Wd[0][0]), stage=stage_npv, lease=lease,
                        defer_build=build, stage_policy=pol, stage_policy_idle=polM, stage_values=V, R0=R0, R_full=Rbig,
                        R_best=Rbest, lease_values=RL)


# ---- single-design helpers (fair comparisons and tests)
def fixed_npv(params: Optional[SiteParams] = None, capacity: float = 500.0, **changes) -> float:
    """NPV of building ``capacity`` now and keeping it (with the exits the inputs allow)."""
    p = (params or SiteParams()).with_(**changes) if changes else (params or SiteParams())
    L = lattice(p)
    rows = [demand_row(p, L, t) for t in range(L.n + 1)]
    R0 = operate(p, L, 0.0, rows=rows)
    RK = operate(p, L, capacity, R0, rows=rows)
    w, dN = _weights(L, L.build_steps), math.pow(L.disc, L.build_steps)
    return float(R0[0][0]) - _ahead(R0, 0, 0, L.build_steps, w, dN) + _ahead(RK, 0, 0, L.build_steps, w, dN) - build_cost(p, capacity, 0)


def lease_npv(params: Optional[SiteParams] = None, capacity: float = 500.0, with_breaks: bool = True, **changes) -> float:
    """NPV of leasing ``capacity`` now, with or without its break clauses."""
    p = (params or SiteParams()).with_(**changes) if changes else (params or SiteParams())
    L = lattice(p)
    rows = [demand_row(p, L, t) for t in range(L.n + 1)]
    R0 = operate(p, L, 0.0, rows=rows)
    Ll = min(L.n, js_round(p.lease_lead / L.dt))
    w, dN = _weights(L, Ll), math.pow(L.disc, Ll)
    RL = lease_value(p, L, capacity, R0, break_steps(p, L, Ll) if with_breaks else None, rows=rows)
    return float(R0[0][0]) - _ahead(R0, 0, 0, Ll, w, dN) + _ahead(RL, 0, 0, Ll, w, dN) - p.lease_fit * p.fixed_cost


def european_call(s0: float, strike: float, sigma: float, r: float, T: float, dt: float) -> float:
    """A European call priced on the same lattice machinery (used to check the lattice against Black–Scholes)."""
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    pu = (math.exp(r * dt) - d) / (u - d)
    n = js_round(T / dt)
    disc = math.exp(-r * dt)
    i = np.arange(n + 1)
    v = np.maximum(0.0, s0 * np.power(u, 2 * i - n) - strike)
    for _ in range(n - 1, -1, -1):
        v = disc * (pu * v[1:] + (1 - pu) * v[:-1])
    return float(v[0])


# ---- simulation along the optimal policies
def simulate(sol: SiteSolution, n: int = 1000, seed: int = 7) -> List[dict]:
    """Paths through the lattice under each strategy's optimal policy. Each path's discounted NPVs average to the
    lattice values (within sampling error); events say when sites were built, sold or mothballed."""
    p, L, lv, N = sol.params, sol.lattice, sol.levels, sol.lattice.n
    rng, out = XorShift32(seed), []
    lead = sol.lease.lead_steps
    for _ in range(n):
        i, df, iW, sumD = 0, 1.0, 0, 0.0
        S = {
            "full": dict(K=0.0, pend=sol.full.capacity, at=L.build_steps, npv=-build_cost(p, sol.full.capacity, 0), R=sol.R_full, sold=None, idle=False, mothballs=0),
            "best": dict(K=0.0, pend=sol.best.capacity, at=L.build_steps, npv=-build_cost(p, sol.best.capacity, 0), R=sol.R_best, sold=None, idle=False, mothballs=0),
            "defer": dict(K=0.0, pend=None, at=-1, npv=0.0, built=False, build_year=None, R=sol.R_full, sold=None, idle=False, mothballs=0),
            "stage": dict(k=-1, pend=0, at=L.build_steps, npv=-build_cost(p, lv[0], 0), expansions=0, first_expansion=None, sold=None, idle=False, mothballs=0, contractions=0),
            "lease": dict(K=0.0, pend=sol.lease.capacity, at=lead, npv=-p.lease_fit * p.fixed_cost, sold=None),
        }
        events = []

        def act(x, t, i, df):
            R = x["R"]
            if R.ex is None or x["K"] <= 0 or x["pend"] is not None:
                return
            c = R.exB[t][i] if x["idle"] else R.ex[t][i]
            if c == 1:
                x["npv"] += df * salvage(p, L, x["K"], t); x["K"] = 0.0; x["sold"] = t * L.dt; x["idle"] = False
            elif c == 2:
                x["npv"] -= df * p.mothball_cost; x["idle"] = True; x["mothballs"] += 1
            elif c == 3:
                x["npv"] -= df * p.reactivate_cost; x["idle"] = False

        def cash(x, D):
            return float(cash_idle(p, D, x["K"]) if x["idle"] else cash_rate(p, D, x["K"]))

        for t in range(N):
            D = demand(p, L, t, i)
            if t == L.w:
                iW = i
            sumD += D
            for s in ("full", "best"):
                x = S[s]
                if x["pend"] is not None and t >= x["at"]:
                    x["K"] = x["pend"]; x["pend"] = None
                act(x, t, i, df)
                x["npv"] += df * cash(x, D) * L.dt
            f = S["defer"]
            if f["pend"] is not None and t >= f["at"]:
                f["K"] = f["pend"]; f["pend"] = None
            if not f["built"] and sol.defer_build[t][i]:
                f["built"] = True; f["pend"] = sol.full.capacity; f["at"] = t + L.build_steps
                f["npv"] -= df * build_cost(p, sol.full.capacity, t * L.dt); f["build_year"] = t * L.dt; events.append(("defer", t, i))
            act(f, t, i, df)
            f["npv"] += df * cash(f, D) * L.dt
            g = S["stage"]
            if g["pend"] is not None and t >= g["at"]:
                g["k"] = g["pend"]; g["pend"] = None
            if g["k"] >= 0 and g["pend"] is None:
                go = True
                if g["idle"]:
                    cm = sol.stage_policy_idle[t][g["k"]][i]
                    if cm == SELL:
                        g["npv"] += df * salvage(p, L, lv[g["k"]], t); g["k"] = -2; g["sold"] = t * L.dt; g["idle"] = False; go = False
                    elif cm == 1:
                        g["npv"] -= df * p.reactivate_cost; g["idle"] = False
                    else:
                        go = False
                guard = 0
                while go and guard <= len(lv) + 1:
                    guard += 1
                    k2 = int(sol.stage_policy[t][g["k"]][i])
                    if k2 == g["k"]:
                        break
                    if k2 == SELL:
                        g["npv"] += df * salvage(p, L, lv[g["k"]], t); g["k"] = -2; g["sold"] = t * L.dt; break
                    if k2 == MOTHBALL:
                        g["npv"] -= df * p.mothball_cost; g["idle"] = True; g["mothballs"] += 1; break
                    if k2 < g["k"]:
                        g["npv"] += df * contract_proceeds(p, L, lv[g["k"]] - lv[k2], t); g["k"] = k2; g["contractions"] += 1; continue
                    g["npv"] -= df * expand_cost(p, lv[k2] - lv[g["k"]]); g["expansions"] += 1
                    if g["first_expansion"] is None:
                        g["first_expansion"] = t * L.dt
                    events.append(("stage", t, i))
                    if L.expand_steps == 0:
                        g["k"] = k2; continue
                    g["pend"] = k2; g["at"] = t + L.expand_steps; break
            if g["k"] >= 0:
                cg = cash_idle(p, D, lv[g["k"]]) if g["idle"] else cash_rate(p, D, lv[g["k"]])
            else:
                cg = cash_rate(p, D, 0.0)
            g["npv"] += df * float(cg) * L.dt
            e = S["lease"]
            if e["pend"] is not None and t >= e["at"]:
                e["K"] = e["pend"]; e["pend"] = None
            if e["K"] > 0 and t in sol.lease.breaks and sol.lease_values.ex[t][i]:
                e["npv"] -= df * p.break_fee * p.rent * e["K"]; e["K"] = 0.0; e["sold"] = t * L.dt
            e["npv"] += df * (float(cash_rate(p, D, e["K"])) - p.rent * e["K"]) * L.dt
            if rng.next() < L.p:
                i += 1
            df *= L.disc
        if L.w >= N:
            iW = i
        endD = demand(p, L, N, i)
        out.append({"window_demand": demand(p, L, L.w, iW), "mean_demand": sumD / N,
                    "full": S["full"]["npv"], "best": S["best"]["npv"], "defer": S["defer"]["npv"], "stage": S["stage"]["npv"], "lease": S["lease"]["npv"],
                    "defer_year": S["defer"]["build_year"],
                    "sold": {k: S[k]["sold"] for k in ("full", "best", "defer", "stage", "lease")},
                    "mothballs": {k: S[k]["mothballs"] for k in ("full", "best", "defer", "stage")},
                    "contractions": S["stage"]["contractions"], "expansions": S["stage"]["expansions"],
                    "first_expansion": S["stage"]["first_expansion"], "end_demand": endD,
                    "growth": math.pow(endD / p.d0, 1 / p.life) - 1, "events": events})
    return out


def params_dict(p: SiteParams) -> dict:
    return asdict(p)
