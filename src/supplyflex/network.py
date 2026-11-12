"""Real options for a two-region network: one hub or two, now or later.

Two regions, A and B, each with demand on its own recombining binomial lattice; the two are correlated, so every step
has four branches (both up, A up and B down, A down and B up, both down). A hub serves its own region first; its spare
capacity serves the other region at a cross-region transport cost; a 3PL covers what is left up to a limit in each
region; the rest is lost with a penalty. A second hub costs less once the first exists (shared systems, people,
supplier terms): the compound option, where opening one hub creates a cheaper option on the next.

Strategies:

* ``central`` — one hub in region A, built now, sized to maximise NPV (it serves B across regions)
* ``two``     — a hub in each region, built now, sizes chosen together to maximise NPV
* ``both``    — a hub in each region, built now, at the planned sizes
* ``pilot``   — a hub at the planned size in one region now (whichever is worth more), and the option to open the other
  region's hub at any time inside the decision window
* ``wait``    — open either hub, or both, at the best moments inside the window

Quarterly steps (a two-dimensional lattice grows with the square of the steps). A growth too strong for its volatility,
or a correlation the four branches cannot carry, is limited and reported in ``warnings``.

Units: money in $M, demand and capacity in thousand units per year, margins in $M per thousand units.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from ._common import XorShift32, check_inputs, js_round


@dataclass(frozen=True)
class NetworkParams:
    """Inputs of the two-region model. Defaults are the web app's."""

    demand_a: float = 500.0
    demand_b: float = 250.0
    sigma_a: float = 0.2
    sigma_b: float = 0.4
    growth_a: float = 0.02
    growth_b: float = 0.08
    rho: float = 0.2             # correlation of the two regions' demand
    ceiling: float = 3.0         # each market saturates at ceiling × its demand today
    life: float = 20.0
    window: float = 8.0
    build_lead: float = 1.5
    margin: float = 0.020
    op_cost: float = 0.006
    out_margin: float = 0.006
    out_cap: float = 1.0
    penalty: float = 0.010
    cross_cost: float = 0.006    # transport cost of serving the other region, $M per thousand units
    basis: str = "wacc"
    wacc: float = 0.10
    risk_free: float = 0.03
    risk_price: float = 0.3
    fixed_cost: float = 8.0
    var_cost: float = 0.030
    escalation: float = 0.06
    learning: float = 0.4        # share of the fixed cost saved on the second hub
    size_a: float = 1.2          # planned hub sizes, × each region's demand today
    size_b: float = 1.4

    def __post_init__(self) -> None:
        extra = []
        if self.basis not in ("wacc", "rn"):
            extra.append(f"basis = {self.basis!r} must be 'wacc' (expected cash flows at the WACC) or 'rn' (risk-neutral)")
        if isinstance(self.life, numbers.Real) and isinstance(self.window, numbers.Real) and self.window > self.life:
            extra.append(f"window = {self.window!r} cannot be longer than life = {self.life!r}")
        if isinstance(self.life, numbers.Real) and isinstance(self.build_lead, numbers.Real) and self.build_lead >= self.life:
            extra.append(f"build_lead = {self.build_lead!r} must be shorter than life = {self.life!r}")
        check_inputs(self, _NET_RANGES, extra, "NetworkParams")

    def with_(self, **changes) -> "NetworkParams":
        """A copy with some inputs changed."""
        return replace(self, **changes)


_NET_RANGES = {
    "demand_a": ("(0, inf)", "region A's demand today, thousand units per year"),
    "demand_b": ("(0, inf)", "region B's demand today, thousand units per year; use a tiny value such as 1e-9 for an empty region"),
    "sigma_a": ("(0, inf)", "volatility of region A's demand per year"),
    "sigma_b": ("(0, inf)", "volatility of region B's demand per year"),
    "growth_a": ("[-1, 1]", "expected growth of region A's demand per year"),
    "growth_b": ("[-1, 1]", "expected growth of region B's demand per year"),
    "rho": ("[-1, 1]", "correlation of the two regions' demand"),
    "ceiling": ("[1, inf)", "market saturation as a multiple of today's demand"),
    "life": ("(0, 100]", "asset life in years"),
    "window": ("[0, inf)", "years during which hubs can be opened"),
    "build_lead": ("[0, inf)", "years from deciding to build until a hub opens"),
    "margin": ("[0, inf)", "$M per thousand units"),
    "op_cost": ("[0, inf)", "$M per thousand units of capacity per year"),
    "out_margin": ("[0, inf)", "3PL margin, $M per thousand units"),
    "out_cap": ("[0, inf)", "3PL limit in each region as a multiple of its demand today"),
    "penalty": ("[0, inf)", "$M per thousand units of lost sales"),
    "cross_cost": ("[0, inf)", "transport cost to serve the other region, $M per thousand units"),
    "wacc": ("[0, 1)", "discount rate per year, e.g. 0.10"),
    "risk_free": ("(-1, 1)", "risk-free rate per year"),
    "risk_price": ("(-inf, inf)", "market price of demand risk (lambda)"),
    "fixed_cost": ("[0, inf)", "fixed cost of a hub, $M"),
    "var_cost": ("[0, inf)", "$M per thousand units of capacity"),
    "escalation": ("(-1, 1)", "yearly escalation of a deferred hub's cost"),
    "learning": ("[0, 1]", "share of the fixed cost saved on the second hub"),
    "size_a": ("(0, inf)", "planned hub size in region A as a multiple of its demand today"),
    "size_b": ("(0, inf)", "planned hub size in region B as a multiple of its demand today"),
}


@dataclass
class Region:
    growth: float
    u: float
    d: float
    p: float


@dataclass
class JointLattice:
    dt: float
    n: int
    w: int
    build_steps: int
    rate: float
    disc: float
    a: Region
    b: Region
    q: List[float]   # branch probabilities: both up, A up only, B up only, both down
    rho: float       # the correlation actually used
    warnings: List[str] = field(default_factory=list)


def lattice(p: NetworkParams) -> JointLattice:
    dt, warnings, rn = 0.25, [], p.basis == "rn"

    def region(name, g, s):
        g0 = g - p.risk_price * s if rn else g
        lim = 0.85 * s / math.sqrt(dt)
        ge = max(-lim, min(lim, g0))
        if abs(ge - g0) > 1e-12:
            warnings.append(f"Region {name}: growth of {g0 * 100:.1f}%/yr is too strong for a volatility of {s * 100:.0f}% "
                            f"on this lattice; {ge * 100:.1f}%/yr is used. Raise the volatility to use your growth.")
        u = math.exp(s * math.sqrt(dt))
        d = 1 / u
        return Region(growth=ge, u=u, d=d, p=(math.exp(ge * dt) - d) / (u - d))

    A, B = region("A", p.growth_a, p.sigma_a), region("B", p.growth_b, p.sigma_b)
    sd = math.sqrt(A.p * (1 - A.p) * B.p * (1 - B.p))
    c = p.rho * sd
    lo = max(-A.p * B.p, -(1 - A.p) * (1 - B.p))
    hi = min(A.p * (1 - B.p), (1 - A.p) * B.p)
    ce = max(lo, min(hi, c))
    if abs(ce - c) > 1e-12:
        warnings.append(f"A correlation of {p.rho:.2f} cannot be represented with these growth rates; {ce / sd:.2f} is used.")
    q = [A.p * B.p + ce, A.p * (1 - B.p) - ce, (1 - A.p) * B.p - ce, (1 - A.p) * (1 - B.p) + ce]
    n = js_round(p.life / dt)
    r = p.risk_free if rn else p.wacc
    return JointLattice(dt=dt, n=n, w=min(n, js_round(p.window / dt)), build_steps=js_round(p.build_lead / dt), rate=r,
                        disc=math.exp(-r * dt), a=A, b=B, q=q, rho=ce / sd, warnings=warnings)


def demands(p: NetworkParams, L: JointLattice):
    """Demand in each region at every step (rows of i up-moves), capped at the market ceiling."""
    a, b = [], []
    for t in range(L.n + 1):
        i = np.arange(t + 1)
        a.append(np.minimum(p.demand_a * np.power(L.a.u, 2 * i - t), p.ceiling * p.demand_a))
        b.append(np.minimum(p.demand_b * np.power(L.b.u, 2 * i - t), p.ceiling * p.demand_b))
    return a, b


def cash(p: NetworkParams, DA, DB, KA: float, KB: float):
    """Cash flow per year: own sales, cross-region service from spare capacity, the 3PL, lost sales, holding cost.
    Cross-region service is used before the 3PL when it earns more; after it otherwise. DA and DB broadcast."""
    DA, DB = np.asarray(DA, dtype=float), np.asarray(DB, dtype=float)
    ownA, ownB = np.minimum(DA, KA), np.minimum(DB, KB)
    spA, spB, shA, shB = KA - ownA, KB - ownB, DA - ownA, DB - ownB
    xm = p.margin - p.cross_cost
    capA, capB = p.out_cap * p.demand_a, p.out_cap * p.demand_b
    x1 = x2 = 0.0
    if xm >= p.out_margin:
        x1 = np.minimum(spA, shB); x2 = np.minimum(spB, shA); shB = shB - x1; shA = shA - x2
        oA = np.minimum(shA, capA); oB = np.minimum(shB, capB); shA = shA - oA; shB = shB - oB
    else:
        oA = np.minimum(shA, capA); oB = np.minimum(shB, capB); shA = shA - oA; shB = shB - oB
        if xm > -p.penalty:
            x1 = np.minimum(spA, shB); x2 = np.minimum(spB, shA); shB = shB - x1; shA = shA - x2
    return p.margin * (ownA + ownB) + xm * (x1 + x2) + p.out_margin * (oA + oB) - p.penalty * (shA + shB) - p.op_cost * (KA + KB)


def hub_cost(p: NetworkParams, K: float, years: float, second: bool) -> float:
    """What a hub costs, built ``years`` from now; a second hub's fixed cost is reduced by the learning share."""
    return 0.0 if K <= 0 else (p.fixed_cost * ((1 - p.learning) if second else 1) + p.var_cost * K) * math.pow(1 + p.escalation, years)


def pair_cost(p: NetworkParams, KA: float, KB: float) -> float:
    if KA > 0 and KB > 0:
        return hub_cost(p, KA, 0, False) + hub_cost(p, KB, 0, True)
    return hub_cost(p, KA, 0, False) + hub_cost(p, KB, 0, False)


def _step(L: JointLattice, nx: np.ndarray) -> np.ndarray:
    q = L.q
    return L.disc * (q[0] * nx[1:, 1:] + q[1] * nx[1:, :-1] + q[2] * nx[:-1, 1:] + q[3] * nx[:-1, :-1])


def operate(p: NetworkParams, L: JointLattice, Dm, KA: float, KB: float) -> list:
    """R[t][i, j]: value of operating capacities KA and KB from node (t, i, j) to the end of life, no decisions."""
    a, b = Dm
    R: list = [None] * (L.n + 1)
    R[L.n] = np.zeros((L.n + 1, L.n + 1))
    for t in range(L.n - 1, -1, -1):
        R[t] = cash(p, a[t][:, None], b[t][None, :], KA, KB) * L.dt + _step(L, R[t + 1])
    return R


def weights2(L: JointLattice, n: int) -> np.ndarray:
    """Joint n-step weights, an (n+1)×(n+1) matrix."""
    w = np.ones((1, 1))
    for k in range(n):
        nw = np.zeros((k + 2, k + 2))
        nw[1:, 1:] += w * L.q[0]
        nw[1:, :-1] += w * L.q[1]
        nw[:-1, 1:] += w * L.q[2]
        nw[:-1, :-1] += w * L.q[3]
        w = nw
    return w


def ahead2(X: list, t: int, n: int, w: np.ndarray, disc_n: float) -> np.ndarray:
    """Discounted n-step expectation of X from every node of step t."""
    if n == 0:
        return X[t].copy()
    row, s = X[t + n], np.zeros((t + 1, t + 1))
    for a in range(n + 1):
        for b in range(n + 1):
            s = s + w[a, b] * row[a:a + t + 1, b:b + t + 1]
    return disc_n * s


def pmf(L: JointLattice) -> list:
    """Probability of every node (forward), for valuing fixed designs as plain expected discounted sums."""
    pm = [np.ones((1, 1))]
    for t in range(L.n):
        cur, nx = pm[t], np.zeros((t + 2, t + 2))
        nx[1:, 1:] += cur * L.q[0]
        nx[1:, :-1] += cur * L.q[1]
        nx[:-1, 1:] += cur * L.q[2]
        nx[:-1, :-1] += cur * L.q[3]
        pm.append(nx)
    return pm


def _expected_cash(p, Dm, pm, t, KA, KB) -> float:
    a, b = Dm
    return float(np.sum(pm[t] * cash(p, a[t][:, None], b[t][None, :], KA, KB)))


def _fixed_value(p, L, Dm, pm, before, KA, KB) -> float:
    """NPV of building KA and KB now and keeping them: the 3PL serves until they arrive."""
    v, df = before - pair_cost(p, KA, KB), math.pow(L.disc, L.build_steps)
    for t in range(L.build_steps, L.n):
        v += df * _expected_cash(p, Dm, pm, t, KA, KB) * L.dt
        df *= L.disc
    return v


def _before_arrival(p, L, Dm, pm) -> float:
    v, df = 0.0, 1.0
    for t in range(min(L.build_steps, L.n)):
        v += df * _expected_cash(p, Dm, pm, t, 0.0, 0.0) * L.dt
        df *= L.disc
    return v


def maximise(f, lo: float, hi: float):
    """Maximise f over [lo, hi]: a coarse scan, then golden-section search around the best point. Returns (x, value)."""
    n, bx, bv = 9, lo, -math.inf
    for k in range(n + 1):
        x = lo + (hi - lo) * k / n
        v = f(x)
        if v > bv:
            bv, bx = v, x
    h = (hi - lo) / n
    a, b, gr = max(lo, bx - h), min(hi, bx + h), (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(14):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d)
    xm, vm = (c if fc > fd else d), max(fc, fd)
    return (xm, vm) if vm >= bv else (bx, bv)


@dataclass
class Design:
    npv: float
    capacity_a: float
    capacity_b: float


@dataclass
class Pilot:
    npv: float
    region: str      # where the first hub goes
    other: float     # the pilot's value had it started in the other region


@dataclass
class NetworkSolution:
    params: NetworkParams
    lattice: JointLattice
    central: Design
    two: Design
    both: Design
    pilot: Pilot
    wait: float
    wait_action: list = field(repr=False, default=None)   # per step: 0 wait, 1 open A, 2 open B, 3 open both
    second_a: dict = field(repr=False, default=None)      # with hub A open: the option on B (values V, policy go)
    second_b: dict = field(repr=False, default=None)
    demand: tuple = field(repr=False, default=None)

    @property
    def warnings(self) -> List[str]:
        return list(self.lattice.warnings)

    def npvs(self) -> Dict[str, float]:
        return {"central": self.central.npv, "two": self.two.npv, "both": self.both.npv, "pilot": self.pilot.npv, "wait": self.wait}

    def option_values(self) -> Dict[str, float]:
        """Flexibility against the best design built now (central or two hubs, whichever is worth more)."""
        fixed = max(self.central.npv, self.two.npv)
        return {"pilot": self.pilot.npv - fixed, "wait": self.wait - fixed}

    def simulate(self, n: int = 1000, seed: int = 7) -> List[dict]:
        return simulate(self, n, seed)


def solve(params: Optional[NetworkParams] = None, **changes) -> NetworkSolution:
    """Value every network strategy. ``solve()`` uses the defaults; ``solve(rho=0.8)`` changes some inputs."""
    p = (params or NetworkParams()).with_(**changes) if changes else (params or NetworkParams())
    L = lattice(p)
    N, W, Lb = L.n, L.w, L.build_steps
    Dm, pm = demands(p, L), pmf(L)
    before = _before_arrival(p, L, Dm, pm)

    def fixed(KA, KB):
        return _fixed_value(p, L, Dm, pm, before, KA, KB)

    Dt = p.demand_a + p.demand_b
    cx, cv = maximise(lambda K: fixed(K, 0.0), 0.2 * Dt, 2.4 * Dt)
    central = Design(npv=cv, capacity_a=cx, capacity_b=0.0)
    KA, KB, two = p.demand_a, p.demand_b, None
    for _ in range(2):
        KA, _va = maximise(lambda K: fixed(K, KB), 0.1 * p.demand_a, 2.6 * p.demand_a)
        KB, vb = maximise(lambda K: fixed(KA, K), 0.1 * p.demand_b, 2.6 * p.demand_b)
        two = Design(npv=vb, capacity_a=KA, capacity_b=KB)
    KAo, KBo = p.size_a * p.demand_a, p.size_b * p.demand_b
    both = Design(npv=fixed(KAo, KBo), capacity_a=KAo, capacity_b=KBo)

    R0, RA, RB, RAB = (operate(p, L, Dm, ka, kb) for ka, kb in ((0, 0), (KAo, 0), (0, KBo), (KAo, KBo)))
    w, dN = weights2(L, Lb), math.pow(L.disc, Lb)

    def second(Rhave, Rnext, Kadd):
        """With one hub open, the option to open the other (second hub: learning applies)."""
        V: list = [None] * (N + 1)
        go: list = [None] * (N + 1)
        V[N] = np.zeros((N + 1, N + 1))
        go[N] = np.zeros((N + 1, N + 1), dtype=np.uint8)
        for t in range(N - 1, -1, -1):
            hold = Rhave[t] - _step(L, Rhave[t + 1]) + _step(L, V[t + 1])
            g = np.zeros((t + 1, t + 1), dtype=np.uint8)
            if t <= W and t + Lb <= N:
                now = Rhave[t] - ahead2(Rhave, t, Lb, w, dN) + ahead2(Rnext, t, Lb, w, dN) - hub_cost(p, Kadd, t * L.dt, True)
                m = now > hold
                hold = np.where(m, now, hold); g[m] = 1
            V[t], go[t] = hold, g
        return {"V": V, "go": go}

    VA, VB = second(RA, RAB, KBo), second(RB, RAB, KAo)
    a, b = Dm
    V0: list = [None] * (N + 1)
    act: list = [None] * (N + 1)
    V0[N] = np.zeros((N + 1, N + 1))
    act[N] = np.zeros((N + 1, N + 1), dtype=np.uint8)
    for t in range(N - 1, -1, -1):
        best = cash(p, a[t][:, None], b[t][None, :], 0.0, 0.0) * L.dt + _step(L, V0[t + 1])
        code = np.zeros((t + 1, t + 1), dtype=np.uint8)
        if t <= W and t + Lb <= N:
            yrs = t * L.dt
            base = R0[t] - ahead2(R0, t, Lb, w, dN)
            oa = base + ahead2(VA["V"], t, Lb, w, dN) - hub_cost(p, KAo, yrs, False)
            ob = base + ahead2(VB["V"], t, Lb, w, dN) - hub_cost(p, KBo, yrs, False)
            oab = base + ahead2(RAB, t, Lb, w, dN) - hub_cost(p, KAo, yrs, False) - hub_cost(p, KBo, yrs, True)
            for val, c in ((oa, 1), (ob, 2), (oab, 3)):
                m = val > best
                best = np.where(m, val, best); code[m] = c
        V0[t], act[t] = best, code
    base0 = float(R0[0][0, 0]) - float(ahead2(R0, 0, Lb, w, dN)[0, 0])
    pilot_a = base0 + float(ahead2(VA["V"], 0, Lb, w, dN)[0, 0]) - hub_cost(p, KAo, 0, False)
    pilot_b = base0 + float(ahead2(VB["V"], 0, Lb, w, dN)[0, 0]) - hub_cost(p, KBo, 0, False)
    pilot = Pilot(npv=pilot_a, region="A", other=pilot_b) if pilot_a >= pilot_b else Pilot(npv=pilot_b, region="B", other=pilot_a)
    return NetworkSolution(params=p, lattice=L, central=central, two=two, both=both, pilot=pilot, wait=float(V0[0][0, 0]),
                           wait_action=act, second_a=VA, second_b=VB, demand=Dm)


def fixed_npv(params: Optional[NetworkParams] = None, capacity_a: float = 600.0, capacity_b: float = 350.0, **changes) -> float:
    """NPV of building both capacities now and keeping them, as an expected discounted sum."""
    p = (params or NetworkParams()).with_(**changes) if changes else (params or NetworkParams())
    L = lattice(p)
    Dm, pm = demands(p, L), pmf(L)
    return _fixed_value(p, L, Dm, pm, _before_arrival(p, L, Dm, pm), capacity_a, capacity_b)


def fixed_npv_backward(params: Optional[NetworkParams] = None, capacity_a: float = 600.0, capacity_b: float = 350.0, **changes) -> float:
    """The same by backward induction (a check on the expected sum)."""
    p = (params or NetworkParams()).with_(**changes) if changes else (params or NetworkParams())
    L = lattice(p)
    Dm = demands(p, L)
    R0, RK = operate(p, L, Dm, 0.0, 0.0), operate(p, L, Dm, capacity_a, capacity_b)
    w, dN = weights2(L, L.build_steps), math.pow(L.disc, L.build_steps)
    return float(R0[0][0, 0]) - float(ahead2(R0, 0, L.build_steps, w, dN)[0, 0]) + float(ahead2(RK, 0, L.build_steps, w, dN)[0, 0]) \
        - pair_cost(p, capacity_a, capacity_b)


def simulate(sol: NetworkSolution, n: int = 1000, seed: int = 7) -> List[dict]:
    """Paths through the joint lattice under each strategy's optimal policy (seeded, so repeatable)."""
    p, L, (Da, Db) = sol.params, sol.lattice, sol.demand
    N, Lb, q = L.n, L.build_steps, L.q
    KAo, KBo = sol.both.capacity_a, sol.both.capacity_b
    rng, out = XorShift32(seed), []

    def cf(DA, DB, KA, KB):
        return float(cash(p, DA, DB, KA, KB))

    for _ in range(n):
        i = j = 0
        df, sumA, sumB = 1.0, 0.0, 0.0

        def fx(KA, KB):
            return dict(KA=0.0, KB=0.0, tA=KA, tB=KB, npv=-pair_cost(p, KA, KB))

        S = {"central": fx(sol.central.capacity_a, 0.0), "two": fx(sol.two.capacity_a, sol.two.capacity_b), "both": fx(KAo, KBo)}
        pa = sol.pilot.region == "A"
        P = dict(KA=0.0, KB=0.0, pend=[[KAo if pa else 0.0, 0.0 if pa else KBo, Lb]],
                 npv=-(hub_cost(p, KAo, 0, False) if pa else hub_cost(p, KBo, 0, False)), second=None, done=False)
        Wt = dict(KA=0.0, KB=0.0, pend=[], npv=0.0, first=None, first_year=None, second=None, state=0)
        for t in range(N):
            DA, DB, yrs = float(Da[t][i]), float(Db[t][j]), t * L.dt
            sumA += DA; sumB += DB
            for x in S.values():
                if t == Lb:
                    x["KA"], x["KB"] = x["tA"], x["tB"]
                x["npv"] += df * cf(DA, DB, x["KA"], x["KB"]) * L.dt
            for x in (P, Wt):
                keep = []
                for pr in x["pend"]:
                    if t >= pr[2]:
                        x["KA"] += pr[0]; x["KB"] += pr[1]
                    else:
                        keep.append(pr)
                x["pend"] = keep
            if not P["done"] and not P["pend"] and (P["KA"] > 0 if pa else P["KB"] > 0):
                VS = sol.second_a if pa else sol.second_b
                if VS["go"][t][i, j]:
                    P["pend"].append([0.0 if pa else KAo, KBo if pa else 0.0, t + Lb])
                    P["npv"] -= df * hub_cost(p, KBo if pa else KAo, yrs, True); P["second"] = yrs; P["done"] = True
            P["npv"] += df * cf(DA, DB, P["KA"], P["KB"]) * L.dt
            if not Wt["pend"]:
                if Wt["state"] == 0:
                    c = sol.wait_action[t][i, j]
                    if c == 1:
                        Wt["pend"].append([KAo, 0.0, t + Lb]); Wt["npv"] -= df * hub_cost(p, KAo, yrs, False); Wt["state"] = 1; Wt["first"] = "A"; Wt["first_year"] = yrs
                    elif c == 2:
                        Wt["pend"].append([0.0, KBo, t + Lb]); Wt["npv"] -= df * hub_cost(p, KBo, yrs, False); Wt["state"] = 2; Wt["first"] = "B"; Wt["first_year"] = yrs
                    elif c == 3:
                        Wt["pend"].append([KAo, KBo, t + Lb]); Wt["npv"] -= df * (hub_cost(p, KAo, yrs, False) + hub_cost(p, KBo, yrs, True))
                        Wt["state"] = 3; Wt["first"] = "both"; Wt["first_year"] = yrs; Wt["second"] = yrs
                elif Wt["state"] in (1, 2):
                    V2 = sol.second_a if Wt["state"] == 1 else sol.second_b
                    if V2["go"][t][i, j]:
                        addA = Wt["state"] == 2
                        Wt["pend"].append([KAo if addA else 0.0, 0.0 if addA else KBo, t + Lb])
                        Wt["npv"] -= df * hub_cost(p, KAo if addA else KBo, yrs, True); Wt["state"] = 3; Wt["second"] = yrs
            Wt["npv"] += df * cf(DA, DB, Wt["KA"], Wt["KB"]) * L.dt
            r = rng.next()
            if r < q[0]:
                i += 1; j += 1
            elif r < q[0] + q[1]:
                i += 1
            elif r < q[0] + q[1] + q[2]:
                j += 1
            df *= L.disc
        out.append({"central": S["central"]["npv"], "two": S["two"]["npv"], "both": S["both"]["npv"], "pilot": P["npv"], "wait": Wt["npv"],
                    "pilot_second": P["second"], "wait_first": Wt["first"], "wait_first_year": Wt["first_year"], "wait_second": Wt["second"],
                    "mean_a": sumA / N, "mean_b": sumB / N})
    return out
