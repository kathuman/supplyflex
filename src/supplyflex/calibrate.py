"""Calibration: volatility and growth from a demand history, with how uncertain each estimate is.

The lattices assume demand follows geometric Brownian motion: log demand moves by (g − σ²/2)·Δt + σ·√Δt·ε each period,
so expected demand grows at g. From n positive observations at f periods a year::

    r_k = ln(x_{k+1} / x_k)      σ = sd(r)·√f      μ = mean(r)·f      g = μ + σ²/2
    se(μ) = σ/√T (T years of history)   se(σ) ≈ σ/√(2(n−1))   se(g)² ≈ se(μ)² + (σ·se(σ))²

The growth estimate is far less certain than the volatility estimate: it depends only on the span of the history, not on
how often it was sampled. Seasonal data inflate the volatility, so with at least two years of weekly, monthly or
quarterly data the season can be removed first (each season's average deviation from the log trend).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

from ._common import XorShift32, js_round

Z90 = 1.6449


@dataclass
class Parsed:
    values: List[float]
    dates: Optional[List[float]]   # milliseconds since 1970 (UTC), as the web app keeps them; None unless every line has one
    warnings: List[str] = field(default_factory=list)


def parse(text: str) -> Parsed:
    """Read a history: one observation per line, as "date,value", "date;value", "date<TAB>value" or just "value".
    A header line and stray spaces, underscores or apostrophes in numbers are tolerated. Dates (yyyy-mm-dd, yyyy-mm,
    yyyy or yyyy-Qn) let the frequency be detected."""
    values, dates, warnings, skipped = [], [], [], 0
    for k, raw in enumerate(re.split(r"\r?\n", str(text or ""))):
        line = raw.strip()
        if not line:
            continue
        parts = [s.strip() for s in re.split(r"[;,\t]|\s{2,}", line) if s.strip()]
        if not parts:
            continue
        vtxt = re.sub(r"[\s_']", "", parts[-1])
        v = _parse_float(vtxt)
        if v is None or not math.isfinite(v):
            if not values and k < 3:
                continue
            skipped += 1
            continue
        values.append(v)
        dates.append(parse_date(parts[0]) if len(parts) > 1 else None)
    if skipped:
        warnings.append(f"{skipped} line{'s' if skipped > 1 else ''} could not be read and were skipped.")
    have_dates = bool(dates) and all(d is not None for d in dates)
    return Parsed(values=values, dates=dates if have_dates else None, warnings=warnings)


def _parse_float(s: str) -> Optional[float]:
    """Like JavaScript's parseFloat: the longest leading number, or None."""
    m = re.match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s)
    return float(m.group(0)) if m else None


def parse_date(s: str) -> Optional[float]:
    s = s.strip()
    m = re.match(r"^(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?$", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc).timestamp() * 1000
        except ValueError:
            return None
    m = re.match(r"^(\d{4})\s*-?\s*Q([1-4])$", s, re.I)
    if m:
        return datetime(int(m.group(1)), (int(m.group(2)) - 1) * 3 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000
    try:
        t = datetime.fromisoformat(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp() * 1000
    except ValueError:
        return None


def detect_frequency(dates: Optional[Sequence[float]]) -> Optional[int]:
    """Periods per year from the median spacing of the dates: 52, 12, 4 or 1."""
    if not dates or len(dates) < 3:
        return None
    gaps = sorted((dates[k] - dates[k - 1]) / 86400000 for k in range(1, len(dates)))
    g = gaps[len(gaps) // 2]
    if g <= 0:
        return None
    return 52 if g < 10 else 12 if g < 50 else 4 if g < 150 else 1


def _trend(logs):
    n = len(logs)
    mx = (n - 1) / 2
    my = sum(logs) / n
    sx = sum((k - mx) ** 2 for k in range(n))
    sxy = sum((k - mx) * (logs[k] - my) for k in range(n))
    return my, sxy / sx, mx


def deseasonalise(logs: Sequence[float], f: int):
    """Remove the season: subtract each season's average deviation from a straight log trend. Returns (logs, factors)."""
    n = len(logs)
    my, b, mx = _trend(logs)
    s, cnt = [0.0] * f, [0] * f
    for k in range(n):
        s[k % f] += logs[k] - (my + b * (k - mx))
        cnt[k % f] += 1
    fac = [s[i] / cnt[i] if cnt[i] else 0.0 for i in range(f)]
    mean = sum(fac) / f
    out = [logs[k] - (fac[k % f] - mean) for k in range(n)]
    return out, [v - mean for v in fac]


def season_strength(logs: Sequence[float], f: int) -> float:
    """Share of the variance of detrended logs explained by the season."""
    d, _ = deseasonalise(logs, f)
    my, b, mx = _trend(logs)
    tot = res = 0.0
    for k in range(len(logs)):
        e = logs[k] - (my + b * (k - mx))
        tot += e * e
        e2 = d[k] - (my + b * (k - mx))
        res += e2 * e2
    return 1 - res / tot if tot > 0 else 0.0


@dataclass
class Estimate:
    n: int
    f: int
    years: float
    sigma: float
    mu: float
    growth: float
    se_sigma: float
    se_mu: float
    se_growth: float
    ci_sigma: tuple        # 90% interval
    ci_growth: tuple       # 90% interval
    d0: float              # demand today as an annual rate in the model's units: the average of the last year
    rho1: float            # first-order autocorrelation of the changes
    seasonal: bool
    factors: Optional[List[float]]
    warnings: List[str]
    logs: Optional[List[float]] = None   # log demand used for the estimate (de-seasoned when the season was removed)

    def site_inputs(self) -> dict:
        """The estimates as inputs to :class:`supplyflex.site.SiteParams`."""
        return {"d0": self.d0, "sigma": self.sigma, "growth": self.growth}


def estimate(values: Iterable[float], f: int, deseason: bool = False, units: str = "period", scale: float = 1.0) -> Estimate:
    """Estimate volatility and growth from a history.

    ``f`` is the number of observations per year (52, 12, 4 or 1). ``units``: "period" when each value is demand in its
    period, "year" when each value is an annual rate. ``scale`` divides values to express demand in the model's
    thousand units (1000 when values are in units). Accepts lists, NumPy arrays or pandas Series.
    """
    values = [float(v) for v in values]
    n, warnings = len(values), []
    if n < 3:
        raise ValueError("At least three observations are needed.")
    if not all(v > 0 for v in values):
        raise ValueError("Every value must be positive: demand on a lattice cannot be zero or negative.")
    if not f > 0:
        raise ValueError("Unknown frequency: say how many observations make a year.")
    logs = [math.log(v) for v in values]
    seasonal, factors = False, None
    can_season = f in (4, 12, 52) and n >= 2 * f
    if deseason and can_season:
        logs, factors = deseasonalise(logs, f)
        seasonal = True
    elif deseason:
        warnings.append("The season can only be removed with at least two years of weekly, monthly or quarterly data.")
    r = [logs[k] - logs[k - 1] for k in range(1, n)]
    m = len(r)
    mr = sum(r) / m
    vr = sum((x - mr) ** 2 for x in r) / max(1, m - 1)
    sigma = math.sqrt(vr * f)
    mu = mr * f
    g = mu + sigma * sigma / 2
    T = m / f
    se_sigma = sigma / math.sqrt(2 * max(1, m - 1))
    se_mu = sigma / math.sqrt(T)
    se_g = math.sqrt(se_mu * se_mu + (sigma * se_sigma) ** 2)
    if T < 3:
        warnings.append("Less than three years of history: the growth estimate is very uncertain.")
    if m < 12:
        warnings.append("Fewer than twelve changes: the volatility estimate is rough.")
    a0 = sum((x - mr) ** 2 for x in r)
    ac = sum((r[k] - mr) * (r[k - 1] - mr) for k in range(1, m))
    rho1 = ac / a0 if a0 > 0 else 0.0
    if m >= 12 and abs(rho1) > 2 / math.sqrt(m):
        warnings.append(f"The changes are autocorrelated ({rho1:.2f}): demand "
                        + ("tends to revert after a move, so the volatility over longer horizons is probably lower" if rho1 < 0
                           else "moves in runs, so the volatility over longer horizons is probably higher") + " than estimated.")
    if not seasonal and can_season and season_strength(logs, f) > 0.3:
        warnings.append("The data look seasonal; removing the season gives a more meaningful volatility.")
    last = min(n, max(1, js_round(f)))
    recent = sum(values[n - last:]) / last
    d0 = (recent if units == "year" else recent * f) / (scale or 1)
    return Estimate(n=n, f=f, years=T, sigma=sigma, mu=mu, growth=g, se_sigma=se_sigma, se_mu=se_mu, se_growth=se_g,
                    ci_sigma=(max(0.0, sigma - Z90 * se_sigma), sigma + Z90 * se_sigma), ci_growth=(g - Z90 * se_g, g + Z90 * se_g),
                    d0=d0, rho1=rho1, seasonal=seasonal, factors=factors, warnings=warnings, logs=list(logs))


def synthetic(f: int, years: float, start: float, growth: float, sigma: float, season: float = 0.0, seed: int = 1) -> List[float]:
    """A seeded synthetic history on geometric Brownian motion, with an optional sine season (the web app's samples)."""
    rng = XorShift32(seed)
    n = js_round(years * f) + 1
    x, dt, out = math.log(start), 1 / f, []
    for k in range(n):
        s = season * math.sin(2 * math.pi * (k % f) / f) if season else 0.0
        out.append(math.exp(x + s))
        x += (growth - sigma * sigma / 2) * dt + sigma * math.sqrt(dt) * rng.gauss()
    return out


def sample_text(f: int, years: float, start: float, growth: float, sigma: float, season: float = 0.0, seed: int = 1) -> str:
    """A synthetic history as CSV text with dates (date,demand), so the expected input format is visible."""
    vals, y0 = synthetic(f, years, start, growth, sigma, season, seed), 2026 - math.ceil(years)
    lines = ["date,demand"]
    for k, v in enumerate(vals):
        if f == 12:
            d = f"{y0 + k // 12}-{k % 12 + 1:02d}"
        elif f == 4:
            d = f"{y0 + k // 4}-{3 * (k % 4) + 1:02d}"
        elif f == 52:
            t = datetime(y0, 1, 1, tzinfo=timezone.utc).timestamp() + k * 7 * 86400
            d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            d = str(y0 + k)
        lines.append(f"{d},{js_round(v)}")
    return "\n".join(lines)


SAMPLES = {
    "Grocery staples (monthly, seasonal)": dict(f=12, years=6, start=40000, growth=0.03, sigma=0.08, season=0.12, seed=109),
    "E-commerce growth (weekly)": dict(f=52, years=3, start=6000, growth=0.15, sigma=0.35, seed=236),
    "Legacy product in decline (quarterly)": dict(f=4, years=9, start=150000, growth=-0.05, sigma=0.14, seed=377),
    "Volatile industrial (monthly)": dict(f=12, years=5, start=45000, growth=0.02, sigma=0.32, seed=176),
}
