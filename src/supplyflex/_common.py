"""Small helpers shared by the models: JavaScript-compatible rounding and a seeded generator.

The engines are ports of the Real Options for Network Design web app. Two of its conventions matter for getting the
same numbers: ``Math.round`` rounds halves up (Python's ``round`` rounds them to even), and the simulations draw from
a 32-bit xorshift generator, so a seed gives the same paths in both.
"""
from __future__ import annotations

import math
import numbers
from typing import Dict, List, Tuple


def check_inputs(obj, ranges: Dict[str, Tuple[str, str]], extra: List[str], what: str) -> None:
    """Raise one ValueError that lists every input of ``obj`` outside its range, with what the input means.

    ``ranges`` maps an input's name to (interval, meaning), the interval written as in mathematics: "(0, inf)",
    "[0, 1]". ``extra`` holds problems the caller found itself (relations between inputs, choices among names).
    """
    problems = []
    for name, (spec, meaning) in ranges.items():
        v = getattr(obj, name)
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
            problems.append(f"{name} = {v!r} must be a finite number ({meaning})")
            continue
        lo, hi = (float(x) for x in spec[1:-1].split(","))
        ok_lo = v > lo if spec[0] == "(" else v >= lo
        ok_hi = v < hi if spec[-1] == ")" else v <= hi
        if not (ok_lo and ok_hi):
            problems.append(f"{name} = {v!r} must be in {spec} ({meaning})")
    problems.extend(extra)
    if problems:
        raise ValueError(f"Invalid {what}:\n  - " + "\n  - ".join(problems))


def js_round(x: float) -> int:
    """Round half up, like JavaScript's ``Math.round``."""
    return int(math.floor(x + 0.5))


class XorShift32:
    """The 32-bit xorshift generator of the web app: ``next()`` returns a float in [0, 1)."""

    def __init__(self, seed: int = 7) -> None:
        self.s = (seed & 0xFFFFFFFF) or 1

    def next(self) -> float:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s & 0xFFFFFFFF
        return self.s / 4294967296.0

    def gauss(self) -> float:
        """A standard normal draw (Box–Muller, cosine branch), as in the web app's calibration samples."""
        u = max(1e-12, self.next())
        v = self.next()
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
