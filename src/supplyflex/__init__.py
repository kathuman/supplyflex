"""supplyflex: valuing flexibility in supply-chain capacity and network decisions.

Modules:

* :mod:`supplyflex.site`      — one site: commit, wait, stage, lease; sell, mothball and contract
* :mod:`supplyflex.network`   — two regions with correlated demand: one hub or two, now or later
* :mod:`supplyflex.calibrate` — volatility and growth from a demand history, with their uncertainty
* :mod:`supplyflex.plots`     — charts for all of the above (needs Matplotlib: ``pip install supplyflex[plot]``)

Quick start::

    from supplyflex import site
    sol = site.solve(sigma=0.35)
    sol.npvs()            # NPV of every strategy, $M
    sol.option_values()   # what each kind of flexibility is worth
    sol.defer_triggers()  # the demand at which waiting turns into building, step by step
"""
from . import calibrate, network, site

__version__ = "0.2.0"
__all__ = ["site", "network", "calibrate", "__version__"]
