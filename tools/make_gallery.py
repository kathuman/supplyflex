"""Render every chart in supplyflex.plots to docs/img/ (the README gallery).  usage: python tools/make_gallery.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from supplyflex import calibrate, network, plots, site  # noqa: E402

out = Path(__file__).resolve().parents[1] / "docs" / "img"
out.mkdir(parents=True, exist_ok=True)

sol = site.solve(sigma=0.35)
sim = sol.simulate(3000, 7)
net = network.solve()
hist = calibrate.synthetic(**calibrate.SAMPLES["Grocery staples (monthly, seasonal)"])

charts = {
    "strategy_values": lambda: plots.strategy_values(sol),
    "decision_map": lambda: plots.decision_map(sol),
    "payoff_by_outcome": lambda: plots.payoff_by_outcome(sol, sim=sim),
    "npv_distribution": lambda: plots.npv_distribution(sol, sim=sim),
    "flexibility_vs_volatility": lambda: plots.flexibility_vs_volatility(),
    "tornado": lambda: plots.tornado(),
    "network_values": lambda: plots.strategy_values(net),
    "network_decision_map": lambda: plots.network_decision_map(net, year=2),
    "pooling": lambda: plots.pooling(),
    "network_npv_distribution": lambda: plots.npv_distribution(net, n=2000),
    "calibration": lambda: plots.calibration(hist, 12, deseason=True),
}
for name, make in charts.items():
    p = make()
    p.save(str(out / f"{name}.png"), dpi=110)
    print("wrote", name)
