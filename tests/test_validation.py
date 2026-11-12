"""Input validation: bad inputs raise one ValueError that names every problem; good ones (the web app's whole range,
and the edge cases the other tests rely on) pass."""
import math

import pytest

from supplyflex import network, site
from supplyflex.network import NetworkParams
from supplyflex.site import SiteParams


@pytest.mark.parametrize("changes, words", [
    (dict(sigma=0), "sigma = 0"),
    (dict(sigma=-0.2), "sigma = -0.2"),
    (dict(d0=0), "d0 = 0"),
    (dict(basis="foo"), "basis = 'foo'"),
    (dict(window=25), "window = 25 cannot be longer than life"),
    (dict(build_lead=20), "build_lead = 20 must be shorter than life"),
    (dict(salvage_factor=1.5), "salvage_factor = 1.5 must be in [0, 1]"),
    (dict(wacc=math.nan), "wacc = nan must be a finite number"),
    (dict(margin=-0.01), "margin = -0.01"),
    (dict(ceiling=0.5), "ceiling = 0.5 must be in [1, inf)"),
    (dict(step_years=2), "step_years = 2"),
    (dict(fixed_cost="8"), "fixed_cost = '8' must be a finite number"),
])
def test_site_rejects(changes, words):
    with pytest.raises(ValueError, match="Invalid SiteParams") as e:
        SiteParams(**changes)
    assert words in str(e.value)


def test_every_problem_is_listed_and_with_checks_too():
    with pytest.raises(ValueError) as e:
        SiteParams().with_(sigma=0, margin=-1, basis="x")
    msg = str(e.value)
    assert "sigma" in msg and "margin" in msg and "basis" in msg
    with pytest.raises(ValueError):
        site.solve(sigma=0)


def test_site_refuses_a_lattice_it_cannot_represent():
    SiteParams(sigma=0.01, growth=0.9)  # each input is in range ...
    with pytest.raises(ValueError, match="cannot represent"):
        site.solve(sigma=0.01, growth=0.9)  # ... but together they leave no valid up-probability
    with pytest.raises(ValueError, match="cannot represent"):
        site.fixed_npv(capacity=500, sigma=0.01, growth=0.9)


@pytest.mark.parametrize("changes", [
    dict(sigma=1e-4, growth=0), dict(window=20), dict(build_lead=0, expand_lead=0), dict(salvage_factor=0),
    dict(out_cap=0), dict(basis="rn", risk_price=0), dict(growth=-0.1), dict(step_years=1 / 26), dict(break_every=0),
])
def test_site_accepts_edge_cases(changes):
    SiteParams(**changes)


@pytest.mark.parametrize("changes, words", [
    (dict(rho=1.5), "rho = 1.5"),
    (dict(rho=-1.01), "rho = -1.01"),
    (dict(learning=1.2), "learning = 1.2"),
    (dict(demand_b=0), "demand_b = 0"),
    (dict(sigma_a=0), "sigma_a = 0"),
    (dict(basis="WACC"), "basis = 'WACC'"),
    (dict(window=21), "window = 21"),
])
def test_network_rejects(changes, words):
    with pytest.raises(ValueError, match="Invalid NetworkParams") as e:
        NetworkParams(**changes)
    assert words in str(e.value)
    with pytest.raises(ValueError):
        network.solve(**changes)


def test_network_accepts_edge_cases():
    for changes in (dict(rho=1), dict(rho=-1), dict(demand_b=1e-9), dict(learning=0), dict(learning=1), dict(window=20)):
        NetworkParams(**changes)
