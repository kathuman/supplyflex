# Parameter reference

Every input of [`site.SiteParams`](../src/supplyflex/site.py) and [`network.NetworkParams`](../src/supplyflex/network.py),
with its default (the web app's), what it means, the range it must lie in, and where to get a value. Out-of-range
inputs raise one `ValueError` that lists every problem.

```python
from supplyflex import site
p = site.SiteParams(d0=410, sigma=0.25, growth=0.03)   # inputs you do not give keep their defaults
q = p.with_(sigma=0.35)                                 # a copy with some inputs changed
site.solve(p) ; site.solve(sigma=0.35)                  # both work
```

## Units

| Quantity | Unit | To convert | Example |
|---|---|---|---|
| Money | $M | — | `fixed_cost = 8.0` is $8 million |
| Demand and capacity | thousand units per year | units per year ÷ 1,000 | `d0 = 500.0` is 500,000 units a year |
| Margins, penalties, unit costs | $M per thousand units | $ per unit ÷ 1,000 | `margin = 0.020` is $20 per unit |
| Holding cost, rent | $M per thousand units of capacity per year | $ per unit-year ÷ 1,000 | `op_cost = 0.006` is $6 per unit of capacity a year |
| Rates (growth, volatility, WACC) | per year, as a fraction | % ÷ 100 | `wacc = 0.10` is 10% |
| Times | years | months ÷ 12 | `build_lead = 1.5` is 18 months |
| `…_factor`, `ceiling`, `out_cap` | multiples or shares | — | `big_factor = 1.4` is 1.4 × today's demand |

## One site: `SiteParams`

### Demand

| Input | Default | Range | Meaning | Where to get it |
|---|---|---|---|---|
| `d0` | 500.0 | > 0 | Demand today, thousand units per year | Last 12 months of shipments; `calibrate.estimate(...).d0` |
| `sigma` | 0.25 | > 0 | Volatility of demand growth per year: the standard deviation of yearly log changes | `calibrate.estimate(...).sigma` (remove the season first). Typical: 0.05–0.15 staples, 0.2–0.4 industrial or new products |
| `growth` | 0.04 | [−1, 1] | Expected demand growth per year | The plan, checked against `calibrate.estimate(...).ci_growth`. A history rarely pins growth down: test the ends of its interval |
| `ceiling` | 3.0 | ≥ 1 | The market saturates at `ceiling × d0`; demand never goes above it | Market sizing. A low ceiling caps the upside and lowers the value of expansion |
| `life` | 20.0 | (0, 100] | Asset life in years; cash flows stop at the end | Depreciation life or lease horizon |
| `window` | 8.0 | [0, `life`] | Build, wait and expand decisions are possible only in the first `window` years; after it capacity stays as it is | How long the site, the land option or the strategy stays open |

### Economics

| Input | Default | Range | Meaning | Where to get it |
|---|---|---|---|---|
| `margin` | 0.020 | ≥ 0 | Margin on each unit the site serves (price less variable cost), $M per thousand units | Unit contribution margin |
| `op_cost` | 0.006 | ≥ 0 | Holding cost of capacity, per unit of capacity per year, whether used or not | Fixed site opex ÷ capacity |
| `penalty` | 0.010 | ≥ 0 | Cost of each unit of demand lost (neither the site nor the 3PL serves it) | Lost future margin, contractual penalties, goodwill |
| `out_margin` | 0.006 | ≥ 0 | Margin on demand served by the 3PL instead | Margin less the 3PL's premium over own handling |
| `out_cap` | 1.0 | ≥ 0 | Most the 3PL can take, as a multiple of `d0` (fixed in units, not growing) | 3PL contract or market capacity; 0 for none |

### Valuation basis

| Input | Default | Range | Meaning | Where to get it |
|---|---|---|---|---|
| `basis` | `"wacc"` | `"wacc"`, `"rn"` | `"wacc"`: expected cash flows discounted at the WACC. `"rn"`: risk-neutral, the drift lowered by `risk_price × sigma` and discounted at `risk_free` | See [the guide](guide.md#which-valuation-basis) |
| `wacc` | 0.10 | [0, 1) | Discount rate for `"wacc"` | Finance; the project's hurdle rate |
| `risk_free` | 0.03 | (−1, 1) | Discount rate for `"rn"` | Government bond yield for the asset's life |
| `risk_price` | 0.3 | any | Market price of demand risk (λ) for `"rn"` | (expected return of a traded asset correlated with demand − risk-free) ÷ its volatility. 0 when demand risk is diversifiable |

### Capacity and its cost

| Input | Default | Range | Meaning | Where to get it |
|---|---|---|---|---|
| `fixed_cost` | 8.0 | ≥ 0 | Fixed cost of building a site, $M | Land, building shell, systems |
| `var_cost` | 0.030 | ≥ 0 | Cost per unit of capacity, $M per thousand units | Racking, equipment, automation per unit |
| `big_factor` | 1.4 | > 0 | Full footprint (`full`, `defer`, and the top staged level) as a multiple of `d0` | The design you would build in one go |
| `small_factor` | 0.8 | > 0 | Initial staged footprint as a multiple of `d0` | The smallest viable first phase |
| `step_factor` | 0.15 | > 0 | Each expansion step as a multiple of `d0` | The size of a module or phase |
| `premium` | 0.25 | ≥ 0 | Premium on the variable cost of capacity added later (25% dearer) | Disruption, smaller orders, brownfield work |
| `escalation` | 0.06 | (−1, 1) | Yearly escalation of a deferred build's cost | Construction cost inflation |
| `build_lead` | 1.5 | [0, `life`) | Years from deciding to build until capacity arrives | Permits plus construction |
| `expand_lead` | 1.0 | ≥ 0 | Years from deciding to expand until the step arrives | Procurement plus installation |

Every expansion also pays **25% of `fixed_cost`** on top of the steps' variable cost. That share is fixed in the code
(it is the web app's), not an input; it is why the staged site often jumps several steps at once. The staged site has
at most 13 capacity levels.

### Exits and switching (owned sites)

| Input | Default | Range | Meaning | Where to get it |
|---|---|---|---|---|
| `salvage_factor` | 0.3 | [0, 1] | Resale value of a new site as a share of its cost, falling linearly to 0 at the end of `life`. 0 removes the option to sell | Second-hand value of the building and equipment |
| `mothball` | True | bool | Owned sites may be idled and reactivated | False if the site cannot be idled |
| `mothball_factor` | 0.2 | [0, 1] | Share of the holding cost still paid while idle | Security, insurance, minimum staff |
| `mothball_cost` | 0.5 | ≥ 0 | One-off cost to idle, $M | Severance, decommissioning |
| `reactivate_cost` | 1.0 | ≥ 0 | One-off cost to restart, $M | Hiring, training, recommissioning |
| `contract` | True | bool | The staged site may sell capacity back in steps (at `salvage_factor` of its variable cost) | False if part of a site cannot be sold |

### Lease

| Input | Default | Range | Meaning | Where to get it |
|---|---|---|---|---|
| `rent` | 0.0065 | ≥ 0 | Rent per unit of leased capacity per year (on top of `op_cost`) | Quoted rent ÷ capacity |
| `lease_lead` | 0.5 | [0, `life`) | Fit-out time, years | Landlord or contractor |
| `lease_fit` | 0.25 | ≥ 0 | Fit-out cost as a multiple of `fixed_cost`, paid now | Fit-out quote ÷ `fixed_cost` |
| `min_term` | 5.0 | ≥ 0 | Years from the lease's start to its first break date | Lease terms |
| `break_every` | 5.0 | ≥ 0 | Years between later break dates; 0 for a single break | Lease terms |
| `break_fee` | 1.0 | ≥ 0 | Fee for breaking, in years of rent | Lease terms |

The leased capacity is not an input: it is chosen to maximise NPV, as is the `best` fixed capacity (both searched
between 0.3 and 2.2 × `d0`).

### Numerics

| Input | Default | Range | Meaning |
|---|---|---|---|
| `step_years` | None | None or (0, 1] | Force a lattice step, for convergence checks. By default the step is quarterly, or monthly or weekly when the growth needs it (a warning says so) |

## Two regions: `NetworkParams`

Inputs shared with `SiteParams` (`ceiling`, `life`, `window`, `build_lead`, `margin`, `op_cost`, `out_margin`,
`out_cap`, `penalty`, `basis`, `wacc`, `risk_free`, `risk_price`, `fixed_cost`, `var_cost`, `escalation`) mean the same
and have the same defaults and ranges; `ceiling` and `out_cap` apply to each region against its own demand today.

| Input | Default | Range | Meaning | Where to get it |
|---|---|---|---|---|
| `demand_a` | 500.0 | > 0 | Region A's demand today | As `d0`. Use a tiny value (1e-9) for an empty region |
| `demand_b` | 250.0 | > 0 | Region B's demand today | As `d0` |
| `sigma_a` | 0.2 | > 0 | Volatility of region A's demand | As `sigma`, per region |
| `sigma_b` | 0.4 | > 0 | Volatility of region B's demand | As `sigma`, per region |
| `growth_a` | 0.02 | [−1, 1] | Expected growth of region A's demand | As `growth` |
| `growth_b` | 0.08 | [−1, 1] | Expected growth of region B's demand | As `growth` |
| `rho` | 0.2 | [−1, 1] | Correlation of the two regions' demand changes | Correlation of the two histories' log changes. Low or negative correlation favours one pooled hub |
| `cross_cost` | 0.006 | ≥ 0 | Extra cost of serving the other region from a hub, per unit | Freight between regions less local delivery |
| `learning` | 0.4 | [0, 1] | Share of `fixed_cost` saved on the second hub | Shared systems, people, supplier terms |
| `size_a` | 1.2 | > 0 | Planned hub size in region A, × `demand_a` (used by `both`, `pilot`, `wait`) | The design for that region |
| `size_b` | 1.4 | > 0 | Planned hub size in region B, × `demand_b` | The design for that region |

The network lattice is always quarterly. A growth too strong for its volatility, or a correlation the four branches
cannot carry, is limited and reported in `sol.warnings` rather than refused.

## Calibration: `calibrate.estimate`

| Argument | Default | Meaning |
|---|---|---|
| `values` | — | Positive observations, oldest first (list, NumPy array or pandas Series); at least 3 |
| `f` | — | Observations per year: 52, 12, 4 or 1 |
| `deseason` | False | Remove the season first (needs 2 years of weekly, monthly or quarterly data). Do it when a warning says the data look seasonal |
| `units` | `"period"` | `"period"`: each value is that period's demand. `"year"`: each value is an annual rate |
| `scale` | 1.0 | Divides values to reach thousand units (1000 when values are in units) |
