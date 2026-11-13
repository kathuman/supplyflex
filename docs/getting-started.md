# Getting started

What `supplyflex` can tell you, what you need to bring, and the steps from data to a decision. To start, copy
[`examples/inputs_template.py`](../examples/inputs_template.py), replace its figures with yours, and run it:

```
python examples/inputs_template.py history.csv
```

Then read the [worked example](worked-example.md) for a full case, the [user guide](guide.md) for how to read each
result, and the [parameter reference](parameters.md) for every input.

- [Questions it answers](#questions-it-answers)
- [What you need](#what-you-need)
- [The workflow](#the-workflow)
- [What it cannot answer yet](#what-it-cannot-answer-yet)

## Questions it answers

**Which strategy?**

- Build the full design now, the right size now, wait, build in phases, or lease? What is each worth?
- What capacity maximises value if we build now? If we lease?
- How much better is the best flexible strategy than the best design built now?

**What is flexibility worth, and so what is it worth paying for?**

- The most to pay extra for a design that can be extended (`option_values()["stage"]`).
- The most extra rent or fee to accept for break clauses (`break_clauses`).
- What the ability to sell or idle the site is worth (`sell`, `mothball`).
- How the value of flexibility grows with uncertainty (`plots.flexibility_vs_volatility`).

**When to act?**

- At what demand do we build, if we wait? At what demand do we expand the first phase?
- How likely is an expansion, and when would it typically happen?
- At what demand would we sell or mothball the site?

**How risky?**

- Each strategy's chance of losing money and its 5th-percentile outcome.
- In which futures (low, middle or high demand) each strategy wins.

**How robust?**

- At what growth rate does the best strategy change?
- Which inputs matter most, and so which are worth researching before deciding?
- Does the answer change between the WACC and risk-neutral valuation?

**About the data**

- How volatile is demand, and how sure can we be of that?
- Is the history long enough to say anything about growth? Is it seasonal or autocorrelated?

**Two regions**

- One central hub, or one in each region?
- Open a pilot hub first, and in which region?
- For each pair of regional demands, what should we open?
- How does the correlation between the regions change the answer?

## What you need

### A demand history

| | |
|---|---|
| Content | Shipments or orders per period, oldest first, all positive |
| Format | A list, NumPy array or pandas Series; or text with one `date,value` line per period (also `;`, a tab, or a value alone) read with `calibrate.parse(text)`. Dates such as `2024-03-01`, `2024-03`, `2024` or `2024-Q1` let the frequency be detected |
| Frequency | Weekly, monthly, quarterly or yearly: pass `f` = 52, 12, 4 or 1 |
| Length | At least 3 points; 2 years to remove the season; **5 years or more** advised. Volatility firms up quickly, growth only with a long history |
| Units | `scale=1000` when values are in units; `units="year"` when each value is an annual rate rather than that period's demand |

For two regions, bring both histories at the same dates. The module does not estimate their correlation; compute it
from the log changes:

```python
import numpy as np
rho = np.corrcoef(np.diff(np.log(history_a)), np.diff(np.log(history_b)))[0, 1]
```

### Business figures

Collect these in everyday units. The template converts them, so you never enter a model unit by hand.

| Group | Collect | Becomes | Conversion |
|---|---|---|---|
| Market | Growth view (% a year); market ceiling | `growth`, `ceiling` | % ÷ 100; ceiling ÷ today's demand |
| Revenue | Contribution margin per unit; cost per lost unit | `margin`, `penalty` | $/unit ÷ 1,000 |
| Outsourcing | 3PL margin per unit; most the 3PL can take | `out_margin`, `out_cap` | $/unit ÷ 1,000; volume ÷ today's demand |
| Building | Fixed site cost; cost per unit of capacity; yearly holding cost per unit of capacity | `fixed_cost`, `var_cost`, `op_cost` | $ ÷ 1,000,000; $/unit ÷ 1,000 |
| Designs | Full design, first phase, one module (units a year) | `big_factor`, `small_factor`, `step_factor` | size ÷ today's demand |
| Change | Premium on later expansion; yearly cost escalation | `premium`, `escalation` | % ÷ 100 |
| Timing | Build and expansion lead times; asset life; how long the decision stays open | `build_lead`, `expand_lead`, `life`, `window` | years (months ÷ 12) |
| Exits | Resale value of a new site; costs to idle and restart; holding cost while idle | `salvage_factor`, `mothball_cost`, `reactivate_cost`, `mothball_factor` | share of cost; $ ÷ 1,000,000; share |
| Lease | Rent per unit of capacity a year; fit-out cost and time; first break; break interval; break fee | `rent`, `lease_fit`, `lease_lead`, `min_term`, `break_every`, `break_fee` | $/unit ÷ 1,000; fit-out ÷ fixed cost; years; years of rent |
| Finance | WACC; optionally the risk-free rate and price of demand risk | `wacc`, `risk_free`, `risk_price` | % ÷ 100 |
| Two regions | Cross-region transport cost; saving on the second hub; planned hub sizes | `cross_cost`, `learning`, `size_a`, `size_b` | $/unit ÷ 1,000; share; size ÷ the region's demand today |

The most common mistake: design sizes are **multiples of today's demand**, not units. A 575,000-unit design with
demand of 410,000 units a year is `big_factor = 575 / 410 = 1.40`.

Inputs you leave out keep the defaults, which are illustrative, not industry benchmarks. Out-of-range inputs raise one
`ValueError` that lists every problem.

## The workflow

| Step | Do | With |
|---|---|---|
| 1. Frame the decision | Write down the real choices: which sites or regions, which designs, whether leasing is available, how long the decision stays open | — |
| 2. Calibrate demand | Estimate volatility and today's demand; read the warnings | `calibrate.estimate(values, f, deseason=True, scale=1000)`, `plots.calibration` |
| 3. Set growth | Take it from the plan; check it sits inside the history's 90% interval | `est.ci_growth` |
| 4. Enter the economics | Convert the business figures | the template's `to_params()`, or `SiteParams(...)` / `NetworkParams(...)` |
| 5. Value the strategies | Compare values and what each kind of flexibility is worth | `site.solve(p)`, `.npvs()`, `.option_values()`, `plots.strategy_values` |
| 6. Get the decision rule | At what demand to build, expand, sell or mothball | `plots.decision_map`, `.defer_triggers()` |
| 7. Check the risk | The spread of outcomes, not only the mean | `plots.npv_distribution` (`.data["stats"]`), `plots.payoff_by_outcome` |
| 8. Stress-test | Re-solve at the ends of the intervals; sweep growth; find the inputs that matter | `p.with_(...)`, `plots.tornado`, `plots.flexibility_vs_volatility` |
| 9. Two regions? | One hub or two, a pilot first, or wait | `network.solve`, `plots.network_decision_map`, `plots.pooling` |
| 10. Decide and monitor | Choose, then track actual demand against the trigger line from step 6 | — |

The same steps in code:

```python
from supplyflex import calibrate, plots, site

est = calibrate.estimate(history, 12, deseason=True, scale=1000)   # 2-3
p = site.SiteParams(d0=est.d0, sigma=est.sigma, growth=0.04,      # 4: plus your economics
                    margin=0.020, fixed_cost=8.0, var_cost=0.030)
sol = site.solve(p)                                                # 5
sol.npvs(), sol.option_values()
plots.decision_map(sol).save("when_to_act.png")                    # 6
plots.npv_distribution(sol).data["stats"]                          # 7
for g in (est.ci_growth[0], 0.04, est.ci_growth[1]):               # 8
    n = site.solve(p.with_(growth=g)).npvs()
    print(f"growth {g:+.1%}: {max(n, key=n.get)}")
plots.tornado(p).save("what_matters.png")
```

## What it cannot answer yet

See [assumptions and limitations](guide.md#assumptions-and-limitations) for the full list.

- Networks of **more than two** sites, or choosing among many candidate locations (planned for 0.3 and 0.4).
- Demand that **mean-reverts, jumps or switches regimes** (planned for 0.3).
- Inventory, service levels, seasonality within a year, or several products.
- Pricing decisions, or demand that responds to them.
- After-tax values: everything is pre-tax, with no depreciation or working capital.
- Expanding a lease; expanding, selling or idling network hubs.
