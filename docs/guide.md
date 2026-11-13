# User guide

How to go from data to a decision with `supplyflex`, how to read what it returns, and what the models leave out. New
here? Start with [getting started](getting-started.md): what to bring and an input template. For every input see the
[parameter reference](parameters.md); for a complete case with numbers see the [worked example](worked-example.md).

- [A workflow](#a-workflow)
- [Reading the results](#reading-the-results)
- [Choosing inputs](#choosing-inputs)
- [Which valuation basis](#which-valuation-basis)
- [Assumptions and limitations](#assumptions-and-limitations)
- [References](#references)

## A workflow

1. **Calibrate demand.** `calibrate.estimate(history, f, deseason=True, scale=1000)` gives `d0`, `sigma` and `growth`
   with 90% intervals. Read its `warnings`. Trust `sigma`; treat `growth` as a range (see
   [Choosing inputs](#choosing-inputs)).
2. **Set the economics** that differ from the defaults: margins, costs, lead times, lease terms. Defaults are
   illustrative, not industry benchmarks.
3. **Solve.** `sol = site.solve(params)`, then `sol.npvs()`, `sol.option_values()` and `sol.warnings`.
4. **Read the policy, not only the value.** `plots.decision_map(sol)` and `sol.defer_triggers()` say at what demand to
   build or expand, which is what you will monitor after the decision.
5. **Look at the risk.** `plots.npv_distribution(sol)` gives each strategy's 5th percentile and chance of a loss. The
   strategy with the highest mean is not always the one to choose.
6. **Test the answer.** Re-solve at the ends of the calibration intervals and run `plots.tornado(params)`. If the
   recommended strategy changes inside the plausible range, the decision depends on that input: say so, and consider
   what it would cost to learn more about it.

## Reading the results

### Strategy values: `sol.npvs()`

The NPV in $M of each strategy, each valued under its own optimal policy against the same demand. All five include
the 3PL and lost-sales rules, so a strategy that builds later still serves (part of) demand while it waits.

| Strategy | What it commits to now | What it keeps open |
|---|---|---|
| `full` | the full footprint (`big_factor × d0`) | selling, mothballing |
| `best` | the capacity that maximises NPV now | selling, mothballing |
| `defer` | nothing | when to build the full footprint, inside the window |
| `stage` | the initial footprint (`small_factor × d0`) | expanding in steps, selling back steps, selling, mothballing |
| `lease` | a leased capacity (chosen to maximise NPV) and its fit-out | handing it back at break dates |

### Option values: `sol.option_values()`

Each kind of flexibility, in $M, **measured against its own baseline**:

| Key | Equals | Reads as |
|---|---|---|
| `wait` | `defer − best` | waiting to build the full footprint, against building the right size now |
| `stage` | `stage − best` | building small and expanding, against building the right size now |
| `lease` | `lease − best` | leasing, against building the right size now |
| `break_clauses` | lease − the same lease without breaks | what the break clauses are worth |
| `sell` | best − the same design with no exits | what selling **and** mothballing are worth together |
| `mothball` | best − the same design without mothballing | what mothballing adds **once selling is possible** |

Three consequences:

- **They do not add up.** Options on the same asset overlap: once you can sell a site, also being able to mothball it is
  worth much less, often nothing, because both protect against the same low-demand futures (Trigeorgis, 1993). Never sum
  option values to value a combined strategy; solve the strategy.
- **`wait` can be negative.** `defer` can build only the full footprint, while `best` picks the best size. A negative
  `wait` means "waiting to build the big design is worse than building the right design now", not that waiting has no
  value. The pure value of timing is `sol.defer − sol.full.npv`, which is never negative.
- **A zero is information.** `mothball = 0` usually means selling beats idling whenever demand falls far enough to
  stop operating. Check by switching selling off (`salvage_factor=0`).

For a network, `option_values()` gives `pilot` and `wait`, each against the better of `central` and `two`.

### Decision rules: when to act

- `sol.defer_triggers()`: for each step of the window, the lowest demand at which the `defer` strategy builds.
  `None` means waiting is better at every demand reachable at that step.
- `plots.decision_map(sol)` draws those triggers with the stage strategy's first-expansion trigger and the demand
  below which the best fixed design is sold or mothballed, over the demand fan. Its `.data` holds the same numbers.
- On a recombining lattice, neighbouring steps sit on alternating sets of demand levels, so a trigger read step by step
  alternates between two values (for example 761 and 861). The true threshold lies between them; a finer lattice
  (`step_years=1/12`) narrows the gap.
- Triggers usually **fall near the end of the window**: with no later chance to act, a smaller margin of safety is
  needed. This is correct behaviour, not noise.

### Simulation: `sol.simulate(n, seed)`

Each path follows every strategy's optimal policy through one demand future. Per path you get each strategy's NPV,
when the `defer` site was built, how many expansions and contractions the staged site made, which sites were sold or
mothballed and when, and the demand at the window's end and the asset's end. The mean of the NPVs matches the lattice
value within sampling error (the tests check three standard errors). Use it for the spread of outcomes, the
chance of a loss, and to describe a policy in words ("half of futures expand once, around year 5").

### Warnings: `sol.warnings`

The single-site lattice switches to monthly or weekly steps when the growth is strong relative to the volatility, and
says so; results stay exact but slower. If no step can represent the inputs, `solve` raises a `ValueError` instead of
returning values. The network lattice limits a growth or correlation it cannot represent and reports what it used.

## Choosing inputs

**Volatility comes from data; growth mostly does not.** The standard error of a volatility estimate shrinks with the
number of observations; the standard error of growth shrinks only with the length of the history (about
`sigma / √years`). Five years of monthly data at 25% volatility give `sigma` to ±0.04 and growth only to about ±0.18.
Use the history for `sigma`, a plan or market view for `growth`, and check that the plan sits inside the history's
interval.

**Remove the season.** Seasonal swings are predictable and do not need flexibility, but they inflate the measured
volatility (by a quarter in the worked example). `calibrate.estimate` warns when data look seasonal.

**Watch autocorrelation.** Positive autocorrelation (demand moving in runs) means longer-horizon volatility is higher
than estimated, negative autocorrelation the reverse. The estimate warns; treat its interval as optimistic.

**Find the inputs that matter.** `plots.tornado(params)` moves each input ±20% and shows how far the value of
flexibility moves. Spend effort on the top few; defaults are fine for the rest at a first pass.

**Check the structure, not only the numbers.** If `stage` loses at high growth, check `big_factor`: the staged site can
never exceed the full footprint, while `best` can go up to 2.2 × `d0`. If `lease` never wins, check `rent` against
`op_cost` and the capital cost it avoids.

## Which valuation basis

- **`"wacc"` (default)**: expected cash flows discounted at the WACC. Familiar, easy to explain, and the way most
  practitioner real-options work is done (de Neufville and Scholtes, 2011). Its weakness: each strategy's risk differs
  from the committed project's. Exits and staging cut the downside and lower it, while waiting holds a call-like claim
  on growth that can raise it. One discount rate for all strategies is therefore an approximation, and its bias can go
  either way.
- **`"rn"`**: risk-neutral valuation. The demand drift is lowered by `risk_price × sigma` and cash flows are discounted
  at `risk_free`. This is consistent for any payoff, but needs the market price of demand risk, which is hard to
  observe (Dixit and Pindyck, 1994; Trigeorgis, 1996). `risk_price = 0` means demand risk is diversifiable.

In practice: decide on `"wacc"`, then check the ranking with `"rn"` for a few plausible `risk_price` values. Levels
will differ (different discount rates); what matters is whether the recommended strategy changes.

## Assumptions and limitations

**Demand**

- Demand follows geometric Brownian motion: a single random driver per region, constant volatility and growth, capped
  at `ceiling × today`. It cannot mean-revert, jump, or switch between regimes. (Mean reversion and regime switching
  are planned for 0.3.)
- Demand does not respond to the decisions: no price response, no lost customers coming back.
- One product, or one aggregate of products; no mix effects.

**Operations and costs**

- Capacity is annual throughput. There is no inventory, no seasonality inside a year and no service-level
  constraint other than the lost-sales penalty.
- Costs, margins, rent and the 3PL terms are known and constant (apart from `escalation` on deferred builds). There are
  no taxes, depreciation or working capital: cash flows are pre-tax.
- Every expansion pays a fixed 25% of `fixed_cost` (fixed in code). The staged site has at most 13 capacity levels and
  cannot go above `big_factor × d0`.
- Decisions are possible only inside the window. After it, capacity stays as it is; the site can still be sold or
  mothballed.
- `defer` builds only the full footprint; the lease's capacity is chosen once and cannot be expanded.

**Network**

- Two regions only (more sites are planned for 0.3). Hubs have planned sizes (`size_a`, `size_b`) in the `both`,
  `pilot` and `wait` strategies; only `central` and `two` choose sizes.
- The central hub is always in region A. To place it in B, swap the regions' inputs.
- Hubs cannot be expanded, sold or mothballed, and the lattice is always quarterly.

**Calibration**

- Assumes the same geometric Brownian motion as the models. Intervals are 90% normal approximations; for short or
  autocorrelated histories they are too narrow.

**Numerics**

- Values are exact on the lattice. The lattice step (quarterly by default) limits how finely triggers and timing can
  be read; capacities are searched on a grid of 0.002 × `d0`.

## References

The methods in `supplyflex`, and where to read more:

- **Binomial lattice.** Cox, J. C., Ross, S. A., and Rubinstein, M. (1979). Option pricing: a simplified approach.
  *Journal of Financial Economics*, 7(3), 229–263.
- **Two correlated variables on a lattice** (the network model). Boyle, P. P. (1988). A lattice framework for option
  pricing with two state variables. *Journal of Financial and Quantitative Analysis*, 23(1), 1–12.
- **The value of waiting.** McDonald, R., and Siegel, D. (1986). The value of waiting to invest. *Quarterly Journal of
  Economics*, 101(4), 707–727.
- **Mothballing, reactivation and hysteresis.** Brennan, M. J., and Schwartz, E. S. (1985). Evaluating natural resource
  investments. *Journal of Business*, 58(2), 135–157. Dixit, A. (1989). Entry and exit decisions under uncertainty.
  *Journal of Political Economy*, 97(3), 620–638.
- **Abandonment (selling).** Myers, S. C., and Majd, S. (1990). Abandonment value and project life. *Advances in
  Futures and Options Research*, 4, 1–21.
- **Why option values do not add up.** Trigeorgis, L. (1993). The nature of option interactions and the valuation of
  investments with multiple real options. *Journal of Financial and Quantitative Analysis*, 28(1), 1–20.
- **Flexibility in global manufacturing networks.** Kogut, B., and Kulatilaka, N. (1994). Operating flexibility, global
  manufacturing, and the option value of a multinational network. *Management Science*, 40(1), 123–139. Huchzermeier,
  A., and Cohen, M. A. (1996). Valuing operational flexibility under exchange rate risk. *Operations Research*, 44(1),
  100–113.
- **Risk pooling** (one hub or two). Eppen, G. D. (1979). Effects of centralization on expected costs in a
  multi-location newsboy problem. *Management Science*, 25(5), 498–501.
- **Least-squares Monte Carlo** (planned for 0.3). Longstaff, F. A., and Schwartz, E. S. (2001). Valuing American
  options by simulation: a simple least-squares approach. *Review of Financial Studies*, 14(1), 113–147.

Books:

- de Neufville, R., and Scholtes, S. (2011). *Flexibility in Engineering Design*. MIT Press. The practitioner's case
  for designing in flexibility, with expected-value methods close to the `"wacc"` basis.
- Trigeorgis, L. (1996). *Real Options: Managerial Flexibility and Strategy in Resource Allocation*. MIT Press.
- Dixit, A. K., and Pindyck, R. S. (1994). *Investment under Uncertainty*. Princeton University Press.
- Copeland, T., and Antikarov, V. (2001). *Real Options: A Practitioner's Guide*. Texere.
