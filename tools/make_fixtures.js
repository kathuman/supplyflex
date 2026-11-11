// Golden values from the JavaScript engine of Real Options for Network Design, for supplyflex's parity tests.
// usage: node tools/make_fixtures.js <path to claude-projects/real-options/src> > tests/fixtures/parity.json
// Inputs are written with the JavaScript names; tests/test_parity.py maps them to supplyflex's names.
const path = require("path");
const src = process.argv[2];
if (!src) { console.error("usage: node tools/make_fixtures.js <real-options/src>"); process.exit(1); }
const M = require(path.resolve(src, "model.js")), N = require(path.resolve(src, "network.js")), C = require(path.resolve(src, "calibrate.js"));

const siteCases = [
  {}, { sigma: 0.45, g: 0 }, { sigma: 0.2, g: -0.06 }, { penalty: 0.03 }, { buildLead: 0, expandLead: 0 },
  { salvF: 0, moth: 0, contract: 0 }, { moth: 0, contract: 0 }, { basis: "rn" }, { basis: "rn", lambda: 0, rf: 0.10 },
  { salvF: 0, mothF: 0.1, opcost: 0.01, sigma: 0.4, g: 0 }, { expandLead: 0 }, { breakEvery: 1, minTerm: 2 },
  { window: 2 }, { escal: 0.12 }, { outCap: 0, penalty: 0.04 }, { sigma: 0.08, g: 0.06 }, { D0: 1200, fFix: 20, ceiling: 2 }
];
const site = siteCases.map((o) => {
  const s = M.solve(o);
  return { params: o, dt: s.lattice.dt, N: s.lattice.N, p: s.lattice.p, warnings: s.lattice.warnings.length, levels: s.levels,
    full: s.full.npv, best: s.best.npv, bestK: s.best.K, noExit: s.best.noExit, noMoth: s.best.noMoth, defer: s.defer.npv,
    stage: s.stage.npv, lease: s.lease.npv, leaseK: s.lease.K, noBreak: s.lease.noBreak,
    deferBuilds: s.defer.build.reduce((a, row) => a + row.reduce((x, y) => x + y, 0), 0) };
});
const fixed = [[{}, 500], [{}, 300], [{ sigma: 0.45, basis: "rn", salvF: 0.5, moth: 0 }, 500], [{ salvF: 0, mothF: 0.1, opcost: 0.01, sigma: 0.4, g: 0 }, 500]]
  .map(([o, K]) => ({ params: o, K, npv: M.fixedNpv(o, K) }));
const lease = [[{}, 500, true], [{}, 500, false], [{ sigma: 0.4, basis: "rn" }, 450, true]].map(([o, K, b]) => ({ params: o, K, breaks: b, npv: M.leaseNpv(o, K, b) }));
const call = { args: [100, 110, 0.3, 0.04, 2, 0.25], value: M.europeanCall(100, 110, 0.3, 0.04, 2, 0.25) };
const simSol = M.solve({ sigma: 0.45, g: 0, salvF: 0.6 }), sim = M.simulate(simSol, 40, 3);
const simulation = { params: { sigma: 0.45, g: 0, salvF: 0.6 }, n: 40, seed: 3,
  paths: sim.slice(0, 5).map((r) => ({ full: r.full, best: r.best, defer: r.defer, stage: r.stage, lease: r.lease, endDemand: r.endDemand })),
  mean: ["full", "best", "defer", "stage", "lease"].reduce((o, k) => (o[k] = sim.reduce((a, r) => a + r[k], 0) / sim.length, o), {}) };

const netCases = [{}, { rho: 0.8 }, { sigA: 0.4, sigB: 0.4 }, { basis: "rn" }, { xcost: 0.03 }, { gB: 0.5 }, { learn: 0, rho: -0.5 }];
const network = netCases.map((o) => {
  const s = N.solve(o);
  return { params: o, rho: s.lattice.rho, q: s.lattice.q, warnings: s.lattice.warnings.length, central: s.central.npv, centralK: s.central.K,
    two: s.two.npv, twoKA: s.two.KA, twoKB: s.two.KB, both: s.both.npv, pilot: s.pilot.npv, pilotRegion: s.pilot.region, pilotOther: s.pilot.other, wait: s.wait.npv };
});
const netFixed = [[{}, 600, 350], [{ rho: 0.8 }, 700, 0]].map(([o, a, b]) => ({ params: o, KA: a, KB: b, sum: N.fixedNpv(o, a, b), backward: N.fixedNpvBackward(o, a, b) }));
const nsim = N.simulate(N.solve({}), 20, 5);
const netSimulation = { n: 20, seed: 5, paths: nsim.slice(0, 3).map((r) => ({ central: r.central, two: r.two, both: r.both, pilot: r.pilot, wait: r.wait })) };

const calibrate = Object.entries(C.SAMPLES).map(([name, o]) => {
  const v = C.synthetic(o), a = C.estimate(v, o.f, {}), b = C.estimate(v, o.f, { deseason: true });
  const pick = (e) => ({ sigma: e.sigma, mu: e.mu, g: e.g, seSigma: e.seSigma, seG: e.seG, D0: e.D0, rho1: e.rho1, seasonal: e.seasonal, warnings: e.warnings.length });
  return { name, sample: o, first: v.slice(0, 5), sum: v.reduce((x, y) => x + y, 0), n: v.length, plain: pick(a), deseasoned: pick(b) };
});
const parsed = C.parse("date,demand\n2024-01,100\n2024-02,110\n2024-03,121\n2024-04,133.1\nbad line\n2024-05,146.41");
const parse = { text: "date,demand\n2024-01,100\n2024-02,110\n2024-03,121\n2024-04,133.1\nbad line\n2024-05,146.41", values: parsed.values, dates: parsed.dates, warnings: parsed.warnings.length, f: C.detectFrequency(parsed.dates) };

console.log(JSON.stringify({ source: "Real Options for Network Design " + M.VERSION, site, fixed, lease, call, simulation, network, netFixed, netSimulation, calibrate, parse }, null, 1));
