// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Method (hand-written, not the house baseline's random walk):
//   Every trophy's earn rate is earners / players-tracked, so the forecast is
//   built from those two counts, not from the rate alone.
//   - Launch-night stale page: when every read in the case's history is
//     identical (same rates, same players-tracked) for hours while the game
//     has just released, the page is a stale Exophase snapshot awaiting its
//     first refresh, not a stable population. RULES documents that refresh:
//     638 tracked players, every rate but the 100% one lower, at the 04:30Z
//     read. The forecast therefore assumes a pending refresh that multiplies
//     the tracked population (growth prior ~1.6x per stale hour, about 5-6x
//     over a 3-hour stale span) and credits the new players with only a
//     small share of each trophy's earners: a new player earns trophy i
//     with probability (r_i)^1.5, well below the old cohort's rate r_i, so
//     every median falls (as the documented refresh did) while the 100%
//     trophy stays at 100%. Spread is wide (sigma ~0.75 logit) because the
//     refresh size is uncertain.
//   - Normal history: per trophy, least-squares fit of logit(rate) against
//     ln(1 + hours since release) over the last 48h of reads, plus a second
//     fit that extrapolates log(earners) and log(players-tracked) separately
//     and divides them; the median blends the two (0.7 / 0.3). ln-time
//     extrapolation saturates, so a 4.5-day horizon does not explode.
//     Spread combines residual scatter with a 0.45/sqrt-day random walk.
//   - Quantiles are logit-normal around the median at the five scored
//     levels. Everything is deterministic: no randomness, no network.
//
//   - read every <id>.in in the input directory (the grader runs one case at
//     a time, with only that case's file there);
//   - write <id>.out to the output directory: {"trophies":[{"id":1,
//     "q":[q05,q25,q50,q75,q95]}, ...]}, one row per trophy, each q five
//     finite numbers from 0 to 100, non-decreasing; at most 65,536 bytes.
//   Outside the grader's container, MAKEFUN_INPUT_DIR and
//   MAKEFUN_OUTPUT_DIR stand in for /input and /output:
//     mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Game {
  release_date?: string;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game?: Game;
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Least-squares line y ~ a + b*x; returns [intercept, slope, residualStd]. */
function ols(xs: number[], ys: number[]): [number, number, number] {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (ys[i] - (a + b * xs[i])) ** 2;
  const resid = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  return [a, b, resid];
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const last = c.history[c.history.length - 1];
  if (last === undefined) throw new Error(`${c.id} has no history`);
  const days = c.horizon_minutes / 1440;
  const checkpoint = Date.parse(c.at);
  const target = Date.parse(c.target);
  const release = c.game?.release_date
    ? Date.parse(`${c.game.release_date}T00:00:00Z`)
    : Date.parse(c.history[0].at);
  const hoursSinceRelease = (tMs: number): number => Math.max(0, (tMs - release) / 3600000);
  const xOf = (tMs: number): number => Math.log1p(hoursSinceRelease(tMs));
  const xTarget = xOf(target);

  // Stale-page test: every read identical in rates and players-tracked.
  const first = c.history[0];
  let stale = c.history.length >= 3 && c.history.every((h) => h.players_tracked === first.players_tracked);
  if (stale) {
    for (let i = 0; i < c.trophies.length; i++) {
      for (const h of c.history) {
        if (h.rates[i] !== first.rates[i]) {
          stale = false;
          break;
        }
      }
      if (!stale) break;
    }
  }
  const staleSpanH = (Date.parse(last.at) - Date.parse(first.at)) / 3600000;
  // Only a launch-window stale page implies a pending first refresh.
  const staleLaunch = stale && staleSpanH >= 1 && hoursSinceRelease(checkpoint) <= 72;

  return c.trophies.map((t, i) => {
    const lastRate = last.rates[i] ?? 0;
    const lastLogit = logit(lastRate);
    let medianLogit: number;
    let sigma: number;

    if (staleLaunch) {
      // Pending-refresh dilution model (see header comment).
      const n0 = Math.max(1, first.players_tracked);
      const growth = 1 + 1.6 * staleSpanH;
      const n1 = n0 * growth;
      if (lastRate >= 99.99) {
        medianLogit = logit(99.99);
        sigma = 0.15;
      } else {
        const earners0 = (clampPct(lastRate) / 100) * n0;
        const newShare = Math.pow(clampPct(lastRate) / 100, 1.5);
        const earners1 = earners0 + newShare * (n1 - n0);
        let med = (100 * earners1) / n1;
        // The documented refresh lowered every rate but the 100% one.
        med = Math.min(med, lastRate * 0.95);
        med = Math.max(med, (100 * earners0) / n1);
        medianLogit = logit(med);
        sigma = 0.75;
      }
      sigma = Math.sqrt(sigma * sigma + (0.45 * Math.sqrt(days)) ** 2);
    } else {
      // Trend model over the last 48h of reads (or all reads if shorter).
      const windowStart = checkpoint - 48 * 3600000;
      const rows = c.history.filter((h) => Date.parse(h.at) >= windowStart);
      const use = rows.length >= 3 ? rows : c.history;
      const xs: number[] = [];
      const ys: number[] = [];
      const lns: number[] = [];
      const les: number[] = [];
      for (const h of use) {
        const r = h.rates[i];
        if (r === null || r === undefined) continue;
        const x = xOf(Date.parse(h.at));
        xs.push(x);
        ys.push(logit(r));
        lns.push(Math.log(Math.max(1, h.players_tracked)));
        les.push(Math.log(Math.max(0.5, (clampPct(r) / 100) * Math.max(1, h.players_tracked))));
      }
      const varied = ys.length >= 3 && Math.max(...ys) - Math.min(...ys) > 1e-9;
      if (varied) {
        const [a, b, resid] = ols(xs, ys);
        const trendLogit = a + b * xTarget;
        // Earners / players decomposition as a cross-check.
        const [an, bn] = ols(xs, lns);
        const [ae, be] = ols(xs, les);
        const nPred = Math.exp(an + bn * xTarget);
        const ePred = Math.exp(ae + be * xTarget);
        const decompLogit = logit(clampPct((100 * ePred) / Math.max(1, nPred)));
        medianLogit = 0.7 * trendLogit + 0.3 * decompLogit;
        // Never let a short-history fit explode: cap the move from the last read.
        medianLogit = clamp(medianLogit, lastLogit - 3.5, lastLogit + 3.5);
        sigma = clamp(Math.sqrt(0.2 * 0.2 + (0.45 * Math.sqrt(days)) ** 2 + resid * resid), 0.35, 1.25);
      } else {
        // No usable movement: persistence with an honest wide spread.
        medianLogit = lastLogit;
        sigma = clamp(Math.sqrt(0.6 * 0.6 + (0.45 * Math.sqrt(days)) ** 2), 0.6, 1.25);
      }
    }

    const q: number[] = [];
    for (let j = 0; j < Z.length; j++) {
      let v = round4(clamp(invLogit(medianLogit + Z[j] * sigma), 0.01, 99.99));
      if (j > 0 && v < q[j - 1]) v = q[j - 1];
      q.push(v);
    }
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
