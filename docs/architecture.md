# Architecture

```
                 data.binance.vision (archives)          fapi.binance.com (live REST)
                            |                                      |
                     rsi.data.vision  ── raw zips + sha256          rsi.live.feed
                            |                                      |
                     rsi.data.lake  ── parquet per symbol           |
                            |                                      |
                     rsi.data.panel ─────── Panel [time x symbol] ──┘   (same type, same aligner)
                            |
        ┌───────────────────┼──────────────────────────────┐
  rsi.universe        rsi.features (registry,         rsi.labels / rsi.splits
  point-in-time       contract, featureset)           fwd returns, walk-forward,
  liquidity top-N     causal, declared lookback       purge+embargo, holdout lock
        └───────────────────┼──────────────────────────────┘
                     rsi.models (lgbm / ridge / factor)
                            |
                     rsi.strategy.TargetBuilder  ── score -> target weights
                            |
                     rsi.engine.portfolio.Portfolio  ── risk overlays + accounting + costs
                      /                     \
       rsi.engine.backtest              rsi.live.runner ── PaperBroker | BinanceBroker
                      \                     /
                     rsi.harness  (spec, runner, gates, ledger, search, holdout, export, screen)
```

## Data choices

| dataset | why | fields kept |
|---|---|---|
| USDT-M perp klines, 30m | the instrument we trade; 30m balances signal half-life against ~10 bps round-trip taker cost | open, high, low, close, volume, quote volume, trade count, taker-buy volume, taker-buy quote volume |
| funding rate prints | dominant carry cost of perps and a crowding signal | rate, settlement interval |
| premium-index klines, 30m | perp-vs-index basis, leads funding | premium close |

* Symbols come from the archive bucket listing (includes delisted contracts → no survivorship bias),
  filtered to USDT-quoted crypto perpetuals (no stablecoin pairs, indices, delivery or TradFi contracts).
* Archives are checksum-verified; ingestion handles header/no-header CSVs and ms/us timestamps.
* The panel keeps symbols that were ever in the top-150 by 30-day ADV; this is harmless for universes
  up to 150 names because every cross-sectional operation runs on point-in-time universe members only.

## Time conventions

* All timestamps are UTC epoch ms; a bar is identified by its open time; its data exists at `open + 30m`.
* Decision at the close of bar *t*, fill at `open[t+1]`. Labels are `log(open[t+1+h] / open[t+1])`.
* Funding printed at `open[t+1]` is charged to the book held across that boundary (before the rebalance).
* Universe membership for UTC day D is decided from data up to the end of D-1.

## Train/serve parity by construction

1. **One feature implementation.** `Feature.compute(FrameView)` is called on the full history in
   research and on a trailing window live.
2. **Declared lookback + contract.** Each feature declares the bars it needs. `rsi.features.contract`
   verifies, for random cut points, that (a) future rows don't change past values and (b) the declared
   window reproduces the full-history value. Only finite rolling windows are allowed (no EWM/expanding).
   The harness runs this automatically for any new or changed feature (cached by source hash).
3. **float32 model inputs** in both paths.
4. **Manifest.** Exported models record the source hash of every feature; the live runner refuses to
   start if the running code would compute anything differently.
5. **Same portfolio/risk code.** Backtest, paper and live call the same `Portfolio.target_qty` and
   accounting; stops are evaluated on the bar close and exited at the next open in both.
6. **Audits.** `rsi parity <model>` replays the live path on archived data; the live decision log can
   be compared against archive recomputation after the fact (`rsi.live.parity.check_journal`).

## Evaluation contract (configs/harness.yaml)

```
2020-07 ─────────────── research ──────────────── 2025-10 ─── holdout ─── 2026-10
            walk-forward: rolling 24m train | purge(h+1)+embargo(1d) | 3m test, from 2022-01
```

* Research runs physically receive no rows at/after `research_end`; labels needing those rows are NaN.
* Holdout is reachable only through `rsi holdout` (≤ 3 looks per 30 days, each one logged).
* Costs: 5 bps taker fee + liquidity-tiered half-spread (`10 / sqrt(ADV/$1M)` bps, clipped 0.5–30)
  + square-root impact `0.5 · σ_daily · sqrt(notional/ADV)` + realised funding + 200 bps on delisting.
* Gates: Sharpe ≥ 1.0, deflated Sharpe ≥ 0.90 (n_trials from the ledger), max DD ≤ 30%,
  ≥ 60% positive folds, ≥ 200 trades; replacing the champion needs +0.15 Sharpe.

## Risk overlays (shared backtest/live)

vol targeting on the strategy book's realised vol · per-name / gross / net caps · ADV participation cap ·
vol-scaled stop with cooldown · drawdown halt (flatten + pause) · daily loss halt · minimum trade size ·
live-only: kill-switch file, stale-data hold, exchange reconciliation, tighten-only overrides.

## Live operation

* Data: mainnet REST (identical numbers to the archives), polled right after each bar closes.
* Execution: `paper` (simulated fills at the reference price, harness cost model),
  `testnet` (Binance demo trading), `mainnet`. One-way mode, cross margin, MARKET orders,
  reduce-only exits, flips split into close + open, idempotent client order ids.
* Every bar appends to `journal.jsonl` (equity, orders, fills with slippage vs reference) and
  `decisions/<day>.parquet` (all model inputs + scores) for parity audits and cost-model calibration.
