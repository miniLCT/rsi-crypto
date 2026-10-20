# RSI protocol (for humans and coding agents)

This repo is a **harness**: the evaluation contract is fixed, and research
iterations only change *candidates*. Improvement is recursive because each
iteration reads the ledger of everything tried before, proposes the next
hypothesis, and the harness decides — not the proposer.

## Objective

Maximise walk-forward, out-of-sample, **net-of-cost** Sharpe on Binance USDT-M
perpetuals subject to the gates in `configs/harness.yaml`, with the deflated
Sharpe (which penalises the total number of trials in the ledger) above threshold.

## What you may change during an RSI loop

| allowed | path |
|---|---|
| experiment specs | `experiments/*.yaml` |
| new / revised factors | `src/rsi/features/custom/*.py` (auto-imported) |
| research notes | `docs/rsi_log.md` |

## What you must not change during an RSI loop

`configs/harness.yaml`, `src/rsi/{splits,labels}.py`, `src/rsi/engine/`, `src/rsi/eval/`,
`src/rsi/harness/`, data periods, cost model. Never call `rsi holdout --force`.
Infra changes are fine as separate, reviewed commits — bump `harness.version`
if they change how results are scored (the ledger then segregates versions).

## Loop

1. **Recall** — `rsi ledger ls --by sharpe` and read `docs/rsi_log.md`. Do not repeat a failed idea without a new reason.
2. **Hypothesise** — one change with an economic story (funding/basis carry, liquidation cascades,
   taker-flow persistence, listing effects, BTC-beta residuals, vol regimes, ...). Write it in `hypothesis:`.
3. **Implement**
   - factor: add `src/rsi/features/custom/f_<id>_<name>.py` using `register(name, lookback, group)`;
     only finite rolling windows/shifts; declare the true lookback.
   - spec: copy the parent spec, set `parent:`, change as little as possible.
4. **Check the factor** — `rsi features check <name>` (causality + lookback contract, mandatory),
   `rsi screen <name>`: look at `ic_t_8`, stability across halves, `max_corr` with existing factors
   (> 0.9 means redundant).
5. **Run** — `rsi run experiments/<spec>.yaml`. The harness trains walk-forward, backtests with fees,
   spread, impact and funding, applies gates and writes the ledger + `runs/<run_id>/report.md`.
6. **Decide** — keep only if the gates pass *and* the champion margin is met; otherwise record why it failed.
7. **Log** — append to `docs/rsi_log.md`: run id, hypothesis, key metrics, decision, what you learned.
8. **Commit** — one commit per iteration, message `rsi: <run_id> <name> <status>`.

`rsi search <spec> --n K` is the automatic proposer (random + hill-climb over `search_space`).
Every candidate counts as a trial, so a lucky search winner is deflated accordingly.

## Promotion ladder

research champion → `rsi holdout <run_id>` (max 3 looks / 30 days, all logged)
→ `rsi export <run_id>` → paper ≥ 2 weeks → `rsi parity <model_dir>` and journal audit
→ testnet → mainnet with tightened `risk_overrides`.

## Overfitting hygiene

- Prefer fewer knobs; a change that needs a lucky parameter is not a result.
- Look at fold consistency and long/short attribution, not only the headline Sharpe.
- A factor whose IC flips sign between halves is noise.
- Costs dominate at 30m: check `turnover_ann`, `fees_frac`; most ideas die here.
