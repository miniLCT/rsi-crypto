# RSI research log

Auto-generated entries per run (HRN-08); add a `- conclusion:` line under each.

### 20261020-174813-616b26 · xs_factor_reversal_1d · **fail**
- date: 2026-10-20 17:48 · parent: - · trials so far: 1
- hypothesis: Model-free control: 1-day cross-sectional reversal among liquid perps (losers bounce, winners fade). Establishes how much of any ML result is plain reversal.
- sharpe -4.21 · dsr 0.00 · cagr -64.1% · mdd 97.9% · turnover 908x · fees 118.6% · funding 3.1% · long -137.7% / short +39.9% · ic 0.0295
- failed gates: sharpe, dsr, max_drawdown, fold_consistency · no incumbent

### 20261020-175432-38c5a6 · xs_factor_reversal_1d~strategy.rebalance_bars=16,strategy.exit_buffer=0.2 · **fail**
- date: 2026-10-20 17:54 · parent: - · trials so far: 2
- hypothesis: Model-free control: 1-day cross-sectional reversal among liquid perps (losers bounce, winners fade). Establishes how much of any ML result is plain reversal.
- sharpe -2.55 · dsr 0.00 · cagr -55.6% · mdd 95.5% · turnover 560x · fees 113.5% · funding 4.9% · long -132.3% / short +37.1% · ic 0.0293
- failed gates: sharpe, dsr, max_drawdown, fold_consistency · no incumbent

### 20261020-181221-acf73a · xs_factor_carry_7d · **fail**
- date: 2026-10-20 18:12 · parent: - · trials so far: 3
- hypothesis: Funding carry: perps with persistently high funding are crowded longs. Shorting the highest 7d-average funding names and buying the lowest collects funding and bets on crowding unwinding; the signal is slow, so turnover (the main cost at 30m) should be low.
- sharpe 0.76 · dsr 0.73 · cagr 18.5% · mdd 26.9% · turnover 43x · fees 33.7% · funding -185.7% · long -8.4% / short +49.1% · ic -0.0074
- failed gates: sharpe, dsr · no incumbent

### 20261020-183056-75d38b · xs_lgbm_base · **fail**
- date: 2026-10-20 18:30 · parent: - · trials so far: 4
- hypothesis: A gradient-boosted ranker over the full feature library, trained on vol-normalised, cross-sectionally demeaned 4h forward returns, extracts a market-neutral signal from the 50 most liquid perps that survives realistic taker costs at a 1h rebalance.
- sharpe -8.02 · dsr 0.00 · cagr -58.0% · mdd 96.1% · turnover 1667x · fees 142.6% · funding -1.9% · long -90.0% / short -6.1% · ic 0.0226
- failed gates: sharpe, dsr, max_drawdown, fold_consistency · no incumbent

### 20261020-184412-9f7f78 · xs_lgbm_1d · **fail**
- date: 2026-10-20 18:44 · parent: xs_lgbm_base · trials so far: 5
- hypothesis: Costs dominate at short horizons (1d reversal: IC 0.03 but 900x turnover). Predict the 1d funding-inclusive return instead, rebalance every 4h with a wide hysteresis band and 2h score smoothing, so the same cross-sectional information is harvested at a fraction of the turnover.
- sharpe -2.13 · dsr 0.00 · cagr -45.4% · mdd 92.1% · turnover 539x · fees 103.6% · funding -15.9% · long -112.0% / short +22.1% · ic 0.0151
- failed gates: sharpe, dsr, max_drawdown, fold_consistency · no incumbent

### 20261020-184421-de037f · xs_ridge_1d · **fail**
- date: 2026-10-20 18:44 · parent: xs_lgbm_1d · trials so far: 5
- hypothesis: Trees stop after 2-3 rounds (patterns do not persist across folds); a ridge on cross-sectional ranks is the low-variance alternative for the same 1d target and turnover controls.
- sharpe 1.20 · dsr 0.86 · cagr 33.9% · mdd 34.1% · turnover 211x · fees 204.7% · funding -84.1% · long -34.5% / short +229.8% · ic 0.0341
- failed gates: dsr, max_drawdown · no incumbent

### 20261020-190227-256cfa · xs_ridge_1d~s1.0 · **fail**
- date: 2026-10-20 19:02 · parent: xs_ridge_1d · trials so far: 7
- hypothesis: search around xs_ridge_1d: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 48, 'strategy.exit_buffer': 0.15, 'strategy.smooth_bars': 4}
- sharpe 0.48 · dsr 0.00 · cagr 10.0% · mdd 47.5% · turnover 176x · fees 101.1% · funding -58.7% · long -38.3% / short +76.6% · ic 0.0308
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-190228-eeea53 · xs_ridge_1d~s2.0 · **fail**
- date: 2026-10-20 19:02 · parent: xs_ridge_1d · trials so far: 7
- hypothesis: search around xs_ridge_1d: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.15, 'strategy.smooth_bars': 4}
- sharpe 0.68 · dsr 0.00 · cagr 16.2% · mdd 51.0% · turnover 363x · fees 235.4% · funding -71.8% · long -49.3% / short +122.7% · ic 0.0308
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-190230-80391a · xs_ridge_1d_bn · **fail**
- date: 2026-10-20 19:02 · parent: xs_ridge_1d · trials so far: 7
- hypothesis: Long leg loses (-35%) while the short leg earns: the book carries residual BTC beta. Rescaling legs to zero beta (constant gross) should remove the market drag on the long side.
- sharpe 1.27 · dsr 0.00 · cagr 35.5% · mdd 34.2% · turnover 214x · fees 213.9% · funding -88.5% · long -5.1% / short +214.5% · ic 0.0341
- failed gates: dsr, max_drawdown · no incumbent

### 20261020-192507-80391a · xs_ridge_1d_bn · **fail**
- date: 2026-10-20 19:25 · parent: xs_ridge_1d · trials so far: 9
- hypothesis: Long leg loses (-35%) while the short leg earns: the book carries residual BTC beta. Rescaling legs to zero beta (constant gross) should remove the market drag on the long side.
- sharpe 1.27 · dsr 0.16 · cagr 35.5% · mdd 34.2% · turnover 214x · fees 213.9% · funding -88.5% · long -5.1% / short +214.5% · ic 0.0341
- failed gates: dsr, max_drawdown · no incumbent

---
## Round 1 conclusions (2026-10-20)

- **Costs decide everything at 30m.** 1d reversal has real signal (OOS IC 0.029, hit 55%) but 1-2h rebalancing gives
  560-900x one-sided turnover and fees > 100% of capital. The 4h-horizon LightGBM (IC 0.023, 1667x turnover) fails the same way.
  Rule of thumb confirmed: edge per trade ~ IC x dispersion x 2.8; at 30m that is ~8bp vs 12-15bp round trip.
- **Funding carry works mostly through funding received** (+186% of capital), price drag on shorts negative (IC -0.007);
  Sharpe 0.76, turnover 43x. Labels now include funding (`label.include_funding`), so models can see carry.
- **Ridge > trees** on 1d funding-inclusive targets: trees early-stop after 2-3 rounds (patterns do not persist across
  folds); ridge on ranks reaches Sharpe 1.20, IC 0.034, 11/15 positive folds, turnover 211x.
- **Beta-neutral legs** (STR-03): Sharpe 1.20 -> 1.27, long-leg drag reduced; drawdown unchanged at 34%.
- **DSR**: with a robust (MAD) estimate of trial-SR dispersion, the expected max SR of 9 trials is ~1.75 annualised,
  so 1.27 is not yet distinguishable from luck (DSR 0.16). Next gains must be structural, not parameter luck.
- Next: search around `xs_ridge_1d_bn` (horizon, rebalance, buffer, smoothing, breadth); then new low-turnover factors
  (funding-adjusted momentum, basis/funding divergence, residual momentum), and 5m-derived intrabar features as a separate test.

### 20261020-193618-f9bc07 · xs_ridge_1d_bn~s11.0 · **fail**
- date: 2026-10-20 19:36 · parent: xs_ridge_1d_bn · trials so far: 10
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 100.0, 'strategy.rebalance_bars': 48, 'strategy.exit_buffer': 0.25, 'strategy.smooth_bars': 4, 'strategy.long_frac': 0.3, 'label.horizon': 48}
- sharpe 0.72 · dsr 0.02 · cagr 17.7% · mdd 32.2% · turnover 114x · fees 84.5% · funding -60.9% · long -36.8% / short +116.6% · ic 0.0341
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-193621-986b20 · xs_ridge_1d_bn~s12.0 · **fail**
- date: 2026-10-20 19:36 · parent: xs_ridge_1d_bn · trials so far: 10
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 100.0, 'strategy.rebalance_bars': 16, 'strategy.exit_buffer': 0.35, 'strategy.smooth_bars': 12, 'strategy.long_frac': 0.3, 'label.horizon': 96}
- sharpe 0.66 · dsr 0.01 · cagr 15.6% · mdd 35.4% · turnover 102x · fees 67.5% · funding -60.8% · long -24.6% / short +86.7% · ic 0.0357
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-194613-7933eb · xs_ridge_1d_bn~s12.1 · **fail**
- date: 2026-10-20 19:46 · parent: xs_ridge_1d_bn · trials so far: 12
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 16, 'strategy.exit_buffer': 0.25, 'strategy.smooth_bars': 4, 'strategy.long_frac': 0.3, 'label.horizon': 96}
- sharpe 0.97 · dsr 0.23 · cagr 26.9% · mdd 34.0% · turnover 169x · fees 125.1% · funding -78.1% · long +50.7% / short +90.7% · ic 0.0324
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-194614-bb5f8f · xs_ridge_1d_bn~s11.1 · **fail**
- date: 2026-10-20 19:46 · parent: xs_ridge_1d_bn · trials so far: 12
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 1000.0, 'strategy.rebalance_bars': 16, 'strategy.exit_buffer': 0.35, 'strategy.smooth_bars': 12, 'strategy.long_frac': 0.1, 'label.horizon': 48}
- sharpe 0.85 · dsr 0.18 · cagr 27.5% · mdd 46.3% · turnover 86x · fees 80.9% · funding -79.3% · long +37.1% / short +107.8% · ic 0.0362
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-195616-a2b2ea · xs_ridge_1d_bn~s12.2 · **fail**
- date: 2026-10-20 19:56 · parent: xs_ridge_1d_bn · trials so far: 14
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 16, 'strategy.exit_buffer': 0.25, 'strategy.smooth_bars': 4, 'strategy.long_frac': 0.1, 'label.horizon': 96}
- sharpe 0.98 · dsr 0.56 · cagr 32.8% · mdd 33.9% · turnover 153x · fees 161.3% · funding -113.8% · long +110.9% / short +72.4% · ic 0.0324
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-195617-491ee6 · xs_ridge_1d_bn~s11.2 · **fail**
- date: 2026-10-20 19:56 · parent: xs_ridge_1d_bn · trials so far: 14
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 100.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.25, 'strategy.smooth_bars': 4, 'strategy.long_frac': 0.3, 'label.horizon': 48}
- sharpe 1.47 · dsr 0.85 · cagr 45.6% · mdd 37.8% · turnover 193x · fees 195.3% · funding -89.3% · long +112.1% / short +194.8% · ic 0.0341
- failed gates: dsr, max_drawdown · no incumbent

### 20261020-200650-3d3d43 · xs_ridge_1d_bn~s12.3 · **fail**
- date: 2026-10-20 20:06 · parent: xs_ridge_1d_bn · trials so far: 16
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 16, 'strategy.exit_buffer': 0.25, 'strategy.smooth_bars': 4, 'strategy.long_frac': 0.1, 'label.horizon': 48}
- sharpe 1.09 · dsr 0.61 · cagr 37.7% · mdd 40.4% · turnover 161x · fees 171.8% · funding -105.5% · long +109.4% / short +113.5% · ic 0.0308
- failed gates: dsr, max_drawdown · no incumbent

### 20261020-200652-0a9cd5 · xs_ridge_1d_bn~s11.3 · **fail**
- date: 2026-10-20 20:06 · parent: xs_ridge_1d_bn · trials so far: 16
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 100.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.25, 'strategy.smooth_bars': 4, 'strategy.long_frac': 0.3, 'label.horizon': 96}
- sharpe 1.08 · dsr 0.59 · cagr 30.3% · mdd 39.5% · turnover 181x · fees 139.5% · funding -83.1% · long +52.0% / short +116.3% · ic 0.0357
- failed gates: dsr, max_drawdown · no incumbent

### 20261020-201650-1f11d3 · xs_ridge_1d_bn~s12.4 · **fail**
- date: 2026-10-20 20:16 · parent: xs_ridge_1d_bn · trials so far: 18
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 16, 'strategy.exit_buffer': 0.15, 'strategy.smooth_bars': 4, 'strategy.long_frac': 0.1, 'label.horizon': 96}
- sharpe 0.76 · dsr 0.34 · cagr 22.6% · mdd 51.3% · turnover 229x · fees 190.5% · funding -88.9% · long +67.0% / short +44.5% · ic 0.0324
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-201653-a63d65 · xs_ridge_1d_bn~s11.4 · **fail**
- date: 2026-10-20 20:16 · parent: xs_ridge_1d_bn · trials so far: 18
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 100.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.25, 'strategy.smooth_bars': 12, 'strategy.long_frac': 0.3, 'label.horizon': 48}
- sharpe 1.31 · dsr 0.73 · cagr 39.0% · mdd 36.0% · turnover 162x · fees 155.4% · funding -83.6% · long +43.1% / short +199.5% · ic 0.0341
- failed gates: dsr, max_drawdown · no incumbent

### 20261020-202628-d3be5e · xs_ridge_1d_bn~s12.5 · **fail**
- date: 2026-10-20 20:26 · parent: xs_ridge_1d_bn · trials so far: 20
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.35, 'strategy.smooth_bars': 1, 'strategy.long_frac': 0.1, 'label.horizon': 96}
- sharpe 1.14 · dsr 0.61 · cagr 40.3% · mdd 35.0% · turnover 146x · fees 165.2% · funding -122.7% · long +138.1% / short +104.6% · ic 0.0324
- failed gates: dsr, max_drawdown · no incumbent

### 20261020-202634-fa57da · xs_ridge_1d_bn~s11.5 · **fail**
- date: 2026-10-20 20:26 · parent: xs_ridge_1d_bn · trials so far: 20
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 100.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.25, 'strategy.smooth_bars': 4, 'strategy.long_frac': 0.1, 'label.horizon': 48}
- sharpe 0.94 · dsr 0.47 · cagr 31.0% · mdd 39.9% · turnover 181x · fees 186.3% · funding -78.2% · long -3.9% / short +178.3% · ic 0.0341
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-203641-050cce · xs_ridge_1d_bn~s12.6 · **fail**
- date: 2026-10-20 20:36 · parent: xs_ridge_1d_bn · trials so far: 22
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 100.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.35, 'strategy.smooth_bars': 1, 'strategy.long_frac': 0.1, 'label.horizon': 96}
- sharpe 0.81 · dsr 0.34 · cagr 25.2% · mdd 39.6% · turnover 135x · fees 114.9% · funding -76.3% · long -8.1% / short +131.7% · ic 0.0357
- failed gates: sharpe, dsr, max_drawdown · no incumbent

### 20261020-203642-57357f · xs_ridge_1d_bn~s11.6 · **champion**
- date: 2026-10-20 20:36 · parent: xs_ridge_1d_bn · trials so far: 22
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.35, 'strategy.smooth_bars': 1, 'strategy.long_frac': 0.2, 'label.horizon': 96}
- sharpe 1.82 · dsr 0.93 · cagr 57.9% · mdd 29.2% · turnover 167x · fees 258.2% · funding -186.6% · long +261.9% / short +167.9% · ic 0.0324
- failed gates: none · no incumbent

### 20261020-204635-b1a348 · xs_ridge_1d_bn~s11.7 · **fail**
- date: 2026-10-20 20:46 · parent: xs_ridge_1d_bn · trials so far: 24
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 16, 'strategy.exit_buffer': 0.35, 'strategy.smooth_bars': 1, 'strategy.long_frac': 0.2, 'label.horizon': 96}
- sharpe 1.56 · dsr 0.84 · cagr 46.5% · mdd 24.7% · turnover 142x · fees 162.3% · funding -143.5% · long +133.3% / short +159.4% · ic 0.0324
- failed gates: dsr · sharpe 1.56 vs champion 1.82 (margin -0.26, need 0.15)

### 20261020-204636-53624a · xs_ridge_1d_bn~s12.7 · **fail**
- date: 2026-10-20 20:46 · parent: xs_ridge_1d_bn · trials so far: 24
- hypothesis: search around xs_ridge_1d_bn: {'model.ridge_alpha': 10.0, 'strategy.rebalance_bars': 8, 'strategy.exit_buffer': 0.35, 'strategy.smooth_bars': 1, 'strategy.long_frac': 0.1, 'label.horizon': 48}
- sharpe 1.14 · dsr 0.59 · cagr 40.2% · mdd 37.1% · turnover 164x · fees 192.9% · funding -123.4% · long +86.3% / short +160.6% · ic 0.0308
- failed gates: dsr, max_drawdown · sharpe 1.14 vs champion 1.82 (margin -0.68, need 0.15)

## Round 2 conclusions (search around xs_ridge_1d_bn, 16 trials)

- **Champion: `20261020-203642-57357f` (xs_ridge_1d_bn~s11.6)** — all gates pass: sharpe 1.82, DSR 0.926 (25 trials counted),
  max DD 29.2%, CAGR 57.9%, 14/15 folds positive, turnover 167x, avg hold ~3.4 days, OOS IC 0.032.
  Changes vs parent: label horizon 48 -> 96 (2 days), ridge alpha 100 -> 10, exit buffer 0.25 -> 0.35, no score smoothing.
- Neighbour s11.7 (sharpe 1.56, DD 24.7%) is also strong -> the gain sits in a region (longer horizon + wide buffer), not a single lucky point.
- Both legs now earn (long +262%, short +168% of capital gross); funding received +187%; fees -258% remain the largest cost.
- Caveat: DSR barely clears 0.90 after 25 trials. Next step on the ladder is a budgeted holdout look (2025-10 .. 2026-09, includes the 2025-10-10 liquidation cascade).

## Holdout look #1 (2026-10-20) — champion 20261020-203642-57357f: **FAIL**

- Holdout 2025-10 .. 2026-09: sharpe 0.41 (need >= 0.73 = 0.4 x 1.82), max DD 38.1%, total +8.5%, turnover 81x.
- Quarters: 2025Q4 -2.3% (sharpe 0.08, DD 22%), 2026Q1 -17.5% (sharpe -2.5), 2026Q2 +32.6% (sharpe 3.9), 2026Q3 +2.3%.
- OOS IC fell from 0.032 (research) to 0.006; funding still received (+39% of capital), long leg -21%, short leg +23%.
- Reading: the edge selected by search over 25 trials did not carry into the new regime; carry survived, the
  cross-sectional return forecast mostly did not. The gate did its job (no promotion to paper on research results alone).
- Rules from here: no tuning against the holdout (2 looks left in this 30-day window). Next research must target
  robustness inside the research period: per-year IC stability of features, adaptive vs. long training windows,
  ensembling the neighbouring good region instead of picking the single best point, and carry-only vs. carry+forecast blends.

### 20261020-224825-a72660 · r3_champ_win12 · **fail**
- date: 2026-10-20 22:48 · parent: 20261020-203642-57357f · trials so far: 26
- hypothesis: Regimes shift (2022 strong, 2023-24 weak, holdout weak): a 12-month rolling training window adapts faster.
- sharpe 1.49 · dsr 0.77 · cagr 44.3% · mdd 34.5% · turnover 162x · fees 163.7% · funding -99.6% · long +119.9% / short +171.9% · ic 0.0225
- failed gates: dsr, max_drawdown · sharpe 1.49 vs champion 1.82 (margin -0.33, need 0.15)

### 20261020-224825-ef88df · r3_champ_stable · **fail**
- date: 2026-10-20 22:48 · parent: 20261020-203642-57357f · trials so far: 26
- hypothesis: Holdout IC collapsed (0.032 -> 0.006). Keep only factors whose yearly IC sign is consistent inside each fold training window (no look at test data), so the forecast rests on persistent effects.
- sharpe 1.45 · dsr 0.77 · cagr 43.1% · mdd 25.6% · turnover 162x · fees 175.6% · funding -155.0% · long +115.6% / short +158.2% · ic 0.0316
- failed gates: dsr · sharpe 1.45 vs champion 1.82 (margin -0.37, need 0.15)

### 20261020-230207-a416a8 · r3_champ_win36 · **fail**
- date: 2026-10-20 23:02 · parent: 20261020-203642-57357f · trials so far: 28
- hypothesis: Counter-hypothesis: a longer 36-month window averages over regimes and is more stable.
- sharpe 1.77 · dsr 0.88 · cagr 56.1% · mdd 37.3% · turnover 180x · fees 271.6% · funding -236.8% · long +383.0% / short +42.2% · ic 0.0353
- failed gates: dsr, max_drawdown · sharpe 1.77 vs champion 1.82 (margin -0.05, need 0.15)

### 20261020-230849-d4f992 · r3_champ_ens · **fail**
- date: 2026-10-20 23:08 · parent: 20261020-203642-57357f · trials so far: 29
- hypothesis: Reduce selection variance: average the standardised scores of the good neighbourhood (horizon 1d/2d x ridge alpha 10/100) instead of trusting the single best search point.
- sharpe 1.44 · dsr 0.73 · cagr 41.5% · mdd 27.4% · turnover 168x · fees 173.9% · funding -119.6% · long +75.5% / short +189.3% · ic 0.0350
- failed gates: dsr · sharpe 1.44 vs champion 1.82 (margin -0.38, need 0.15)

### 20261020-231849-b7adfc · r3_carry_blend · **fail**
- date: 2026-10-20 23:18 · parent: 20261020-203642-57357f · trials so far: 30
- hypothesis: Holdout kept the funding income but lost the forecast IC. Anchor the score on 7d funding carry (model-free, persistent) and blend in the ridge forecast at equal weight, so the book degrades to carry when the forecast fades.
- sharpe 0.53 · dsr 0.14 · cagr 11.2% · mdd 47.9% · turnover 40x · fees 26.4% · funding -161.5% · long -31.7% / short +60.0% · ic 0.0189
- failed gates: sharpe, dsr, max_drawdown · sharpe 0.53 vs champion 1.82 (margin -1.29, need 0.15)

### 20261021-105741-8b52ed · r3_champ_expand · **fail**
- date: 2026-10-21 10:57 · parent: 20261020-203642-57357f · trials so far: 31
- hypothesis: Expanding window: use all past data; most stable estimate if effects are persistent.
- sharpe 1.75 · dsr 0.88 · cagr 55.1% · mdd 47.1% · turnover 183x · fees 264.2% · funding -265.4% · long +339.4% / short +61.9% · ic 0.0386
- failed gates: dsr, max_drawdown · sharpe 1.75 vs champion 1.82 (margin -0.07, need 0.15)

### 20261021-105742-b7adfc · r3_carry_blend · **fail**
- date: 2026-10-21 10:57 · parent: 20261020-203642-57357f · trials so far: 30
- hypothesis: Holdout kept the funding income but lost the forecast IC. Anchor the score on 7d funding carry (model-free, persistent) and blend in the ridge forecast at equal weight, so the book degrades to carry when the forecast fades.
- sharpe 0.53 · dsr 0.14 · cagr 11.2% · mdd 47.9% · turnover 40x · fees 26.4% · funding -161.5% · long -31.7% / short +60.0% · ic 0.0189
- failed gates: sharpe, dsr, max_drawdown · sharpe 0.53 vs champion 1.82 (margin -1.29, need 0.15)

## Round 3 — robustness review after the failed holdout (7 trials, research period only)

Method review: structure choices made by looking at the full 2022-2025 walk-forward (feature set, model family,
horizon) are soft selection on the same period we report as OOS; DSR only counts specs. Hence this round moved
selection *inside* each fold (stable-sign feature selection on the training window only) and tested variance
reducers (training window, neighbourhood ensemble, carry anchor). No holdout was used.

| variant | sharpe | max DD | 2023-24 mean fold SR | min fold SR | OOS IC |
|---|---|---|---|---|---|
| champion s11.6 | 1.82 | 29.2% | 1.08 | -0.42 | 0.032 |
| in-fold stable selection | 1.45 | 25.6% | 1.19 | -1.90 | 0.032 |
| 12m window | 1.49 | 34.5% | 0.58 | -3.03 | 0.023 |
| 36m window | 1.77 | 37.3% | 0.65 | -2.28 | 0.035 |
| expanding window | 1.75 | 47.1% | 0.41 | -4.42 | 0.039 |
| 4-member ensemble | 1.44 | 27.4% | 0.80 | -1.28 | 0.035 |
| carry + ridge blend | 0.53 | 47.9% | 0.59 | -4.42 | 0.019 |

Conclusions
- Every reasonable variant of this information set lands at research SR 1.4-1.8; picking among them is noise
  next to the research->holdout gap (1.82 -> 0.41). The limit is the information, not the model plumbing.
- Longer windows raise IC (0.039 expanding) but blow up in 2023-24 (DD 47%): the edge is regime-dependent
  (strong in high-dispersion 2022 / 2025, weak in 2023-24 and in the holdout).
- Naive carry blending hurts (squeeze episodes dominate when carry is the anchor at this rebalance cadence).
- None beats the champion by the 0.15 margin; champion unchanged. No new holdout look is justified.
- Next: new information rather than new plumbing — positioning data (open interest, top-trader and taker
  long/short ratios from the `metrics` archive), market-wide state (breadth, aggregate funding, dispersion; DEC-03)
  as regime conditioners, and 5m intrabar features.

### 20261021-120124-c9bdf9 · r4_disp_scale · **fail**
- date: 2026-10-21 12:01 · parent: 20261020-203642-57357f · trials so far: 32
- hypothesis: The xs edge is strong in high-dispersion regimes (2022, 2025) and weak in low-dispersion ones (2023-24, holdout) while costs stay constant: size the book by current dispersion vs its 30d median.
- sharpe 1.11 · dsr 0.49 · cagr 28.4% · mdd 29.4% · turnover 198x · fees 159.9% · funding -94.0% · long +64.7% / short +91.6% · ic 0.0318
- failed gates: dsr · sharpe 1.11 vs champion 1.82 (margin -0.71, need 0.15)

### 20261021-121857-2084da · r4_regime_lgbm · **fail**
- date: 2026-10-21 12:18 · parent: 20261020-203642-57357f · trials so far: 33
- hypothesis: Trees can condition on regime directly: rank features plus raw market-state features, heavily regularised, corr early stopping.
- sharpe -0.21 · dsr 0.01 · cagr -7.8% · mdd 57.0% · turnover 142x · fees 62.4% · funding -12.6% · long -123.1% / short +69.1% · ic 0.0170
- failed gates: sharpe, dsr, max_drawdown, fold_consistency · sharpe -0.21 vs champion 1.82 (margin -2.03, need 0.15)

### 20261021-122024-ca6ea0 · r4_control_89 · **fail**
- date: 2026-10-21 12:20 · parent: 20261020-203642-57357f · trials so far: 33
- hypothesis: Control: champion spec unchanged; the feature selector now also picks up the 7 new broadcast market-state columns.
- sharpe 1.35 · dsr 0.66 · cagr 38.7% · mdd 30.6% · turnover 171x · fees 165.1% · funding -119.4% · long +98.1% / short +144.6% · ic 0.0318
- failed gates: dsr, max_drawdown · sharpe 1.35 vs champion 1.82 (margin -0.46, need 0.15)

### 20261021-122503-0abcb3 · r4_regime_interact · **fail**
- date: 2026-10-21 12:25 · parent: 20261020-203642-57357f · trials so far: 35
- hypothesis: Let the linear model condition each factor on the regime: add factor x dispersion-ratio and factor x aggregate-funding-z interactions.
- sharpe 1.44 · dsr 0.71 · cagr 43.1% · mdd 28.9% · turnover 236x · fees 334.8% · funding -144.3% · long +147.6% / short +140.2% · ic 0.0346
- failed gates: dsr · sharpe 1.44 vs champion 1.82 (margin -0.38, need 0.15)

## Round 4 — market-state (regime) information (DEC-03)

New broadcast features over the 120 most liquid names (identical in research and live panels):
`mkt_breadth_48`, `mkt_ret_48`, `xs_disp_48`, `xs_disp_ratio`, `mkt_funding`, `mkt_funding_z`, `btc_rv_336`.
Broadcast columns cannot rank coins, so regime enters via (a) exposure scaling, (b) factor x regime interactions, (c) trees.

| variant | sharpe | max DD | note |
|---|---|---|---|
| champion (82 cols) | 1.82 | 29.2% | |
| control: champion + 7 broadcast cols | 1.35 | 30.6% | selector `all` picked them up; nothing else changed |
| factor x (dispersion, aggregate funding) | 1.44 | 28.9% | +0.09 vs control: noise level |
| exposure scaled by dispersion ratio | 1.11 | 29.4% | worse |
| LightGBM with regime features | -0.21 | 57.0% | trees fail again |

- Hypothesis refuted: corr(fold sharpe, quarter mean dispersion) = 0.02. Dispersion does not explain the regime dependence.
- Key finding: adding 7 near-irrelevant columns moves the champion from 1.82 to 1.35. The champion is a sharp peak
  (selection luck), consistent with the holdout failure. Harness gap closed: new `rsi robust` (feature-dropout
  re-runs, excluded from DSR trial counts) is now a required step before any holdout look.

### 20261021-130643-0f3777 · xs_ridge_1d_bn~s11.6~robust0 · **fail**
- date: 2026-10-21 13:06 · parent: 20261020-203642-57357f · trials so far: 36
- hypothesis: robustness: 15% feature dropout of 20261020-203642-57357f
- sharpe 1.77 · dsr 0.86 · cagr 55.7% · mdd 28.3% · turnover 169x · fees 205.6% · funding -145.2% · long +222.8% / short +195.0% · ic 0.0328
- failed gates: dsr · sharpe 1.77 vs champion 1.82 (margin -0.05, need 0.15)

### 20261021-130644-0707f4 · xs_ridge_1d_bn~s11.6~robust0 · **fail**
- date: 2026-10-21 13:06 · parent: 20261020-203642-57357f · trials so far: 36
- hypothesis: robustness: 15% feature dropout of 20261020-203642-57357f
- sharpe 1.11 · dsr 0.47 · cagr 30.6% · mdd 32.8% · turnover 203x · fees 177.0% · funding -103.4% · long +44.6% / short +123.8% · ic 0.0327
- failed gates: dsr, max_drawdown · sharpe 1.11 vs champion 1.82 (margin -0.70, need 0.15)

### 20261021-131507-c75d88 · xs_ridge_1d_bn~s11.6~robust1 · **fail**
- date: 2026-10-21 13:15 · parent: 20261020-203642-57357f · trials so far: 36
- hypothesis: robustness: 15% feature dropout of 20261020-203642-57357f
- sharpe 1.65 · dsr 0.86 · cagr 50.9% · mdd 23.0% · turnover 187x · fees 234.1% · funding -135.0% · long +108.2% / short +250.7% · ic 0.0320
- failed gates: dsr · sharpe 1.65 vs champion 1.82 (margin -0.17, need 0.15)

### 20261021-131508-f606d7 · xs_ridge_1d_bn~s11.6~robust1 · **fail**
- date: 2026-10-21 13:15 · parent: 20261020-203642-57357f · trials so far: 36
- hypothesis: robustness: 15% feature dropout of 20261020-203642-57357f
- sharpe 1.43 · dsr 0.69 · cagr 42.5% · mdd 34.6% · turnover 195x · fees 189.7% · funding -104.9% · long +129.0% / short +144.3% · ic 0.0310
- failed gates: dsr, max_drawdown · sharpe 1.43 vs champion 1.82 (margin -0.38, need 0.15)

### 20261021-132330-78ea10 · xs_ridge_1d_bn~s11.6~robust2 · **fail**
- date: 2026-10-21 13:23 · parent: 20261020-203642-57357f · trials so far: 36
- hypothesis: robustness: 15% feature dropout of 20261020-203642-57357f
- sharpe 1.67 · dsr 0.84 · cagr 51.3% · mdd 26.0% · turnover 162x · fees 186.8% · funding -139.7% · long +98.1% / short +254.7% · ic 0.0349
- failed gates: dsr · sharpe 1.67 vs champion 1.82 (margin -0.15, need 0.15)

### 20261021-132331-422a70 · xs_ridge_1d_bn~s11.6~robust2 · **fail**
- date: 2026-10-21 13:23 · parent: 20261020-203642-57357f · trials so far: 36
- hypothesis: robustness: 15% feature dropout of 20261020-203642-57357f
- sharpe 1.40 · dsr 0.68 · cagr 40.4% · mdd 33.9% · turnover 164x · fees 156.9% · funding -107.7% · long +22.8% / short +226.2% · ic 0.0327
- failed gates: dsr, max_drawdown · sharpe 1.40 vs champion 1.82 (margin -0.42, need 0.15)
- Robustness of champion 20261020-203642-57357f (6 re-runs, 15% feature dropout, 2 seeds):
  sharpe [1.77, 1.65, 1.67, 1.11, 1.43, 1.40] -> median 1.54, min 1.11; max DD median 30.4% (gate 30%) -> borderline FAIL.
  Reading: a real research-period edge (~1.5) exists and survives feature dropout, but the reported 1.82 carries
  ~0.3 of selection optimism. The holdout drop (0.41) is a second, separate problem: decay over time, not feature luck.
