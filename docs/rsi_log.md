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
