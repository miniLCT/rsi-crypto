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
