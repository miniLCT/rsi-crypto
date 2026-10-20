# rsi-crypto

Research harness and trading engine for Binance USDT-M perpetual futures, built for
fast, recursive self-improvement (RSI) iterations: a locked evaluation contract,
an append-only experiment ledger, promotion gates, and one decision path shared by
backtest, paper, testnet and mainnet.

- Specification: [`docs/spec.md`](docs/spec.md) · plan: [`docs/plan.md`](docs/plan.md) · architecture: [`docs/architecture.md`](docs/architecture.md)
- Research protocol for humans and agents: [`AGENTS.md`](AGENTS.md) · iteration log: [`docs/rsi_log.md`](docs/rsi_log.md)

## Quickstart

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ln -sf ../../scripts/check_secrets.py .git/hooks/pre-commit   # blocks credentials / personal data in commits

rsi data sync && rsi data ingest && rsi data quality      # mirror + lake (data.binance.vision, incl. delisted)
rsi data panel && rsi features build --jobs 4             # aligned panel + feature cache
rsi run experiments/xs_factor_carry.yaml                  # walk-forward, costs, gates, ledger
rsi ledger ls --by sharpe
rsi search experiments/xs_lgbm_base.yaml --n 20           # automatic proposer
rsi holdout <run_id> && rsi export <run_id>               # budgeted holdout, deployable artifact
rsi live -c configs/live/paper.yaml                       # paper | testnet | mainnet
```

## Credentials and privacy

- API keys are read **only from environment variables** named in `configs/live/*.yaml`
  (`BINANCE_API_KEY`, `BINANCE_API_SECRET` or `BINANCE_ED25519_KEY` = path to a PEM file,
  `RSI_ALERT_WEBHOOK`). Never put keys, webhook URLs or account identifiers in configs, code or docs.
- Keep key files outside the repository or under `secrets/` (git-ignored). `*.pem`, `*.key`, `.env*`
  are ignored as well.
- Data, caches, ledgers, run artifacts, exported models and live state (`data/`, `cache/`, `runs/`,
  `models/`, `live_state/`, `logs/`) are reproducible or private and are never committed.
- `scripts/check_secrets.py` (pre-commit hook) rejects private keys, token-like strings, webhook URLs,
  non-noreply e-mail addresses and absolute home paths. Run `python scripts/check_secrets.py --all` before publishing.
- Use a GitHub noreply address for commits: `git config user.email <id>+<user>@users.noreply.github.com`.

## Disclaimer

Research software. Trading leveraged perpetual futures can lose more than the deposited margin;
nothing here is investment advice.
