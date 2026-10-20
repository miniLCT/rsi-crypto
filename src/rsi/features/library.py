"""Core feature library (30m bars). Windows are in bars: 2=1h, 8=4h, 48=1d, 336=7d, 1440=30d."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rsi.features.registry import FrameView, register, register_partial

BTC = "BTCUSDT"
ETH = "ETHUSDT"


def _safe_log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


# ---------------------------------------------------------------- momentum / reversal
def _ret(fv: FrameView, n: int) -> pd.DataFrame:
    return np.log(fv.close / fv.close.shift(n))


for _n in (1, 2, 4, 8, 16, 48, 96, 336, 672):
    register_partial(f"ret_{_n}", _ret, lookback=_n + 1, group="momentum", doc=f"log return over {_n} bars", n=_n)


def _ret_skip(fv: FrameView, n: int, skip: int) -> pd.DataFrame:
    return np.log(fv.close.shift(skip) / fv.close.shift(n))


register_partial("ret_336_skip48", _ret_skip, lookback=337, group="momentum",
                 doc="7d momentum skipping the last day", n=336, skip=48)


def _ret_voladj(fv: FrameView, n: int, vol_n: int) -> pd.DataFrame:
    vol = fv.logret.rolling(vol_n, min_periods=vol_n).std()
    return np.log(fv.close / fv.close.shift(n)) / (vol * np.sqrt(n))


for _n in (8, 48, 336):
    register_partial(f"ret_voladj_{_n}", _ret_voladj, lookback=max(_n, 336) + 1, group="momentum",
                     doc="return scaled by 7d bar vol", n=_n, vol_n=336)


def _ma_gap(fv: FrameView, n: int) -> pd.DataFrame:
    return np.log(fv.close / fv.close.rolling(n, min_periods=n).mean())


for _n in (16, 48, 336, 1440):
    register_partial(f"ma_gap_{_n}", _ma_gap, lookback=_n, group="momentum", doc="log distance to SMA", n=_n)


def _ma_cross(fv: FrameView, fast: int, slow: int) -> pd.DataFrame:
    return np.log(fv.close.rolling(fast, min_periods=fast).mean() / fv.close.rolling(slow, min_periods=slow).mean())


register_partial("ma_cross_8_48", _ma_cross, lookback=48, group="momentum", fast=8, slow=48)
register_partial("ma_cross_48_336", _ma_cross, lookback=336, group="momentum", fast=48, slow=336)


def _rsi(fv: FrameView, n: int) -> pd.DataFrame:
    d = fv.close.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-d).clip(lower=0).rolling(n, min_periods=n).mean()
    return up / (up + dn) - 0.5


for _n in (14, 48, 336):
    register_partial(f"rsi_{_n}", _rsi, lookback=_n + 1, group="momentum", doc="SMA RSI centred at 0", n=_n)


def _range_pos(fv: FrameView, n: int) -> pd.DataFrame:
    hh = fv.high.rolling(n, min_periods=n).max()
    ll = fv.low.rolling(n, min_periods=n).min()
    return (fv.close - ll) / (hh - ll) - 0.5


for _n in (48, 336, 1440):
    register_partial(f"range_pos_{_n}", _range_pos, lookback=_n, group="momentum", doc="stochastic position in n-bar range", n=_n)


def _dist_high(fv: FrameView, n: int) -> pd.DataFrame:
    return np.log(fv.close / fv.high.rolling(n, min_periods=n).max())


def _dist_low(fv: FrameView, n: int) -> pd.DataFrame:
    return np.log(fv.close / fv.low.rolling(n, min_periods=n).min())


for _n in (336, 1440):
    register_partial(f"dist_high_{_n}", _dist_high, lookback=_n, group="momentum", n=_n)
    register_partial(f"dist_low_{_n}", _dist_low, lookback=_n, group="momentum", n=_n)


# ---------------------------------------------------------------- volatility / distribution
def _rv(fv: FrameView, n: int) -> pd.DataFrame:
    return fv.logret.rolling(n, min_periods=n).std()


for _n in (16, 48, 336):
    register_partial(f"rv_{_n}", _rv, lookback=_n + 1, group="volatility", doc="std of bar log returns", n=_n)


def _rv_ratio(fv: FrameView, fast: int, slow: int) -> pd.DataFrame:
    lr = fv.logret
    return np.log(lr.rolling(fast, min_periods=fast).std() / lr.rolling(slow, min_periods=slow).std())


register_partial("rv_ratio_16_336", _rv_ratio, lookback=337, group="volatility", fast=16, slow=336)
register_partial("rv_ratio_48_1440", _rv_ratio, lookback=1441, group="volatility", fast=48, slow=1440)


def _parkinson(fv: FrameView, n: int) -> pd.DataFrame:
    hl = _safe_log(fv.high / fv.low) ** 2
    return np.sqrt(hl.rolling(n, min_periods=n).mean() / (4 * np.log(2)))


for _n in (16, 48):
    register_partial(f"pvol_{_n}", _parkinson, lookback=_n, group="volatility", doc="Parkinson high-low vol", n=_n)


@register("range_expansion", lookback=48, group="volatility", doc="last bar range vs 1d Parkinson vol")
def _range_expansion(fv: FrameView) -> pd.DataFrame:
    hl = _safe_log(fv.high / fv.low)
    return hl / np.sqrt((hl ** 2).rolling(48, min_periods=48).mean())


def _downside_share(fv: FrameView, n: int) -> pd.DataFrame:
    lr = fv.logret
    dn = (lr.clip(upper=0) ** 2).rolling(n, min_periods=n).sum()
    tot = (lr ** 2).rolling(n, min_periods=n).sum()
    return dn / tot - 0.5


register_partial("downside_share_48", _downside_share, lookback=49, group="volatility", n=48)
register_partial("downside_share_336", _downside_share, lookback=337, group="volatility", n=336)


def _skew(fv: FrameView, n: int) -> pd.DataFrame:
    return fv.logret.rolling(n, min_periods=n).skew()


register_partial("skew_336", _skew, lookback=337, group="volatility", n=336)


def _kurt(fv: FrameView, n: int) -> pd.DataFrame:
    return fv.logret.rolling(n, min_periods=n).kurt()


register_partial("kurt_336", _kurt, lookback=337, group="volatility", n=336)


@register("max_ret_48", lookback=49, group="volatility", doc="largest single-bar return over 1d (lottery)")
def _max_ret(fv: FrameView) -> pd.DataFrame:
    return fv.logret.rolling(48, min_periods=48).max()


# ---------------------------------------------------------------- volume / liquidity / order flow
def _qv_ratio(fv: FrameView, n: int, base: int) -> pd.DataFrame:
    qv = fv.quote_volume
    return _safe_log(qv.rolling(n, min_periods=n).mean() / qv.rolling(base, min_periods=base).mean())


for _n in (1, 4, 16, 48):
    register_partial(f"vol_surge_{_n}", _qv_ratio, lookback=336, group="flow", doc="quote volume vs 7d mean", n=_n, base=336)


@register("log_adv_30d", lookback=1440, group="flow", doc="log average daily quote volume (size proxy)")
def _log_adv(fv: FrameView) -> pd.DataFrame:
    return _safe_log(fv.quote_volume.rolling(1440, min_periods=1440).mean() * 48)


def _taker_imb(fv: FrameView, n: int) -> pd.DataFrame:
    qv = fv.quote_volume.rolling(n, min_periods=n).sum()
    tb = fv.taker_buy_quote_volume.rolling(n, min_periods=n).sum()
    return (2 * tb - qv) / qv.where(qv > 0)


for _n in (1, 4, 16, 48, 336):
    register_partial(f"taker_imb_{_n}", _taker_imb, lookback=_n, group="flow", doc="taker buy-sell imbalance", n=_n)


@register("taker_imb_shift", lookback=336, group="flow", doc="1d taker imbalance minus 7d")
def _taker_imb_shift(fv: FrameView) -> pd.DataFrame:
    return _taker_imb(fv, 48) - _taker_imb(fv, 336)


@register("trade_size_z", lookback=336, group="flow", doc="avg trade size (1d) vs 7d, log")
def _trade_size(fv: FrameView) -> pd.DataFrame:
    qv, tr = fv.quote_volume, fv.trades
    s48 = qv.rolling(48, min_periods=48).sum() / tr.rolling(48, min_periods=48).sum().where(lambda x: x > 0)
    s336 = qv.rolling(336, min_periods=336).sum() / tr.rolling(336, min_periods=336).sum().where(lambda x: x > 0)
    return _safe_log(s48 / s336)


@register("amihud_48", lookback=49, group="flow", doc="log Amihud illiquidity over 1d")
def _amihud(fv: FrameView) -> pd.DataFrame:
    ill = fv.logret.abs() / fv.quote_volume.where(fv.quote_volume > 0)
    return _safe_log(ill.rolling(48, min_periods=24).mean() * 1e9)


def _vwap_gap(fv: FrameView, n: int) -> pd.DataFrame:
    vwap = fv.quote_volume.rolling(n, min_periods=n).sum() / fv.volume.rolling(n, min_periods=n).sum()
    return np.log(fv.close / vwap)


register_partial("vwap_gap_48", _vwap_gap, lookback=48, group="flow", n=48)
register_partial("vwap_gap_336", _vwap_gap, lookback=336, group="flow", n=336)


@register("pv_corr_48", lookback=49, group="flow", doc="corr(|return|, volume) over 1d")
def _pv_corr(fv: FrameView) -> pd.DataFrame:
    return fv.logret.abs().rolling(48, min_periods=48).corr(_safe_log(fv.quote_volume))


# ---------------------------------------------------------------- candle structure
@register("clv_4", lookback=4, group="candle", doc="close location value, 2h mean")
def _clv(fv: FrameView) -> pd.DataFrame:
    rng = (fv.high - fv.low).where(lambda x: x > 0)
    return ((2 * fv.close - fv.high - fv.low) / rng).rolling(4, min_periods=4).mean()


@register("upper_wick_16", lookback=16, group="candle", doc="upper shadow share of range, 8h")
def _upper_wick(fv: FrameView) -> pd.DataFrame:
    rng = (fv.high - fv.low).where(lambda x: x > 0)
    top = fv.high - np.maximum(fv.open, fv.close)
    return (top / rng).rolling(16, min_periods=16).mean()


@register("lower_wick_16", lookback=16, group="candle", doc="lower shadow share of range, 8h")
def _lower_wick(fv: FrameView) -> pd.DataFrame:
    rng = (fv.high - fv.low).where(lambda x: x > 0)
    bot = np.minimum(fv.open, fv.close) - fv.low
    return (bot / rng).rolling(16, min_periods=16).mean()


@register("body_ratio_16", lookback=16, group="candle", doc="|body|/range, trendiness of bars")
def _body_ratio(fv: FrameView) -> pd.DataFrame:
    rng = (fv.high - fv.low).where(lambda x: x > 0)
    return ((fv.close - fv.open).abs() / rng).rolling(16, min_periods=16).mean()


# ---------------------------------------------------------------- perp specific: funding & basis
def _funding_8h(fv: FrameView) -> pd.DataFrame:
    def f():
        fr = fv.funding_rate.ffill(limit=47)
        ih = fv.funding_interval_h.ffill(limit=47)
        return fr * 8.0 / ih
    return fv.memo("_funding_8h", f)


@register("funding_last", lookback=48, group="carry", doc="last settled funding, 8h-equivalent")
def _funding_last(fv: FrameView) -> pd.DataFrame:
    return _funding_8h(fv)


@register("funding_avg_7d", lookback=48 + 336, group="carry", doc="mean settled funding over 7d")
def _funding_avg(fv: FrameView) -> pd.DataFrame:
    return _funding_8h(fv).rolling(336, min_periods=48).mean()


@register("funding_chg_1d", lookback=48 + 49, group="carry")
def _funding_chg(fv: FrameView) -> pd.DataFrame:
    f = _funding_8h(fv)
    return f - f.shift(48)


def _prem(fv: FrameView, n: int) -> pd.DataFrame:
    return fv.premium.rolling(n, min_periods=max(1, n // 2)).mean()


for _n in (1, 16, 48, 336):
    register_partial(f"premium_{_n}", _prem, lookback=_n, group="carry", doc="mean perp premium vs index", n=_n)


@register("premium_z_336", lookback=336, group="carry", doc="1d premium z-score vs 7d")
def _prem_z(fv: FrameView) -> pd.DataFrame:
    p = fv.premium
    return (p.rolling(48, min_periods=24).mean() - p.rolling(336, min_periods=168).mean()) / p.rolling(336, min_periods=168).std()


@register("premium_mom_16", lookback=48, group="carry", doc="premium now minus 1d mean")
def _prem_mom(fv: FrameView) -> pd.DataFrame:
    p = fv.premium
    return p.rolling(2, min_periods=1).mean() - p.rolling(48, min_periods=24).mean()


# ---------------------------------------------------------------- market context (BTC) & residuals
def _btc_ret(fv: FrameView, n: int) -> pd.DataFrame:
    c = fv.column("close", BTC)
    return fv.broadcast(np.log(c / c.shift(n)))


for _n in (2, 8, 48, 336):
    register_partial(f"btc_ret_{_n}", _btc_ret, lookback=_n + 1, group="market_ctx", n=_n)


@register("btc_rv_48", lookback=49, group="market_ctx")
def _btc_rv(fv: FrameView) -> pd.DataFrame:
    c = fv.column("close", BTC)
    return fv.broadcast(np.log(c).diff().rolling(48, min_periods=48).std())


def _beta(fv: FrameView, n: int) -> pd.DataFrame:
    def f():
        b = np.log(fv.column("close", BTC)).diff()
        lr = fv.logret
        cov = lr.rolling(n, min_periods=n).cov(b)
        var = b.rolling(n, min_periods=n).var()
        return cov.div(var, axis=0)
    return fv.memo(f"_beta_{n}", f)


@register("beta_btc_336", lookback=337, group="market", doc="7d beta to BTC")
def _beta_336(fv: FrameView) -> pd.DataFrame:
    return _beta(fv, 336)


def _resid_ret(fv: FrameView, n: int) -> pd.DataFrame:
    btc = fv.column("close", BTC)
    br = np.log(btc / btc.shift(n))
    return _ret(fv, n) - _beta(fv, 336).mul(br, axis=0)


for _n in (8, 48, 336):
    register_partial(f"resid_ret_{_n}", _resid_ret, lookback=max(_n, 336) + 1, group="market",
                     doc="return net of BTC beta", n=_n)


@register("corr_btc_336", lookback=337, group="market")
def _corr_btc(fv: FrameView) -> pd.DataFrame:
    b = np.log(fv.column("close", BTC)).diff()
    return fv.logret.rolling(336, min_periods=336).corr(b)


# ---------------------------------------------------------------- calendar / lifecycle
@register("hour_utc", lookback=1, group="calendar")
def _hour(fv: FrameView) -> pd.DataFrame:
    h = pd.Series((fv.times // 1_800_000 % 48) / 2.0, index=fv.times)
    return fv.broadcast(h)


@register("dow_utc", lookback=1, group="calendar")
def _dow(fv: FrameView) -> pd.DataFrame:
    d = pd.Series(((fv.times // 86_400_000) + 3) % 7, index=fv.times).astype(float)  # 0 = Monday
    return fv.broadcast(d)


@register("bars_to_funding", lookback=1, group="calendar", doc="bars until next 8h funding boundary")
def _bars_to_funding(fv: FrameView) -> pd.DataFrame:
    s = pd.Series((16 - (fv.times // 1_800_000) % 16) % 16, index=fv.times).astype(float)
    return fv.broadcast(s)


@register("age_30d", lookback=1440, group="calendar", doc="fraction of last 30d the contract traded (new listings < 1)")
def _age(fv: FrameView) -> pd.DataFrame:
    return fv.close.notna().astype(float).rolling(1440, min_periods=1).mean()
