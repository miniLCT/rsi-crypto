"""Minimal async Binance USDT-M futures REST client (public market data + signed trading)."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import time
import urllib.parse
from pathlib import Path

import aiohttp

from rsi.data.vision import ssl_context
from rsi.utils.log import get_logger

log = get_logger("binance")

ENDPOINTS = {
    "mainnet": "https://fapi.binance.com",
    "testnet": "https://demo-fapi.binance.com",
}


class BinanceError(RuntimeError):
    def __init__(self, status: int, payload):
        super().__init__(f"HTTP {status}: {payload}")
        self.status = status
        self.payload = payload
        self.code = payload.get("code") if isinstance(payload, dict) else None


class BinanceFutures:
    def __init__(self, env: str = "mainnet", api_key: str | None = None, api_secret: str | None = None,
                 private_key_path: str | None = None, recv_window: int = 5000, max_weight_1m: int = 1800):
        self.base = ENDPOINTS[env]
        self.env = env
        self.api_key = api_key
        self.api_secret = api_secret
        self._ed25519 = None
        if private_key_path:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            self._ed25519 = load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
        self.recv_window = recv_window
        self.max_weight = max_weight_1m
        self.used_weight = 0
        self._time_offset = 0
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Semaphore(16)

    async def __aenter__(self):
        await self.session()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
            self._session = aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=ssl_context(), limit=32),
                                                  timeout=aiohttp.ClientTimeout(total=20))
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------ transport
    def _sign(self, params: dict) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        params["timestamp"] = int(time.time() * 1000) + self._time_offset
        params["recvWindow"] = self.recv_window
        qs = urllib.parse.urlencode(params)
        if self._ed25519 is not None:
            params["signature"] = base64.b64encode(self._ed25519.sign(qs.encode())).decode()
        elif self.api_secret:
            params["signature"] = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        else:
            raise RuntimeError("signed endpoint requires api credentials")
        return params

    async def request(self, method: str, path: str, params: dict | None = None, signed: bool = False, retries: int = 3):
        s = await self.session()
        for attempt in range(retries + 1):
            if self.used_weight > self.max_weight:
                wait = 61 - time.time() % 60
                log.warning(f"weight {self.used_weight} near limit, sleeping {wait:.0f}s")
                await asyncio.sleep(wait)
                self.used_weight = 0
            p = self._sign(dict(params or {})) if signed else {k: v for k, v in (params or {}).items() if v is not None}
            try:
                async with self._lock, s.request(method, self.base + path, params=p) as r:
                    self.used_weight = int(r.headers.get("X-MBX-USED-WEIGHT-1M", self.used_weight))
                    payload = await r.json(content_type=None)
                    if r.status == 200:
                        return payload
                    if r.status in (418, 429):
                        await asyncio.sleep(int(r.headers.get("Retry-After", 30)))
                        continue
                    if isinstance(payload, dict) and payload.get("code") == -1021 and attempt < retries:
                        await self.sync_time()
                        continue
                    if r.status >= 500 and attempt < retries:
                        await asyncio.sleep(1 + attempt)
                        continue
                    raise BinanceError(r.status, payload)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == retries:
                    raise
                log.warning(f"{method} {path} transport error {e!r}, retry {attempt + 1}")
                await asyncio.sleep(1 + attempt)
        raise RuntimeError("unreachable")

    async def sync_time(self) -> None:
        t0 = time.time()
        server = (await self.request("GET", "/fapi/v1/time"))["serverTime"]
        self._time_offset = int(server - (t0 + time.time()) / 2 * 1000)

    # ------------------------------------------------------------------ market data
    async def exchange_info(self) -> dict:
        return await self.request("GET", "/fapi/v1/exchangeInfo")

    async def klines(self, symbol: str, interval: str, limit: int = 1500, start: int | None = None, end: int | None = None) -> list:
        return await self.request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit,
                                                             "startTime": start, "endTime": end})

    async def premium_klines(self, symbol: str, interval: str, limit: int = 1500, start: int | None = None) -> list:
        return await self.request("GET", "/fapi/v1/premiumIndexKlines", {"symbol": symbol, "interval": interval,
                                                                        "limit": limit, "startTime": start})

    async def funding_history(self, symbol: str, limit: int = 1000, start: int | None = None) -> list:
        return await self.request("GET", "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit, "startTime": start})

    async def funding_info(self) -> list:
        return await self.request("GET", "/fapi/v1/fundingInfo")

    async def ticker_price(self) -> dict[str, float]:
        rows = await self.request("GET", "/fapi/v2/ticker/price")
        return {r["symbol"]: float(r["price"]) for r in rows}

    async def book_ticker(self) -> dict[str, tuple[float, float]]:
        rows = await self.request("GET", "/fapi/v1/ticker/bookTicker")
        return {r["symbol"]: (float(r["bidPrice"]), float(r["askPrice"])) for r in rows}

    # ------------------------------------------------------------------ account / trading
    async def account(self) -> dict:
        return await self.request("GET", "/fapi/v3/account", signed=True)

    async def positions(self) -> list:
        return await self.request("GET", "/fapi/v3/positionRisk", signed=True)

    async def position_mode(self) -> dict:
        return await self.request("GET", "/fapi/v1/positionSide/dual", signed=True)

    async def set_one_way_mode(self) -> None:
        try:
            await self.request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"}, signed=True)
        except BinanceError as e:
            if e.code != -4059:  # no need to change
                raise

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self.request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str = "CROSSED") -> None:
        try:
            await self.request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type}, signed=True)
        except BinanceError as e:
            if e.code != -4046:  # already set
                raise

    async def market_order(self, symbol: str, side: str, quantity: str, client_id: str, reduce_only: bool = False) -> dict:
        return await self.request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity, "newClientOrderId": client_id,
            "reduceOnly": "true" if reduce_only else None, "newOrderRespType": "RESULT"}, signed=True, retries=0)

    async def get_order(self, symbol: str, client_id: str) -> dict:
        return await self.request("GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": client_id}, signed=True)

    async def user_trades(self, symbol: str, order_id: int) -> list:
        return await self.request("GET", "/fapi/v1/userTrades", {"symbol": symbol, "orderId": order_id}, signed=True)

    async def income(self, income_type: str | None = None, start: int | None = None, limit: int = 1000) -> list:
        return await self.request("GET", "/fapi/v1/income", {"incomeType": income_type, "startTime": start, "limit": limit},
                                  signed=True)


def kline_rows_to_frame(rows: list):
    import pandas as pd
    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                                     "taker_buy_volume", "taker_buy_quote_volume"])
    df = pd.DataFrame(rows).iloc[:, :11]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                  "taker_buy_volume", "taker_buy_quote_volume"]
    df = df.drop(columns=["close_time"])
    df["open_time"] = df["open_time"].astype("int64")
    for c in df.columns[1:]:
        df[c] = df[c].astype("float64")
    return df
