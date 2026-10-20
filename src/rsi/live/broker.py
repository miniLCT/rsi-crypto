"""Order routing. Paper fills are simulated; Binance fills are real (testnet or mainnet)."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from rsi.live.binance import BinanceError, BinanceFutures
from rsi.utils.log import get_logger

log = get_logger("broker")

OK_STATUS = ("FILLED", "PARTIALLY_FILLED", "NEW")


@dataclass
class Fill:
    symbol: str
    dq: float            # signed executed quantity
    price: float         # average fill price
    fee_usd: float
    client_id: str
    status: str
    ref_price: float = float("nan")
    reduce_only: bool = False

    @property
    def slippage_bps(self) -> float:
        if not math.isfinite(self.ref_price) or self.ref_price <= 0 or self.dq == 0 or not math.isfinite(self.price):
            return float("nan")
        return (self.price / self.ref_price - 1) * 1e4 * (1 if self.dq > 0 else -1)


@dataclass
class SymbolRules:
    step: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: float
    tick: Decimal = Decimal("0.0001")

    def round_qty(self, q: float) -> Decimal:
        d = (Decimal(str(abs(q))) / self.step).to_integral_value(rounding=ROUND_DOWN) * self.step
        return min(d, self.max_qty)

    def round_price(self, p: float) -> Decimal:
        return (Decimal(str(p)) / self.tick).to_integral_value() * self.tick


def parse_rules(exchange_info: dict) -> dict[str, SymbolRules]:
    out = {}
    for s in exchange_info.get("symbols", []):
        f = {x["filterType"]: x for x in s.get("filters", [])}
        lot = f.get("MARKET_LOT_SIZE") or f.get("LOT_SIZE") or {}
        lot2 = f.get("LOT_SIZE", lot)
        step = max(Decimal(lot.get("stepSize", "0.001")), Decimal(lot2.get("stepSize", "0.001")))
        out[s["symbol"]] = SymbolRules(
            step=step if step > 0 else Decimal("0.001"),
            min_qty=Decimal(lot2.get("minQty", "0")),
            max_qty=Decimal(lot.get("maxQty", "1e12")),
            min_notional=float(f.get("MIN_NOTIONAL", {}).get("notional", 5.0)),
            tick=Decimal(f.get("PRICE_FILTER", {}).get("tickSize", "0.0001")),
        )
    return out


def plan_legs(old: float, dq: float) -> list[tuple[float, bool]]:
    """Split a position change into (signed qty, reduce_only) legs; flips become close + open."""
    new = old + dq
    if old != 0 and (new == 0 or math.copysign(1, new) != math.copysign(1, old)):
        legs = [(-old, True)]
        if new != 0:
            legs.append((new, False))
        return legs
    return [(dq, old != 0 and abs(new) < abs(old))]


class PaperBroker:
    """Fills at the current reference price; costs are charged by the Portfolio's cost model."""
    kind = "paper"

    async def execute(self, orders: dict[str, float], ref_px: dict[str, float], bar_no: int,
                      current: dict[str, float] | None = None, tag: str = "rsi", open_deadline: float | None = None) -> list[Fill]:
        return [Fill(s, dq, ref_px[s], 0.0, f"{tag}-{bar_no}-{s}-0", "FILLED", ref_px[s]) for s, dq in orders.items()
                if dq != 0 and math.isfinite(ref_px.get(s, float("nan")))]

    async def sync_stops(self, stops: dict[str, tuple[int, float]]) -> None:
        return None


class BinanceBroker:
    kind = "binance"

    def __init__(self, client: BinanceFutures, rules: dict[str, SymbolRules], leverage: int = 3, taker_fee_bps: float = 5.0):
        if not 1 <= leverage <= 10:
            raise ValueError("DEC-06: exchange leverage must be within 1-10")
        self.c = client
        self.rules = rules
        self.leverage = leverage
        self.fee_bps = taker_fee_bps
        self._configured: set[str] = set()
        self._configured_lev: dict[str, int] = {}
        self.recent: list[bool] = []          # rolling order outcomes for the reject-rate breaker
        self.blacklist: set[str] = set()
        self._stops: dict[str, tuple[str, str, str]] = {}   # symbol -> (id kind, id, price) of the protective stop
        self._brackets: dict[str, list[dict]] = {}
        self.margin_ratio: float | None = None

    async def prepare(self) -> None:
        await self.c.sync_time()
        await self.c.set_one_way_mode()

    async def _ensure_symbol(self, sym: str, notional: float = 0.0) -> None:
        """Cross margin + leverage = min(configured, bracket limit for the intended notional)."""
        lev = self.leverage
        try:
            if not self._brackets:
                rows = await self.c.request("GET", "/fapi/v1/leverageBracket", signed=True)
                self._brackets = {r["symbol"]: r["brackets"] for r in rows}
            for b in self._brackets.get(sym, []):
                if b["notionalFloor"] <= notional < b["notionalCap"]:
                    lev = min(lev, int(b["initialLeverage"]))
                    break
        except BinanceError as e:
            log.warning(f"leverage brackets unavailable: {e}")
        if sym in self._configured and self._configured_lev.get(sym) == lev:
            return
        await self.c.set_margin_type(sym, "CROSSED")
        await self.c.set_leverage(sym, lev)
        self._configured.add(sym)
        self._configured_lev[sym] = lev

    async def sync(self) -> tuple[dict[str, float], float]:
        """Exchange truth: signed position amounts and total margin balance (wallet + unrealised)."""
        acct, pos = await asyncio.gather(self.c.account(), self.c.positions())
        positions = {p["symbol"]: float(p["positionAmt"]) for p in pos if float(p["positionAmt"]) != 0}
        bal = float(acct["totalMarginBalance"])
        self.margin_ratio = float(acct.get("totalMaintMargin", 0) or 0) / bal if bal > 0 else 1.0
        return positions, bal

    def reject_rate(self, window: int = 20) -> float:
        r = self.recent[-window:]
        return 0.0 if not r else 1 - sum(r) / len(r)

    async def _send(self, sym: str, side: str, qty: Decimal, cid: str, reduce_only: bool, ref: float) -> Fill:
        try:
            r = await self.c.market_order(sym, side, format(qty.normalize(), "f"), cid, reduce_only)
        except (BinanceError, asyncio.TimeoutError, OSError) as e:
            # the order may still have been accepted: look it up by client id before deciding
            try:
                r = await self.c.get_order(sym, cid)
            except Exception:  # noqa: BLE001
                return Fill(sym, 0.0, float("nan"), 0.0, cid, f"ERROR:{getattr(e, 'code', '')}:{e}", ref, reduce_only)
        for _ in range(6):
            if r.get("status") in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                break
            await asyncio.sleep(0.5)
            r = await self.c.get_order(sym, cid)
        exq = float(r.get("executedQty", 0) or 0)
        avg = float(r.get("avgPrice", 0) or 0) or float("nan")
        sign = 1 if side == "BUY" else -1
        fee = abs(exq) * (avg if math.isfinite(avg) else ref) * self.fee_bps * 1e-4
        return Fill(sym, sign * exq, avg, fee, cid, r.get("status", "?"), ref, reduce_only)

    async def _route_leg(self, sym: str, q: float, reduce_only: bool, cid: str, ref: float) -> Fill | None:
        rules = self.rules[sym]
        qty = rules.round_qty(q)
        if qty <= 0 or qty < rules.min_qty:
            return None
        if not reduce_only and float(qty) * ref < rules.min_notional:
            return None
        f = await self._send(sym, "BUY" if q > 0 else "SELL", qty, cid, reduce_only, ref)
        if f.status.startswith("ERROR") and not reduce_only:
            # one retry with a smaller size (e.g. margin / max-qty rejections), then blacklist
            f2 = await self._send(sym, "BUY" if q > 0 else "SELL", rules.round_qty(float(qty) * 0.5), cid + "r", reduce_only, ref)
            if f2.status.startswith("ERROR"):
                self.blacklist.add(sym)
                log.error(f"{sym} rejected twice, blacklisted: {f2.status}")
            f = f2
        self.recent.append(f.status in OK_STATUS)
        return f

    async def execute(self, orders: dict[str, float], ref_px: dict[str, float], bar_no: int,
                      current: dict[str, float] | None = None, tag: str = "rsi", open_deadline: float | None = None) -> list[Fill]:
        """LIVE-03: reduce/close legs first, then opens; opens not sent before `open_deadline` (epoch s) are dropped."""
        current = current or {}
        reduce, opens = [], []
        for sym, dq in orders.items():
            if dq == 0 or sym not in self.rules or not math.isfinite(ref_px.get(sym, float("nan"))):
                continue
            for seq, (q, ro) in enumerate(plan_legs(current.get(sym, 0.0), dq)):
                (reduce if ro else opens).append((sym, q, ro, f"{tag}-{bar_no}-{sym}-{seq}"[:36]))
        fills: list[Fill] = []

        async def go(sym, q, ro, cid):
            await self._ensure_symbol(sym, abs(current.get(sym, 0.0) + orders[sym]) * ref_px[sym])
            f = await self._route_leg(sym, q, ro, cid, ref_px[sym])
            if f:
                fills.append(f)

        await asyncio.gather(*(go(*leg) for leg in reduce))
        if open_deadline is not None and time.time() > open_deadline:
            log.warning(f"open deadline passed, dropping {len(opens)} opening legs")
            opens = []
        await asyncio.gather(*(go(*leg) for leg in opens if leg[0] not in self.blacklist))
        return fills

    async def sync_stops(self, stops: dict[str, tuple[int, float]]) -> None:
        """Keep exactly one protective STOP_MARKET (close-position) per held symbol at the requested price."""
        for sym in list(self._stops):
            if sym not in stops:
                await self._cancel_stop(sym)
        for sym, (side, px) in stops.items():
            rules = self.rules.get(sym)
            if rules is None or not math.isfinite(px):
                continue
            price = format(rules.round_price(px).normalize(), "f")
            cur = self._stops.get(sym)
            if cur and cur[2] == price:
                continue
            if cur:
                await self._cancel_stop(sym)
            order_side = "SELL" if side > 0 else "BUY"
            try:
                r = await self.c.request("POST", "/fapi/v1/algoOrder", {
                    "algoType": "CONDITIONAL", "symbol": sym, "side": order_side, "type": "STOP_MARKET",
                    "triggerPrice": price, "closePosition": "true", "workingType": "MARK_PRICE"}, signed=True, retries=0)
                self._stops[sym] = ("algo", str(r.get("algoId")), price)
            except BinanceError as e:
                try:   # older API generation: conditional orders on the regular order endpoint
                    r = await self.c.request("POST", "/fapi/v1/order", {
                        "symbol": sym, "side": order_side, "type": "STOP_MARKET", "stopPrice": price,
                        "closePosition": "true", "workingType": "MARK_PRICE"}, signed=True, retries=0)
                    self._stops[sym] = ("order", str(r.get("orderId")), price)
                except BinanceError as e2:
                    log.error(f"cannot place hard stop for {sym}: {e} / {e2}")

    async def _cancel_stop(self, sym: str) -> None:
        kind, oid, _ = self._stops.pop(sym)
        try:
            if kind == "algo":
                await self.c.request("DELETE", "/fapi/v1/algoOrder", {"algoId": oid}, signed=True, retries=0)
            else:
                await self.c.request("DELETE", "/fapi/v1/order", {"symbol": sym, "orderId": oid}, signed=True, retries=0)
        except BinanceError as e:
            log.warning(f"cancel stop {sym} {oid}: {e}")
