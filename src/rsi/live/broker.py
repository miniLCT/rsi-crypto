"""Order routing. Paper fills are simulated; Binance fills are real (testnet or mainnet)."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from rsi.live.binance import BinanceError, BinanceFutures
from rsi.utils.log import get_logger

log = get_logger("broker")


@dataclass
class Fill:
    symbol: str
    dq: float            # signed executed quantity
    price: float         # average fill price
    fee_usd: float
    client_id: str
    status: str
    ref_price: float = float("nan")

    @property
    def slippage_bps(self) -> float:
        if not math.isfinite(self.ref_price) or self.ref_price <= 0 or self.dq == 0:
            return float("nan")
        return (self.price / self.ref_price - 1) * 1e4 * (1 if self.dq > 0 else -1)


@dataclass
class SymbolRules:
    step: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: float

    def round_qty(self, q: float) -> Decimal:
        d = (Decimal(str(abs(q))) / self.step).to_integral_value(rounding=ROUND_DOWN) * self.step
        return min(d, self.max_qty)


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
        )
    return out


class PaperBroker:
    """Fills at the current reference price; costs are charged by the Portfolio's cost model."""
    kind = "paper"

    async def execute(self, orders: dict[str, float], ref_px: dict[str, float], bar_no: int) -> list[Fill]:
        return [Fill(s, dq, ref_px[s], 0.0, f"paper-{bar_no}-{s}", "FILLED", ref_px[s]) for s, dq in orders.items()
                if dq != 0 and math.isfinite(ref_px.get(s, float("nan")))]

    async def sync(self):
        return None


class BinanceBroker:
    kind = "binance"

    def __init__(self, client: BinanceFutures, rules: dict[str, SymbolRules], leverage: int = 3, taker_fee_bps: float = 5.0):
        self.c = client
        self.rules = rules
        self.leverage = leverage
        self.fee_bps = taker_fee_bps
        self._configured: set[str] = set()

    async def prepare(self) -> None:
        await self.c.sync_time()
        await self.c.set_one_way_mode()

    async def _ensure_symbol(self, sym: str) -> None:
        if sym in self._configured:
            return
        await self.c.set_margin_type(sym, "CROSSED")
        await self.c.set_leverage(sym, self.leverage)
        self._configured.add(sym)

    async def sync(self) -> tuple[dict[str, float], float]:
        """Exchange truth: signed position amounts and total margin balance (wallet + unrealised)."""
        acct, pos = await asyncio.gather(self.c.account(), self.c.positions())
        positions = {p["symbol"]: float(p["positionAmt"]) for p in pos if float(p["positionAmt"]) != 0}
        return positions, float(acct["totalMarginBalance"])

    async def _send(self, sym: str, side: str, qty: Decimal, cid: str, reduce_only: bool, ref: float) -> Fill | None:
        try:
            r = await self.c.market_order(sym, side, format(qty.normalize(), "f"), cid, reduce_only)
        except (BinanceError, asyncio.TimeoutError, OSError) as e:
            # the order may still have been accepted: look it up by client id before giving up
            try:
                r = await self.c.get_order(sym, cid)
            except Exception:  # noqa: BLE001
                log.error(f"order {cid} {side} {sym} {qty} failed: {e}")
                return Fill(sym, 0.0, float("nan"), 0.0, cid, f"ERROR:{e}", ref)
        for _ in range(5):
            if r.get("status") in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                break
            await asyncio.sleep(0.5)
            r = await self.c.get_order(sym, cid)
        exq = float(r.get("executedQty", 0) or 0)
        avg = float(r.get("avgPrice", 0) or 0) or float("nan")
        sign = 1 if side == "BUY" else -1
        fee = abs(exq) * (avg if math.isfinite(avg) else ref) * self.fee_bps * 1e-4
        return Fill(sym, sign * exq, avg, fee, cid, r.get("status", "?"), ref)

    async def execute(self, orders: dict[str, float], ref_px: dict[str, float], bar_no: int,
                      current: dict[str, float] | None = None) -> list[Fill]:
        current = current or {}
        fills: list[Fill] = []

        async def route(sym: str, dq: float):
            rules = self.rules.get(sym)
            ref = ref_px.get(sym, float("nan"))
            if rules is None or not math.isfinite(ref):
                return
            await self._ensure_symbol(sym)
            old = current.get(sym, 0.0)
            new = old + dq
            legs: list[tuple[float, bool]] = []
            if old != 0 and (new == 0 or math.copysign(1, new) != math.copysign(1, old)):
                legs.append((-old, True))
                if new != 0:
                    legs.append((new, False))
            else:
                legs.append((dq, old != 0 and abs(new) < abs(old)))
            for j, (q, reduce_only) in enumerate(legs):
                qty = rules.round_qty(q)
                if qty <= 0 or qty < rules.min_qty:
                    continue
                if not reduce_only and float(qty) * ref < rules.min_notional:
                    continue
                cid = f"rsi{bar_no}{sym[:18]}{j}"
                f = await self._send(sym, "BUY" if q > 0 else "SELL", qty, cid, reduce_only, ref)
                if f:
                    fills.append(f)

        await asyncio.gather(*(route(s, dq) for s, dq in orders.items() if dq != 0))
        return fills
