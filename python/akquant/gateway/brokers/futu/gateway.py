"""Futu OpenD market and trader gateway implementations."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Sequence

from ....akquant import DataFeed
from ...broker_event_mapper import BrokerEventMapper, create_default_mapper
from ...broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedExecutionReport,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedPosition,
    UnifiedTrade,
    validate_execution_semantics,
)

logger = logging.getLogger(__name__)

try:
    from futu import (
        OpenQuoteContext,
        OpenSecTradeContext,
        RET_OK,
        SubType,
        TrdEnv,
        TrdMarket,
        TrdSide,
        OrderType as FutuOrderType,
        OrderStatus as FutuOrderStatus,
        StockQuoteHandlerBase,
        OrderBookHandlerBase,
        TradeOrderHandlerBase,
        TradeDealHandlerBase,
    )

    _HAS_FUTU = True
except ImportError:
    _HAS_FUTU = False


def _require_futu() -> None:
    if not _HAS_FUTU:
        raise ImportError(
            "futu-api is required for Futu integration. "
            "Install with: pip install futu-api"
        )


_FUTU_STATUS_MAP: dict[str, UnifiedOrderStatus] = {}

if _HAS_FUTU:
    _FUTU_STATUS_MAP = {
        "SUBMITTED": UnifiedOrderStatus.SUBMITTED,
        "SUBMITTING": UnifiedOrderStatus.NEW,
        "FILLED_ALL": UnifiedOrderStatus.FILLED,
        "FILLED_PART": UnifiedOrderStatus.PARTIALLY_FILLED,
        "CANCELLED_ALL": UnifiedOrderStatus.CANCELLED,
        "CANCELLED_PART": UnifiedOrderStatus.CANCELLED,
        "FAILED": UnifiedOrderStatus.REJECTED,
        "DISABLED": UnifiedOrderStatus.REJECTED,
        "DELETED": UnifiedOrderStatus.CANCELLED,
    }


def _map_futu_order_status(status_str: str) -> UnifiedOrderStatus:
    return _FUTU_STATUS_MAP.get(status_str, UnifiedOrderStatus.SUBMITTED)


class FutuMarketGateway:
    """Futu OpenD market data gateway.

    Connects to Futu OpenD via OpenQuoteContext, subscribes to real-time
    quotes and feeds data into the AKQuant DataFeed.
    """

    def __init__(
        self,
        feed: DataFeed,
        symbols: Sequence[str],
        host: str = "127.0.0.1",
        port: int = 11111,
        **kwargs: Any,
    ) -> None:
        _require_futu()
        self.feed = feed
        self.symbols = list(symbols)
        self._host = host
        self._port = port
        self.kwargs = kwargs
        self.connected = False
        self._ctx: Any = None
        self._running = False
        self.tick_callback: Callable[[dict[str, Any]], None] | None = None
        self.bar_callback: Callable[[dict[str, Any]], None] | None = None

    def _to_futu_codes(self, symbols: Sequence[str]) -> list[str]:
        """Convert AKQuant symbols to Futu format.

        ``00700.HK`` → ``HK.00700``, ``AAPL.US`` → ``US.AAPL``.
        """
        result = []
        for sym in symbols:
            parts = sym.rsplit(".", 1)
            if len(parts) == 2:
                code, mkt = parts
                result.append(f"{mkt}.{code}")
            else:
                result.append(sym)
        return result

    def connect(self) -> None:
        self._ctx = OpenQuoteContext(host=self._host, port=self._port)
        self.connected = True
        logger.info("Futu market gateway connected to %s:%s", self._host, self._port)

    def disconnect(self) -> None:
        self._running = False
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None
        self.connected = False

    def subscribe(self, symbols: Sequence[str]) -> None:
        self.symbols = list(symbols)
        if self._ctx is None:
            return
        futu_codes = self._to_futu_codes(symbols)
        ret, err = self._ctx.subscribe(futu_codes, [SubType.QUOTE, SubType.K_1M])
        if ret != RET_OK:
            logger.error("Subscribe failed: %s", err)
            raise RuntimeError(f"Futu subscribe failed: {err}")
        logger.info("Subscribed to %d symbols", len(futu_codes))

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        removed = set(symbols)
        self.symbols = [s for s in self.symbols if s not in removed]
        if self._ctx is None:
            return
        futu_codes = self._to_futu_codes(symbols)
        self._ctx.unsubscribe(futu_codes, [SubType.QUOTE, SubType.K_1M])

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.tick_callback = callback

    def on_bar(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.bar_callback = callback

    def start(self) -> None:
        self.connect()
        self.subscribe(self.symbols)
        self._running = True

        if self.tick_callback or self.bar_callback:
            self._setup_handlers()

    def _setup_handlers(self) -> None:
        gateway = self

        class _QuoteHandler(StockQuoteHandlerBase):
            def on_recv_rsp(self, rsp_pb):
                ret, data = super().on_recv_rsp(rsp_pb)
                if ret != RET_OK or data is None:
                    return ret, data
                for _, row in data.iterrows():
                    tick_data = {
                        "symbol": row.get("code", ""),
                        "last_price": float(row.get("last_price", 0)),
                        "volume": float(row.get("volume", 0)),
                        "turnover": float(row.get("turnover", 0)),
                        "bid_price": float(row.get("bid_price", 0)),
                        "ask_price": float(row.get("ask_price", 0)),
                        "bid_vol": float(row.get("bid_vol", 0)),
                        "ask_vol": float(row.get("ask_vol", 0)),
                        "timestamp": row.get("data_date", "")
                        + " "
                        + row.get("data_time", ""),
                    }
                    if gateway.tick_callback:
                        gateway.tick_callback(tick_data)
                return ret, data

        self._ctx.set_handler(_QuoteHandler())


class FutuTraderGateway:
    """Futu OpenD trader gateway.

    Provides order placement, cancellation, position/account queries
    through Futu's OpenSecTradeContext.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        trd_env: str = "SIMULATE",
        trd_market: str = "HK",
        security_firm: str = "FUTUINC",
        **kwargs: Any,
    ) -> None:
        _require_futu()
        self._host = host
        self._port = port
        self._trd_env = TrdEnv.SIMULATE if trd_env.upper() == "SIMULATE" else TrdEnv.REAL
        self._trd_market_str = trd_market.upper()
        self._trd_market = getattr(TrdMarket, trd_market.upper(), TrdMarket.HK)
        self._security_firm_str = security_firm
        self.kwargs = kwargs
        self.connected = False
        self._ctx: Any = None
        self.orders: dict[str, UnifiedOrderSnapshot] = {}
        self.trades: list[UnifiedTrade] = []
        self.client_to_broker_order_ids: dict[str, str] = {}
        self.broker_to_client_order_ids: dict[str, str] = {}
        self._order_seq = 0
        self.mapper: BrokerEventMapper = kwargs.get(
            "event_mapper", create_default_mapper()
        )
        self.order_callback: Callable[[UnifiedOrderSnapshot], None] | None = None
        self.trade_callback: Callable[[UnifiedTrade], None] | None = None
        self.execution_callback: Callable[[UnifiedExecutionReport], None] | None = None

    def _to_futu_code(self, symbol: str) -> str:
        parts = symbol.rsplit(".", 1)
        if len(parts) == 2:
            code, mkt = parts
            return f"{mkt}.{code}"
        return symbol

    def connect(self) -> None:
        self._ctx = OpenSecTradeContext(
            host=self._host,
            port=self._port,
            security_firm=self._security_firm_str,
        )
        ret, data = self._ctx.unlock_trade(self.kwargs.get("trade_password", ""))
        if ret != RET_OK:
            logger.warning("Unlock trade failed (may be simulate env): %s", data)
        self.connected = True
        logger.info(
            "Futu trader gateway connected (%s, %s)",
            self._trd_env,
            self._trd_market_str,
        )

    def disconnect(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None
        self.connected = False

    def place_order(self, req: UnifiedOrderRequest) -> str:
        self._ensure_connected()
        req.position_effect = validate_execution_semantics(
            self.get_capabilities(),
            req.position_effect,
            req.reduce_only,
        )
        futu_code = self._to_futu_code(req.symbol)
        side = TrdSide.BUY if req.side.upper() in ("BUY", "B") else TrdSide.SELL

        if req.order_type.upper() == "MARKET":
            order_type = FutuOrderType.MARKET
        else:
            order_type = FutuOrderType.NORMAL

        ret, data = self._ctx.place_order(
            price=req.price or 0,
            qty=req.quantity,
            code=futu_code,
            trd_side=side,
            order_type=order_type,
            trd_env=self._trd_env,
            trd_market=self._trd_market,
            remark=req.client_order_id,
        )
        if ret != RET_OK:
            raise RuntimeError(f"Futu place_order failed: {data}")

        broker_order_id = str(data["order_id"].iloc[0])
        now_ns = time.time_ns()
        snapshot = UnifiedOrderSnapshot(
            client_order_id=req.client_order_id,
            broker_order_id=broker_order_id,
            symbol=req.symbol,
            status=UnifiedOrderStatus.SUBMITTED,
            timestamp_ns=now_ns,
            position_effect=req.position_effect,
        )
        self.orders[broker_order_id] = snapshot
        self.client_to_broker_order_ids[req.client_order_id] = broker_order_id
        self.broker_to_client_order_ids[broker_order_id] = req.client_order_id
        self._emit_order(snapshot)
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> None:
        self._ensure_connected()
        ret, data = self._ctx.modify_order(
            modify_order_op="CANCEL",
            order_id=broker_order_id,
            qty=0,
            price=0,
            trd_env=self._trd_env,
        )
        if ret != RET_OK:
            logger.error("Cancel order %s failed: %s", broker_order_id, data)
            raise RuntimeError(f"Futu cancel_order failed: {data}")
        order = self.orders.get(broker_order_id)
        if order is not None:
            order.status = UnifiedOrderStatus.CANCELLED
            order.timestamp_ns = time.time_ns()
            self._emit_order(order)

    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        return self.orders.get(broker_order_id)

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        if since is None:
            return list(self.trades)
        return [t for t in self.trades if t.timestamp_ns >= since]

    def query_account(self) -> UnifiedAccount | None:
        self._ensure_connected()
        ret, data = self._ctx.accinfo_query(trd_env=self._trd_env)
        if ret != RET_OK:
            logger.error("Account query failed: %s", data)
            return None
        row = data.iloc[0]
        return UnifiedAccount(
            account_id=str(row.get("trd_env", "futu")),
            equity=float(row.get("total_assets", 0)),
            cash=float(row.get("cash", 0)),
            available_cash=float(row.get("avl_withdrawal_cash", 0)),
            timestamp_ns=time.time_ns(),
        )

    def query_positions(self) -> list[UnifiedPosition]:
        self._ensure_connected()
        ret, data = self._ctx.position_list_query(trd_env=self._trd_env)
        if ret != RET_OK:
            logger.error("Position query failed: %s", data)
            return []
        positions = []
        for _, row in data.iterrows():
            code = str(row.get("code", ""))
            parts = code.split(".", 1)
            if len(parts) == 2:
                mkt, sym = parts
                akquant_symbol = f"{sym}.{mkt}"
            else:
                akquant_symbol = code
            qty = float(row.get("qty", 0))
            positions.append(
                UnifiedPosition(
                    symbol=akquant_symbol,
                    quantity=qty,
                    available_quantity=qty,
                    direction="long" if qty > 0 else "short",
                    avg_price=float(row.get("cost_price", 0)),
                    timestamp_ns=time.time_ns(),
                )
            )
        return positions

    def on_order(self, callback: Callable[[UnifiedOrderSnapshot], None]) -> None:
        self.order_callback = callback

    def on_trade(self, callback: Callable[[UnifiedTrade], None]) -> None:
        self.trade_callback = callback

    def on_execution_report(
        self, callback: Callable[[UnifiedExecutionReport], None]
    ) -> None:
        self.execution_callback = callback

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        self._ensure_connected()
        ret, data = self._ctx.order_list_query(
            trd_env=self._trd_env,
            trd_market=self._trd_market,
        )
        if ret != RET_OK:
            return []
        open_orders = []
        for _, row in data.iterrows():
            status = _map_futu_order_status(str(row.get("order_status", "")))
            if status in (
                UnifiedOrderStatus.NEW,
                UnifiedOrderStatus.SUBMITTED,
                UnifiedOrderStatus.PARTIALLY_FILLED,
            ):
                broker_id = str(row.get("order_id", ""))
                snapshot = UnifiedOrderSnapshot(
                    client_order_id=str(row.get("remark", broker_id)),
                    broker_order_id=broker_id,
                    symbol=str(row.get("code", "")),
                    status=status,
                    filled_quantity=float(row.get("dealt_qty", 0)),
                    avg_fill_price=float(row.get("dealt_avg_price", 0)),
                    timestamp_ns=time.time_ns(),
                )
                self.orders[broker_id] = snapshot
                open_orders.append(snapshot)
        return open_orders

    def sync_today_trades(self) -> list[UnifiedTrade]:
        self._ensure_connected()
        ret, data = self._ctx.deal_list_query(
            trd_env=self._trd_env,
            trd_market=self._trd_market,
        )
        if ret != RET_OK:
            return []
        trades = []
        for _, row in data.iterrows():
            trade = UnifiedTrade(
                trade_id=str(row.get("deal_id", "")),
                broker_order_id=str(row.get("order_id", "")),
                client_order_id=str(row.get("order_id", "")),
                symbol=str(row.get("code", "")),
                side="BUY" if str(row.get("trd_side", "")).upper() == "BUY" else "SELL",
                quantity=float(row.get("qty", 0)),
                price=float(row.get("price", 0)),
                timestamp_ns=time.time_ns(),
            )
            trades.append(trade)
        self.trades = trades
        return trades

    def heartbeat(self) -> bool:
        return self.connected

    def get_capabilities(self) -> BrokerCapability:
        return BrokerCapability(
            broker_name="futu",
            broker_live=True,
            client_order_id=True,
            order_type=True,
            time_in_force_str=False,
            position_effect=False,
            reduce_only=False,
            position_details=False,
            supports_short_sell=True,
            supported_position_effects=("auto",),
        )

    def start(self) -> None:
        self.connect()

    def _emit_order(self, order: UnifiedOrderSnapshot) -> None:
        if self.order_callback is not None:
            self.order_callback(order)

    def _emit_trade(self, trade: UnifiedTrade) -> None:
        if self.trade_callback is not None:
            self.trade_callback(trade)

    def _emit_execution_report(self, report: UnifiedExecutionReport) -> None:
        if self.execution_callback is not None:
            self.execution_callback(report)

    def _ensure_connected(self) -> None:
        if self._ctx is None:
            raise RuntimeError("Futu trader not connected. Call connect() first.")
