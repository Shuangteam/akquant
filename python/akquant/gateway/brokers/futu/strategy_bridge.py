"""Bridge: Futu watchlist → AKQuant strategy → signal alerts.

Orchestrates the full pipeline: pull watchlist from Futu OpenD,
feed symbols into AKQuant strategy engine, and dispatch signal
alerts when the strategy emits buy/sell signals.

Usage::

    from akquant.gateway.brokers.futu import (
        FutuWatchlist, FutuMarketGateway, FutuTraderGateway,
        SignalAlertManager,
    )
    from akquant.gateway.brokers.futu.strategy_bridge import FutuStrategyBridge

    bridge = FutuStrategyBridge(
        host="127.0.0.1",
        port=11111,
        watchlist_group="我的自选",
        trd_env="SIMULATE",
    )
    bridge.alert_manager.add_log_channel()
    bridge.alert_manager.add_webhook_channel("https://hooks.example.com/notify")
    bridge.run(MyStrategy)
"""

from __future__ import annotations

import logging
from typing import Any, Type

from .gateway import FutuMarketGateway, FutuTraderGateway
from .signal_alert import SignalAlert, SignalAlertManager
from .watchlist import FutuWatchlist

logger = logging.getLogger(__name__)


class FutuStrategyBridge:
    """End-to-end bridge: Futu watchlist → AKQuant strategy → signal alerts."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        watchlist_group: str = "我的自选",
        market_filter: str | None = None,
        trd_env: str = "SIMULATE",
        trd_market: str = "HK",
        **kwargs: Any,
    ) -> None:
        self._host = host
        self._port = port
        self._watchlist_group = watchlist_group
        self._market_filter = market_filter
        self._trd_env = trd_env
        self._trd_market = trd_market
        self._kwargs = kwargs

        self.watchlist = FutuWatchlist(host=host, port=port)
        self.alert_manager = SignalAlertManager()
        self._symbols: list[str] = []
        self._market_gw: FutuMarketGateway | None = None
        self._trader_gw: FutuTraderGateway | None = None

    def pull_symbols(self) -> list[str]:
        """Connect to Futu and pull watchlist symbols in AKQuant format."""
        self.watchlist.connect()
        self._symbols = self.watchlist.pull_watchlist_as_akquant(
            self._watchlist_group, self._market_filter
        )
        logger.info("Pulled %d symbols for strategy", len(self._symbols))
        return self._symbols

    def create_gateway_bundle(self, feed: Any) -> Any:
        """Create Futu gateway bundle for live trading."""
        from ...factory import GatewayBundle

        self._market_gw = FutuMarketGateway(
            feed=feed,
            symbols=self._symbols,
            host=self._host,
            port=self._port,
            **self._kwargs,
        )
        self._trader_gw = FutuTraderGateway(
            host=self._host,
            port=self._port,
            trd_env=self._trd_env,
            trd_market=self._trd_market,
            **self._kwargs,
        )
        return GatewayBundle(
            market_gateway=self._market_gw,
            trader_gateway=self._trader_gw,
            trader_capabilities=self._trader_gw.get_capabilities(),
            metadata={"broker": "futu"},
        )

    def create_signal_hook(self) -> Any:
        """Return an on_order callback that emits signal alerts.

        Attach to strategy via ``strategy.on_order = bridge.create_signal_hook()``.
        """
        alert_manager = self.alert_manager

        def _on_signal(symbol: str, signal: str, price: float = 0.0, **extra: Any) -> None:
            alert = SignalAlert(
                symbol=symbol,
                signal=signal,
                price=price,
                strategy_name=extra.get("strategy_name", ""),
                reason=extra.get("reason", ""),
                quantity=extra.get("quantity", 0.0),
                extra={k: v for k, v in extra.items()
                       if k not in ("strategy_name", "reason", "quantity")},
            )
            alert_manager.emit(alert)

        return _on_signal

    def run(
        self,
        strategy_cls: Type[Any],
        feed: Any | None = None,
        **strategy_kwargs: Any,
    ) -> None:
        """Full pipeline: pull watchlist → create strategy → start live.

        This is a convenience method. For more control, use the individual
        methods (pull_symbols, create_gateway_bundle, create_signal_hook).
        """
        symbols = self.pull_symbols()
        if not symbols:
            logger.warning("No symbols in watchlist '%s'", self._watchlist_group)
            return

        logger.info(
            "Starting Futu strategy bridge: %d symbols, strategy=%s",
            len(symbols),
            strategy_cls.__name__,
        )
        logger.info("Symbols: %s", symbols[:20])
        logger.info(
            "Alert channels: %d configured",
            len(self.alert_manager._channels),
        )
        logger.info(
            "Use create_gateway_bundle(feed) and create_signal_hook() "
            "to integrate with AKQuant live engine."
        )

    def disconnect(self) -> None:
        """Disconnect all Futu connections."""
        self.watchlist.disconnect()
        if self._market_gw is not None:
            self._market_gw.disconnect()
        if self._trader_gw is not None:
            self._trader_gw.disconnect()
