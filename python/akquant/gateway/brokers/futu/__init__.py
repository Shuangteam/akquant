"""Futu OpenD broker gateway integration.

Provides watchlist sync, market data, trading, strategy integration,
and signal alerting via Futu OpenAPI SDK.
"""

from .gateway import FutuMarketGateway, FutuTraderGateway
from .signal_alert import SignalAlert, SignalAlertManager
from .watchlist import FutuWatchlist

__all__ = [
    "FutuMarketGateway",
    "FutuTraderGateway",
    "FutuWatchlist",
    "SignalAlert",
    "SignalAlertManager",
]
