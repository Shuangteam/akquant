"""Futu watchlist (自选股) synchronization."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from futu import (
        KLType,
        Market,
        OpenQuoteContext,
        RET_OK,
        SubType,
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


class FutuWatchlist:
    """Pull and manage watchlist stocks from Futu OpenD.

    Usage::

        wl = FutuWatchlist(host="127.0.0.1", port=11111)
        wl.connect()
        groups = wl.list_groups()
        symbols = wl.pull_watchlist("我的自选")
        wl.disconnect()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 11111) -> None:
        _require_futu()
        self._host = host
        self._port = port
        self._ctx: Any = None

    def connect(self) -> None:
        self._ctx = OpenQuoteContext(host=self._host, port=self._port)
        logger.info("Futu watchlist connected to %s:%s", self._host, self._port)

    def disconnect(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def list_groups(self) -> list[str]:
        """Return all watchlist group names."""
        self._ensure_connected()
        ret, data = self._ctx.get_user_security_group()
        if ret != RET_OK:
            raise RuntimeError(f"Failed to list watchlist groups: {data}")
        return list(data["group_name"])

    def pull_watchlist(self, group_name: str = "我的自选") -> list[str]:
        """Pull stock codes from a named watchlist group.

        Returns a list of Futu-format codes like ``['HK.00700', 'US.AAPL']``.
        """
        self._ensure_connected()
        ret, data = self._ctx.get_user_security(group_name)
        if ret != RET_OK:
            raise RuntimeError(
                f"Failed to pull watchlist '{group_name}': {data}"
            )
        codes = list(data["code"])
        logger.info(
            "Pulled %d symbols from watchlist '%s': %s",
            len(codes),
            group_name,
            codes[:10],
        )
        return codes

    def pull_watchlist_as_akquant(
        self, group_name: str = "我的自选", market: str | None = None
    ) -> list[str]:
        """Pull watchlist and convert codes to AKQuant format.

        Futu ``HK.00700`` → AKQuant ``00700.HK``,
        Futu ``US.AAPL`` → AKQuant ``AAPL.US``,
        Futu ``SH.600000`` → AKQuant ``600000.SH``.
        """
        futu_codes = self.pull_watchlist(group_name)
        results = []
        for code in futu_codes:
            parts = code.split(".", 1)
            if len(parts) == 2:
                mkt, symbol = parts
                if market and mkt != market:
                    continue
                results.append(f"{symbol}.{mkt}")
            else:
                results.append(code)
        return results

    def get_snapshot(self, codes: list[str]) -> Any:
        """Get real-time snapshot for given Futu codes."""
        self._ensure_connected()
        ret, data = self._ctx.get_market_snapshot(codes)
        if ret != RET_OK:
            raise RuntimeError(f"Failed to get snapshot: {data}")
        return data

    def _ensure_connected(self) -> None:
        if self._ctx is None:
            raise RuntimeError("Not connected. Call connect() first.")
