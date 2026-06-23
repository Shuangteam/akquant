"""Signal alerting for Futu-integrated AKQuant strategies.

Monitors strategy signals and dispatches alerts via configurable channels
(logging, callback, webhook, Futu app push).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    LOG = "log"
    CALLBACK = "callback"
    WEBHOOK = "webhook"


@dataclass
class SignalAlert:
    """A single trading signal alert."""

    symbol: str
    signal: str
    price: float = 0.0
    quantity: float = 0.0
    strategy_name: str = ""
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "price": self.price,
            "quantity": self.quantity,
            "strategy_name": self.strategy_name,
            "reason": self.reason,
            "timestamp": self.timestamp,
            **self.extra,
        }

    def to_message(self) -> str:
        parts = [
            f"[{self.strategy_name or 'Strategy'}]",
            f"{self.signal.upper()} {self.symbol}",
        ]
        if self.price:
            parts.append(f"@ {self.price:.4f}")
        if self.quantity:
            parts.append(f"qty={self.quantity}")
        if self.reason:
            parts.append(f"({self.reason})")
        return " ".join(parts)


class SignalAlertManager:
    """Manages signal alert dispatching across multiple channels.

    Usage::

        mgr = SignalAlertManager()
        mgr.add_log_channel()
        mgr.add_callback_channel(my_handler)
        mgr.add_webhook_channel("https://hooks.example.com/notify")

        # In strategy on_bar / on_tick:
        mgr.emit(SignalAlert(symbol="00700.HK", signal="BUY", price=350.0))
    """

    def __init__(self) -> None:
        self._channels: list[tuple[AlertChannel, Any]] = []
        self._history: list[SignalAlert] = []
        self._max_history = 1000
        self._cooldown_seconds: float = 0
        self._last_alert_time: dict[str, float] = {}

    def add_log_channel(self, level: int = logging.INFO) -> SignalAlertManager:
        self._channels.append((AlertChannel.LOG, level))
        return self

    def add_callback_channel(
        self, callback: Callable[[SignalAlert], None]
    ) -> SignalAlertManager:
        self._channels.append((AlertChannel.CALLBACK, callback))
        return self

    def add_webhook_channel(
        self, url: str, headers: dict[str, str] | None = None
    ) -> SignalAlertManager:
        self._channels.append((AlertChannel.WEBHOOK, (url, headers or {})))
        return self

    def set_cooldown(self, seconds: float) -> SignalAlertManager:
        self._cooldown_seconds = seconds
        return self

    def emit(self, alert: SignalAlert) -> bool:
        """Dispatch an alert to all channels. Returns False if throttled."""
        key = f"{alert.symbol}:{alert.signal}"
        now = time.time()
        if self._cooldown_seconds > 0:
            last = self._last_alert_time.get(key, 0)
            if now - last < self._cooldown_seconds:
                return False
        self._last_alert_time[key] = now

        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for channel, config in self._channels:
            try:
                if channel == AlertChannel.LOG:
                    logger.log(config, alert.to_message())
                elif channel == AlertChannel.CALLBACK:
                    config(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook(alert, *config)
            except Exception:
                logger.exception("Alert channel %s failed", channel)
        return True

    def get_history(self, limit: int = 50) -> list[SignalAlert]:
        return self._history[-limit:]

    @staticmethod
    def _send_webhook(
        alert: SignalAlert, url: str, headers: dict[str, str]
    ) -> None:
        payload = json.dumps(alert.to_dict()).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
