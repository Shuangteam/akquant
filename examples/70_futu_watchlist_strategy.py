"""Example: Futu watchlist → AKQuant strategy → signal alerts.

Prerequisites:
  1. Install futu-api: pip install futu-api
  2. Run Futu OpenD on localhost:11111
  3. Have a watchlist group (default: "我的自选") with stocks

This example demonstrates:
  - Pulling watchlist symbols from Futu
  - Setting up signal alert channels (log + webhook)
  - Creating a simple moving-average crossover strategy
  - Bridging everything together for live signal monitoring
"""

import logging

from akquant.gateway.brokers.futu import (
    FutuWatchlist,
    SignalAlert,
    SignalAlertManager,
)
from akquant.gateway.brokers.futu.strategy_bridge import FutuStrategyBridge

logging.basicConfig(level=logging.INFO)


# --- 1. Standalone watchlist usage ---

def demo_watchlist():
    """Pull and display watchlist stocks."""
    wl = FutuWatchlist(host="127.0.0.1", port=11111)
    wl.connect()
    try:
        groups = wl.list_groups()
        print(f"Watchlist groups: {groups}")

        symbols = wl.pull_watchlist("我的自选")
        print(f"Futu format: {symbols}")

        akquant_symbols = wl.pull_watchlist_as_akquant("我的自选")
        print(f"AKQuant format: {akquant_symbols}")
    finally:
        wl.disconnect()


# --- 2. Signal alert usage ---

def demo_alerts():
    """Set up and test signal alerts."""
    mgr = SignalAlertManager()
    mgr.add_log_channel()
    mgr.set_cooldown(30)

    # Custom callback
    def my_handler(alert: SignalAlert):
        print(f"🔔 {alert.to_message()}")

    mgr.add_callback_channel(my_handler)

    # Emit test signal
    mgr.emit(SignalAlert(
        symbol="00700.HK",
        signal="BUY",
        price=350.0,
        strategy_name="MA_Cross",
        reason="MA5 crossed above MA20",
    ))

    # Check history
    history = mgr.get_history()
    print(f"Alert history: {len(history)} alerts")


# --- 3. Full pipeline: watchlist → strategy → alerts ---

def demo_full_pipeline():
    """Full bridge example (requires Futu OpenD running)."""
    bridge = FutuStrategyBridge(
        host="127.0.0.1",
        port=11111,
        watchlist_group="我的自选",
        trd_env="SIMULATE",
        trd_market="HK",
    )

    # Configure alert channels
    bridge.alert_manager.add_log_channel()
    bridge.alert_manager.set_cooldown(60)

    # Optional: webhook for external notifications
    # bridge.alert_manager.add_webhook_channel(
    #     "https://hooks.example.com/trading-alerts",
    #     headers={"Authorization": "Bearer YOUR_TOKEN"},
    # )

    # Pull symbols from Futu watchlist
    symbols = bridge.pull_symbols()
    print(f"Strategy will monitor: {symbols}")

    # Get the signal hook to use in your strategy's on_bar/on_tick
    signal_hook = bridge.create_signal_hook()

    # In your strategy, call signal_hook when conditions are met:
    # signal_hook("00700.HK", "BUY", price=350.0, reason="MA crossover")

    bridge.disconnect()


if __name__ == "__main__":
    # These require Futu OpenD running:
    # demo_watchlist()
    # demo_full_pipeline()

    # This works standalone:
    demo_alerts()
