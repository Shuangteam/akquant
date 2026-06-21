# -*- coding: utf-8 -*-
"""
策略名称：MACD 绿柱转红 + 5/10日均线金叉（多头排列）策略
数据源：  AKShare 美股日线数据接口 ak.stock_us_daily（新浪财经源）
回测框架：AKQuant  https://akquant.akfamily.xyz

安装依赖：
    pip install akquant akshare plotly

策略逻辑
--------
1. MACD(12,26,9) 柱状图（histogram = MACD - Signal）连续 >= GREEN_MIN_DAYS（默认10）
   个交易日为负值（即"绿柱"，国内常见绘图习惯：柱为负=绿色）
2. 当日柱状图由负转正（绿柱转红柱的转折当天）
3. 同时 5 日均线（MA5）相对 10 日均线（MA10）出现"金叉后的多头排列"：
   - 最近 CROSS_LOOKBACK 天内（默认3天，含当天）发生过 MA5 上穿 MA10 的金叉
   - 且当前 MA5 > MA10（已经处于多头排列状态）
4. 同时满足 2 和 3，且当前空仓 -> 全仓买入（目标仓位 TARGET_PERCENT，默认 1.0 即100%）
5. 必须先买入持仓，才有可能触发卖出（空仓状态下任何"卖出信号"都不会被执行）。
   持仓期间，以买入当天为起点（第1天），满足以下任一条件 -> 当天全仓卖出平仓：
   a) 从买入当天起，收盘价连续上涨 >= UP_STREAK_DAYS 天（默认3天，达到当天即卖）
      —— 注意这个"连涨"是从买入那天开始重新计数的，不是全局连涨天数
   b) 持仓交易日数达到 MAX_HOLD_DAYS（默认5天，买入当天记为第1天），无论涨跌强制平仓
   卖出当天不会在同一根K线上再次买入，必须等下一次买入信号触发。

说明 / 假设
-----------
- "MACD绿柱"按国内常见画图习惯定义为 histogram < 0（红涨绿跌的反向配色，与美股惯用红绿
  含义相反，如果你的图表配色相反，把判断条件里的符号对调即可）。
- "5日线突破10日线的多头排列"理解为：金叉 + 当前多头排列，而不要求与MACD转红信号
  严格发生在同一天，因此引入 CROSS_LOOKBACK 容差，可按需调整为 1 表示严格同一天。
- 图表上额外叠加了 MA20（MA_MID）、MA60（MA_LONG）两条均线，纯粹用于参考大趋势，
  不参与 build_signals 里的买入/卖出判断逻辑（这两点只跟 MA5/MA10 有关）。
- 图上的买卖点同时画在 K线图和 MACD 图上；卖出点旁边的文字标签（红=赚，绿=亏，
  跟K线红涨绿跌配色一致）展示的是"这一笔交易"的盈亏百分比和盈亏金额，金额按
  INITIAL_CASH 起始资金、全仓滚动复利模拟得出，仅用于图表直观展示，不代表
  AKQuant 实际回测引擎里含手续费/滑点后的精确盈亏（更精确的数字以 result 对象
  和 result.report() 里的绩效报告为准）。
- 买入/卖出条件（含"是否在持仓中""从买入起算连涨/持有天数"）依赖时间顺序上的状态机，
  无法用纯 pandas 向量化一次算完，因此在 build_signals 里用一次前向遍历（forward loop）
  模拟整段历史上的持仓状态，生成最终的 buy_signal / sell_signal 列；这两列已经是"position
  -aware"的最终执行信号（不会出现空仓时卖出、或持仓时重复买入的情况），on_bar 里再做一次
  current_pos 校验只是双重保险，理论上应该始终一致。
- 该状态机假设买入/卖出都在信号触发的当天以收盘价附近成交（与 AKQuant 默认的成交时点假设
  一致），不考虑次日才成交带来的1天延迟误差。
- 指标（EMA/MACD/MA）在回测前用 pandas 在全量历史数据上向量化计算一次，而不是在
  on_bar 里用滑动窗口重新计算 EMA —— 这样可以避免"EMA 在固定窗口内重新初始化"带来的
  数值误差，结果与做图软件的 MACD 完全一致。计算好的信号以额外列（buy_signal /
  sell_signal）随 OHLCV 一起喂给 AKQuant，在 on_bar 里通过 get_history(field=...) 取用
  （AKQuant 支持读取 adj_close 等自定义列，本质相同）。
- 数据缓存：默认会把 AKShare 下载的全量历史数据缓存到本地 CACHE_DIR 目录下的 CSV 文件
  （文件名形如 AAPL_us_daily_qfq.csv），同一个股票代码再次运行时会直接读本地文件，
  不会重复联网下载；改 START_DATE/END_DATE 不会触发重新下载（用缓存里的全量数据再做
  日期截取）。想强制刷新数据就把 FORCE_REFRESH 改成 True，或者直接删掉对应的缓存文件。
- 本脚本仅用于学习/研究演示，不构成任何投资建议。
"""

import os

import akshare as ak
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from akquant import Strategy, run_backtest

# ============================================================
# 1. 参数配置
# ============================================================
SYMBOL = "AAPL"          # 美股代码，可换成 TSLA / MSFT / NVDA / NFLX 等
START_DATE = "2018-01-01"  # 回测起始日期，None 表示不限制
END_DATE = None             # 回测结束日期，None 表示取到最新数据

FAST, SLOW, SIGNAL_P = 12, 26, 9   # MACD 参数
MA_FAST, MA_SLOW = 5, 10           # 均线参数（用于买入信号判断：金叉 + 多头排列）
MA_MID, MA_LONG = 20, 60           # 仅作图表参考显示的均线，不参与买卖信号判断
GREEN_MIN_DAYS = 10                 # 绿柱最少持续天数（"10天以上"）
CROSS_LOOKBACK = 3                  # 金叉确认容差窗口（天），改成 1 表示严格同一天
UP_STREAK_DAYS = 3                  # 买入后连续上涨卖出阈值（从买入当天起重新计数，不是全局连涨天数）
MAX_HOLD_DAYS = 5                   # 最大持有交易日数（买入当天记为第1天），到期无条件强制卖出

TARGET_PERCENT = 1.0                # 买入时的目标仓位比例（1.0 = 全仓买入；卖出本身就是全部平仓）
INITIAL_CASH = 100000.0             # 初始资金，同时用于 run_backtest 和图表上"每笔交易盈亏金额"的计算

CACHE_DIR = "data_cache"            # 本地缓存目录（相对于脚本运行目录）
FORCE_REFRESH = False               # True 则忽略本地缓存，强制重新从网络下载


# ============================================================
# 2. 获取美股数据（AKShare，新浪财经源，前复权）—— 带本地缓存
# ============================================================
def load_us_stock_data(
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = FORCE_REFRESH,
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_us_daily_qfq.csv")

    if (not force_refresh) and os.path.exists(cache_path):
        print(f"[缓存] 从本地文件读取 {symbol}: {cache_path}")
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
    else:
        print(f"[网络] 从 AKShare(新浪财经) 下载 {symbol} 历史数据...")
        df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        df = df.reset_index() if df.index.name == "date" else df
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        print(f"[缓存] 已写入本地文件: {cache_path}")

    df = df.sort_values("date").reset_index(drop=True)

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# ============================================================
# 3. 向量化计算 MACD 与均线信号
# ============================================================
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    ema_fast = close.ewm(span=FAST, adjust=False).mean()
    ema_slow = close.ewm(span=SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=SIGNAL_P, adjust=False).mean()
    hist = macd_line - signal_line

    ma_fast = close.rolling(MA_FAST).mean()
    ma_slow = close.rolling(MA_SLOW).mean()

    ma_mid = close.rolling(MA_MID).mean()
    ma_long = close.rolling(MA_LONG).mean()

    green_streak = np.zeros(len(df), dtype=int)
    cnt = 0
    for i, h in enumerate(hist):
        if pd.isna(h):
            cnt = 0
        elif h < 0:
            cnt += 1
        else:
            cnt = 0
        green_streak[i] = cnt
    green_streak_prev = pd.Series(green_streak).shift(1).fillna(0).values

    macd_turn_red = (hist > 0) & (hist.shift(1) <= 0) & (green_streak_prev >= GREEN_MIN_DAYS)

    golden_cross = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    recent_golden_cross = (
        golden_cross.astype(int).rolling(CROSS_LOOKBACK, min_periods=1).max().astype(bool)
    )
    bullish_alignment = ma_fast > ma_slow

    raw_buy_trigger = (macd_turn_red & recent_golden_cross & bullish_alignment).fillna(False).to_numpy()

    close_arr = close.to_numpy()
    n = len(close_arr)
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    holding_days_arr = np.zeros(n, dtype=int)
    up_streak_since_entry_arr = np.zeros(n, dtype=int)
    trade_pnl_pct_arr = np.full(n, np.nan)
    trade_pnl_amount_arr = np.full(n, np.nan)

    in_position = False
    holding_days = 0
    up_streak_since_entry = 0
    entry_price = 0.0
    equity = INITIAL_CASH

    for i in range(n):
        if in_position:
            holding_days += 1
            if i > 0 and close_arr[i] > close_arr[i - 1]:
                up_streak_since_entry += 1
            else:
                up_streak_since_entry = 0
            holding_days_arr[i] = holding_days
            up_streak_since_entry_arr[i] = up_streak_since_entry

            if up_streak_since_entry >= UP_STREAK_DAYS or holding_days >= MAX_HOLD_DAYS:
                sell_signal[i] = True
                exit_price = close_arr[i]
                pnl_pct = (exit_price - entry_price) / entry_price
                pnl_amount = equity * pnl_pct
                equity *= (1.0 + pnl_pct)
                trade_pnl_pct_arr[i] = pnl_pct
                trade_pnl_amount_arr[i] = pnl_amount
                in_position = False
                holding_days = 0
                up_streak_since_entry = 0
                continue

        if (not in_position) and raw_buy_trigger[i]:
            buy_signal[i] = True
            in_position = True
            entry_price = close_arr[i]
            holding_days = 1
            up_streak_since_entry = 0
            holding_days_arr[i] = holding_days
            up_streak_since_entry_arr[i] = up_streak_since_entry

    df = df.copy()
    df["macd_line"] = macd_line
    df["signal_line"] = signal_line
    df["hist"] = hist
    df["ma_fast"] = ma_fast
    df["ma_slow"] = ma_slow
    df["ma_mid"] = ma_mid
    df["ma_long"] = ma_long
    df["green_streak_prev"] = green_streak_prev
    df["holding_days"] = holding_days_arr
    df["up_streak_since_entry"] = up_streak_since_entry_arr
    df["trade_pnl_pct"] = trade_pnl_pct_arr
    df["trade_pnl_amount"] = trade_pnl_amount_arr
    df["buy_signal"] = buy_signal.astype(float)
    df["sell_signal"] = sell_signal.astype(float)
    return df


# ============================================================
# 4. 组装 AKQuant 需要的数据格式
# ============================================================
def to_backtest_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = symbol
    df["date"] = df["date"].dt.tz_localize("America/New_York")
    cols = [
        "date", "open", "high", "low", "close", "volume", "symbol",
        "buy_signal", "sell_signal",
    ]
    return df[cols].reset_index(drop=True)


# ============================================================
# 5. 策略定义
# ============================================================
class MacdGreenToRedMaCrossStrategy(Strategy):
    def __init__(self, target_percent: float = TARGET_PERCENT):
        self.target_percent = target_percent
        self.warmup_period = SLOW + SIGNAL_P + GREEN_MIN_DAYS + 5

    def on_bar(self, bar):
        symbol = bar.symbol

        buy_arr = self.get_history(count=1, symbol=symbol, field="buy_signal")
        sell_arr = self.get_history(count=1, symbol=symbol, field="sell_signal")
        if buy_arr is None or len(buy_arr) == 0:
            return

        buy_signal = buy_arr[-1] > 0.5
        sell_signal = bool(sell_arr is not None and len(sell_arr) and sell_arr[-1] > 0.5)

        current_pos = self.get_position(symbol)

        if buy_signal and current_pos == 0:
            self.order_target_percent(symbol=symbol, target_percent=self.target_percent)

        elif current_pos > 0 and sell_signal:
            self.close_position(symbol=symbol)


# ============================================================
# 6. 可视化：K线+均线+买卖点（上）+ MACD快慢线与红绿柱（下）
# ============================================================
def plot_macd_chart(signal_df: pd.DataFrame, symbol: str, output_path: str | None = None):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.30],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{symbol} 价格 & MA{MA_FAST}/MA{MA_SLOW}/MA{MA_MID}/MA{MA_LONG}",
            "成交量",
            f"MACD({FAST},{SLOW},{SIGNAL_P})",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=signal_df["date"],
            open=signal_df["open"],
            high=signal_df["high"],
            low=signal_df["low"],
            close=signal_df["close"],
            name="K线",
            increasing_line_color="red",
            decreasing_line_color="green",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signal_df["date"], y=signal_df["ma_fast"],
            name=f"MA{MA_FAST}", line=dict(width=1, color="#1f77b4"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signal_df["date"], y=signal_df["ma_slow"],
            name=f"MA{MA_SLOW}", line=dict(width=1, color="#9467bd"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signal_df["date"], y=signal_df["ma_mid"],
            name=f"MA{MA_MID}", line=dict(width=1, color="#ff7f0e"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signal_df["date"], y=signal_df["ma_long"],
            name=f"MA{MA_LONG}", line=dict(width=1, color="#555555"),
        ),
        row=1, col=1,
    )

    buys = signal_df[signal_df["buy_signal"] > 0.5]
    sells = signal_df[signal_df["sell_signal"] > 0.5]

    buy_hover = [
        f"买入：{d:%Y-%m-%d}<br>价格：{p:.2f}"
        for d, p in zip(buys["date"], buys["close"])
    ]
    sell_labels = [
        f"{'+' if pct >= 0 else ''}{pct * 100:.1f}% ({'+' if amt >= 0 else ''}{amt:,.0f}元)"
        for pct, amt in zip(sells["trade_pnl_pct"], sells["trade_pnl_amount"])
    ]
    sell_hover = [
        f"卖出：{d:%Y-%m-%d}<br>价格：{p:.2f}<br>本笔盈亏：{label}"
        for d, p, label in zip(sells["date"], sells["close"], sell_labels)
    ]
    sell_text_colors = ["red" if pct >= 0 else "green" for pct in sells["trade_pnl_pct"]]

    fig.add_trace(
        go.Scatter(
            x=buys["date"], y=buys["low"] * 0.97, mode="markers",
            marker=dict(symbol="triangle-up", color="red", size=11, line=dict(width=1, color="black")),
            name="买入信号",
            hovertext=buy_hover, hoverinfo="text",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sells["date"], y=sells["high"] * 1.03, mode="markers+text",
            marker=dict(symbol="triangle-down", color="green", size=11, line=dict(width=1, color="black")),
            text=sell_labels, textposition="top center",
            textfont=dict(size=10, color=sell_text_colors),
            name="卖出信号（标签=本笔盈亏%/金额）",
            hovertext=sell_hover, hoverinfo="text",
        ),
        row=1, col=1,
    )

    volume_colors = np.where(signal_df["close"] >= signal_df["open"], "red", "green")
    fig.add_trace(
        go.Bar(
            x=signal_df["date"], y=signal_df["volume"],
            name="成交量", marker_color=volume_colors, showlegend=False,
        ),
        row=2, col=1,
    )

    bar_colors = np.where(signal_df["hist"] >= 0, "red", "green")
    fig.add_trace(
        go.Bar(
            x=signal_df["date"], y=signal_df["hist"],
            name="MACD柱 (DIF-DEA)", marker_color=bar_colors,
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signal_df["date"], y=signal_df["macd_line"],
            name="DIF (快线)", line=dict(width=1.2, color="orange"),
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signal_df["date"], y=signal_df["signal_line"],
            name="DEA (慢线)", line=dict(width=1.2, color="blue"),
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line=dict(width=1, color="gray", dash="dot"), row=3, col=1)

    hist_min = float(signal_df["hist"].min(skipna=True))
    hist_max = float(signal_df["hist"].max(skipna=True))
    macd_margin = (hist_max - hist_min) * 0.08 or 0.1
    fig.add_trace(
        go.Scatter(
            x=buys["date"], y=np.full(len(buys), hist_min - macd_margin), mode="markers",
            marker=dict(symbol="triangle-up", color="red", size=9, line=dict(width=1, color="black")),
            name="买入信号（MACD）", showlegend=False,
            hovertext=buy_hover, hoverinfo="text",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sells["date"], y=np.full(len(sells), hist_max + macd_margin), mode="markers",
            marker=dict(symbol="triangle-down", color="green", size=9, line=dict(width=1, color="black")),
            name="卖出信号（MACD）", showlegend=False,
            hovertext=sell_hover, hoverinfo="text",
        ),
        row=3, col=1,
    )

    fig.update_layout(
        title=f"{symbol}  MACD绿转红 + MA{MA_FAST}/{MA_SLOW}金叉策略 — 信号与指标图",
        height=1000,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    all_dates = pd.date_range(
        start=signal_df["date"].min(), end=signal_df["date"].max(), freq="D"
    )
    trading_dates = pd.to_datetime(signal_df["date"].dt.date)
    missing_dates = all_dates[~all_dates.normalize().isin(trading_dates)]
    fig.update_xaxes(rangebreaks=[dict(values=missing_dates)])

    if output_path:
        fig.write_html(output_path)
        print(f"图表已保存到: {output_path}")
    return fig


# ============================================================
# 7. 运行回测
# ============================================================
def main():
    raw = load_us_stock_data(SYMBOL, START_DATE, END_DATE)
    if raw.empty:
        raise RuntimeError(f"未获取到 {SYMBOL} 的美股数据，请检查代码或网络。")

    signal_df = build_signals(raw)
    bt_df = to_backtest_frame(signal_df, SYMBOL)

    print(f"数据区间: {bt_df['date'].min()} ~ {bt_df['date'].max()}，共 {len(bt_df)} 条")
    print(f"触发买入信号天数: {int(signal_df['buy_signal'].sum())}")

    result = run_backtest(
        data=bt_df, strategy=MacdGreenToRedMaCrossStrategy, lot_size=1,
        initial_cash=INITIAL_CASH,
    )

    print(result)

    plot_macd_chart(signal_df, SYMBOL, output_path=f"{SYMBOL}_macd_signal_chart.html")

    result.report(show=True, filename=f"{SYMBOL}_macd_ma_cross_report.html", market_data=bt_df)


if __name__ == "__main__":
    main()
