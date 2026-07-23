"""
テクニカル指標の計算モジュール。
入力は yfinance などから取得した OHLCV の DataFrame
(columns: Open, High, Low, Close, Volume / index: DatetimeIndex) を想定。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame, windows=(5, 25, 75, 200)) -> pd.DataFrame:
    """単純移動平均線を追加する。"""
    out = df.copy()
    for w in windows:
        out[f"MA{w}"] = out["Close"].rolling(window=w, min_periods=w).mean()
    return out


def detect_ma_cross(df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.DataFrame:
    """
    2本の移動平均線のゴールデンクロス／デッドクロスを検出する。
    戻り値には 'cross' 列が追加され、'golden' / 'dead' / None が入る。
    """
    out = df.copy()
    fast = out[fast_col]
    slow = out[slow_col]
    diff = fast - slow
    prev_diff = diff.shift(1)

    cross = pd.Series(index=out.index, dtype=object)
    cross[(prev_diff < 0) & (diff >= 0)] = "golden"
    cross[(prev_diff > 0) & (diff <= 0)] = "dead"
    out["cross"] = cross
    return out


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD・シグナル・ヒストグラムを追加し、クロス点も検出する。"""
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    out["MACD"] = macd_line
    out["MACD_signal"] = signal_line
    out["MACD_hist"] = hist

    prev_hist = hist.shift(1)
    cross = pd.Series(index=out.index, dtype=object)
    cross[(prev_hist < 0) & (hist >= 0)] = "golden"
    cross[(prev_hist > 0) & (hist <= 0)] = "dead"
    out["MACD_cross"] = cross
    return out


def add_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """RSI(相対力指数)を追加する。"""
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # avg_loss=0 (連続上昇)の場合は100寄りが自然だが、簡易的に50埋め回避
    rsi[avg_loss == 0] = 100
    out[f"RSI{period}"] = rsi
    return out


def add_bollinger_bands(df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
    """ボリンジャーバンドを追加する。"""
    out = df.copy()
    mid = out["Close"].rolling(window=window, min_periods=window).mean()
    std = out["Close"].rolling(window=window, min_periods=window).std()
    out["BB_mid"] = mid
    out["BB_upper"] = mid + num_std * std
    out["BB_lower"] = mid - num_std * std
    return out


def add_ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou_b=52) -> pd.DataFrame:
    """
    一目均衡表を追加する。
    転換線・基準線・先行スパン1/2・遅行スパン。
    先行スパンは kijun 期間分未来にシフトされる（一目均衡表の標準的な描画方法）。
    """
    out = df.copy()
    high = out["High"]
    low = out["Low"]
    close = out["Close"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou_span = close.shift(-kijun)

    out["ichimoku_tenkan"] = tenkan_sen
    out["ichimoku_kijun"] = kijun_sen
    out["ichimoku_senkou_a"] = senkou_span_a
    out["ichimoku_senkou_b"] = senkou_span_b
    out["ichimoku_chikou"] = chikou_span
    return out


def add_volume_ma(df: pd.DataFrame, window=25) -> pd.DataFrame:
    """出来高移動平均を追加する。"""
    out = df.copy()
    out[f"VolMA{window}"] = out["Volume"].rolling(window=window, min_periods=window).mean()
    return out


def compute_margin_ratio(margin_df: pd.DataFrame) -> pd.DataFrame:
    """
    信用残データ(買い残高・売り残高の時系列)から信用倍率を計算する。
    margin_df は columns: ['date', 'buy_balance', 'sell_balance'] を想定。
    信用倍率 = 買い残高 / 売り残高
    """
    out = margin_df.copy()
    out["margin_ratio"] = out["buy_balance"] / out["sell_balance"].replace(0, np.nan)
    return out


# 表示期間ラベル -> 末尾から遡る期間。
# 移動平均線(200日)等を正しく計算するため、株価データ自体は常に長期間(既定10年)を
# 取得しておき、画面表示の絞り込みはこのオフセットを使って行う
# （こうすることで「1ヶ月」表示を選んでも、その1ヶ月間のMA200が
# ちゃんと過去のデータに基づいて計算された状態で表示できる）。
PERIOD_OFFSETS = {
    "1日": pd.DateOffset(days=1),
    "1週間": pd.DateOffset(weeks=1),
    "1ヶ月": pd.DateOffset(months=1),
    "6ヶ月": pd.DateOffset(months=6),
    "1年": pd.DateOffset(years=1),
    "2年": pd.DateOffset(years=2),
    "5年": pd.DateOffset(years=5),
    "10年": pd.DateOffset(years=10),
}


def slice_by_period(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """指標計算済みのDataFrameから、表示期間ラベルに応じた末尾部分を切り出す。"""
    if df.empty or period_label not in PERIOD_OFFSETS:
        return df
    cutoff = df.index.max() - PERIOD_OFFSETS[period_label]
    return df.loc[df.index >= cutoff]


def build_all_technical_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """全テクニカル指標をまとめて計算するヘルパー。"""
    df = price_df.copy()
    df = add_moving_averages(df)
    df = detect_ma_cross(df, "MA25", "MA75")
    df = add_macd(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_ichimoku(df)
    df = add_volume_ma(df)
    return df
