"""
完全無料のルールベース診断コメント生成モジュール。
Claude APIを一切呼び出さず、fundamentals.py で計算済みの5段階評価・PPM分類・
注意フラグと、テクニカル指標（RSI・クロス・信用倍率）から定型文を組み立てる。
API課金は一切発生しない。
"""
from __future__ import annotations

import pandas as pd

AXIS_ORDER = ["安全性", "収益性", "効率性", "成長性"]


def _stars(score: int | None) -> str:
    if not score:
        return "評価不可"
    return "★" * score + "☆" * (5 - score)


def comment_ma(latest_row) -> str:
    """移動平均線の状況から、短期的な方向感についてのコメントを組み立てる（ルールベース）。"""
    price = latest_row.get("Close")
    ma25 = latest_row.get("MA25")
    ma75 = latest_row.get("MA75")
    cross = latest_row.get("cross")

    if cross == "golden":
        return ("直近でMA25がMA75を上抜く「ゴールデンクロス」が発生しました。短期的には上昇トレンドへの"
                "転換シグナルとされますが、だまし（一時的な交差）のケースもあるため、他の指標と合わせて確認することが望まれます。")
    if cross == "dead":
        return ("直近でMA25がMA75を下抜く「デッドクロス」が発生しました。短期的には下落トレンドへの"
                "転換シグナルとされますが、だましのケースもあるため、他の指標と合わせて確認することが望まれます。")
    if price is not None and ma25 is not None and ma75 is not None and \
            not any(pd.isna(v) for v in [price, ma25, ma75]):
        if price > ma25 > ma75:
            return "株価は短期・中期の移動平均線をいずれも上回っており、上昇トレンドが継続しやすい地合いと考えられます。"
        if price < ma25 < ma75:
            return "株価は短期・中期の移動平均線をいずれも下回っており、下落トレンドが継続しやすい地合いと考えられます。"
        return "株価と移動平均線が交錯しており、方向感に乏しいレンジ相場となっている可能性があります。"
    return "移動平均線を判定するためのデータが不足しています（データ取得期間が短い可能性があります）。"


def comment_macd(latest_row) -> str:
    """MACDの状況から、短期モメンタムについてのコメントを組み立てる（ルールベース）。"""
    hist = latest_row.get("MACD_hist")
    cross = latest_row.get("MACD_cross")

    if cross == "golden":
        return "MACDがシグナルを上抜けるゴールデンクロスが発生しており、短期的に上昇モメンタムが強まっている可能性があります。"
    if cross == "dead":
        return "MACDがシグナルを下抜けるデッドクロスが発生しており、短期的に下落モメンタムが強まっている可能性があります。"
    if hist is not None and pd.notna(hist):
        if hist > 0:
            return "MACDヒストグラムはプラス圏で推移しており、上昇モメンタムが優勢な状態が続いていると考えられます。"
        return "MACDヒストグラムはマイナス圏で推移しており、下落モメンタムが優勢な状態が続いていると考えられます。"
    return "MACDを判定するためのデータが不足しています。"


def comment_rsi(latest_row) -> str:
    """RSIの水準から、短期的な過熱感についてのコメントを組み立てる（ルールベース）。"""
    rsi = latest_row.get("RSI14")
    if rsi is None or pd.isna(rsi):
        return "RSIを判定するためのデータが不足しています。"
    if rsi >= 70:
        return (f"RSIは{rsi:.1f}と70を超え、短期的な買われすぎ水準にあります。過熱感から一旦調整が"
                "入りやすい局面とされますが、強いトレンドが続く場合は高水準のまま推移することもあります。")
    if rsi <= 30:
        return (f"RSIは{rsi:.1f}と30を下回り、短期的な売られすぎ水準にあります。自律反発が入りやすい局面と"
                "されますが、下落トレンドが強い場合は低水準のまま推移することもあります。")
    return f"RSIは{rsi:.1f}で中立圏にあり、方向感を判断しづらい状態です。"


def comment_bb(latest_row) -> str:
    """ボリンジャーバンドとの位置関係から、短期的な過熱感についてのコメントを組み立てる（ルールベース）。"""
    price = latest_row.get("Close")
    upper = latest_row.get("BB_upper")
    lower = latest_row.get("BB_lower")
    if price is None or upper is None or lower is None or \
            any(pd.isna(v) for v in [price, upper, lower]):
        return "ボリンジャーバンドを判定するためのデータが不足しています。"
    if price >= upper:
        return ("株価が+2σ(上限)に接近・突破しており、短期的な過熱感から反落しやすい局面とされますが、"
                "バンドに沿って上昇が続く「バンドウォーク」となる可能性もあります。")
    if price <= lower:
        return ("株価が-2σ(下限)に接近・突破しており、短期的な自律反発が入りやすい局面とされますが、"
                "バンドに沿って下落が続く可能性もあります。")
    return "株価はバンド内の中央付近で推移しており、方向感に乏しい状態と考えられます。"


def comment_ichimoku(latest_row) -> str:
    """一目均衡表の雲との位置関係から、トレンドについてのコメントを組み立てる（ルールベース）。"""
    price = latest_row.get("Close")
    span_a = latest_row.get("ichimoku_senkou_a")
    span_b = latest_row.get("ichimoku_senkou_b")
    if price is None or span_a is None or span_b is None or \
            any(pd.isna(v) for v in [price, span_a, span_b]):
        return ("一目均衡表を判定するためのデータが不足しています"
                "（先行スパンは未来方向にずれて描画されるため、直近データでは値が入らない場合があります）。")
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    if price > cloud_top:
        return "株価は雲の上に位置しており、上昇トレンド優勢のシグナルとされます。"
    if price < cloud_bottom:
        return "株価は雲の下に位置しており、下落トレンド優勢のシグナルとされます。"
    return "株価は雲の中に位置しており、方向感に乏しいトレンド転換期にある可能性があります。"


def comment_volume(latest_row) -> str:
    """出来高の水準から、市場の関心度合いについてのコメントを組み立てる（ルールベース）。"""
    vol = latest_row.get("Volume")
    vol_ma = latest_row.get("VolMA25")
    if vol is None or vol_ma is None or pd.isna(vol) or pd.isna(vol_ma) or vol_ma == 0:
        return "出来高を判定するためのデータが不足しています。"
    ratio = vol / vol_ma
    if ratio >= 1.5:
        return f"出来高が25日平均の{ratio:.1f}倍と大きく増加しており、株価の方向性に対する市場の関心が高まっている可能性があります。"
    if ratio <= 0.5:
        return f"出来高が25日平均の{ratio:.1f}倍と閑散気味で、様子見ムードが強い可能性があります。"
    return "出来高は平均的な水準で推移しています。"


def comment_margin(margin_ratio: float | None) -> str:
    """信用倍率から、需給についてのコメントを組み立てる（ルールベース）。"""
    if margin_ratio is None or pd.isna(margin_ratio):
        return "信用倍率を判定するためのデータが不足しています。"
    if margin_ratio >= 3:
        return f"信用倍率は{margin_ratio:.2f}倍と買い残が多く、将来的な戻り売り圧力（利益確定売り）に注意が必要と考えられます。"
    if margin_ratio <= 1:
        return f"信用倍率は{margin_ratio:.2f}倍と売り残が多く、踏み上げ（買い戻し）による上昇の可能性があります。"
    return f"信用倍率は{margin_ratio:.2f}倍で、買い方・売り方の勢力が概ね拮抗しています。"


def build_technical_sentence(latest_tech, ma_cross: str | None, macd_cross: str | None,
                              margin_ratio: float | None) -> str:
    """テクニカル指標から状況説明の文章を組み立てる（無料・ルールベース）。"""
    sentences = []

    if ma_cross == "golden":
        sentences.append("直近で移動平均線のゴールデンクロスが発生しています。")
    elif ma_cross == "dead":
        sentences.append("直近で移動平均線のデッドクロスが発生しています。")

    if macd_cross == "golden":
        sentences.append("MACDもゴールデンクロス（上昇シグナル）となっています。")
    elif macd_cross == "dead":
        sentences.append("MACDはデッドクロス（下降シグナル）となっています。")

    rsi = latest_tech.get("RSI14") if hasattr(latest_tech, "get") else None
    if rsi is not None:
        if rsi >= 70:
            sentences.append(f"RSIは{rsi:.1f}と70を超え、短期的には買われすぎの水準です。")
        elif rsi <= 30:
            sentences.append(f"RSIは{rsi:.1f}と30を下回り、短期的には売られすぎの水準です。")
        else:
            sentences.append(f"RSIは{rsi:.1f}で中立圏です。")

    if margin_ratio is not None:
        if margin_ratio >= 3:
            sentences.append(f"信用倍率は{margin_ratio:.2f}倍と買い残が多く、将来的な戻り売り圧力に注意が必要です。")
        elif margin_ratio <= 1:
            sentences.append(f"信用倍率は{margin_ratio:.2f}倍と売り残が多く、踏み上げ（買い戻し）による上昇の可能性があります。")
        else:
            sentences.append(f"信用倍率は{margin_ratio:.2f}倍で、買い方・売り方の勢力は概ね拮抗しています。")

    return " ".join(sentences) if sentences else "テクニカル面で特筆すべきシグナルは検出されませんでした。"


def generate_free_diagnosis(
    company_name: str,
    scores: dict | None,
    ppm: dict | None,
    flags: list[str],
    technical_sentence: str,
) -> str:
    """
    Claude APIを使わない総合診断コメントを組み立てる。無料・課金なしで動作する。
    """
    parts = [f"### {company_name} 総合診断（無料・ルールベース版）"]

    if scores:
        summary_line = "、".join(
            f"{axis}は{scores[axis]['label']}({_stars(scores[axis]['score'])})"
            for axis in AXIS_ORDER if scores.get(axis)
        )
        parts.append(f"経営分析の観点では、{summary_line}という評価です。")
        for axis in AXIS_ORDER:
            r = scores.get(axis)
            if r:
                parts.append(f"- **{axis}**: {r['detail']}")

    if ppm:
        parts.append(f"事業ポートフォリオの観点では**{ppm['quadrant']}**に位置づけられます。{ppm['detail']}")

    if flags:
        parts.append("財務面で機械的に検出された注意点: " + " / ".join(flags))
    else:
        parts.append("財務面で機械的に検出された特段の注意点はありません。")

    parts.append(f"テクニカル面では、{technical_sentence}")

    parts.append(
        "\n※本診断は定型ルールに基づく機械的な生成であり、Claude等の生成AIは使用していません。"
        "そのため完全無料でご利用いただけます。より柔軟で読みやすい自然文の診断コメントが欲しい場合は、"
        "Anthropic APIキーを設定すると生成AIによる診断も追加で利用できます（その場合のみ従量課金）。"
    )

    return "\n\n".join(parts)
