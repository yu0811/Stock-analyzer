"""
完全無料のルールベース診断コメント生成モジュール。
Claude APIを一切呼び出さず、fundamentals.py で計算済みの5段階評価・PPM分類・
注意フラグと、テクニカル指標（RSI・クロス・信用倍率）から定型文を組み立てる。
API課金は一切発生しない。
"""
from __future__ import annotations

AXIS_ORDER = ["安全性", "収益性", "効率性", "成長性"]


def _stars(score: int | None) -> str:
    if not score:
        return "評価不可"
    return "★" * score + "☆" * (5 - score)


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
