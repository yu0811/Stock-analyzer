"""
Claude API を使った診断コメント生成モジュール。
中小企業診断士の視点で、財務指標・テクニカル指標をもとにした定性的な
コメントを生成する。既定モデルは Claude Sonnet 5
（コストと速度のバランスが良く、この用途に十分な性能があるため）。
"""
from __future__ import annotations

import anthropic

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """\
あなたは日本の中小企業診断士です。個人投資家向けに、上場企業の財務データと
株価のテクニカル指標をもとに、専門的だが分かりやすい診断コメントを作成します。

守るべきルール:
- 断定的な「買い」「売り」の投資助言はしない。あくまで経営状態・株価動向の解説に徹する。
- 数字の裏付けがある指摘を優先する。憶測は「〜と考えられます」等、断定を避ける表現にする。
- 収益性・安全性・成長性・効率性の観点を踏まえる。
- 与えられた5段階評価・PPM(事業ポートフォリオ)分類があれば、それらの評価結果を根拠とともに
  自然な文章に織り込んで解説する（単なる数値の繰り返しにしない）。
- 日本語、300〜500字程度で簡潔にまとめる。
"""


def _build_user_prompt(company_name: str, fundamentals_summary: str,
                        technical_summary: str, flags: list[str]) -> str:
    flags_text = "\n".join(f"- {f}" for f in flags) if flags else "特になし"
    return f"""\
【対象企業】{company_name}

【財務指標サマリー】
{fundamentals_summary}

【テクニカル指標サマリー】
{technical_summary}

【機械的に検出された注意フラグ】
{flags_text}

上記を踏まえ、中小企業診断士目線での総合診断コメントを作成してください。
"""


def generate_diagnosis(
    api_key: str,
    company_name: str,
    fundamentals_summary: str,
    technical_summary: str,
    flags: list[str],
    model: str = DEFAULT_MODEL,
) -> str:
    """診断コメントを生成する。API呼び出しに失敗した場合は例外を投げる。"""
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _build_user_prompt(
                company_name, fundamentals_summary, technical_summary, flags)}
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def build_fundamentals_summary(ratios_latest, forecast: dict | None = None,
                                scores: dict | None = None, ppm: dict | None = None) -> str:
    """ratios_latest (pandas.Series) から、プロンプト用のテキストサマリーを作る。"""
    lines = []
    for key in ["ROE(%)", "売上高営業利益率(%)", "自己資本比率(%)", "流動比率(%)",
                "総資産回転率(回)", "売上高成長率(%)", "経常利益成長率(%)", "フリーCF"]:
        val = ratios_latest.get(key)
        if val is not None:
            lines.append(f"{key}: {val:.2f}" if isinstance(val, (int, float)) else f"{key}: {val}")
    if forecast:
        lines.append(f"来期会社予想(売上高): {forecast.get('revenue_forecast', '未開示')}")
        lines.append(f"来期会社予想(純利益): {forecast.get('net_income_forecast', '未開示')}")
    if scores:
        lines.append("【経営分析5段階評価】")
        for axis, result in scores.items():
            lines.append(f"{axis}: {result['score']}/5（{result['label']}） - {result['detail']}")
    if ppm:
        lines.append(f"【PPM観点の事業分類（簡易）】{ppm['quadrant']} - {ppm['detail']}")
    return "\n".join(lines)


def build_technical_summary(latest_row, ma_cross: str | None, macd_cross: str | None,
                             margin_ratio: float | None = None) -> str:
    """最新のテクニカル指標行から、プロンプト用のテキストサマリーを作る。"""
    lines = []
    for key in ["Close", "MA25", "MA75", "RSI14"]:
        val = latest_row.get(key)
        if val is not None:
            lines.append(f"{key}: {val:.2f}" if isinstance(val, (int, float)) else f"{key}: {val}")
    if ma_cross:
        lines.append(f"直近の移動平均線クロス: {ma_cross}")
    if macd_cross:
        lines.append(f"直近のMACDクロス: {macd_cross}")
    if margin_ratio is not None:
        lines.append(f"信用倍率(買い残/売り残): {margin_ratio:.2f}")
    return "\n".join(lines)
