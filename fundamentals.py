"""
ファンダメンタルズ分析モジュール。
中小企業診断士的な経営診断フレームワーク
(収益性・安全性・成長性・効率性) に沿った財務比率を計算する。

入力の financials DataFrame は、決算期(fiscal_year, 例: '2025-03')を index に持ち、
以下の列を想定 (単位は円。取得できない項目は NaN でよい):

  revenue              売上高
  operating_income     営業利益
  ordinary_income      経常利益
  net_income           当期純利益
  total_assets         総資産
  equity               自己資本(純資産)
  current_assets       流動資産
  current_liabilities  流動負債
  operating_cf         営業活動によるキャッシュフロー
  investing_cf         投資活動によるキャッシュフロー
  financing_cf         財務活動によるキャッシュフロー
  shares_outstanding   発行済株式数(期末)
"""
from __future__ import annotations

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "revenue", "operating_income", "ordinary_income", "net_income",
    "total_assets", "equity", "current_assets", "current_liabilities",
    "operating_cf", "investing_cf", "financing_cf", "shares_outstanding",
]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """不足している列を NaN で補完する（データソース側の欠損に強くするため）。"""
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def calculate_fundamental_ratios(financials: pd.DataFrame) -> pd.DataFrame:
    """
    過去複数期分の財務データから経営分析指標を計算する。
    financials は fiscal_year 昇順（古い年度が先）を想定。
    """
    df = ensure_columns(financials).sort_index()
    out = pd.DataFrame(index=df.index)

    # --- 収益性 ---
    out["ROE(%)"] = df["net_income"] / df["equity"] * 100
    out["売上高営業利益率(%)"] = df["operating_income"] / df["revenue"] * 100
    out["売上高経常利益率(%)"] = df["ordinary_income"] / df["revenue"] * 100

    # --- 安全性 ---
    out["自己資本比率(%)"] = df["equity"] / df["total_assets"] * 100
    out["流動比率(%)"] = df["current_assets"] / df["current_liabilities"] * 100

    # --- 効率性 ---
    out["総資産回転率(回)"] = df["revenue"] / df["total_assets"]

    # --- 成長性（前期比） ---
    out["売上高成長率(%)"] = df["revenue"].pct_change(fill_method=None) * 100
    out["経常利益成長率(%)"] = df["ordinary_income"].pct_change(fill_method=None) * 100
    out["純利益成長率(%)"] = df["net_income"].pct_change(fill_method=None) * 100

    # --- キャッシュフロー ---
    out["営業CF"] = df["operating_cf"]
    out["投資CF"] = df["investing_cf"]
    out["財務CF"] = df["financing_cf"]
    out["フリーCF"] = df["operating_cf"] + df["investing_cf"]

    # --- EPS / BPS ---
    out["EPS(円)"] = df["net_income"] / df["shares_outstanding"]
    out["BPS(円)"] = df["equity"] / df["shares_outstanding"]

    return out


def calculate_valuation(current_price: float, eps: float, bps: float) -> dict:
    """現在株価からPER・PBRを計算する。"""
    per = current_price / eps if eps and eps > 0 else np.nan
    pbr = current_price / bps if bps and bps > 0 else np.nan
    return {"PER(倍)": per, "PBR(倍)": pbr}


def last_5_years(ratios: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """直近n期分を抽出する。"""
    return ratios.tail(n)


# PL(損益計算書)の実績表示に使う項目。(内部列名, 表示ラベル) の順。
# 売上総利益(粗利)は、J-Quants無料/Lightプランの財務情報サマリーAPIには含まれず、
# 詳細BS/PL/CF(/fins/details、Standardプラン以上)でのみ取得可能なため、
# データがない場合は空欄で表示される。
PL_ITEMS = [
    ("revenue", "売上高"),
    ("gross_profit", "売上総利益(粗利)"),
    ("operating_income", "営業利益"),
    ("ordinary_income", "経常利益"),
    ("net_income", "純利益"),
]


def build_pl_table(financials: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    過去n期分のPL実績推移表を作る（単位: 億円）。
    各項目について「実績(億円)」と「前期比(%)」を並べて表示する。
    financials は fiscal_year 昇順（古い年度が先）を想定。
    """
    df = financials.copy().sort_index()
    if "gross_profit" not in df.columns:
        df["gross_profit"] = np.nan

    out = pd.DataFrame(index=df.tail(n).index)
    out.index.name = "決算期"
    for col, label in PL_ITEMS:
        oku = df[col] / 1e8
        yoy = df[col].pct_change(fill_method=None) * 100
        out[f"{label}(億円)"] = oku.tail(n).round(1)
        out[f"{label} 前期比(%)"] = yoy.tail(n).round(1)
    return out


def format_company_forecast(forecast: dict) -> pd.DataFrame:
    """
    会社予想(会社が決算短信・EDINETで開示する次期業績予想)を整形する。
    forecast は例えば:
      {
        'fiscal_year_end': '2027-03',
        'revenue_forecast': ...,
        'operating_income_forecast': ...,
        'ordinary_income_forecast': ...,
        'net_income_forecast': ...,
        'eps_forecast': ...,
      }
    を想定。値が None の項目は「未開示」として表示する。
    """
    label_map = {
        "revenue_forecast": "売上高予想",
        "operating_income_forecast": "営業利益予想",
        "ordinary_income_forecast": "経常利益予想",
        "net_income_forecast": "純利益予想",
        "eps_forecast": "EPS予想(円)",
    }
    rows = []
    for key, label in label_map.items():
        val = forecast.get(key)
        rows.append({"項目": label, "会社予想値": val if val is not None else "未開示"})
    fy = forecast.get("fiscal_year_end", "不明")
    df = pd.DataFrame(rows).set_index("項目")
    df.attrs["fiscal_year_end"] = fy
    return df


def _score_from_thresholds(value, thresholds: list[float]) -> int | None:
    """
    thresholds は昇順4つの境界値 [t1,t2,t3,t4]。
    value < t1 -> 1, t1<=value<t2 -> 2, t2<=value<t3 -> 3, t3<=value<t4 -> 4, value>=t4 -> 5
    値が高いほど良い、という前提の指標に使う。
    """
    if value is None or pd.isna(value):
        return None
    t1, t2, t3, t4 = thresholds
    if value >= t4:
        return 5
    if value >= t3:
        return 4
    if value >= t2:
        return 3
    if value >= t1:
        return 2
    return 1


_LEVEL_LABELS = {5: "優良", 4: "良好", 3: "平均的", 2: "やや低い", 1: "低い", None: "判定不可"}


def _combine_scores(scores: list[int]) -> int | None:
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid))


def score_fundamentals(ratios_latest: pd.Series) -> dict:
    """
    中小企業診断士的な経営分析フレームワーク（安全性・収益性・効率性・成長性）で
    直近期の財務指標を5段階評価する。判定基準は一般的な経営分析のベンチマークに
    基づく簡易的なものであり、業種によって適正水準は大きく異なる点に注意。

    戻り値: {'安全性': {'score':1-5, 'label':.., 'detail':..}, '収益性': {...}, ...}
    """
    equity_ratio = ratios_latest.get("自己資本比率(%)")
    current_ratio = ratios_latest.get("流動比率(%)")
    roe = ratios_latest.get("ROE(%)")
    op_margin = ratios_latest.get("売上高営業利益率(%)")
    asset_turnover = ratios_latest.get("総資産回転率(回)")
    sales_growth = ratios_latest.get("売上高成長率(%)")
    ordinary_growth = ratios_latest.get("経常利益成長率(%)")

    # --- 安全性: 自己資本比率・流動比率 ---
    s_equity = _score_from_thresholds(equity_ratio, [20, 30, 40, 50])
    s_current = _score_from_thresholds(current_ratio, [100, 120, 150, 200])
    safety_score = _combine_scores([s_equity, s_current])

    # --- 収益性: ROE・売上高営業利益率 ---
    s_roe = _score_from_thresholds(roe, [3, 5, 8, 10])
    s_opm = _score_from_thresholds(op_margin, [1, 4, 7, 10])
    profitability_score = _combine_scores([s_roe, s_opm])

    # --- 効率性: 総資産回転率 ---
    s_turnover = _score_from_thresholds(asset_turnover, [0.5, 0.7, 1.0, 1.2])
    efficiency_score = s_turnover

    # --- 成長性: 売上高成長率・経常利益成長率 ---
    s_sales_growth = _score_from_thresholds(sales_growth, [-5, 0, 5, 10])
    s_profit_growth = _score_from_thresholds(ordinary_growth, [-5, 0, 5, 10])
    growth_score = _combine_scores([s_sales_growth, s_profit_growth])

    def fmt(v, unit=""):
        return f"{v:.1f}{unit}" if v is not None and pd.notna(v) else "データなし"

    return {
        "安全性": {
            "score": safety_score,
            "label": _LEVEL_LABELS[safety_score],
            "detail": f"自己資本比率{fmt(equity_ratio, '%')}、流動比率{fmt(current_ratio, '%')}。"
                      "一般的に自己資本比率は40%以上、流動比率は150%以上が安全性の目安とされます。",
        },
        "収益性": {
            "score": profitability_score,
            "label": _LEVEL_LABELS[profitability_score],
            "detail": f"ROE{fmt(roe, '%')}、売上高営業利益率{fmt(op_margin, '%')}。"
                      "ROEは8〜10%以上、営業利益率は業種平均以上であれば収益性が高いと判断されます。",
        },
        "効率性": {
            "score": efficiency_score,
            "label": _LEVEL_LABELS[efficiency_score],
            "detail": f"総資産回転率{fmt(asset_turnover, '回')}。"
                      "1回転前後が目安ですが、装置産業など資本集約型の業種では低めに出る傾向があるため、"
                      "同業他社との比較が重要です。",
        },
        "成長性": {
            "score": growth_score,
            "label": _LEVEL_LABELS[growth_score],
            "detail": f"売上高成長率{fmt(sales_growth, '%')}、経常利益成長率{fmt(ordinary_growth, '%')}。"
                      "5〜10%以上の成長が続いていれば成長性が高いと判断されます。",
        },
    }


def classify_ppm(ratios_latest: pd.Series) -> dict:
    """
    PPM(プロダクト・ポートフォリオ・マネジメント/BCGマトリクス)の考え方を援用した
    簡易的な事業分類。本来は「市場成長率」と「相対市場シェア」の2軸で分類するが、
    個別銘柄の財務データだけでは市場シェアは分からないため、ここでは
    「売上高成長率」を市場成長性の代理指標、「収益性(ROEないし営業利益率)」を
    競争優位性の代理指標として、簡易的に4象限に分類する。
    あくまで参考程度の簡易分類であり、実際の市場シェアに基づく厳密なPPM分析ではない点に注意。
    """
    sales_growth = ratios_latest.get("売上高成長率(%)")
    roe = ratios_latest.get("ROE(%)")
    op_margin = ratios_latest.get("売上高営業利益率(%)")

    if sales_growth is None or pd.isna(sales_growth) or \
            (roe is None or pd.isna(roe)) and (op_margin is None or pd.isna(op_margin)):
        return {"quadrant": "判定不可", "detail": "成長率・収益性データが不足しているため分類できません。"}

    profitability = roe if roe is not None and pd.notna(roe) else op_margin
    growth_high = sales_growth >= 5
    profit_high = profitability >= 8

    if growth_high and profit_high:
        quadrant = "花形（スター）"
        detail = "売上が伸びており収益性も高い、成長事業の段階にあると考えられます。積極投資により将来の収益源となり得ますが、競争も激しい局面です。"
    elif not growth_high and profit_high:
        quadrant = "金のなる木（キャッシュカウ）"
        detail = "売上成長は鈍化しているものの収益性は高く、安定的にキャッシュを生み出している成熟事業と考えられます。"
    elif growth_high and not profit_high:
        quadrant = "問題児（クエスチョンマーク）"
        detail = "売上は伸びているものの収益性がまだ低く、投資段階にあると考えられます。今後シェアを拡大し収益化できるかが焦点です。"
    else:
        quadrant = "負け犬（ドッグ）"
        detail = "売上成長・収益性ともに低調です。事業の見直しや選択と集中が必要な局面である可能性があります。"

    return {
        "quadrant": quadrant,
        "detail": detail,
        "sales_growth": sales_growth,
        "profitability": profitability,
    }


def diagnose_flags(ratios_latest: pd.Series) -> list[str]:
    """
    直近期の指標から、簡易的な注意フラグ（悪化シグナル）を機械的に抽出する。
    Claude への診断コメント生成プロンプトの材料としても使う。
    """
    flags = []
    if ratios_latest.get("自己資本比率(%)", np.nan) < 30:
        flags.append("自己資本比率が30%を下回っており、財務の安全性にやや懸念があります。")
    if ratios_latest.get("流動比率(%)", np.nan) < 100:
        flags.append("流動比率が100%を下回っており、短期的な支払い能力に注意が必要です。")
    if ratios_latest.get("売上高成長率(%)", np.nan) < 0:
        flags.append("売上高が前期比で減少しています。")
    if ratios_latest.get("フリーCF", np.nan) < 0:
        flags.append("フリーキャッシュフローがマイナスです。")
    if ratios_latest.get("ROE(%)", np.nan) < 0:
        flags.append("ROEがマイナスであり、純損失が発生している可能性があります。")
    return flags
