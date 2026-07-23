"""
外部データ取得モジュール。

構成:
  - yfinance: 株価(OHLCV)取得。テクニカル分析の元データ。
  - 日本証券金融(日証金): 信用倍率のスナップショットを無料・登録不要で取得。
    「銘柄別残高一覧」(zandaka.csv) を毎日ダウンロードし、蓄積することで
    無料のまま信用倍率の推移グラフを作れる（詳しくは get_taisyaku_margin_snapshot
    のdocstring参照）。J-Quants Standardプラン(3,300円/月)を使わずに済む。
  - J-Quants API: 財務情報(決算短信ベース)・信用残高(週次、Standardプラン以上)を取得。
    無料登録が必要 (https://jpx-jquants.com/ 。ダッシュボードの「API Key」ページで
    発行される文字列を .streamlit/secrets.toml か環境変数 JQUANTS_API_KEY に設定する)。

【重要な注意】
2025年12月にJ-Quants APIがV2に移行し、(1)認証方式が「リフレッシュトークン」から
「APIキー」(ヘッダー x-api-key)に変更、(2)レスポンスの項目名(カラム名)が短縮形に
変更、という2点の破壊的変更があった(V1は2026年6月1日に廃止済み)。
本モジュールは認証部分はV2の x-api-key 方式に対応済みだが、
J-Quants /fins/statements のレスポンス項目名(FIELD_MAP内)は、V1時点の
公式ドキュメント(https://jpx.gitbook.io/j-quants-ja/api-reference/statements)を
もとにしたbest-effortの対応表であり、V2のカラム名短縮により実際のキー名と
ずれている可能性が高い。財務データ・会社予想が空欄/おかしい値になる場合は、
`debug_dump_raw_statement()` で生JSONを1度出力し、実際のキー名を確認のうえ
FIELD_MAP を書き換えること。

同様に、日証金CSVの列位置(TAISYAKU_COL_*)も、開発環境のネットワーク制限により
実際のCSVを取得してのカラム名検証ができておらず、公開されている解説ページの記述と
サンプル行の手計算による突き合わせで組み立てた位置推定です。実行後、値が想定と
異なる場合は `debug_dump_taisyaku_raw()` で生データを確認し、列位置を調整してください。
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

JQUANTS_BASE_URL = "https://api.jquants.com/v2"
TAISYAKU_ZANDAKA_URL = "http://www.taisyaku.jp/data/zandaka.csv"

# zandaka.csv の列位置（0始まり）。日証金の解説ページと、公開データの手動突き合わせによる推定値。
# 列0: 申込日 / 列1: 受渡日 / 列2: 銘柄コード / 列3: 銘柄名 / 列4: 市場 / 列6: 品貸区分
# 列7-13: 株数ベース(融資新規,融資返済,融資残,貸株新規,貸株返済,貸株残,差引残)
TAISYAKU_COL_DATE = 0
TAISYAKU_COL_CODE = 2
TAISYAKU_COL_NAME = 3
TAISYAKU_COL_FINANCING_BALANCE = 9   # 融資残（信用買い残の代理指標）
TAISYAKU_COL_LENDING_BALANCE = 12    # 貸株残（信用売り残の代理指標）

MARGIN_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "margin_history")


# ---------------------------------------------------------------------------
# yfinance: 株価データ
# ---------------------------------------------------------------------------

# Streamlit Community Cloud等、共有IPからのアクセスはYahoo Finance側に
# 「Too Many Requests」としてレート制限されることがある。ブラウザに近い
# User-Agentを付けたセッションを使い回すことで発生頻度を下げる。
_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
})


def _call_with_retry(fn, retries: int = 3, base_delay: float = 2.0):
    """
    yfinance呼び出し用のリトライヘルパー。
    「Too Many Requests」等の一時的なレート制限エラーの場合のみ、
    指数バックオフ（2秒→4秒→8秒）で再試行する。
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            is_rate_limit = "Too Many Requests" in msg or "429" in msg or "Rate limited" in msg
            if not is_rate_limit or attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last_err  # pragma: no cover


def to_yfinance_ticker(code: str) -> str:
    """4桁の証券コードを yfinance 形式（末尾に .T）に変換する。"""
    code = code.strip()
    if code.endswith(".T"):
        return code
    return f"{code}.T"


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_price_history(code: str, period: str = "2y") -> pd.DataFrame:
    """
    株価の時系列(OHLCV)を取得する。
    period: yfinance の period 指定 ('6mo','1y','2y','5y' など)
    """
    ticker = to_yfinance_ticker(code)

    def _fetch():
        df = yf.Ticker(ticker, session=_YF_SESSION).history(period=period, auto_adjust=False)
        if df.empty:
            raise ValueError(f"株価データが取得できませんでした: {ticker}")
        return df

    df = _call_with_retry(_fetch)
    df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_company_profile(code: str) -> dict:
    """yfinanceから会社の簡易プロフィール(社名・セクター等)を取得する。"""
    ticker = to_yfinance_ticker(code)
    info = _call_with_retry(lambda: yf.Ticker(ticker, session=_YF_SESSION).info)
    return {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
    }


# ---------------------------------------------------------------------------
# 日証金: 信用倍率（無料・登録不要）
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _fetch_taisyaku_zandaka_raw() -> pd.DataFrame:
    """
    日証金が無料公開している「銘柄別残高一覧」CSVを丸ごと取得する。
    登録・認証不要。1日1回程度の更新頻度。Shift_JIS(cp932)エンコード。
    """
    resp = requests.get(TAISYAKU_ZANDAKA_URL, timeout=20)
    resp.raise_for_status()
    resp.encoding = "cp932"
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), header=0, dtype=str)
    return df


def debug_dump_taisyaku_raw(code: str) -> None:
    """開発・検証用: 日証金CSVの該当銘柄の生の行を表示する(列位置調整に使う)。"""
    df = _fetch_taisyaku_zandaka_raw()
    row = df[df.iloc[:, TAISYAKU_COL_CODE] == code]
    st.write(row.T if not row.empty else "該当銘柄が見つかりませんでした（貸借銘柄でない可能性があります）")


def get_taisyaku_margin_snapshot(code: str) -> dict | None:
    """
    日証金CSVから、指定銘柄の最新の信用残高スナップショットを取得する。
    貸借銘柄（比較的流動性の高い銘柄）でないと該当データが存在しない。
    戻り値: {'date':.., 'financing_balance':.., 'lending_balance':.., 'margin_ratio':..} or None
    """
    df = _fetch_taisyaku_zandaka_raw()
    match = df[df.iloc[:, TAISYAKU_COL_CODE] == code]
    if match.empty:
        return None
    row = match.iloc[0]
    financing = _to_float(row.iloc[TAISYAKU_COL_FINANCING_BALANCE])
    lending = _to_float(row.iloc[TAISYAKU_COL_LENDING_BALANCE])
    if financing is None or lending is None:
        return None
    return {
        "date": row.iloc[TAISYAKU_COL_DATE],
        "financing_balance": financing,   # 融資残（信用買い残の代理指標）
        "lending_balance": lending,       # 貸株残（信用売り残の代理指標）
        "margin_ratio": financing / lending if lending else None,
    }


def append_and_load_margin_history(code: str, snapshot: dict) -> pd.DataFrame:
    """
    日証金の無料データは「最新1日分」のスナップショットしか提供されないため、
    アプリを起動するたびに今日の値を追記して、自前でローカルに履歴を蓄積する。
    これにより無料のまま徐々に信用倍率の推移グラフが作れるようになる
    （使い始めた日からの履歴のみ。過去に遡った履歴が最初から欲しい場合は
    J-Quants Standardプラン(3,300円/月)の週次信用残高データを使うこと）。

    注意: Streamlit Community Cloudなど一部のホスティングでは、再デプロイ時に
    ローカルファイルが消える場合がある。長期の履歴を確実に残したい場合は、
    外部ストレージ(例: Google Sheets, S3等)への保存に置き換えることを推奨する。
    """
    os.makedirs(MARGIN_HISTORY_DIR, exist_ok=True)
    path = os.path.join(MARGIN_HISTORY_DIR, f"{code}.csv")

    new_row = pd.DataFrame([{
        "date": snapshot["date"],
        "buy_balance": snapshot["financing_balance"],
        "sell_balance": snapshot["lending_balance"],
    }])

    if os.path.exists(path):
        history = pd.read_csv(path)
        history = pd.concat([history, new_row], ignore_index=True)
        history = history.drop_duplicates(subset="date", keep="last")
    else:
        history = new_row

    history.to_csv(path, index=False)
    history["date"] = pd.to_datetime(history["date"])
    return history.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# J-Quants: 認証
# ---------------------------------------------------------------------------

class JQuantsClient:
    """
    J-Quants APIクライアント（無料プラン想定）。

    2025年12月のJ-Quants API V2移行により、認証方式が
    「リフレッシュトークン→IDトークン」の2段階方式から、
    シンプルな「APIキー」方式（ヘッダー x-api-key）に変更された。
    APIキーは https://jpx-jquants.com/ のダッシュボード内
    「API Keys」ページで発行・確認できる。
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, path: str, params: dict) -> dict:
        headers = {"x-api-key": self.api_key}
        resp = requests.get(f"{JQUANTS_BASE_URL}{path}", headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_list(data: dict, preferred_keys: tuple[str, ...]) -> list[dict]:
        """
        レスポンスJSONの中からデータのリスト部分を取り出す。
        V2移行でレスポンスのラッパーキー名も変わった可能性があるため、
        想定キー名を優先しつつ、見つからなければ「リスト型の値を持つ最初のキー」を
        使うフォールバックを行う（キー名が想定と違っていても壊れないようにするため）。
        """
        for key in preferred_keys:
            if isinstance(data.get(key), list):
                return data[key]
        for v in data.values():
            if isinstance(v, list):
                return v
        return []

    def get_statements_raw(self, code: str) -> list[dict]:
        """
        決算短信ベースの財務情報サマリーの生データを取得する。
        V2移行で /fins/statements から /fins/summary にエンドポイント名が変更された
        （旧パスは403 Forbiddenになる）。
        """
        data = self._get("/fins/summary", {"code": code})
        return self._extract_list(data, ("statements", "summary"))

    def get_weekly_margin_interest_raw(self, code: str) -> list[dict]:
        """
        週次信用残高の生データを取得する（Standardプラン以上が必要）。
        注意: V2でのエンドポイント名は未確認（この開発環境では実地検証できていない）。
        取得に失敗した場合、呼び出し元(app.py)は日証金の無料データに自動フォールバックする。
        """
        data = self._get("/markets/weekly_margin_interest", {"code": code})
        return self._extract_list(data, ("weekly_margin_interest",))

    def get_short_selling_raw(self, sector33_code: str) -> list[dict]:
        """
        /markets/short_selling の生データを取得する（Standardプラン以上が必要）。
        注意: このエンドポイントは業種(33業種)単位の空売り集計であり、
        個別銘柄ごとの空売り比率ではない。参考情報として扱うこと。
        V2でのエンドポイント名は未確認（この開発環境では実地検証できていない）。
        """
        data = self._get("/markets/short_selling", {"sector33code": sector33_code})
        return self._extract_list(data, ("short_selling",))


def debug_dump_raw_statement(client: JQuantsClient, code: str) -> None:
    """開発・検証用: 生のJSONをそのまま表示する(FIELD_MAPの調整に使う)。"""
    raw = client.get_statements_raw(code)
    st.json(raw[:2] if raw else raw)


# ---------------------------------------------------------------------------
# J-Quants /fins/statements -> fundamentals.py 用スキーマへの変換
# ---------------------------------------------------------------------------

# 内部で使うキー -> J-Quants APIのレスポンス項目名。
# V2移行(2025年12月)でカラム名が短縮形に変更された。以下は実際にV2の
# /fins/summary を呼び出した実例(コミュニティ記事)で確認されたカラム名
# ['DiscDate','DiscTime','Code','DiscNo','DocType','CurPerType','CurPerSt',
#  'CurPerEn','CurFYSt','CurFYEn','NxtFYSt','NxtFYEn','Sales','OP','OdP','NP',
#  'EPS','DEPS','TA','Eq','EqAR','BPS','CFO','CFI','CFF','CashEq', ...
#  'NxFSales','NxFOP','NxFOdP','NxFNp','NxFEPS', ...] に基づく対応表。
# 銘柄やプランによって欠けている列がある可能性はあるため、値が想定と違う場合は
# `debug_dump_raw_statement()` で実際のキー名を確認すること。
FIELD_MAP = {
    "fiscal_year_end": "CurFYEn",
    "period_type": "CurPerType",   # 'FY','1Q','2Q','3Q'
    "doc_type": "DocType",
    "revenue": "Sales",
    "operating_income": "OP",
    "ordinary_income": "OdP",
    "net_income": "NP",
    "eps": "EPS",
    "total_assets": "TA",
    "equity": "Eq",
    "equity_ratio": "EqAR",
    "bps": "BPS",
    "operating_cf": "CFO",
    "investing_cf": "CFI",
    "financing_cf": "CFF",
    "shares_outstanding": "ShOutFY",
    # 会社予想（来期。本決算開示時に同時に開示される。通期分を使う）
    "next_fy_revenue_forecast": "NxFSales",
    "next_fy_operating_income_forecast": "NxFOP",
    "next_fy_ordinary_income_forecast": "NxFOdP",
    "next_fy_net_income_forecast": "NxFNp",
    "next_fy_eps_forecast": "NxFEPS",
}


def _to_float(v):
    if v in (None, "", "－", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def statements_to_financials_df(raw_statements: list[dict]) -> pd.DataFrame:
    """
    J-Quants /fins/summary の生データを、fundamentals.py が期待する
    financials DataFrame（fiscal_year をindexに、revenue/operating_income等を列に持つ）
    に変換する。本決算(FY)の開示のみを対象に、年度ごとの実績値を抽出する。
    """
    rows = []
    for s in raw_statements:
        if s.get(FIELD_MAP["period_type"]) != "FY":
            continue  # 本決算(通期)のみを使う。四半期は対象外。
        fy_end = s.get(FIELD_MAP["fiscal_year_end"])
        if not fy_end:
            continue
        rows.append({
            "fiscal_year": fy_end,
            "revenue": _to_float(s.get(FIELD_MAP["revenue"])),
            "operating_income": _to_float(s.get(FIELD_MAP["operating_income"])),
            "ordinary_income": _to_float(s.get(FIELD_MAP["ordinary_income"])),
            "net_income": _to_float(s.get(FIELD_MAP["net_income"])),
            "total_assets": _to_float(s.get(FIELD_MAP["total_assets"])),
            "equity": _to_float(s.get(FIELD_MAP["equity"])),
            "operating_cf": _to_float(s.get(FIELD_MAP["operating_cf"])),
            "investing_cf": _to_float(s.get(FIELD_MAP["investing_cf"])),
            "financing_cf": _to_float(s.get(FIELD_MAP["financing_cf"])),
            "shares_outstanding": _to_float(s.get(FIELD_MAP["shares_outstanding"])),
            # 流動資産・流動負債は決算短信サマリーには含まれないことが多いため NaN のままにする
            "current_assets": None,
            "current_liabilities": None,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset="fiscal_year", keep="last")
    df = df.set_index("fiscal_year").sort_index()
    return df


def extract_latest_company_forecast(raw_statements: list[dict]) -> dict:
    """直近の開示から、来期の会社予想を抽出する。"""
    if not raw_statements:
        return {}
    # 開示日が最も新しいものを採用（V2では DiscDate というカラム名になった）
    date_field = "DiscDate"
    latest = sorted(raw_statements, key=lambda s: s.get(date_field, ""), reverse=True)[0]
    return {
        "fiscal_year_end": latest.get(FIELD_MAP["fiscal_year_end"]),
        "revenue_forecast": _to_float(latest.get(FIELD_MAP["next_fy_revenue_forecast"])),
        "operating_income_forecast": _to_float(latest.get(FIELD_MAP["next_fy_operating_income_forecast"])),
        "ordinary_income_forecast": _to_float(latest.get(FIELD_MAP["next_fy_ordinary_income_forecast"])),
        "net_income_forecast": _to_float(latest.get(FIELD_MAP["next_fy_net_income_forecast"])),
        "eps_forecast": _to_float(latest.get(FIELD_MAP["next_fy_eps_forecast"])),
    }


# ---------------------------------------------------------------------------
# J-Quants /markets/weekly_margin_interest -> indicators.py 用スキーマへの変換
# ---------------------------------------------------------------------------

def weekly_margin_to_df(raw: list[dict]) -> pd.DataFrame:
    """
    週次信用残高の生データを、indicators.compute_margin_ratio が期待する
    DataFrame(columns: date, buy_balance, sell_balance) に変換する。
    """
    if not raw:
        return pd.DataFrame(columns=["date", "buy_balance", "sell_balance"])
    rows = []
    for r in raw:
        rows.append({
            "date": r.get("Date"),
            "buy_balance": _to_float(r.get("LongMarginOutstanding")),
            "sell_balance": _to_float(r.get("ShortMarginOutstanding")),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
