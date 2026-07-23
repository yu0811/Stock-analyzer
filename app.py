"""
株式総合分析ツール — メインStreamlitアプリ。

画面構成:
  サイドバー: お気に入り銘柄一覧（クリックで切替）
  上部: 銘柄コード入力・お気に入り追加/解除
  タブ1: ファンダメンタル分析 (PER/PBR/ROE/CF/自己資本比率 等 + 過去5年業績 + 会社予想
         + 安全性/収益性/効率性/成長性の5段階評価 + PPM観点の簡易事業分類)
  タブ2: テクニカル分析 (ローソク足+MA/BB/一目均衡表切替、MACD、RSI、出来高、信用倍率)
  タブ3: 総合診断 (無料のルールベース診断がデフォルト。Claude APIキー設定時はAI診断も追加可能)

起動: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ai_commentary
import data_sources as ds
import favorites as fav
import free_diagnosis
import fundamentals as fnd
import indicators as ind

st.set_page_config(page_title="株式総合分析ツール", layout="wide")


def get_secret(key: str, default=""):
    """
    st.secrets.get() は secrets.toml が一つも存在しない場合に例外を投げるため、
    それを吸収するラッパー。（ローカルでsecrets.tomlを用意していない場合でも
    アプリが落ちないようにするため）
    """
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# 簡易パスワードゲート（個人利用向け。公開URLで誰でも見られる状態を避けるため）
# ---------------------------------------------------------------------------
def check_password() -> bool:
    app_password = get_secret("APP_PASSWORD")
    if not app_password:
        return True  # secrets未設定ならゲートをスキップ（ローカル利用など）
    if st.session_state.get("password_ok"):
        return True
    pw = st.text_input("パスワード", type="password")
    if pw == app_password:
        st.session_state["password_ok"] = True
        st.rerun()
    elif pw:
        st.error("パスワードが違います。")
    return False


if not check_password():
    st.stop()


# ---------------------------------------------------------------------------
# データ取得（キャッシュ付きラッパー）
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_financials(code: str, _api_key: str):
    client = ds.JQuantsClient(_api_key)
    raw_statements = client.get_statements_raw(code)
    financials_df = ds.statements_to_financials_df(raw_statements)
    forecast = ds.extract_latest_company_forecast(raw_statements)
    return financials_df, forecast


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_margin(code: str, _api_key: str):
    client = ds.JQuantsClient(_api_key)
    raw = client.get_weekly_margin_interest_raw(code)
    return ds.weekly_margin_to_df(raw)


# ---------------------------------------------------------------------------
# サイドバー: お気に入り銘柄一覧
# ---------------------------------------------------------------------------
st.sidebar.header("⭐ お気に入り銘柄")
favorites_list = fav.load_favorites()
if not favorites_list:
    st.sidebar.caption("まだお気に入りがありません。銘柄を検索して「お気に入りに追加」から登録できます。")
else:
    for f in favorites_list:
        if st.sidebar.button(f"{f['name']}（{f['code']}）", key=f"fav_btn_{f['code']}", width='stretch'):
            st.session_state["code"] = f["code"]
            st.rerun()


# ---------------------------------------------------------------------------
# 入力欄
# ---------------------------------------------------------------------------
st.title("株式総合分析ツール")

if "code" not in st.session_state:
    st.session_state["code"] = "7203"

col_input, col_period = st.columns([2, 1])
with col_input:
    code = st.text_input("銘柄コード（例: 7203 = トヨタ自動車）", key="code")
with col_period:
    period_options = ["1日", "1週間", "1ヶ月", "6ヶ月", "1年", "2年", "5年", "10年"]
    period_label = st.selectbox("株価表示期間", period_options, index=5)

jquants_token = get_secret("JQUANTS_API_KEY", "")
anthropic_key = get_secret("ANTHROPIC_API_KEY", "")

if not code:
    st.info("銘柄コードを入力してください。")
    st.stop()

# --- 会社プロフィール・現在株価・お気に入り追加/解除 ---
try:
    profile = ds.get_company_profile(code)
    col_title, col_fav = st.columns([4, 1])
    with col_title:
        st.subheader(f"{profile.get('name') or code}（{code}）")
        st.caption(f"セクター: {profile.get('sector') or '不明'} / 現在株価: {profile.get('current_price')}")
    with col_fav:
        if fav.is_favorite(code):
            if st.button("★ お気に入り解除"):
                fav.remove_favorite(code)
                st.rerun()
        else:
            if st.button("☆ お気に入りに追加"):
                fav.add_favorite(code, profile.get("name") or code)
                st.rerun()
except Exception as e:
    st.warning(f"会社プロフィールの取得に失敗しました: {e}")
    profile = {}

tab_fundamental, tab_technical, tab_overall = st.tabs(
    ["ファンダメンタル分析", "テクニカル分析", "総合診断"]
)


# ---------------------------------------------------------------------------
# タブ1: ファンダメンタル分析
# ---------------------------------------------------------------------------
ratios = None
forecast = {}
scores_result = None
ppm_result = None
with tab_fundamental:
    if not jquants_token:
        st.warning(
            "J-Quants の API Key が設定されていません。"
            ".streamlit/secrets.toml（またはStreamlit CloudのSecrets）に "
            "JQUANTS_API_KEY を設定してください。"
            "（財務データ・会社予想・信用残高の取得に使用します）"
        )
    else:
        try:
            financials_df, forecast = fetch_financials(code, jquants_token)
            if financials_df.empty:
                st.info("財務データが見つかりませんでした。銘柄コードをご確認ください。")
            else:
                ratios = fnd.calculate_fundamental_ratios(financials_df)
                latest = ratios.iloc[-1]

                current_price = profile.get("current_price")
                if current_price:
                    valuation = fnd.calculate_valuation(
                        current_price, latest.get("EPS(円)"), latest.get("BPS(円)")
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("PER(倍)", f"{valuation['PER(倍)']:.2f}" if pd.notna(valuation["PER(倍)"]) else "N/A")
                    c2.metric("PBR(倍)", f"{valuation['PBR(倍)']:.2f}" if pd.notna(valuation["PBR(倍)"]) else "N/A")
                    c3.metric("ROE(%)", f"{latest.get('ROE(%)', float('nan')):.2f}")
                    c4.metric("自己資本比率(%)", f"{latest.get('自己資本比率(%)', float('nan')):.2f}")

                st.markdown("#### 過去5年の業績（PL実績）")
                pl_table = fnd.build_pl_table(financials_df, 5)
                st.dataframe(pl_table, width='stretch')
                st.caption(
                    "単位: 億円。前期比はその期の前年同期比(%)。"
                    "「売上総利益(粗利)」はJ-Quantsの無料/Lightプランの財務情報サマリーには含まれないため、"
                    "空欄になる場合があります（詳細BS/PL/CFはStandardプラン以上が必要）。"
                )

                st.markdown("#### 過去5年の財務指標（比率）推移")
                st.dataframe(fnd.last_5_years(ratios, 5).round(2), width='stretch')

                st.markdown("#### 来期 会社予想（決算短信ベース）")
                if forecast:
                    fc_df = fnd.format_company_forecast(forecast)
                    st.caption(f"対象事業年度: {fc_df.attrs.get('fiscal_year_end', '不明')}")
                    st.dataframe(fc_df, width='stretch')
                else:
                    st.info("会社予想データが見つかりませんでした。")

                # --- 経営分析5段階評価（安全性・収益性・効率性・成長性） ---
                st.markdown("#### 経営分析 5段階評価")
                st.caption("一般的な経営分析のベンチマークに基づく簡易評価です。業種により適正水準は異なります。")
                scores_result = fnd.score_fundamentals(latest)
                score_cols = st.columns(4)
                for col, (axis, result) in zip(score_cols, scores_result.items()):
                    with col:
                        score = result["score"]
                        stars = "★" * score + "☆" * (5 - score) if score else "評価不可"
                        st.metric(axis, stars, result["label"])
                        st.caption(result["detail"])

                # --- PPM観点の簡易事業分類 ---
                st.markdown("#### 事業ポートフォリオ分類（PPM観点・簡易版）")
                ppm_result = fnd.classify_ppm(latest)
                st.info(f"**{ppm_result['quadrant']}**\n\n{ppm_result['detail']}")
                st.caption(
                    "※本来のPPM(プロダクト・ポートフォリオ・マネジメント)分析は市場成長率と相対市場シェアの"
                    "2軸で判定しますが、個別銘柄の財務データだけでは市場シェアが分からないため、"
                    "ここでは売上高成長率とROE(収益性)を代理指標とした簡易分類にとどめています。"
                    "参考情報としてご利用ください。"
                )

                flags = fnd.diagnose_flags(latest)
                if flags:
                    st.markdown("#### 注意フラグ")
                    for f in flags:
                        st.warning(f)
        except Exception as e:
            st.error(f"財務データの取得・計算でエラーが発生しました: {e}")


# ---------------------------------------------------------------------------
# タブ2: テクニカル分析
# ---------------------------------------------------------------------------
tech_df = None
tech_df_full = None
margin_df = None
with tab_technical:
    try:
        # 移動平均線(200日)等が短い表示期間でも正しく計算されるよう、
        # 常に長期間(既定10年)の株価データを取得してから指標を計算し、
        # 画面表示だけを選択された期間で絞り込む。
        price_df = ds.get_price_history(code)
        tech_df_full = ind.build_all_technical_indicators(price_df)
        tech_df = ind.slice_by_period(tech_df_full, period_label)
        latest_row = tech_df_full.iloc[-1]

        overlay_choice = st.radio(
            "価格チャートに重ねる指標を選択",
            ["移動平均線", "ボリンジャーバンド", "一目均衡表"],
            horizontal=True,
        )
        show_ma5 = st.checkbox("MA5を表示", value=False)
        show_ma200 = st.checkbox("MA200を表示", value=False)

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.15, 0.2, 0.2],
            vertical_spacing=0.03,
            subplot_titles=("価格", "出来高", "MACD", "RSI"),
        )

        fig.add_trace(go.Candlestick(
            x=tech_df.index, open=tech_df["Open"], high=tech_df["High"],
            low=tech_df["Low"], close=tech_df["Close"], name="株価",
        ), row=1, col=1)

        if overlay_choice == "移動平均線":
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MA25"], name="MA25", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MA75"], name="MA75", line=dict(width=1)), row=1, col=1)
            if show_ma5:
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MA5"], name="MA5", line=dict(width=1)), row=1, col=1)
            if show_ma200:
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MA200"], name="MA200", line=dict(width=1)), row=1, col=1)
            golden = tech_df[tech_df["cross"] == "golden"]
            dead = tech_df[tech_df["cross"] == "dead"]
            fig.add_trace(go.Scatter(x=golden.index, y=golden["MA25"], mode="markers",
                                      marker=dict(symbol="triangle-up", size=10, color="red"),
                                      name="ゴールデンクロス"), row=1, col=1)
            fig.add_trace(go.Scatter(x=dead.index, y=dead["MA25"], mode="markers",
                                      marker=dict(symbol="triangle-down", size=10, color="blue"),
                                      name="デッドクロス"), row=1, col=1)
        elif overlay_choice == "ボリンジャーバンド":
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["BB_upper"], name="+2σ", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["BB_mid"], name="中心線", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["BB_lower"], name="-2σ", line=dict(width=1)), row=1, col=1)
        else:  # 一目均衡表
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["ichimoku_tenkan"], name="転換線", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["ichimoku_kijun"], name="基準線", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["ichimoku_senkou_a"], name="先行スパン1", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["ichimoku_senkou_b"], name="先行スパン2", line=dict(width=1), fill="tonexty"), row=1, col=1)

        fig.add_trace(go.Bar(x=tech_df.index, y=tech_df["Volume"], name="出来高"), row=2, col=1)
        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["VolMA25"], name="出来高MA25", line=dict(width=1)), row=2, col=1)

        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MACD"], name="MACD", line=dict(width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MACD_signal"], name="シグナル", line=dict(width=1)), row=3, col=1)
        fig.add_trace(go.Bar(x=tech_df.index, y=tech_df["MACD_hist"], name="ヒストグラム"), row=3, col=1)

        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["RSI14"], name="RSI14", line=dict(width=1)), row=4, col=1)
        fig.add_hline(y=70, line_dash="dot", row=4, col=1)
        fig.add_hline(y=30, line_dash="dot", row=4, col=1)

        fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(fig, width='stretch')

        # --- 各指標の見立て（無料・ルールベースの自動コメント） ---
        st.markdown("#### 各指標の見立て（ルールベース・自動生成）")
        st.markdown(f"**移動平均線:** {free_diagnosis.comment_ma(latest_row)}")
        st.markdown(f"**MACD:** {free_diagnosis.comment_macd(latest_row)}")
        st.markdown(f"**RSI:** {free_diagnosis.comment_rsi(latest_row)}")
        st.markdown(f"**ボリンジャーバンド:** {free_diagnosis.comment_bb(latest_row)}")
        st.markdown(f"**一目均衡表:** {free_diagnosis.comment_ichimoku(latest_row)}")
        st.markdown(f"**出来高:** {free_diagnosis.comment_volume(latest_row)}")
        st.caption(
            "※定型ルールに基づく機械的なコメントであり、将来の値動きを保証するものではありません。"
            "投資判断はご自身の責任で行ってください。"
        )

        # --- 信用倍率 ---
        # 優先順位: ①J-Quants(Standardプラン以上)の週次データがあれば最初からフル履歴を表示
        #           ②なければ日証金の無料CSV(登録不要)を使い、今日のスナップショットを
        #             ローカルに蓄積して自前で履歴グラフを育てていく
        st.markdown("#### 信用倍率（融資残高 ÷ 貸株残高）")
        margin_shown = False

        if jquants_token:
            try:
                margin_df = fetch_margin(code, jquants_token)
                if not margin_df.empty:
                    margin_df = ind.compute_margin_ratio(margin_df)
                    fig_margin = go.Figure()
                    fig_margin.add_trace(go.Scatter(x=margin_df["date"], y=margin_df["margin_ratio"], name="信用倍率"))
                    fig_margin.update_layout(height=300)
                    st.plotly_chart(fig_margin, width='stretch')
                    st.caption("J-Quants(Standardプラン以上)の週次信用残高データに基づく。")
                    st.markdown(f"**信用倍率:** {free_diagnosis.comment_margin(margin_df['margin_ratio'].iloc[-1])}")
                    margin_shown = True
            except Exception as e:
                st.info(f"J-Quantsの信用残高データは利用できませんでした（Standardプラン未契約の可能性があります）: {e}")

        if not margin_shown:
            try:
                snapshot = ds.get_taisyaku_margin_snapshot(code)
                if snapshot is None:
                    st.info(
                        "信用残高データが見つかりませんでした。"
                        "この銘柄が「貸借銘柄」でない可能性があります（貸借銘柄以外は無料データの対象外）。"
                    )
                else:
                    margin_df = ds.append_and_load_margin_history(code, snapshot)
                    margin_df = ind.compute_margin_ratio(margin_df)
                    c1, c2 = st.columns(2)
                    c1.metric("信用倍率（本日時点）", f"{snapshot['margin_ratio']:.2f}" if snapshot["margin_ratio"] else "N/A")
                    c2.metric("対象日", str(snapshot["date"]))
                    fig_margin = go.Figure()
                    fig_margin.add_trace(go.Scatter(x=margin_df["date"], y=margin_df["margin_ratio"], name="信用倍率"))
                    fig_margin.update_layout(height=300)
                    st.plotly_chart(fig_margin, width='stretch')
                    st.caption(
                        "日本証券金融(日証金)が無料公開する日次データ（登録不要）に基づく。"
                        "無料データは最新日のみの提供のため、このアプリを使うたびに1日分ずつ記録を蓄積しています。"
                        "グラフは使い始めた日からの履歴のみとなります。"
                        "最初から数年分の履歴が欲しい場合はJ-Quants Standardプラン(3,300円/月)をご検討ください。"
                    )
                    st.markdown(f"**信用倍率:** {free_diagnosis.comment_margin(snapshot['margin_ratio'])}")
            except Exception as e:
                st.warning(f"信用残高データの取得に失敗しました: {e}")

    except Exception as e:
        st.error(f"株価・テクニカル指標の取得でエラーが発生しました: {e}")


# ---------------------------------------------------------------------------
# タブ3: 総合診断
# ---------------------------------------------------------------------------
with tab_overall:
    if ratios is None or tech_df_full is None:
        st.info("ファンダメンタル・テクニカル両方のデータが揃うと診断コメントを生成できます。")
    else:
        latest_ratios = ratios.iloc[-1]
        # 表示期間の絞り込み(period_label)に関わらず、常に最新の状態で判定するため
        # ここでは画面表示用に切り出す前の tech_df_full を使う。
        latest_tech = tech_df_full.iloc[-1]
        ma_cross_series = tech_df_full[tech_df_full["cross"].notna()]
        macd_cross_series = tech_df_full[tech_df_full["MACD_cross"].notna()]
        last_ma_cross = ma_cross_series["cross"].iloc[-1] if not ma_cross_series.empty else None
        last_macd_cross = macd_cross_series["MACD_cross"].iloc[-1] if not macd_cross_series.empty else None
        margin_ratio_latest = margin_df["margin_ratio"].iloc[-1] if margin_df is not None and not margin_df.empty else None
        flags = fnd.diagnose_flags(latest_ratios)

        # --- 無料・ルールベース診断（デフォルト表示。Claude APIは使わないため課金なし） ---
        technical_sentence = free_diagnosis.build_technical_sentence(
            latest_tech, last_ma_cross, last_macd_cross, margin_ratio_latest
        )
        free_comment = free_diagnosis.generate_free_diagnosis(
            company_name=profile.get("name") or code,
            scores=scores_result,
            ppm=ppm_result,
            flags=flags,
            technical_sentence=technical_sentence,
        )
        st.markdown(free_comment)

        # --- AI(Claude)による診断コメントは、APIキーを設定した場合のみ任意で利用可能(従量課金) ---
        st.divider()
        if not anthropic_key:
            st.caption(
                "より自然な文章での診断コメントが欲しい場合は、Anthropic APIキーを設定すると"
                "Claude(AI)による診断も追加で利用できます（その場合のみ従量課金。未設定なら上記の無料診断のみで完結します）。"
            )
        else:
            if st.button("AIによる診断コメントも生成する（この操作のみ課金対象）"):
                with st.spinner("Claude が診断中..."):
                    try:
                        fundamentals_summary = ai_commentary.build_fundamentals_summary(
                            latest_ratios, forecast, scores_result, ppm_result
                        )
                        technical_summary = ai_commentary.build_technical_summary(
                            latest_tech, last_ma_cross, last_macd_cross, margin_ratio_latest
                        )
                        comment = ai_commentary.generate_diagnosis(
                            api_key=anthropic_key,
                            company_name=profile.get("name") or code,
                            fundamentals_summary=fundamentals_summary,
                            technical_summary=technical_summary,
                            flags=flags,
                        )
                        st.markdown("#### AIによる診断コメント")
                        st.markdown(comment)
                    except Exception as e:
                        st.error(f"診断コメントの生成でエラーが発生しました: {e}")

st.caption(
    "本ツールは情報提供を目的としており、投資判断を推奨するものではありません。"
    "データの正確性・最新性は保証されません。投資判断はご自身の責任で行ってください。"
)
