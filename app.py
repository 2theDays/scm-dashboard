import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import requests

# ============================================
# 페이지 기본 설정
# ============================================
st.set_page_config(
    page_title="SCM 시계열 분석 대시보드",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SCM 시계열 분석 대시보드")
st.markdown("""
<p style="
    font-size: 17px;
    font-weight: 600;
    color: #4FC3F7;
    letter-spacing: 0.04em;
    margin-top: -12px;
    margin-bottom: 16px;
    font-style: italic;
">📊 지수평활법(SES / Holt) 기반 주가 예측</p>
""", unsafe_allow_html=True)

# ============================================
# 커스텀 CSS
# ============================================
st.markdown("""
<style>
/* 셀렉트박스 좌측(값 표시) 영역 - 텍스트 입력 커서(I빔) */
div[data-baseweb="select"] > div > div:first-child {
    cursor: text !important;
}
/* 드롭다운 화살표(꺽쇠) 영역만 포인터 커서 */
div[data-baseweb="select"] > div > div:last-child {
    cursor: pointer !important;
}
div[data-baseweb="select"] svg {
    cursor: pointer !important;
}
/* 드롭다운 열렸을 때 옵션 목록 */
div[data-baseweb="popover"] li {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# 인기 종목 목록
# ============================================
POPULAR_STOCKS = {
    "🔍 직접 입력": "CUSTOM",
    "쿠팡 (CPNG)": "CPNG",
    "애플 (AAPL)": "AAPL",
    "테슬라 (TSLA)": "TSLA",
    "엔비디아 (NVDA)": "NVDA",
    "마이크로소프트 (MSFT)": "MSFT",
    "구글 (GOOGL)": "GOOGL",
    "아마존 (AMZN)": "AMZN",
    "메타 (META)": "META",
    "삼성전자 (005930.KS)": "005930.KS",
    "SK하이닉스 (000660.KS)": "000660.KS",
    "카카오 (035720.KS)": "035720.KS",
    "네이버 (035420.KS)": "035420.KS",
    "현대차 (005380.KS)": "005380.KS",
}

# ============================================
# RMSE 함수
# ============================================
def calculate_rmse(actual, predicted):
    actual = np.array(actual).ravel()
    predicted = np.array(predicted).ravel()
    return np.sqrt(mean_squared_error(actual, predicted))

# ============================================
# 데이터 로딩 (캐싱 - 종목/날짜 바뀔 때만 재다운로드)
# ============================================
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df_stock = yf.download(ticker, start=str(start), end=str(end), progress=False, auto_adjust=True)
    if df_stock.empty:
        return pd.DataFrame()
    # MultiIndex 컬럼 처리 (yfinance 버전에 따라 다름)
    if isinstance(df_stock.columns, pd.MultiIndex):
        df_stock.columns = df_stock.columns.droplevel(1)
    if "Close" not in df_stock.columns:
        return pd.DataFrame()
    df = df_stock[["Close"]].copy()
    df = df.dropna()
    df.columns = ["Close"]
    return df

@st.cache_data(show_spinner=False, ttl=300)
def search_ticker(query):
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "lang": "en-US", "quotesCount": 8, "newsCount": 0}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=5)
        quotes = r.json().get("quotes", [])
        results = []
        for q in quotes:
            symbol = q.get("symbol", "")
            name = q.get("longname") or q.get("shortname", "")
            q_type = q.get("quoteType", "")
            if symbol and name and q_type in ("EQUITY", "ETF"):
                results.append(f"{name} ({symbol})")
        return results
    except:
        return []

# ============================================
# 사이드바: 파라미터 설정
# ============================================
with st.sidebar:
    st.header("⚙️ 설정")

    # 종목 검색
    st.subheader("🔎 종목 검색")

    search_query = st.text_input(
        "회사명 또는 종목 코드 입력",
        placeholder="예: Palantir, 삼성, AAPL...",
        help="회사명이나 티커 코드를 입력하면 자동으로 검색됩니다"
    )

    if search_query:
        with st.spinner("검색 중..."):
            results = search_ticker(search_query)

        if results:
            selected_result = st.selectbox("검색 결과", options=results)
            company = selected_result.split("(")[-1].rstrip(")")
            st.caption(f"📌 티커 코드: `{company}`")
        else:
            st.warning("검색 결과가 없습니다. 인기 종목에서 선택하거나 코드를 직접 입력하세요.")
            company = search_query.strip().upper()
    else:
        selected_label = st.selectbox(
            "인기 종목 선택",
            options=list(POPULAR_STOCKS.keys()),
            index=1,
        )
        if POPULAR_STOCKS[selected_label] == "CUSTOM":
            company = st.text_input(
                "종목 코드 직접 입력",
                value="CPNG",
                placeholder="예: AAPL, 005930.KS, TSLA",
            )
        else:
            company = POPULAR_STOCKS[selected_label]
            st.caption(f"📌 티커 코드: `{company}`")

    st.divider()

    # 날짜 설정
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", value=date(2020, 1, 1))
    with col2:
        end_date = st.date_input("종료일", value=date.today())

    st.divider()

    # 테스트 기간
    test_size = st.slider("테스트 기간 (일)", min_value=10, max_value=90, value=30, step=5)

    st.divider()

    # SES 설정
    st.subheader("SES 설정")
    alpha_manual = st.slider("α 값 (수동 SES)", min_value=0.01, max_value=0.99, value=0.2, step=0.01)

    st.divider()

    # SCM 재고 설정
    st.subheader("🏭 SCM 재고 설정")
    lead_time = st.slider(
        "리드타임 (일, Lead Time)", min_value=1, max_value=30, value=5,
        help="발주 후 입고까지 걸리는 기간"
    )
    service_level = st.selectbox(
        "목표 서비스 수준 (Service Level)",
        options=["90%", "95%", "99%", "99.9%"],
        index=1,
        help="재고 부족 없이 수요를 충족시킬 확률"
    )
    daily_demand = st.number_input(
        "일평균 수요량 (단위)", min_value=1, max_value=10000, value=100, step=10,
        help="하루 평균 판매/소비량 (개, 박스 등)"
    )

# ============================================
# 데이터 로딩
# ============================================
company = company.strip().upper() if company else "CPNG"

with st.spinner(f"📡 {company} 데이터 불러오는 중..."):
    df = load_data(company, start_date, end_date)

if df.empty:
    st.error(f"❌ '{company}' 데이터를 찾을 수 없습니다. 종목 코드를 확인해주세요.")
    st.stop()

if len(df) < test_size + 30:
    st.error(f"❌ 데이터가 부족합니다. 현재 {len(df)}일치 데이터 (최소 {test_size + 30}일 필요). 기간을 늘려주세요.")
    st.stop()

# ============================================
# 모델 학습
# ============================================
train = df.iloc[:-test_size].copy()
test = df.iloc[-test_size:].copy()

with st.spinner("🧠 모델 학습 중..."):
    # SES 수동 α
    ses_model = ExponentialSmoothing(
        train["Close"], trend=None, seasonal=None, initialization_method="estimated"
    ).fit(smoothing_level=alpha_manual, optimized=False)
    train["SES"] = ses_model.fittedvalues
    test["SES"] = ses_model.forecast(steps=len(test)).values

    # SES α=0.8
    ses_08 = ExponentialSmoothing(
        train["Close"], trend=None, seasonal=None, initialization_method="estimated"
    ).fit(smoothing_level=0.8, optimized=False)
    train["SES_08"] = ses_08.fittedvalues
    test["SES_08"] = ses_08.forecast(steps=len(test)).values

    # SES 자동 α
    ses_auto = ExponentialSmoothing(
        train["Close"], trend=None, seasonal=None, initialization_method="estimated"
    ).fit()
    train["SES_AUTO"] = ses_auto.fittedvalues
    test["SES_AUTO"] = ses_auto.forecast(steps=len(test)).values
    alpha_auto = ses_auto.params["smoothing_level"]

    # Holt 모델
    holt_model = ExponentialSmoothing(
        train["Close"], trend="add", seasonal=None, initialization_method="estimated"
    ).fit()
    train["HOLT"] = holt_model.fittedvalues
    test["HOLT"] = holt_model.forecast(steps=len(test)).values
    alpha_holt = holt_model.params["smoothing_level"]
    beta_holt = holt_model.params["smoothing_trend"]

    # 이동평균 & 차분
    train["MA_20"] = train["Close"].rolling(window=20).mean()
    df["Diff"] = df["Close"].diff()

    # RMSE
    rmse_ses = calculate_rmse(test["Close"], test["SES"])
    rmse_ses_08 = calculate_rmse(test["Close"], test["SES_08"])
    rmse_ses_auto = calculate_rmse(test["Close"], test["SES_AUTO"])
    rmse_holt = calculate_rmse(test["Close"], test["HOLT"])

# ============================================
# SCM 재고 계산
# ============================================
sl_z_map = {"90%": 1.28, "95%": 1.645, "99%": 2.326, "99.9%": 3.09}
z_score = sl_z_map[service_level]

is_kr = company.endswith(".KS") or company.endswith(".KQ")
currency_symbol = "₩" if is_kr else "$"
unit_price = float(df["Close"].iloc[-1])
price_display = f"{unit_price:,.0f}" if is_kr else f"{unit_price:.2f}"

# 수요 표준편차: 일평균 수요의 20% (변동계수 0.2 가정)
demand_cv = 0.20
demand_std = daily_demand * demand_cv

# 안전재고 (Safety Stock): Z × σ_d × √LT
safety_stock = z_score * demand_std * np.sqrt(lead_time)

# 재주문점 (ROP): D̄ × LT + SS
reorder_point = daily_demand * lead_time + safety_stock

# 경제적 주문량 (EOQ): √(2DS/H)
annual_demand_total = daily_demand * 365
ordering_cost_per_order = max(unit_price * daily_demand * 0.01, 1.0)  # 1회 발주비용
holding_cost_per_unit = unit_price * 0.25  # 연간 보유비용 (단가의 25%)
eoq = np.sqrt(2 * annual_demand_total * ordering_cost_per_order / holding_cost_per_unit) if unit_price > 0 else daily_demand * 30
eoq = max(eoq, daily_demand)  # 최소 하루치 이상

# 재고 시뮬레이션 (120일)
np.random.seed(42)
sim_days = 120
inventory_levels = []
current_inv = reorder_point + eoq
orders_in_transit = []  # (도착일, 수량)

for day in range(sim_days):
    # 발주 도착 처리
    arrived = sum(q for (d, q) in orders_in_transit if d == day)
    current_inv += arrived
    orders_in_transit = [(d, q) for (d, q) in orders_in_transit if d != day]

    # 실제 수요 발생 (정규분포)
    actual_demand = max(0, np.random.normal(daily_demand, demand_std))
    stockout = max(0, actual_demand - current_inv)
    current_inv = max(0.0, current_inv - actual_demand)

    inventory_levels.append({
        "Day": day,
        "재고량": current_inv,
        "재주문점": reorder_point,
        "안전재고": safety_stock,
        "재고부족": stockout > 0
    })

    # 재주문점 이하 & 발주 중 없으면 발주
    if current_inv <= reorder_point and not orders_in_transit:
        orders_in_transit.append((day + lead_time, eoq))

sim_df = pd.DataFrame(inventory_levels)
stockout_count = sim_df["재고부족"].sum()

# 연간 비용 계산
num_orders_per_year = annual_demand_total / eoq
annual_ordering_cost = num_orders_per_year * ordering_cost_per_order
avg_inventory = eoq / 2 + safety_stock
annual_holding_cost = avg_inventory * holding_cost_per_unit
annual_total_cost = annual_ordering_cost + annual_holding_cost

# ============================================
# ③ 공급업체 안정성 스코어카드 계산
# ============================================
daily_returns = df["Close"].pct_change().dropna()
annual_vol = float(daily_returns.std() * np.sqrt(252))
avg_price = float(df["Close"].mean())

# 1. 가격 안정성: 연간 변동성이 낮을수록 높은 점수
price_stability = max(0.0, min(100.0, 100.0 - annual_vol * 100.0))

# 2. 추세 안정성: 20일 MA 방향과 가격 방향 일치율
ma20_dir = df["Close"].rolling(20).mean().diff()
price_dir_s = df["Close"].diff()
trend_consistency = float(((price_dir_s > 0) == (ma20_dir > 0)).dropna().mean() * 100.0)

# 3. 회복 탄력성: 최대 낙폭(MDD)이 작을수록 높은 점수
mdd = float(((df["Close"] - df["Close"].cummax()) / df["Close"].cummax()).min())
recovery_score = max(0.0, min(100.0, 100.0 + mdd * 150.0))

# 4. 예측 가능성: Holt RMSE 기준 (낮을수록 높은 점수)
mape_pct = (rmse_holt / avg_price * 100.0) if avg_price > 0 else 50.0
predictability = max(0.0, min(100.0, 100.0 - mape_pct * 8.0))

# 5. 충격 저항성: 일간 -5% 이하 하락 빈도가 낮을수록 높은 점수
big_drops_pct = float((daily_returns < -0.05).mean() * 100.0)
shock_resistance = max(0.0, min(100.0, 100.0 - big_drops_pct * 15.0))

score_labels = ["가격 안정성", "추세 안정성", "회복 탄력성", "예측 가능성", "충격 저항성"]
score_values = [price_stability, trend_consistency, recovery_score, predictability, shock_resistance]
total_score = float(np.mean(score_values))

# ============================================
# ② 공급 충격 이벤트 탐지 (일간 -5% 이하)
# ============================================
shock_series = daily_returns[daily_returns < -0.05]
shock_records = []
for shock_date in shock_series.index:
    shock_ret = float(daily_returns.loc[shock_date])
    before = df.loc[:shock_date, "Close"]
    pre_price = float(before.iloc[-2]) if len(before) >= 2 else float(before.iloc[-1])
    after = df.loc[shock_date:, "Close"]
    recovered = after[after >= pre_price]
    rec_days = int((recovered.index[0] - shock_date).days) if len(recovered) > 0 else None
    buffer_needed = (rec_days if rec_days is not None else 90) * int(daily_demand)
    shock_records.append({
        "날짜": shock_date.strftime("%Y-%m-%d"),
        "하락폭": f"{shock_ret * 100:.1f}%",
        "하락폭_값": shock_ret * 100.0,
        "회복일수": rec_days if rec_days is not None else "미회복",
        "필요 안전재고": f"{buffer_needed:,}단위",
    })
shock_df = pd.DataFrame(shock_records) if shock_records else pd.DataFrame()

# ============================================
# ① RMSE → 재고비용 연결 계산
# ============================================
models_rmse = {
    f"SES α={alpha_manual:.2f}": rmse_ses,
    "SES α=0.8": rmse_ses_08,
    f"자동 SES α={alpha_auto:.2f}": rmse_ses_auto,
    "Holt": rmse_holt,
}
rmse_cost_records = []
for model_name, rmse_val in models_rmse.items():
    err_rate = (rmse_val / avg_price) if avg_price > 0 else 0.1
    demand_err = int(daily_demand) * err_rate
    ss_m = z_score * demand_err * np.sqrt(lead_time)
    holding = ss_m * unit_price * 0.25
    rmse_cost_records.append({
        "모델": model_name,
        "RMSE": rmse_val,
        "예측오차율(%)": round(err_rate * 100, 2),
        "필요 안전재고(단위)": ss_m,
        "연간 보유비용": holding,
    })
rmse_cost_df = pd.DataFrame(rmse_cost_records).sort_values("RMSE").reset_index(drop=True)

# ============================================
# 상단 요약 지표
# ============================================
st.subheader(f"📊 {company} 분석 요약")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("전체 데이터", f"{len(df):,}일")
m2.metric("훈련 데이터", f"{len(train):,}일")
m3.metric("테스트 데이터", f"{len(test):,}일")
m4.metric("자동 SES α", f"{alpha_auto:.4f}")
m5.metric("Holt β", f"{beta_holt:.4f}")

st.divider()

# ============================================
# 탭 구성
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📉 원본 데이터",
    "🔵 이동평균",
    "🟠 SES 분석",
    "🟢 Holt 모델",
    "🏆 모델 비교",
    "🏭 SCM 공급망 리스크"
])

# ---------- 탭 1: 원본 데이터 ----------
with tab1:
    st.subheader(f"{company} 종가 원본 데이터")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="종가",
        line=dict(color="#1f77b4")
    ))
    fig.update_layout(
        xaxis_title="날짜", yaxis_title="주가",
        hovermode="x unified", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 원본 데이터 테이블 보기"):
        st.dataframe(df.tail(30), use_container_width=True)

# ---------- 탭 2: 이동평균 ----------
with tab2:
    st.subheader("이동평균 복습 (20일 MA)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=train.index, y=train["Close"],
        mode="lines", name="실제 종가",
        line=dict(color="#1f77b4"), opacity=0.5
    ))
    fig2.add_trace(go.Scatter(
        x=train.index, y=train["MA_20"],
        mode="lines", name="20일 이동평균",
        line=dict(color="orange", width=2)
    ))
    fig2.update_layout(
        xaxis_title="날짜", yaxis_title="주가",
        hovermode="x unified", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info("이동평균은 여러 날의 데이터를 같은 비중으로 평균냅니다. 모든 과거 데이터를 동일하게 취급하는 것이 한계점입니다.")

# ---------- 탭 3: SES 분석 ----------
with tab3:
    st.subheader("단순 지수평활법 (SES)")

    sub1, sub2 = st.tabs([f"수동 α={alpha_manual:.2f} vs α=0.8", "자동 α 비교"])

    with sub1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=train.index, y=train["Close"], name="훈련 실제값", line=dict(color="#aec7e8"), opacity=0.5))
        fig3.add_trace(go.Scatter(x=test.index, y=test["Close"], name="테스트 실제값", line=dict(color="#1f77b4", width=2)))
        fig3.add_trace(go.Scatter(x=train.index, y=train["SES"], name=f"SES α={alpha_manual:.2f} (적합)", line=dict(color="orange", dash="dash")))
        fig3.add_trace(go.Scatter(x=test.index, y=test["SES"], name=f"SES α={alpha_manual:.2f} (예측) RMSE={rmse_ses:.2f}", line=dict(color="orange", width=2)))
        fig3.add_trace(go.Scatter(x=train.index, y=train["SES_08"], name="SES α=0.8 (적합)", line=dict(color="green", dash="dash")))
        fig3.add_trace(go.Scatter(x=test.index, y=test["SES_08"], name=f"SES α=0.8 (예측) RMSE={rmse_ses_08:.2f}", line=dict(color="green", width=2)))
        fig3.add_vrect(
            x0=test.index[0], x1=test.index[-1],
            fillcolor="yellow", opacity=0.08,
            annotation_text="테스트 구간", annotation_position="top left"
        )
        fig3.update_layout(xaxis_title="날짜", yaxis_title="주가", hovermode="x unified", height=450,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig3, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric(f"RMSE (α={alpha_manual:.2f})", f"{rmse_ses:.4f}")
        c2.metric("RMSE (α=0.8)", f"{rmse_ses_08:.4f}")
        st.info("α 값이 작으면 그래프가 부드럽고, α 값이 크면 최근 변화에 민감하게 반응합니다.")

    with sub2:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=train.index, y=train["Close"], name="훈련 실제값", line=dict(color="#aec7e8"), opacity=0.5))
        fig4.add_trace(go.Scatter(x=test.index, y=test["Close"], name="테스트 실제값", line=dict(color="#1f77b4", width=2)))
        fig4.add_trace(go.Scatter(x=train.index, y=train["SES_AUTO"], name=f"자동 SES α={alpha_auto:.4f} (적합)", line=dict(color="purple", dash="dash")))
        fig4.add_trace(go.Scatter(x=test.index, y=test["SES_AUTO"], name=f"자동 SES (예측) RMSE={rmse_ses_auto:.2f}", line=dict(color="purple", width=2)))
        fig4.add_vrect(
            x0=test.index[0], x1=test.index[-1],
            fillcolor="yellow", opacity=0.08,
            annotation_text="테스트 구간", annotation_position="top left"
        )
        fig4.update_layout(xaxis_title="날짜", yaxis_title="주가", hovermode="x unified", height=450,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig4, use_container_width=True)

        st.metric("자동 최적 α", f"{alpha_auto:.4f}", help="모델이 자동으로 선택한 α 값")
        st.metric("RMSE (자동 SES)", f"{rmse_ses_auto:.4f}")
        st.info(f"모델이 자동으로 선택한 최적 α = {alpha_auto:.4f}")

# ---------- 탭 4: Holt 모델 ----------
with tab4:
    st.subheader("Holt 추세 모델 (이중 지수평활법)")

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=train.index, y=train["Close"], name="훈련 실제값", line=dict(color="#aec7e8"), opacity=0.5))
    fig5.add_trace(go.Scatter(x=test.index, y=test["Close"], name="테스트 실제값", line=dict(color="#1f77b4", width=2)))
    fig5.add_trace(go.Scatter(x=train.index, y=train["HOLT"], name="Holt 적합값", line=dict(color="red", dash="dash")))
    fig5.add_trace(go.Scatter(x=test.index, y=test["HOLT"], name=f"Holt 예측값 RMSE={rmse_holt:.2f}", line=dict(color="red", width=2)))
    fig5.add_vrect(
        x0=test.index[0], x1=test.index[-1],
        fillcolor="yellow", opacity=0.08,
        annotation_text="테스트 구간", annotation_position="top left"
    )
    fig5.update_layout(xaxis_title="날짜", yaxis_title="주가", hovermode="x unified", height=450,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig5, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Holt α (수준)", f"{alpha_holt:.4f}")
    c2.metric("Holt β (추세)", f"{beta_holt:.4f}")
    c3.metric("RMSE", f"{rmse_holt:.4f}")
    st.info("Holt 모델은 SES에 추세(Trend)를 추가합니다. 수준(Level)과 추세(Trend)를 모두 반영하므로 이중 지수평활법이라고도 합니다.")

    st.subheader("차분 (Differencing) - ARIMA 예고")
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=df.index, y=df["Diff"], name="1차 차분", line=dict(color="#17becf")))
    fig6.update_layout(xaxis_title="날짜", yaxis_title="변화량", hovermode="x unified", height=350)
    st.plotly_chart(fig6, use_container_width=True)
    st.info("차분(Differencing)은 오늘 값에서 어제 값을 빼는 방법입니다. ARIMA의 I(Integrated)가 바로 이 차분을 의미합니다.")

# ---------- 탭 5: 모델 비교 ----------
with tab5:
    st.subheader("🏆 모델 성능 비교 (RMSE)")

    rmse_df = pd.DataFrame({
        "모델": [f"SES α={alpha_manual:.2f}", "SES α=0.8", f"자동 SES α={alpha_auto:.2f}", "Holt"],
        "RMSE": [rmse_ses, rmse_ses_08, rmse_ses_auto, rmse_holt]
    }).sort_values("RMSE")

    best_model = rmse_df.iloc[0]["모델"]
    colors = ["gold" if m == best_model else "#636efa" for m in rmse_df["모델"]]

    fig7 = go.Figure(go.Bar(
        x=rmse_df["모델"],
        y=rmse_df["RMSE"],
        marker_color=colors,
        text=rmse_df["RMSE"].round(2),
        textposition="outside"
    ))
    fig7.update_layout(
        xaxis_title="모델", yaxis_title="RMSE",
        height=400, showlegend=False
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.dataframe(
        rmse_df.style.highlight_min(subset=["RMSE"], color="lightgreen"),
        use_container_width=True
    )

    st.success(f"✅ 가장 낮은 RMSE: **{best_model}** (RMSE = {rmse_df.iloc[0]['RMSE']:.4f})")
    st.warning("⚠️ RMSE가 낮다고 해서 미래 주가를 정확히 맞춘다는 뜻은 아닙니다. 오늘의 목표는 예측 구조와 평가 방법을 이해하는 것입니다.")

    # 전체 모델 통합 차트
    st.subheader("전체 모델 통합 비교 차트")
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=train.index, y=train["Close"], name="훈련 실제값", line=dict(color="#aec7e8"), opacity=0.4))
    fig8.add_trace(go.Scatter(x=test.index, y=test["Close"], name="테스트 실제값", line=dict(color="#1f77b4", width=2.5)))
    fig8.add_trace(go.Scatter(x=test.index, y=test["SES"], name=f"SES α={alpha_manual:.2f}", line=dict(dash="dash")))
    fig8.add_trace(go.Scatter(x=test.index, y=test["SES_08"], name="SES α=0.8", line=dict(dash="dash")))
    fig8.add_trace(go.Scatter(x=test.index, y=test["SES_AUTO"], name=f"자동 SES α={alpha_auto:.2f}", line=dict(dash="dot")))
    fig8.add_trace(go.Scatter(x=test.index, y=test["HOLT"], name="Holt", line=dict(color="red", dash="dot", width=2)))
    fig8.add_vrect(
        x0=test.index[0], x1=test.index[-1],
        fillcolor="yellow", opacity=0.08,
        annotation_text="테스트 구간", annotation_position="top left"
    )
    fig8.update_layout(xaxis_title="날짜", yaxis_title="주가", hovermode="x unified", height=500,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig8, use_container_width=True)

# ---------- 탭 6: SCM 공급망 리스크 분석 ----------
with tab6:
    st.subheader("🏭 SCM 공급망 리스크 분석")
    st.caption(f"분석 종목: **{company}** | 현재가: **{currency_symbol}{price_display}** | 기간: {start_date} ~ {end_date}")

    # ════════════════════════════════════════════
    # ③ 공급업체 안정성 스코어카드
    # ════════════════════════════════════════════
    st.markdown("### 🎯 공급업체 안정성 스코어카드")
    st.caption("이 종목 기업을 공급업체로 볼 때 얼마나 안정적인가를 5개 지표로 평가합니다.")

    # 등급 결정
    if total_score >= 80:
        score_color, score_grade, score_emoji = "#4CAF50", "매우 안정", "🟢"
    elif total_score >= 60:
        score_color, score_grade, score_emoji = "#FFC107", "양호", "🟡"
    elif total_score >= 40:
        score_color, score_grade, score_emoji = "#FF9800", "주의", "🟠"
    else:
        score_color, score_grade, score_emoji = "#F44336", "위험", "🔴"

    col_gauge, col_radar = st.columns([1, 2])

    with col_gauge:
        st.markdown(f"""
        <div style="text-align:center; padding:28px 16px; border-radius:14px;
                    border:2px solid {score_color}; margin:8px 0;">
            <div style="font-size:56px; font-weight:900; color:{score_color}; line-height:1.1;">
                {total_score:.0f}
            </div>
            <div style="font-size:18px; color:{score_color}; margin-top:6px; font-weight:600;">
                {score_emoji} {score_grade}
            </div>
            <div style="font-size:12px; color:#888; margin-top:6px;">
                종합 공급 안정성 (100점)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        for lbl, val in zip(score_labels, score_values):
            c = "#4CAF50" if val >= 80 else "#FFC107" if val >= 60 else "#FF9800" if val >= 40 else "#F44336"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;font-size:13px;">
                    <span>{lbl}</span>
                    <span style="color:{c};font-weight:700;">{val:.0f}점</span>
                </div>
                <div style="background:#2a2a2a;border-radius:4px;height:7px;margin-top:3px;">
                    <div style="background:{c};width:{val:.0f}%;height:7px;border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_radar:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=score_values + [score_values[0]],
            theta=score_labels + [score_labels[0]],
            fill="toself",
            fillcolor="rgba(79,195,247,0.18)",
            line=dict(color="#4FC3F7", width=2.5),
            name=company,
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10), gridcolor="#444"),
                angularaxis=dict(tickfont=dict(size=13)),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            height=400,
            margin=dict(t=30, b=30, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ════════════════════════════════════════════
    # ① 예측 정확도 → 재고비용 연결
    # ════════════════════════════════════════════
    st.markdown("### 📐 예측 정확도 → 재고비용 연결")
    st.caption("앞 탭의 RMSE를 그대로 활용합니다. 예측이 정확할수록 안전재고를 줄여 비용을 낮출 수 있습니다.")

    best_row = rmse_cost_df.iloc[0]
    worst_row = rmse_cost_df.iloc[-1]
    saving = worst_row["연간 보유비용"] - best_row["연간 보유비용"]
    ss_diff = worst_row["필요 안전재고(단위)"] - best_row["필요 안전재고(단위)"]

    ka, kb, kc = st.columns(3)
    ka.metric("최적 모델", best_row["모델"], help="RMSE가 가장 낮은 모델")
    kb.metric("안전재고 절감", f"{ss_diff:,.0f} 단위", help="최악 모델 대비 최적 모델의 안전재고 차이")
    kc.metric("연간 비용 절감", f"{currency_symbol}{saving:,.0f}", help="모델 선택으로 줄일 수 있는 연간 보유비용")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        best_name = best_row["모델"]
        bar_colors_ss = ["gold" if m == best_name else "#636efa" for m in rmse_cost_df["모델"]]
        fig_ss = go.Figure(go.Bar(
            x=rmse_cost_df["모델"],
            y=rmse_cost_df["필요 안전재고(단위)"].round(1),
            marker_color=bar_colors_ss,
            text=rmse_cost_df["필요 안전재고(단위)"].round(1),
            textposition="outside",
        ))
        fig_ss.update_layout(
            title="모델별 필요 안전재고 (단위)",
            yaxis_title="안전재고 (단위)",
            height=330, showlegend=False,
        )
        st.plotly_chart(fig_ss, use_container_width=True)

    with col_b2:
        bar_colors_cost = ["gold" if m == best_name else "#EF553B" for m in rmse_cost_df["모델"]]
        fig_bc = go.Figure(go.Bar(
            x=rmse_cost_df["모델"],
            y=rmse_cost_df["연간 보유비용"].round(0),
            marker_color=bar_colors_cost,
            text=[f"{currency_symbol}{v:,.0f}" for v in rmse_cost_df["연간 보유비용"].round(0)],
            textposition="outside",
        ))
        fig_bc.update_layout(
            title="모델별 연간 안전재고 보유비용",
            yaxis_title=f"연간 비용 ({currency_symbol})",
            height=330, showlegend=False,
        )
        st.plotly_chart(fig_bc, use_container_width=True)

    with st.expander("📋 상세 수치 보기"):
        display_cost = rmse_cost_df.copy()
        display_cost["RMSE"] = display_cost["RMSE"].round(4)
        display_cost["예측오차율(%)"] = display_cost["예측오차율(%)"].round(2)
        display_cost["필요 안전재고(단위)"] = display_cost["필요 안전재고(단위)"].round(1)
        display_cost["연간 보유비용"] = display_cost["연간 보유비용"].apply(lambda x: f"{currency_symbol}{x:,.0f}")
        st.dataframe(display_cost.style.highlight_min(subset=["RMSE"], color="lightgreen"), use_container_width=True)

    st.info(f"💡 **{best_name}** 모델을 사용하면 최대 **{currency_symbol}{saving:,.0f}**의 연간 안전재고 보유비용을 절감할 수 있습니다.")

    st.divider()

    # ════════════════════════════════════════════
    # ② 공급 충격 이벤트 탐지
    # ════════════════════════════════════════════
    st.markdown("### ⚡ 공급 충격 이벤트 탐지")
    st.caption("일간 주가 하락 -5% 이하를 '공급 충격 이벤트'로 정의합니다. 이 시점에 재고가 부족하면 공급 차질이 발생합니다.")

    if shock_df.empty:
        st.success(f"✅ {start_date} ~ {end_date} 기간 동안 주요 공급 충격(-5% 이하)이 발생하지 않았습니다.")
    else:
        sa, sb, sc_ = st.columns(3)
        sa.metric("충격 발생 횟수", f"{len(shock_df)}회")
        avg_drop = shock_df["하락폭_값"].mean()
        sa2_val = shock_df["하락폭_값"].min()
        sb.metric("최대 단일 하락", f"{sa2_val:.1f}%")
        numeric_rec = [r for r in shock_df["회복일수"] if isinstance(r, int)]
        sc_.metric("평균 회복일수", f"{np.mean(numeric_rec):.0f}일" if numeric_rec else "미회복 포함")

        # 충격 이벤트 타임라인 차트
        fig_shock = go.Figure()
        fig_shock.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name="종가",
            line=dict(color="#4FC3F7", width=1.5),
        ))

        shock_dates_idx = pd.to_datetime(shock_df["날짜"])
        shock_prices_list = [
            float(df.loc[d, "Close"]) if d in df.index else None
            for d in shock_dates_idx
        ]
        fig_shock.add_trace(go.Scatter(
            x=shock_dates_idx,
            y=shock_prices_list,
            mode="markers",
            name="공급 충격 발생",
            marker=dict(color="red", size=10, symbol="triangle-down"),
            hovertemplate="%{x}<br>주가: %{y:.2f}<extra></extra>",
        ))

        fig_shock.update_layout(
            xaxis_title="날짜", yaxis_title="주가",
            hovermode="x unified", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_shock, use_container_width=True)

        # 충격 TOP 10 테이블
        st.markdown("**충격 이벤트 (하락폭 상위 10개)**")
        top10 = (
            shock_df.sort_values("하락폭_값")
            .head(10)[["날짜", "하락폭", "회복일수", "필요 안전재고"]]
            .reset_index(drop=True)
        )
        top10.index += 1
        st.dataframe(top10, use_container_width=True)

        worst_shock = shock_df.sort_values("하락폭_값").iloc[0]
        st.warning(
            f"⚠️ 가장 큰 충격 ({worst_shock['날짜']}, {worst_shock['하락폭']}) 시 "
            f"**{worst_shock['필요 안전재고']}**의 완충 재고가 필요했습니다."
        )
