import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# ============================================
# 페이지 기본 설정
# ============================================
st.set_page_config(
    page_title="SCM 시계열 분석 대시보드",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SCM 시계열 분석 대시보드")
st.caption("지수평활법(SES / Holt) 기반 주가 예측")

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
    df_stock = yf.download(ticker, start=str(start), end=str(end), progress=False)
    if df_stock.empty:
        return pd.DataFrame()
    df = df_stock[["Close"]].copy()
    df = df.dropna()
    df.columns = ["Close"]
    return df

@st.cache_data(show_spinner=False)
def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except:
        return ticker

# ============================================
# 사이드바: 파라미터 설정
# ============================================
with st.sidebar:
    st.header("⚙️ 설정")

    # 종목 검색
    st.subheader("🔎 종목 검색")
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
            help="야후 파이낸스 기준 티커 코드를 입력하세요"
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 원본 데이터",
    "🔵 이동평균",
    "🟠 SES 분석",
    "🟢 Holt 모델",
    "🏆 모델 비교"
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
