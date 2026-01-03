import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pytrends.request import TrendReq
from streamlit_gsheets import GSheetsConnection

# 1. 페이지 설정 및 구글 시트 연결
st.set_page_config(page_title="K-Investment AI Pro", layout="wide", page_icon="📈")
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    conn = None

# 2. 사이드바 메뉴
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.info(f"현재 날짜: {datetime.now().strftime('%Y-%m-%d')}")

# --- [기능: 로그 기록 함수] ---
def save_log(code, name):
    if conn:
        try:
            existing_data = conn.read(worksheet="Sheet1", ttl=0)
            new_log = pd.DataFrame([{"날짜": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "종목코드": code, "종목명": name}])
            updated_df = pd.concat([existing_data, new_log], ignore_index=True)
            conn.update(worksheet="Sheet1", data=updated_df)
        except: pass

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 실전 투자용 AI 패턴 분석기")
    st.write("사용자님의 Ridge 회귀 엔진이 5가지 주요 변수를 학습하여 미래를 예측합니다.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    with col2:
        st.write("") 
        run_button = st.button("AI 분석 및 로그 기록 시작", use_container_width=True)

    if run_button:
        try:
            status = st.empty()
            status.info("최근 2년 데이터를 수집하고 분석 중입니다...")

            # [1] 데이터 수집 및 로그 기록
            stocks_krx = fdr.StockListing('KRX')
            stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
            save_log(stock_code, stock_name)
            st.toast(f"'{stock_name}' 검색 내역이 구글 시트에 기록되었습니다.")

            df = fdr.DataReader(stock_code, datetime.now()-timedelta(days=730), datetime.now())
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})

            # [2] 구글 트렌드 (차단 시 방어 로직)
            df['구글트렌드'] = 0
            try:
                pytrends = TrendReq(hl='ko', tz=360)
                pytrends.build_payload([stock_name], cat=0, timeframe='today 2-y', geo='KR')
                trends_df = pytrends.interest_over_time()
                if not trends_df.empty:
                    trends_df = trends_df[[stock_name]].resample('D').interpolate(method='linear')
                    df = df.join(trends_df).fillna(method='ffill').fillna(0)
                    df['구글트렌드'] = df[stock_name]
            except:
                st.warning("구글 트렌드 일시 차단됨: 주가 데이터 위주로 정밀 분석을 수행합니다.")

            # [3] AI 변수 생성 및 Ridge 학습
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target_return'] = df['종가'].pct_change().shift(-1)
            df_train = df.dropna().copy()

            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', '구글트렌드']
            X_scaled = StandardScaler().fit_transform(df_train[features])
            y = df_train['target_return']
            
            # Ridge 모델: 과적합 방지를 위해 L2 규제를 사용합니다.
            # $$J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^m \theta_j^2$$
            model = Ridge(alpha=1.0).fit(X_scaled, y)

            # [4] 미래 7거래일 예측
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5:
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    pred = model.predict(StandardScaler().fit(df_train[features]).transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [5] 시각화
            st.subheader(f"📊 {stock_name} AI 분석 결과")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="실제 시세"))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, name="AI 예측(7일)", line=dict(dash='dash', color='red')))
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            status.success("분석 완료!")

        except Exception as e:
            st.error(f"오류 발생: {e}")

# --- [관리자 페이지] ---
elif menu == "관리자 대시보드":
    pw = st.text_input("비밀번호", type="password")
    if pw == st.secrets.get("admin_password", "0000"):
        if conn:
            data = conn.read(worksheet="Sheet1", ttl=0)
            st.dataframe(data.iloc[::-1], use_container_width=True)
