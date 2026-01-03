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

# 구글 시트 연결 (Secrets에 설정한 주소 사용)
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception:
    conn = None

# 2. 사이드바 메뉴
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info(f"접속 시간: {datetime.now().strftime('%H:%M:%S')}")

# --- [기능: 구글 시트 로그 기록 함수] ---
def record_log(code, name):
    if conn:
        try:
            # 실시간 로그 기록 (image_3ea728.png 구조에 맞춤)
            existing_data = conn.read(worksheet="Sheet1", ttl=0)
            new_entry = pd.DataFrame([{
                "날짜": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "종목코드": code,
                "종목명": name
            }])
            updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
            conn.update(worksheet="Sheet1", data=updated_df)
        except Exception:
            pass # 로그 실패 시에도 분석은 계속 진행

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 실전 투자용 AI 패턴 분석기")
    st.write("사용자님의 Ridge 회귀 모델이 5가지 주요 변수를 학습하여 미래를 예측합니다.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    with col2:
        st.write("") 
        run_button = st.button("AI 분석 시작", use_container_width=True)

    if run_button:
        try:
            status = st.empty()
            status.info("데이터를 수집하고 분석 중입니다...")

            # [1] 데이터 수집 및 로그 기록
            stocks_krx = fdr.StockListing('KRX')
            stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
            record_log(stock_code, stock_name) # 시트에 기록
            st.toast(f"'{stock_name}' 검색 데이터가 서버에 기록되었습니다.")

            # 최근 2년 데이터
            df = fdr.DataReader(stock_code, datetime.now()-timedelta(days=730), datetime.now())
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})

            # [2] 구글 트렌드 방어 로직 (오류 발생 시 패스)
            df['구글트렌드'] = 0
            try:
                pytrends = TrendReq(hl='ko', tz=360)
                pytrends.build_payload([stock_name], cat=0, timeframe='today 2-y', geo='KR')
                trends_df = pytrends.interest_over_time()
                if not trends_df.empty:
                    trends_df = trends_df[[stock_name]].resample('D').interpolate(method='linear')
                    df = df.join(trends_df).fillna(method='ffill').fillna(0)
                    df['구글트렌드'] = df[stock_name]
            except Exception:
                # Expecting value 오류 등을 여기서 차단합니다.
                st.warning("구글 트렌드 통신 제한으로 주가 지표 위주로 분석합니다.")

            # [3] AI 학습 (Ridge Regression)
            # 수식: $J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m \theta_j^2$
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target_return'] = df['종가'].pct_change().shift(-1)
            df_train = df.dropna().copy()

            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', '구글트렌드']
            X = df_train[features]
            y = df_train['target_return']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=1.0).fit(X_scaled, y)

            # [4] 미래 7거래일 예측
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5: # 영업일만
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    pred = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price)
                    future_dates.append(curr_d)

            # [5] 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="최근 시세", line=dict(color='#00CCFF')))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, name="AI 예측(7일)", line=dict(dash='dash', color='red')))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            status.success(f"{stock_name} 분석 완료!")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 대시보드")
    pw = st.text_input("관리자 비밀번호", type="password")
    
    if pw == st.secrets.get("admin_password", "0000"):
        if conn:
            try:
                data = conn.read(worksheet="Sheet1", ttl=0)
                st.subheader("🔥 실시간인기 종목 TOP 5")
                st.bar_chart(data['종목명'].value_counts().head(5), color='#FF4B4B')
                st.write("---")
                st.subheader("📝 전체 검색 로그")
                st.dataframe(data.sort_index(ascending=False), use_container_width=True)
            except:
                st.info("데이터를 불러오는 중입니다...")
    else:
        st.info("비밀번호를 입력해 주세요.")
