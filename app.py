import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# 1. 페이지 및 한국 시간(KST) 설정
st.set_page_config(page_title="재현&정우의 5개년 AI 분석기", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. 구글 시트 연결
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    conn = None

# 3. 사이드바
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.info(f"접속 시간(KST): {current_time_str}")

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 5개년 통합 AI 패턴 분석기")
    st.write("주가 데이터와 함께 **VIX(공포지수)** 및 **RSI(상대강도지수)**를 학습합니다.")
    
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    run_button = st.button("5년치 정밀 분석 시작", use_container_width=True)

    if run_button:
        try:
            # [단계 1] 5년치 데이터 수집
            start_date = KST_NOW - timedelta(days=365 * 5)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # [단계 2] 핵심 심리지표 생성 (설명 포함)
            # 1. VIX(시장 공포지수)
            try:
                vix = fdr.DataReader('^VIX', start_date)
                df = df.join(vix[['Close']].rename(columns={'Close': 'VIX(시장공포지수)'})).fillna(method='ffill').fillna(20)
            except:
                df['VIX(시장공포지수)'] = 20

            # 2. RSI(상대강도지수) 계산 로직
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI(매수·매도강도)'] = (100 - (100 / (1 + rs))).fillna(50)

            # 3. 기타 변수
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target'] = df['종가'].pct_change().shift(-1)
            
            # [단계 3] AI 학습 (Ridge)
            df_train = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX(시장공포지수)', 'RSI(매수·매도강도)']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_train[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_train['target'])

            # [단계 4] 막대그래프 시각화 (변수 설명 포함)
            st.subheader("💡 5개년 학습 AI 변수 기여도")
            st.write("막대가 길수록 AI가 미래 예측 시 해당 지표를 더 중요하게 고려했다는 뜻입니다.")
            importance = pd.DataFrame({'변수': features, '비중(%)': (np.abs(model.coef_) / np.sum(np.abs(model.coef_))) * 100})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 5] 7거래일 미래 예측
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df_train[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5:
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    pred = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [단계 6] 결과 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['종가'].iloc[-60:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 7일 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("RSI를 포함한 5년치 정밀 분석이 성공적으로 완료되었습니다!")

            # 로그 기록 (JSON 서비스 계정)
            if conn:
                try:
                    data = conn.read(worksheet="Sheet1", ttl=0)
                    new_row = pd.DataFrame([{"날짜": current_time_str, "종목코드": stock_code, "종목명": "RSI분석성공"}])
                    conn.update(worksheet="Sheet1", data=pd.concat([data, new_row], ignore_index=True))
                except: pass

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링 (KST)")
    pw = st.text_input("비밀번호", type="password")
    if pw == st.secrets.get("admin_password", "0801"):
        if conn:
            data = conn.read(worksheet="Sheet1", ttl=0)
            st.dataframe(data.iloc[::-1], use_container_width=True)
