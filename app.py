import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# 1. 페이지 설정 (KST 한국 시간 기준)
st.set_page_config(page_title="주식 AI 분석기 v4.5", layout="wide", page_icon="📊")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. 구글 시트 연결 (Secrets에 설정된 JSON 키 활용)
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    conn = None

# 3. 사이드바 내비게이션
with st.sidebar:
    st.title("🚀 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info(f"현재 시간(KST): {current_time_str}")

# --- [정밀 분석용 보조 함수] ---

def get_vix_data(start_date):
    """^VIX 심볼을 사용하여 공포지수를 수집합니다."""
    try:
        vix = fdr.DataReader('^VIX', start_date)
        if not vix.empty:
            return vix[['Close']].rename(columns={'Close': 'VIX(시장공포지수)'})
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📊 주식 AI 분석기")
    st.write("최근 5개년 데이터를 기반으로 **VIX(시장공포지수)**와 **RSI(매수·매도강도)**를 정밀 분석합니다.")
    
    stock_code = st.text_input("종목 번호 6자리 (예: 삼성전자 005930):", value="005930")
    run_button = st.button("5개년 데이터 정밀 분석 및 7일 예측 시작", use_container_width=True)

    if run_button:
        try:
            # [단계 1] 5년치 데이터 수집 (timedelta 1825일)
            start_date = KST_NOW - timedelta(days=365 * 5)
            df = fdr.DataReader(stock_code, start_date)
            
            if df.empty:
                st.error("주가 데이터를 가져올 수 없습니다. 코드를 확인하세요.")
                st.stop()
            
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # VIX(시장 공포지수) 수집 및 설명 포함
            vix_df = get_vix_data(start_date)
            if not vix_df.empty:
                df = df.join(vix_df, how='left').fillna(method='ffill').fillna(20)
            else:
                df['VIX(시장공포지수)'] = 20

            # [단계 2] RSI(매수·매도강도) 계산
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI(매수·매도강도)'] = (100 - (100 / (1 + rs))).fillna(50)

            # AI 학습용 변수 생성
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target'] = df['종가'].pct_change().shift(-1)
            
            # [단계 3] AI 학습 (Ridge Regression)
            # $J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m \theta_j^2$
            df_train = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX(시장공포지수)', 'RSI(매수·매도강도)']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_train[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_train['target'])

            # [단계 4] AI 변수 기여도 시각화
            st.subheader("💡 AI 모델이 분석한 지표별 중요도")
            importance = pd.DataFrame({'변수': features, '비중(%)': (np.abs(model.coef_) / np.sum(np.abs(model.coef_))) * 100})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 5] 7거래일(영업일 기준) 예측 루프
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df_train[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5: # 영업일만 포함
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    pred = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [단계 6] 결과 시각화
            fig = go.Figure()
            # 최근 60거래일 시세 (5년치라 60일이 보기 편함)
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['종가'].iloc[-60:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            # 미래 7일 예측 (빨간 점선)
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 7일 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"최근 5개년 데이터를 기반으로 한 7거래일 분석이 완료되었습니다.")

            # [단계 7] 구글 시트에 로그 기록
            if conn:
                try:
                    data = conn.read(worksheet="Sheet1", ttl=0)
                    new_row = pd.DataFrame([{"날짜": current_time_str, "종목코드": stock_code, "종목명": "AI_분석완료"}])
                    conn.update(worksheet="Sheet1", data=pd.concat([data, new_row], ignore_index=True))
                except: pass

        except Exception as e:
            st.error(f"분석 도중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링")
    pw = st.text_input("비밀번호", type="password")
    if pw == st.secrets.get("admin_password", "0801"):
        st.success("로그인 성공")
        if conn:
            try:
                data = conn.read(worksheet="Sheet1", ttl=0)
                st.dataframe(data.iloc[::-1], use_container_width=True)
            except Exception as e:
                st.error(f"시트 데이터를 불러올 수 없습니다: {e}")
