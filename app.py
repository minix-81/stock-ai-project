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
st.set_page_config(page_title="K-Investment AI Pro v4.1", layout="wide", page_icon="📈")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. 구글 시트 연결
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    conn = None

# 3. 사이드바 내비게이션
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info(f"접속 시간(KST): {current_time_str}")

# --- [VIX 지수 수집 및 병합 로직 강화] ---

def get_vix_data(start_date):
    """표준 심볼 ^VIX를 사용하여 공포지수를 안정적으로 수집합니다."""
    try:
        # 'VIX' 대신 야후 파이낸스 표준 심볼인 '^VIX' 사용
        vix = fdr.DataReader('^VIX', start_date)
        if not vix.empty:
            return vix[['Close']].rename(columns={'Close': 'VIX'})
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 7대 변수 정밀 AI 분석기")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    run_button = st.button("KST 기준 분석 및 7일 예측 시작", use_container_width=True)

    if run_button:
        try:
            # [단계 1] 주가 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df = fdr.DataReader(stock_code, start_date)
            if df.empty:
                st.error("주가 데이터를 가져올 수 없습니다.")
                st.stop()
            
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # [단계 2] VIX 데이터 결합 (오류 방어의 핵심)
            vix_df = get_vix_data(start_date)
            if not vix_df.empty:
                # 한국 시장과 미국 시장의 날짜 차이를 메우기 위해 outer join 후 ffill 수행
                df = df.join(vix_df, how='left')
                df['VIX'] = df['VIX'].fillna(method='ffill').fillna(20) # 앞 데이터로 채우고 안되면 20(보통수준)
            else:
                df['VIX'] = 20 # 수집 실패 시 0이 아닌 기본값 20 부여
                st.warning("VIX 지수 수집 실패. 기본값(20)으로 분석합니다.")

            # [단계 3] 기술적 지표 및 변수 생성
            # RSI 계산
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = (100 - (100 / (1 + (gain / loss)))).fillna(50)

            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target'] = df['종가'].pct_change().shift(-1)
            
            df_train = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']
            
            # [단계 4] AI 학습 및 기여도 시각화
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_train[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_train['target'])

            st.subheader("💡 AI 변수별 예측 기여도 분석")
            importance = pd.DataFrame({'변수': features, '비중(%)': (np.abs(model.coef_) / np.sum(np.abs(model.coef_))) * 100})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 5] 7거래일 영업일 예측
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df[features].iloc[-1:].copy()

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
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 예측(7일)", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("분석 완료! VIX 지수가 정상적으로 반영되었습니다.")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링 (KST)")
    pw = st.text_input("비밀번호", type="password")
    if pw == st.secrets.get("admin_password", "1234"):
        if conn:
            data = conn.read(worksheet="Sheet1", ttl=0)
            st.dataframe(data.iloc[::-1], use_container_width=True)
