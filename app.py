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
st.set_page_config(page_title="K-Investment AI Pro v4.0", layout="wide", page_icon="📈")
# 서버 시간이 아닌 실제 한국 시간 계산
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. 구글 시트 연결 (image_408b67.jpg 오류 방지를 위한 예외 처리)
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception:
    conn = None

# 3. 사이드바 내비게이션
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info(f"접속 시간(KST): {current_time_str}")

# --- [정밀 분석용 보조 함수] ---

def calculate_rsi(df, period=14):
    """RSI(상대강도지수) 계산: 과매수/과매도 심리를 수치화합니다."""
    delta = df['종가'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 7대 변수 통합 AI 패턴 분석기")
    st.write("구글 트렌드와 뉴스 심리 대신, 시장 공포지수(VIX)와 기술적 심리(RSI)를 활용해 더 안정적으로 분석합니다.")
    
    stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    run_button = st.button("정밀 분석 및 7일 예측 시작", use_container_width=True)

    if run_button:
        try:
            # [단계 1] 주가 데이터 수집 (네이버 금융 루트)
            df = fdr.DataReader(stock_code, KST_NOW - timedelta(days=730))
            if df.empty:
                st.error("주가 데이터를 가져올 수 없습니다. 코드를 확인하세요.")
                st.stop()
            
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # 종목명 확보
            try:
                stocks = fdr.StockListing('KRX')
                stock_name = stocks[stocks['Code'] == stock_code]['Name'].values[0]
            except: stock_name = f"종목({stock_code})"

            # [단계 2] 7대 핵심 변수 생성 (에러 유발 요소 제거)
            df['VIX'] = 20 # 기본값 설정
            try:
                vix = fdr.DataReader('VIX', KST_NOW - timedelta(days=730))
                df = df.join(vix[['Close']].rename(columns={'Close': 'VIX'})).fillna(method='ffill').fillna(20)
            except: pass
            
            df['RSI'] = calculate_rsi(df).fillna(50)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target'] = df['종가'].pct_change().shift(-1)
            
            # [단계 3] AI 학습 (Ridge Regression)
            df_train = df.dropna().copy()
            # 7대 변수: 날짜지수, 요일, 거래량, 변동성, 감성지수, VIX, RSI
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_train[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_train['target'])

            # [단계 4] 변수 기여도 막대그래프 (요청 사항)
            st.subheader("💡 AI 변수별 예측 기여도 분석")
            importance = pd.DataFrame({'변수': features, '비중(%)': (np.abs(model.coef_) / np.sum(np.abs(model.coef_))) * 100})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 5] 7거래일 영업일 기준 예측 (복구 완료)
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5: # 영업일만 계산
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    # 예측 수행
                    pred = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [단계 6] 결과 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 예측(7일)", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"{stock_name} 분석 및 7일 예측 완료!")

            # 로그 기록 (image_3ea728.png 구조 유지)
            if conn:
                try:
                    data = conn.read(worksheet="Sheet1", ttl=0)
                    new_row = pd.DataFrame([{"날짜": current_time_str, "종목코드": stock_code, "종목명": stock_name}])
                    conn.update(worksheet="Sheet1", data=pd.concat([data, new_row], ignore_index=True))
                except: pass

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링 (KST)")
    pw = st.text_input("비밀번호", type="password")
    
    if pw == st.secrets.get("admin_password", "0000"):
        st.success("인증 성공")
        if conn:
            try:
                # image_408b67.jpg의 HTTPError 방지를 위해 데이터 읽기 로직 보강
                data = conn.read(worksheet="Sheet1", ttl="0")
                if not data.empty:
                    st.subheader(f"🔥 실시간 인기 종목 TOP 5")
                    st.bar_chart(data['종목명'].value_counts().head(5), color='#FF4B4B')
                    st.write("---")
                    st.subheader("📝 상세 검색 로그")
                    st.dataframe(data.iloc[::-1], use_container_width=True)
                else:
                    st.info("기록된 데이터가 없습니다.")
            except Exception as e:
                st.error(f"대시보드 데이터를 불러올 수 없습니다: {e}")
                st.info("Streamlit Secrets에 설정된 구글 시트 URL이 올바른지 확인하세요.")
    else:
        st.info("관리자 비밀번호를 입력해 주세요.")
