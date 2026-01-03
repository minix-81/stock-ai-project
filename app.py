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

# 1. 페이지 설정 및 한국 시간(KST) 정의
st.set_page_config(page_title="K-Investment AI Pro", layout="wide", page_icon="📈")

# 서버 시간(UTC)을 한국 시간(KST, UTC+9)으로 변환
KST = datetime.now() + timedelta(hours=9)
current_time_str = KST.strftime('%Y-%m-%d %H:%M:%S')

# 2. 구글 시트 연결 (관리자 대시보드용)
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

# --- [기능: 로그 기록 함수] ---
def record_log(code, name):
    if conn:
        try:
            # image_3ea728.png 시트 구조에 맞게 기록
            existing_data = conn.read(worksheet="Sheet1", ttl=0)
            new_entry = pd.DataFrame([{"날짜": current_time_str, "종목코드": code, "종목명": name}])
            updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
            conn.update(worksheet="Sheet1", data=updated_df)
        except: pass

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 7거래일 AI 패턴 분석기")
    st.write("최근 2년 데이터를 학습하여 주말을 제외한 향향 7거래일을 정밀 예측합니다.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리:", value="005930")
    with col2:
        st.write("") 
        run_button = st.button("영업일 기준 정밀 분석 시작", use_container_width=True)

    if run_button:
        try:
            status = st.empty()
            status.info("데이터를 수집하고 분석 중입니다...")

            # [1] 데이터 수집 (네이버 금융 루트 우선)
            df = fdr.DataReader(stock_code, KST - timedelta(days=730))
            if df.empty:
                st.error("데이터 수집 실패! 코드를 확인하세요.")
                st.stop()
            
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # 종목명 확보 및 로그 기록
            try:
                stocks = fdr.StockListing('KRX')
                stock_name = stocks[stocks['Code'] == stock_code]['Name'].values[0]
            except: stock_name = f"종목({stock_code})"
            record_log(stock_code, stock_name)

            # [2] 구글 트렌드 (오류 방어 로직 강화)
            df['구글트렌드'] = 0
            try:
                pytrends = TrendReq(hl='ko', tz=360)
                pytrends.build_payload([stock_name], timeframe='today 2-y', geo='KR')
                trends = pytrends.interest_over_time()
                if not trends.empty:
                    df['구글트렌드'] = trends[stock_name].reindex(df.index, method='ffill').fillna(0)
            except:
                st.warning("시장 심리 지표를 일시적으로 불러올 수 없어 주가 데이터로 분석합니다.")

            # [3] AI 변수 생성 및 Ridge 학습
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target'] = df['종가'].pct_change().shift(-1)
            
            df_train = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', '구글트렌드']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_train[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_train['target'])

            # [4] ★ 요청하신 변수 기여도 막대그래프 ★
            st.subheader("AI 분석 결과: 2개년 데이터 변수별 기여도")
            raw_importance = np.abs(model.coef_)
            total_raw = np.sum(raw_importance)
            ai_weights = (raw_importance / total_raw) * 100.0
            weight_data = pd.DataFrame({'변수': features, '비중(%)': list(ai_weights)})
            st.bar_chart(weight_data.set_index('변수'), color='#00CCFF')

            # [5] 미래 7거래일 예측 로직 (영업일 기준)
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5: # 월~금 영업일만
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    pred = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [6] 결과 시각화 (최근 1개월 집중)
            st.subheader(f"📊 {stock_name} 최근 흐름 및 7거래일 예측")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 예측(7일)", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            status.success("KST 기준 7거래일 분석이 완료되었습니다!")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링")
    pw = st.text_input("비밀번호", type="password")
    
    # Secrets의 admin_password와 대조
    if pw == st.secrets.get("admin_password", "0000"):
        st.success("인증 성공")
        if conn:
            try:
                data = conn.read(worksheet="Sheet1", ttl=0)
                st.subheader(f"🔥 인기 종목 TOP 5 (누적 {len(data)}건)")
                st.bar_chart(data['종목명'].value_counts().head(5), color='#FF4B4B')
                st.write("---")
                st.subheader("📝 최근 검색 로그 (KST 기준)")
                st.dataframe(data.iloc[::-1], use_container_width=True)
            except:
                st.warning("데이터베이스 연결 대기 중... requirements.txt 설정을 확인하세요.")
        else:
            st.error("구글 시트 연결 라이브러리가 설치되지 않았습니다.")
    else:
        st.info("비밀번호를 입력해 주세요.")
