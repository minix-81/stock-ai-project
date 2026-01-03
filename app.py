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

# 1. 페이지 및 구글 시트 연결 설정
st.set_page_config(page_title="K-Investment AI Pro", layout="wide", page_icon="📈")

# 구글 시트 연결 시도 (관리자 대시보드용)
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    conn = None

# 2. 사이드바 메뉴 (관리자 페이지 전환 핵심)
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info(f"현재 시각: {datetime.now().strftime('%H:%M:%S')}")

# --- [기능: 로그 기록 함수] ---
def record_log(code, name):
    if conn:
        try:
            # image_3ea728.png 시트 구조에 맞게 기록
            existing_data = conn.read(worksheet="Sheet1", ttl=0)
            new_entry = pd.DataFrame([{"날짜": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                     "종목코드": code, "종목명": name}])
            updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
            conn.update(worksheet="Sheet1", data=updated_df)
        except: pass

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 7거래일 AI 패턴 분석기")
    st.write("네이버 금융 데이터를 메인으로 Ridge 회귀 모델이 미래 시세를 예측합니다.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    with col2:
        st.write("") 
        run_button = st.button("AI 분석 시작", use_container_width=True)

    if run_button:
        try:
            # [단계 1] 종목 정보 확보 (실패 시 코드번호 사용)
            try:
                # KRX 정보 수집 시도
                stocks = fdr.StockListing('KRX')
                stock_name = stocks[stocks['Code'] == stock_code]['Name'].values[0]
            except:
                stock_name = f"종목({stock_code})"
            
            record_log(stock_code, stock_name)

            # [단계 2] ★ 주가 데이터 수집 (네이버 금융 루트 고정) ★
            # fdr.DataReader는 KRX 차단 시에도 네이버 소스를 통해 데이터를 가져올 수 있습니다.
            df = fdr.DataReader(stock_code, datetime.now()-timedelta(days=730))
            
            if df.empty:
                st.error("데이터를 수집할 수 없습니다. 종목 코드를 다시 확인해 주세요.")
                st.stop()
            
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})

            # [단계 3] 구글 트렌드 (오류 격리 방어)
            df['구글트렌드'] = 0
            try:
                pytrends = TrendReq(hl='ko', timeout=(5, 10))
                pytrends.build_payload([stock_name], timeframe='today 2-y', geo='KR')
                trends = pytrends.interest_over_time()
                if not trends.empty:
                    df['구글트렌드'] = trends[stock_name].reindex(df.index, method='ffill').fillna(0)
            except:
                st.warning("시장 심리 지표를 불러오지 못해 주가 데이터 위주로 분석합니다.")

            # [단계 4] AI 학습 (Ridge Regression)
            # 변수: 날짜지수, 요일, 거래량, 변동성, 감성지수
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

            # [단계 5] ★ 핵심: 7거래일 예측 루프 복구 ★
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5: # 영업일 기준
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    pred_ret = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred_ret)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [단계 6] 결과 시각화
            fig = go.Figure()
            # 실제 데이터 (최근 30일)
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="최근 시세", line=dict(color='#00CCFF', width=3)))
            # 예측 데이터 (빨간색 점선)
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 예측(7일)", line=dict(dash='dash', color='red', width=4)))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"{stock_name} 7거래일 분석 완료!")

        except Exception as e:
            st.error(f"분석 도중 오류가 발생했습니다: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링")
    pw = st.text_input("비밀번호", type="password")
    
    if pw == st.secrets.get("admin_password", "0000"):
        st.success("인증 성공")
        if conn:
            try:
                data = conn.read(worksheet="Sheet1", ttl=0)
                st.subheader(f"🔥 인기 종목 TOP 5 (누적 {len(data)}건)")
                st.bar_chart(data['종목명'].value_counts().head(5), color='#FF4B4B')
                st.write("---")
                st.subheader("📝 최근 검색 로그")
                st.dataframe(data.iloc[::-1], use_container_width=True)
            except:
                st.warning("데이터베이스 연결 대기 중... 시트 공유 설정을 확인하세요.")
    else:
        st.info("관리자 비밀번호를 입력해 주세요.")
