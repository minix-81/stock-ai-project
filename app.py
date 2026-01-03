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
conn = st.connection("gsheets", type=GSheetsConnection)

# 2. 사이드바 내비게이션
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info("데이터 기반 분석 엔진 가동 중")

# --- [기능: 로그 기록 함수] ---
def record_search_log(code, name):
    try:
        # 기존 시트 데이터 읽기 (ttl=0으로 실시간성 확보)
        existing_data = conn.read(worksheet="Sheet1", ttl=0)
        # 새 로그 생성
        new_log = pd.DataFrame([{
            "날짜": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "종목코드": code,
            "종목명": name
        }])
        # 데이터 합치기 및 시트 업데이트
        updated_df = pd.concat([existing_data, new_log], ignore_index=True)
        conn.update(worksheet="Sheet1", data=updated_df)
    except:
        pass # 시트 연결 전이라도 분석은 진행됨

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 실전 투자용 AI 패턴 분석기")
    st.write("최근 2년 데이터를 Ridge 회귀 모델로 학습하여 향후 7거래일을 예측합니다.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    with col2:
        st.write("") 
        run_button = st.button("AI 분석 및 로그 기록 시작", use_container_width=True)

    if run_button:
        try:
            # [1] 데이터 수집 및 종목명 확인
            stocks_krx = fdr.StockListing('KRX')
            stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
            
            # 실시간 로그 기록 실행
            record_search_log(stock_code, stock_name)
            st.toast(f"'{stock_name}' 검색 내역이 구글 시트에 기록되었습니다.")

            # 주가 데이터 (최근 2년)
            df = fdr.DataReader(stock_code, datetime.now()-timedelta(days=730), datetime.now())
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})

            # [2] 구글 트렌드 (오류 방어)
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
                st.warning("구글 트렌드 일시 차단으로 인해 기술 지표 위주로 분석합니다.")

            # [3] 변수 생성 (사용자님의 핵심 로직)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target_return'] = df['종가'].pct_change().shift(-1)
            df_train = df.dropna().copy()

            # [4] Ridge AI 학습
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', '구글트렌드']
            X = df_train[features]
            y = df_train['target_return']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)

            # [5] 가중치 분석 시각화
            st.subheader("💡 AI 변수 기여도 분석")
            importance = pd.DataFrame({'변수': features, '비중': np.abs(model.coef_)}).set_index('변수')
            st.bar_chart(importance)

            # [6] 미래 7거래일 예측 (영업일 기준)
            last_price = df['종가'].iloc[-1]
            last_date = df.index[-1]
            future_prices, future_dates = [], []
            current_price = last_price
            last_feats = df[features].iloc[-1:].copy()

            check_date = last_date
            while len(future_prices) < 7:
                check_date += timedelta(days=1)
                if check_date.weekday() < 5: # 주말 제외
                    last_feats['날짜지수'] += 1
                    last_feats['요일'] = check_date.weekday()
                    pred_ret = model.predict(scaler.transform(last_feats))[0]
                    current_price *= (1 + pred_ret)
                    future_prices.append(current_price)
                    future_dates.append(check_date)

            # [7] 결과 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="최근 시세", line=dict(color='#00CCFF', width=2)))
            fig.add_trace(go.Scatter(x=[last_date] + future_dates, y=[last_price] + future_prices, name="AI 예측(7일)", line=dict(color='#FF3300', dash='dash')))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 검색 데이터 통계")
    pw = st.text_input("관리자 비밀번호", type="password")
    if pw == st.secrets.get("admin_password", "0000"):
        try:
            data = conn.read(worksheet="Sheet1", ttl=0)
            st.subheader(f"🔥 실시간 인기 종목 (누적 {len(data)}건)")
            st.bar_chart(data['종목명'].value_counts().head(5))
            st.write("---")
            st.subheader("📝 상세 로그")
            st.dataframe(data.sort_index(ascending=False), use_container_width=True)
        except:
            st.info("데이터를 불러오는 중입니다...")
