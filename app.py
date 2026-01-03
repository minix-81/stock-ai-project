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

# 1. 페이지 설정
st.set_page_config(page_title="K-Investment AI Pro", layout="wide", page_icon="📈")

# 2. 구글 시트 연결 (관리자 대시보드용)
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    conn = None

# 3. 사이드바 메뉴 (관리자 페이지가 여기서 전환됩니다)
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info(f"현재 시각: {datetime.now().strftime('%H:%M:%S')}")

# --- [기능: 로그 기록 함수] ---
def record_log(code, name):
    if conn:
        try:
            # image_3ea728.png 구조에 맞게 기록
            existing_data = conn.read(worksheet="Sheet1", ttl=0)
            new_entry = pd.DataFrame([{"날짜": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                     "종목코드": code, "종목명": name}])
            updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
            conn.update(worksheet="Sheet1", data=updated_df)
        except: pass

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 7거래일 AI 패턴 분석기")
    st.write("사용자님의 Ridge 회귀 모델이 5가지 핵심 변수를 학습하여 미래를 예측합니다.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리:", value="005930")
    with col2:
        st.write("") 
        run_button = st.button("AI 분석 시작", use_container_width=True)

    if run_button:
        try:
            # [1] 데이터 수집 및 로그
            stocks_krx = fdr.StockListing('KRX')
            stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
            record_log(stock_code, stock_name)
            
            df = fdr.DataReader(stock_code, datetime.now()-timedelta(days=730), datetime.now())
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})

            # [2] 구글 트렌드 (오류 방어)
            df['구글트렌드'] = 0
            try:
                pytrends = TrendReq(hl='ko', tz=360)
                pytrends.build_payload([stock_name], timeframe='today 2-y', geo='KR')
                trends_df = pytrends.interest_over_time()
                if not trends_df.empty:
                    df['구글트렌드'] = trends_df[stock_name].reindex(df.index, method='ffill').fillna(0)
            except:
                st.warning("구글 트렌드 데이터를 일시적으로 불러올 수 없습니다.")

            # [3] AI 변수 생성 (사용자님의 핵심 로직)
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

            # [4] ★ 핵심: 7거래일 예측 루프 (복구 완료) ★
            last_price = df['종가'].iloc[-1]
            last_date = df.index[-1]
            future_prices, future_dates = [], []
            temp_price = last_price
            last_feat = df[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5: # 주말 제외 영업일 기준
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    # 예측 수행
                    pred_ret = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred_ret)
                    future_prices.append(temp_price)
                    future_dates.append(curr_d)

            # [5] 시각화
            fig = go.Figure()
            # 실제 데이터 (최근 30일)
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="최근 시세", line=dict(color='#00CCFF')))
            # 예측 데이터 (7일)
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 예측(7일)", line=dict(dash='dash', color='red')))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"{stock_name} 7거래일 예측 완료!")

        except Exception as e:
            st.error(f"오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링")
    pw = st.text_input("관리자 비밀번호를 입력하세요", type="password")
    
    # 비밀번호 확인 로직 (Secrets에 설정한 비밀번호 사용)
    if pw == st.secrets.get("admin_password", "0000"):
        st.success("인증 성공")
        if conn:
            try:
                # 구글 시트에서 데이터 읽기
                data = conn.read(worksheet="Sheet1", ttl=0)
                st.subheader("🔥 실시간인기 종목 TOP 5")
                st.bar_chart(data['종목명'].value_counts().head(5))
                st.write("---")
                st.subheader("📝 상세 검색 로그")
                st.dataframe(data.iloc[::-1], use_container_width=True)
            except Exception as e:
                st.warning(f"데이터 로드 중: {e}")
        else:
            st.error("구글 시트 연결 설정(Secrets)이 필요합니다.")
    else:
        st.info("관리자 비밀번호를 입력해 주세요.")
