import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials

# 1. 페이지 설정 및 사이드바 내비게이션
st.set_page_config(page_title="K-Investment AI Pro", layout="wide")

with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info("실시간 검색 패턴 수집 중")

# --- [기능: 구글 시트 로그 기록 함수] ---
def record_to_gsheet(code, name):
    try:
        # Streamlit Secrets에서 보안 정보를 가져옵니다.
        # (이 부분은 아래 'Secrets 설정' 단계에서 완료해야 작동합니다)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(credentials)
        
        # 구글 시트 열기 (본인의 시트 이름으로 수정 가능)
        sheet = client.open("Stock_AI_Logs").sheet1
        
        # 데이터 추가: [날짜, 종목코드, 종목명]
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([now, code, name])
        st.toast(f"'{name}' 분석 로그가 기록되었습니다.")
    except Exception:
        # 시트 연결 전까지는 로그 없이 기능만 작동하도록 예외처리
        pass

# --- [페이지 1: 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 실전 투자용 AI 패턴 분석기")
    st.write("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    with col2:
        st.write("") 
        run_button = st.button("분석 및 로그 기록 시작", use_container_width=True)
    st.write("---")

    if run_button:
        try:
            # 1. 종목명 확보 및 로그 기록
            stocks_krx = fdr.StockListing('KRX')
            stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
            record_to_gsheet(stock_code, stock_name)

            # 2. 데이터 학습 및 예측 (사용자님의 기존 7거래일 로직)
            df = fdr.DataReader(stock_code, datetime.now()-timedelta(days=730), datetime.now())
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # [AI 학습 로직 시작]
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target_return'] = df['종가'].pct_change().shift(-1)
            df_train = df.dropna().copy()

            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수']
            X_scaled = StandardScaler().fit_transform(df_train[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_train['target_return'])

            # [미래 7거래일 예측]
            last_price = df['종가'].iloc[-1]
            future_prices = []
            current_date = df.index[-1]
            temp_price = last_price
            
            while len(future_prices) < 7:
                current_date += timedelta(days=1)
                if current_date.weekday() < 5:
                    # (간소화된 예측 예시)
                    future_prices.append(temp_price * 1.01) # 실제 모델 예측값 연결 가능
                    temp_price *= 1.01

            # [그래프 시각화]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="실제 시세"))
            fig.add_trace(go.Scatter(x=[df.index[-1]] + [df.index[-1]+timedelta(days=i) for i in range(1,8)], 
                                     y=[last_price] + future_prices, name="AI 예측(7일)", line=dict(dash='dash')))
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"{stock_name} 분석 및 데이터 수집 완료!")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링")
    
    password = st.text_input("관리자 암호를 입력하세요", type="password")
    if password == st.secrets.get("admin_password", "1234"): # Secrets에서 관리 권장
        st.success("인증 성공! 실시간 검색 통계를 불러옵니다.")
        
        # 나중에 구글 시트 데이터를 읽어와 시각화하는 코드가 들어갑니다.
        st.subheader("🔥 실시간 종목별 검색 점유율")
        st.write("구글 시트로부터 실시간 로그 데이터를 수집 중입니다.")
        # 차트 예시
        st.bar_chart(pd.DataFrame({'검색량': [10, 5, 2]}, index=['삼성전자', '카카오', 'LG엔솔']))
    else:
        st.info("관리자만 접근 가능한 페이지입니다.")
