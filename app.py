import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pytrends.request import TrendReq
import gspread
from google.oauth2.service_account import Credentials

# 1. 페이지 및 사이드바 설정
st.set_page_config(page_title="실전 투자용 AI 패턴 분석기", layout="wide", page_icon="📈")

with st.sidebar:
    st.title("🚀 메뉴")
    menu = st.radio("이동할 페이지", ["종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info("데이터 기반 투자 보조 도구")

# 2. 구글 시트 로그 기록 함수 (gspread 사용)
def save_log_to_gsheet(code, name):
    try:
        # Secrets에 저장된 JSON 인증 정보를 사용합니다.
        creds_info = st.secrets["gcp_service_account"]
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(credentials)
        
        # 시트 열기 (이름: Stock_AI_Logs)
        sheet = client.open("Stock_AI_Logs").sheet1
        
        # 데이터 추가 [날짜, 종목코드, 종목명] 순서
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([now, code, name])
        st.toast(f"'{name}' 분석 내역이 기록되었습니다.")
    except Exception:
        pass # 기록 실패 시에도 분석은 계속 진행

# --- [페이지 1: 종목 분석기] ---
if menu == "종목 분석기":
    st.title("📈 실전 투자용 AI 패턴 분석기")
    st.write("최근 2년 데이터를 학습하여 향후 7거래일(영업일 기준)의 시세를 예측합니다.")
    
    st.write("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    with col2:
        st.write("") 
        run_button = st.button("분석 시작", use_container_width=True)
    st.write("---")

    if run_button:
        try:
            status = st.empty()
            status.info("데이터 수집 및 AI 학습 중...")

            # [1] 주가 데이터 수집
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 2) 
            df = fdr.DataReader(stock_code, start_date, end_date)

            if df.empty:
                st.error("데이터를 수집할 수 없습니다. 종목 코드를 확인하세요.")
            else:
                df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
                
                # 종목명 확보 및 로그 기록
                stocks_krx = fdr.StockListing('KRX')
                stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
                save_log_to_gsheet(stock_code, stock_name)

                # [2] 구글 트렌드 (Expecting value 오류 방어)
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
                    st.warning("구글 트렌드 API 제한으로 인해 주가 지표 위주로 분석합니다.")

                # [3] AI 학습 (Ridge Regression)
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
                model = Ridge(alpha=1.0)
                model.fit(X_scaled, y)

                # [4] 미래 7거래일 예측
                last_real_price = df['종가'].iloc[-1]
                last_date = df.index[-1]
                future_prices, future_dates = [], []
                current_price = last_real_price
                last_features = df[features].iloc[-1:].copy()

                check_date = last_date
                while len(future_prices) < 7:
                    check_date += timedelta(days=1)
                    if check_date.weekday() < 5: # 영업일만 포함
                        last_features['날짜지수'] += 1
                        last_features['요일'] = check_date.weekday()
                        pred_return = model.predict(scaler.transform(last_features))[0]
                        current_price *= (1 + pred_return)
                        future_prices.append(current_price)
                        future_dates.append(check_date)

                # [5] 시각화
                st.subheader(f"📊 {stock_name} AI 분석 및 예측")
                fig = go.Figure()
                display_df = df.iloc[-30:] 
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df['종가'], name="실제 시세", line=dict(color='#00CCFF', width=3)))
                fig.add_trace(go.Scatter(x=[last_date] + future_dates, y=[last_real_price] + future_prices, name="AI 예측(7일)", line=dict(color='#FF3300', dash='dash', width=4)))
                fig.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                status.success("분석 완료!")

        except Exception as e:
            st.error(f"오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 관리자 전용 데이터 대시보드")
    password = st.text_input("관리자 비밀번호를 입력하세요", type="password")
    
    if password == st.secrets.get("admin_password", "1234"):
        st.success("인증 성공")
        try:
            # 구글 시트에서 전체 로그 읽어오기
            creds_info = st.secrets["gcp_service_account"]
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_info(creds_info, scopes=scope)
            client = gspread.authorize(credentials)
            sheet = client.open("Stock_AI_Logs").sheet1
            data = pd.DataFrame(sheet.get_all_records())
            
            if not data.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🔥 실시간 검색 순위")
                    st.bar_chart(data['종목명'].value_counts().head(5), color='#FF4B4B')
                with col2:
                    st.subheader("📈 누적 이용 통계")
                    st.metric("총 분석 횟수", f"{len(data)}회")
                
                st.write("---")
                st.subheader("📝 상세 로그 (최신순)")
                st.dataframe(data.iloc[::-1], use_container_width=True)
            else:
                st.info("기록된 데이터가 없습니다.")
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
    else:
        st.info("비밀번호를 입력해 주세요.")
