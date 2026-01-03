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

# 2. 구글 시트 연결 (Secrets 설정 기반)
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception:
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info("사용자의 검색 패턴을 학습 중입니다.")

# --- [기능: 로그 기록 함수] ---
def record_search(code, name):
    if conn:
        try:
            # 기존 시트 데이터 읽기 (ttl=0으로 실시간성 확보)
            existing_data = conn.read(worksheet="Sheet1", ttl=0)
            # 새 로그 데이터 생성
            new_log = pd.DataFrame([{
                "날짜": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "종목코드": code,
                "종목명": name
            }])
            # 데이터 합치기 및 시트 업데이트
            updated_df = pd.concat([existing_data, new_log], ignore_index=True)
            conn.update(worksheet="Sheet1", data=updated_df)
            st.toast(f"'{name}' 분석 로그가 서버에 기록되었습니다.")
        except Exception:
            pass # 로그 기록 실패 시에도 사용자 분석은 계속 진행

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 실전 투자용 AI 패턴 분석기")
    st.write("최근 2년 데이터를 학습하여 향후 7거래일(영업일 기준)의 흐름을 예측합니다.")
    
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
            status = st.empty()
            status.info("최근 2년 데이터를 수집하고 분석 중입니다...")

            # [1] 주가 데이터 수집 (최근 2년)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 2) 
            df = fdr.DataReader(stock_code, start_date, end_date)

            if df.empty:
                st.error("데이터 수집 실패! 종목 코드를 확인하세요.")
            else:
                df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
                
                # 종목명 가져오기 및 로그 기록 호출
                stocks_krx = fdr.StockListing('KRX')
                stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
                record_search(stock_code, stock_name)

                # [2] 구글 트렌드 (오류 방어 로직 강화)
                df['구글트렌드'] = 0
                try:
                    pytrends = TrendReq(hl='ko', tz=360)
                    pytrends.build_payload([stock_name], cat=0, timeframe='today 2-y', geo='KR')
                    trends_df = pytrends.interest_over_time()
                    if not trends_df.empty:
                        trends_df = trends_df[[stock_name]].resample('D').interpolate(method='linear')
                        df = df.join(trends_df).fillna(method='ffill').fillna(0)
                        df['구글트렌드'] = df[stock_name]
                except Exception:
                    # 'Expecting value' 등 통신 오류 발생 시 경고만 띄우고 주가 데이터로만 분석 진행
                    st.warning("구글 트렌드 데이터를 일시적으로 불러올 수 없어 기술적 지표 중심으로 분석합니다.")

                # [3] 변수 생성 및 AI 학습
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

                # [4] 미래 7거래일 예측 (주말 제외 영업일 기준)
                last_real_price = df['종가'].iloc[-1]
                last_date = df.index[-1]
                future_prices, future_dates = [], []
                current_price = last_real_price
                last_features = df[features].iloc[-1:].copy()

                check_date = last_date
                while len(future_prices) < 7:
                    check_date += timedelta(days=1)
                    if check_date.weekday() < 5: # 월~금만 포함
                        last_features['날짜지수'] += 1
                        last_features['요일'] = check_date.weekday()
                        pred_return = model.predict(scaler.transform(last_features))[0]
                        current_price *= (1 + pred_return)
                        future_prices.append(current_price)
                        future_dates.append(check_date)

                # [5] 결과 시각화
                st.subheader(f"📊 {stock_name} AI 분석 및 예측 결과")
                fig = go.Figure()
                display_df = df.iloc[-30:] # 최근 한 달간의 흐름
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df['종가'], name="실제 시세", line=dict(color='#00CCFF', width=3)))
                fig.add_trace(go.Scatter(x=[last_date] + future_dates, y=[last_real_price] + future_prices, name="AI 예측(7거래일)", line=dict(color='#FF3300', dash='dash', width=4)))
                fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                status.success(f"{stock_name} 분석 완료!")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링")
    
    password = st.text_input("관리자 비밀번호를 입력하세요", type="password")
    
    # Secrets에 설정한 비밀번호와 대조 (기본값 0000)
    if password == st.secrets.get("admin_password", "0000"):
        st.success("인증 성공! 실시간 데이터를 불러옵니다.")
        
        if conn:
            try:
                logs = conn.read(worksheet="Sheet1", ttl=0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🔥 실시간 인기 종목 TOP 5")
                    if not logs.empty:
                        top_stocks = logs['종목명'].value_counts().head(5)
                        st.bar_chart(top_stocks, color='#FF4B4B')
                
                with col2:
                    st.subheader("📈 누적 이용 통계")
                    st.metric("총 검색 횟수", f"{len(logs)}회")
                    st.metric("오늘의 분석", f"{len(logs[logs['날짜'].str.contains(datetime.now().strftime('%Y-%m-%d'))])}회")

                st.write("---")
                st.subheader("📝 최근 검색 로그 (최근 20건)")
                st.dataframe(logs.sort_index(ascending=False).head(20), use_container_width=True)
            except Exception:
                st.warning("데이터베이스 연결 대기 중... 시트의 공유 설정(편집자)을 확인하세요.")
        else:
            st.error("데이터베이스 연결 설정(Secrets)이 필요합니다.")
    else:
        st.info("관리자만 접근 가능한 페이지입니다.")
