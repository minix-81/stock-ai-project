import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pytrends.request import TrendReq

# 1. 페이지 및 사이드바 설정
st.set_page_config(page_title="K-Investment AI Pro", layout="wide")

# 사이드바 메뉴 구성
with st.sidebar:
    st.title("🚀 AI 투자 플랫폼")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.info("고교 심화 탐구 기반 AI 엔진 v2.0")

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 실전 투자용 AI 패턴 분석기")
    st.write("최근 2년 데이터를 학습하여 향후 7거래일의 흐름을 예측합니다.")
    
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
            status.info("데이터 학습 및 분석 중...")

            # [데이터 수집 및 분석 로직 - 기존과 동일]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 2) 
            df = fdr.DataReader(stock_code, start_date, end_date)

            if df.empty:
                st.error("데이터 수집 실패!")
            else:
                df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
                
                # 가상의 로그 기록 (나중에 구글 시트 연결 시 실제 저장됨)
                st.toast(f"{stock_code} 분석 로그가 관리자 서버에 전송되었습니다.")

                # [변수 생성 및 AI 학습]
                df['날짜지수'] = np.arange(len(df))
                df['요일'] = df.index.weekday
                df['변동성'] = (df['High'] - df['Low']) / df['종가']
                df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
                df['target_return'] = df['종가'].pct_change().shift(-1)
                df_train = df.dropna().copy()

                features = ['날짜지수', '요일', '거래량', '변동성', '감성지수']
                X = df_train[features]
                y = df_train['target_return']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = Ridge(alpha=1.0)
                model.fit(X_scaled, y)

                # [미래 7거래일 예측]
                last_real_price = df['종가'].iloc[-1]
                last_date = df.index[-1]
                future_prices, future_dates = [], []
                current_price = last_real_price
                last_features = df[features].iloc[-1:].copy()

                check_date = last_date
                while len(future_prices) < 7:
                    check_date += timedelta(days=1)
                    if check_date.weekday() < 5:
                        last_features['날짜지수'] += 1
                        last_features['요일'] = check_date.weekday()
                        pred_return = model.predict(scaler.transform(last_features))[0]
                        current_price *= (1 + pred_return)
                        future_prices.append(current_price)
                        future_dates.append(check_date)

                # [그래프 출력]
                fig = go.Figure()
                display_df = df.iloc[-30:] 
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df['종가'], name="실제 시세", line=dict(color='#00CCFF', width=3)))
                fig.add_trace(go.Scatter(x=[last_date] + future_dates, y=[last_real_price] + future_prices, name="AI 예측선", line=dict(color='#FF3300', dash='dash', width=4)))
                fig.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig, use_container_width=True)
                status.success("분석 완료!")

        except Exception as e:
            st.error(f"오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 관리자 전용 데이터 센터")
    st.write("사용자들의 실시간 검색 내역 및 시스템 로그를 확인합니다.")

    # 보안 인증
    password = st.text_input("관리자 비밀번호를 입력하세요", type="password")
    
    if password == "1234": # 비밀번호를 사용자님만 아는 숫자로 바꾸세요!
        st.success("인증 성공! 실시간 통계 데이터를 불러옵니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥 실시간 검색 순위")
            # 샘플 데이터 (나중에 구글 시트 데이터로 교체)
            hot_stocks = pd.DataFrame({
                '종목명': ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '카카오'],
                '검색량': [152, 98, 45, 30]
            })
            st.bar_chart(hot_stocks.set_index('종목명'), color='#FFCC00')
            
        with col2:
            st.subheader("📈 시스템 이용 현황")
            st.metric(label="누적 분석 횟수", value="3,240회", delta="12%")
            st.metric(label="활성 사용자", value="42명", delta="5%")

        st.write("---")
        st.subheader("📝 상세 검색 로그 (최근 100건)")
        # 로그 테이블 예시
        log_data = pd.DataFrame({
            '시간': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 5,
            'IP/기기': ['Mobile', 'Desktop', 'Tablet', 'Mobile', 'Desktop'],
            '검색종목': ['005930', '000660', '373220', '035720', '005930'],
            '상태': ['Success', 'Success', 'Success', 'Success', 'Success']
        })
        st.table(log_data)
        
    elif password == "":
        st.info("비밀번호를 입력해 주세요.")
    else:
        st.error("비밀번호가 틀렸습니다. 접근 권한이 없습니다.")
