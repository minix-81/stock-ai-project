import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# 1. 페이지 및 시간 설정
st.set_page_config(page_title="주식 AI 분석기 v5.3 (Lite)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. 내부 메모리(Session State) 초기화 (시트 대신 사용)
if 'global_knowledge' not in st.session_state:
    st.session_state['global_knowledge'] = pd.DataFrame() # 지능 저장소
if 'search_logs' not in st.session_state:
    st.session_state['search_logs'] = [] # 검색 기록

# 3. 사이드바
with st.sidebar:
    st.title("🚀 로컬 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.info("시트 연결 없이 앱 내부 메모리를 사용 중입니다.")

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📊 주식 AI 분석기 (세션 성장형)")
    st.write("사용자가 검색할수록 현재 세션의 **AI 지식**이 풍부해집니다.")
    
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    if st.button("데이터 분석 및 지능 통합 시작", width='stretch'):
        try:
            # [단계 1] 데이터 수집 및 지표 계산 (5년치)
            start_date = KST_NOW - timedelta(days=365 * 5)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # 지표 생성 (VIX, RSI)
            try:
                vix = fdr.DataReader('^VIX', start_date)
                df = df.join(vix[['Close']].rename(columns={'Close': 'VIX'})).ffill().fillna(20)
            except: df['VIX'] = 20
            
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = (100 - (100 / (1 + (gain / loss)))).fillna(50)
            
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            
            df_current = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']

            # [단계 2] 내부 지식 통합 (감쇄계수 0.15)
            global_df = st.session_state['global_knowledge']
            decay_coeff = 0.15

            if not global_df.empty:
                X_total = pd.concat([df_current[features], global_df[features]])
                y_total = pd.concat([df_current['target'], global_df['target']])
                weights = np.array([1.0] * len(df_current) + [decay_coeff] * len(global_df))
                st.info(f"💡 현재 세션에 축적된 {len(global_df)}개의 지식을 반영하여 학습했습니다.")
            else:
                X_total, y_total = df_current[features], df_current['target']
                weights = np.array([1.0] * len(X_total))

            # [단계 3] AI 학습 (Ridge Regression)
            # $$J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m \theta_j^2$$
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            model = Ridge(alpha=1.0).fit(X_scaled, y_total, sample_weight=weights)
            
            # 결과 차트
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['종가'].iloc[-60:], name="최근 시세"))
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, width='stretch')
            
            # [단계 4] 지능 기부 및 로그 기록 (메모리에 저장)
            # 지식 저장 (최대 1000개 행 유지)
            new_knowledge = pd.concat([global_df, df_current[features + ['target']].tail(30)]).tail(1000)
            st.session_state['global_knowledge'] = new_knowledge
            
            # 로그 저장
            st.session_state['search_logs'].append(stock_code)
            
            st.success("분석 완료! 현재 세션 지능이 업데이트되었습니다.")

        except Exception as e:
            st.error(f"분석 오류: {e}")

# --- [페이지 2: 관리자 대시보드 - 메모리 데이터 요약] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 세션 모니터링")
    pw = st.text_input("비밀번호", type="password")
    
    if pw == "0801": # 간단한 비밀번호 확인
        st.success("인증 성공 (로컬 모드)")
        
        logs = st.session_state['search_logs']
        if logs:
            # 총 검색량
            st.markdown(f"### 🚩 현재 세션 총 검색량: `{len(logs)}회`")
            st.write("---")
            
            # 종목별 횟수 요약
            st.subheader("📈 종목별 검색 현황")
            log_series = pd.Series(logs)
            counts = log_series.value_counts()
            for code, count in counts.items():
                st.write(f"📍 **{code}**: {count}회")
        else:
            st.info("아직 검색 데이터가 없습니다.")
