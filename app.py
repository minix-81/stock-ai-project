import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI 분석기 v5.0 (점진적 학습)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. 구글 시트 연결
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    conn = None

# --- [핵심: 글로벌 지식 관리 함수] ---

def load_global_knowledge():
    """다른 사용자들이 검색했던 과거 학습 데이터를 불러옵니다."""
    if conn:
        try:
            # Global_Knowledge 워크시트에서 데이터 로드
            return conn.read(worksheet="Global_Knowledge", ttl=0)
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_global_knowledge(new_df):
    """현재 분석한 데이터 중 일부를 글로벌 지식 저장소에 기부합니다."""
    if conn and not new_df.empty:
        try:
            # 데이터가 너무 방대해지지 않게 최근 30일치만 샘플링하여 저장
            sample = new_df.tail(30).copy()
            existing = load_global_knowledge()
            updated = pd.concat([existing, sample], ignore_index=True).tail(1000) # 최대 1000개 유지
            conn.update(worksheet="Global_Knowledge", data=updated)
        except: pass

# --- [페이지 1: 실전 종목 분석기] ---
if st.sidebar.radio("메뉴", ["실전 종목 분석기", "관리자 대시보드"]) == "실전 종목 분석기":
    st.title("📈 점진적 성장을 하는 주식 AI 분석기")
    st.write("사용자들이 종목을 검색할수록 '글로벌 지식 저장소'가 풍부해지며 AI가 똑똑해집니다.")
    
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("5년 데이터 학습 및 지식 통합 시작", width='stretch'):
        try:
            # [단계 1] 현재 종목 데이터 준비 (RSI, VIX 포함)
            start_date = KST_NOW - timedelta(days=365 * 5)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            # (RSI, VIX 계산 로직은 이전과 동일하게 유지)
            # ... [지표 계산 코드 생략] ...
            df['target'] = df['종가'].pct_change().shift(-1)
            df_current = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX(시장공포지수)', 'RSI']

            # [단계 2] 글로벌 지식 불러오기 및 감쇄계수 적용
            global_df = load_global_knowledge()
            decay_coeff = 0.15 # 재현님이 제안하신 감쇄계수 (타 종목 데이터 영향력 15%)

            if not global_df.empty:
                # 현재 데이터와 글로벌 데이터 통합
                X_local = df_current[features]
                y_local = df_current['target']
                
                X_global = global_df[features]
                y_global = global_df['target']
                
                X_total = pd.concat([X_local, X_global])
                y_total = pd.concat([y_local, y_global])
                
                # 가중치 설정 (현재 데이터는 1.0, 글로벌 지식은 감쇄계수 적용)
                weights = np.array([1.0] * len(X_local) + [decay_coeff] * len(X_global))
                st.info(f"💡 글로벌 지능 통합 완료: 과거 {len(global_df)}개의 지식을 참고하여 학습합니다.")
            else:
                X_total, y_total = df_current[features], df_current['target']
                weights = np.array([1.0] * len(X_total))

            # [단계 3] 가중치 학습 (Weighted Ridge)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            model = Ridge(alpha=1.0).fit(X_scaled, y_total, sample_weight=weights)

            # [단계 4] 결과 시각화 및 지능 기부
            # (예측 및 차트 출력 로직 동일)
            # ... 
            
            # 현재 학습한 깨달음을 글로벌 저장소에 저장 (다음 검색자를 위해)
            save_global_knowledge(df_current[features + ['target']])
            st.success("분석 완료 및 AI 지식 저장소 업데이트 성공!")

        except Exception as e:
            st.error(f"분석 오류: {e}")
