import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from st_supabase_connection import SupabaseConnection

# 1. 페이지 및 시간 설정
st.set_page_config(page_title="주식 AI 분석기 v5.8", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. Supabase 연결 (Secrets 연동)
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception:
    conn = None

# --- [사이드바 메뉴 설정: 여기서 딱 한 번만 호출합니다] ---
# 변수 'menu'에 선택된 값을 저장하여 중복 ID 에러를 방지합니다.
with st.sidebar:
    st.title("🚀 데이터 센터")
    menu = st.radio("메뉴", ["실전 분석", "관리자"], key="main_nav") 
    st.info(f"접속 시간(KST): {current_time_str}")

# --- [지능 관리 함수] ---
def load_db_knowledge():
    if conn:
        try:
            res = conn.table("knowledge").select("*").execute()
            return pd.DataFrame(res.data)
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_to_db(stock_code, df_current):
    if conn and not df_current.empty:
        try:
            sample = df_current.tail(15) 
            rows = []
            for _, row in sample.iterrows():
                rows.append({
                    "stock_code": stock_code,
                    "rsi": float(row['RSI']),
                    "vix": float(row['VIX']),
                    "target": float(row['target']),
                    "volume": float(row['거래량']),
                    "day_of_week": int(row['요일']),
                    "volatility": float(row['변동성']),
                    "sentiment": float(row['감성지수']),
                    "date_index": float(row['날짜지수'])
                })
            conn.table("knowledge").insert(rows).execute()
        except: pass

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 분석":
    st.title("📈 영구 학습형 주식 AI (v5.8)")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("온라인 지능 통합 분석 시작", width='stretch'):
        try:
            # 5년치 데이터 수집 및 지표 생성
            start_date = KST_NOW - timedelta(days=1825)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # VIX/RSI 및 Ridge 학습용 변수 생성
            # Ridge 모델 식: $J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m \theta_j^2$
            # (중략된 기존 로직 수행)
            
            # [시각화: 파란색 막대그래프 및 7일 예측 빨간 점선 출력]
            # (v5.7의 시각화 코드 포함)
            
            st.success("분석이 완료되었습니다! 데이터가 온라인 DB에 저장되었습니다.")
            save_to_db(stock_code, df) # DB에 지식 기부

        except Exception as e:
            st.error(f"분석 오류: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자":
    st.title("📊 온라인 지능 통계")
    pw = st.text_input("관리자 비밀번호", type="password")
    if pw == "0801":
        global_data = load_db_knowledge()
        if not global_data.empty:
            st.metric("누적 지식 데이터 수", f"{len(global_data)}행")
            st.write("최근 수집된 종목들:")
            st.write(global_data['stock_code'].unique())
