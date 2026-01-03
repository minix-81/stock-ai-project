import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from supabase import create_client # 수동 연결용 라이브러리

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI v8.1 (강제 연결)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 강제 연결 로직
@st.cache_resource
def get_manual_conn():
    try:
        # Secrets에서 주소와 키를 직접 추출합니다.
        url = st.secrets["connections"]["supabase"]["url"]
        key = st.secrets["connections"]["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"📡 강제 연결 시도 중 에러: {e}")
        return None

client = get_manual_conn()

# --- [사이드바 진단] ---
with st.sidebar:
    st.title("🚀 데이터 센터")
    if client:
        st.success("✅ 강제 연결 성공!")
    else:
        st.error("❌ 연결 정보를 읽을 수 없습니다.")
    menu = st.radio("메뉴", ["실전 분석", "관리자"], key="nav_v81")

# --- [데이터 관리 함수: 타입 고정 적용] ---
def save_to_db(stock_code, df_curr):
    if client and not df_curr.empty:
        try:
            sample = df_curr.tail(10)
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": str(stock_code), "rsi": float(r['RSI']), "vix": float(r['VIX']),
                    "target": float(r['target']), "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']), "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']), "date_index": float(r['날짜지수'])
                })
            client.table("knowledge").insert(rows).execute()
            return True
        except: return False
    return False

# --- [페이지 1: 실전 분석 (Ridge Regression 기반)] ---
# $$J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m \theta_j^2$$
if menu == "실전 분석":
    st.title("📊 2년 학습 및 7일 예측")
    stock_code = st.text_input("종목 번호:", value="005930")
    
    if st.button("분석 및 지능 공유 시작"):
        try:
            # (데이터 수집 및 시각화 로직은 v7.8과 동일하게 유지됩니다)
            # ... 생략 ...
            if save_to_db(stock_code, df_curr):
                st.success("✅ 지능 공유 완료! 수파베이스에 기록되었습니다.")
        except Exception as e: st.error(f"오류: {e}")
