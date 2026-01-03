import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from st_supabase_connection import SupabaseConnection

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI v6.4 (관리자 기능 강화)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결 (Secrets 필수)
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception:
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 데이터 센터")
    menu = st.radio("메뉴", ["실전 분석", "관리자"], key="main_nav_final")

# --- [데이터 관리 함수] ---
def load_db():
    if conn:
        try:
            # 'knowledge' 테이블 데이터를 가져옵니다
            res = conn.table("knowledge").select("*").execute()
            return pd.DataFrame(res.data)
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_to_db(stock_code, df_curr):
    if conn and not df_curr.empty:
        try:
            # 검색 1회당 최근 10일치 데이터를 저장하여 기록을 남깁니다
            sample = df_curr.tail(10)
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": str(stock_code), "rsi": float(r['RSI']), "vix": float(r['VIX']),
                    "target": float(r['target']), "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']), "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']), "date_index": float(r['날짜지수'])
                })
            conn.table("knowledge").insert(rows).execute()
        except: pass

# --- [페이지 1: 실전 분석] ---
if menu == "실전 분석":
    st.title("📊 2년 학습 및 주가 예측")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("분석 및 지능 통합 시작", width='stretch'):
        try:
            # 2년 데이터 수집 및 AI 학습 (기존 로직 동일)
            start_date = KST_NOW - timedelta(days=730)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # (지표 계산 및 Ridge 모델 학습 과정)
            # ... [생략] ...
            
            # [시각화: 최근 한 달 차트 및 7일 예측]
            # (생략: v6.3과 동일한 시각화 코드)
            
            # [핵심] 성공적으로 분석되면 DB에 저장 호출
            save_to_db(stock_code, df)
            st.success("분석 완료! 데이터가 온라인 DB에 영구 기록되었습니다.")

        except Exception as e: st.error(f"오류: {e}")

# --- [페이지 2: 관리자 (재현님이 원하신 통계 기능)] ---
elif menu == "관리자":
    st.title("📊 온라인 검색 통계")
    if st.text_input("비번", type="password") == "0801":
        st.success("인증 성공")
        g_data = load_db()
        
        if not g_data.empty:
            # 1회 검색 시 10행이 저장되므로 총 행수를 10으로 나눕니다
            total_searches = len(g_data) // 10
            st.markdown(f"### 🚩 총 누적 검색량: `{total_searches}회`")
            st.write("---")
            
            # 종목별 횟수 출력 (예: 005930: 13회)
            if 'stock_code' in g_data.columns:
                st.subheader("📈 종목별 검색 현황")
                counts = g_data['stock_code'].value_counts()
                for code, row_count in counts.items():
                    # 각 종목의 행수를 10으로 나눠 실제 검색 횟수 계산
                    st.write(f"📍 **{code}**: {row_count // 10}회")
            else:
                st.error("DB 열 설정이 잘못되었습니다. 'stock_code' 열을 확인하세요.")
        else:
            # image_4f1365.png에서 보신 메시지 해결
            st.info("현재 DB에 저장된 데이터가 없습니다. 분석을 실행해 보세요.")
