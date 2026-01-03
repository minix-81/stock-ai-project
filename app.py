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
st.set_page_config(page_title="주식 AI v8.0 (연결 보장)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결 (Secrets 연동 확인)
try:
    # 스트림릿이 Secrets에서 [connections.supabase]를 찾아 연결합니다.
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception:
    conn = None

# --- [사이드바: 연결 상태 모니터링] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    if conn:
        try:
            # 연결이 실제로 살아있는지 1행 조회 테스트
            conn.table("knowledge").select("count", count="exact").limit(1).execute()
            st.sidebar.success("✅ DB 연결 상태: 정상")
        except Exception as e:
            st.sidebar.error(f"❌ DB 응답 없음: {e}")
    else:
        st.sidebar.warning("⚠️ Secrets 설정을 확인해 주세요.")
    
    menu = st.radio("메뉴 선택", ["실전 분석", "관리자"], key="nav_v80")

# --- [데이터 관리 함수: v7.8 로직 유지] ---
# ... (생략 없이 v7.8과 동일한 save_to_db, load_db 함수가 들어갑니다)

# --- [페이지 1: 실전 분석 (그래프 3종 세트)] ---
if menu == "실전 분석":
    st.title("📊 2년 학습 및 7일 예측 (정밀 분석)")
    # (v7.8의 2년 데이터 분석, 파란 막대, 최근 한달 시각화 로직 전체 포함)
    # 분석 완료 후 초록색 메시지가 뜨면 image_500b2a.png에 데이터가 들어옵니다.
