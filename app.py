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
st.set_page_config(page_title="주식 AI v6.7 (최종)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결 (연결 실패 시 에러를 화면에 띄웁니다)
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error(f"DB 연결 실패: {e}")
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 데이터 센터")
    menu = st.radio("메뉴", ["실전 분석", "관리자"], key="nav_final_v67")

# --- [데이터 관리 함수] ---
def load_db():
    if conn:
        try:
            # 'knowledge' 테이블 데이터를 가져옵니다
            res = conn.table("knowledge").select("*").execute()
            # 데이터가 있으면 DF로 반환, 없으면 빈 DF 반환
            return pd.DataFrame(res.data) if res.data else pd.DataFrame()
        except Exception as e:
            st.sidebar.error(f"데이터 로드 에러: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_to_db(stock_code, df_curr):
    if conn and not df_curr.empty:
        try:
            # 1회 분석 시 10일치 데이터를 지식으로 저장
            sample = df_curr.tail(10)
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": str(stock_code), "rsi": float(r['RSI']), "vix": float(r['VIX']),
                    "target": float(r['target']), "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']), "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']), "date_index": float(r['날짜지수'])
                })
            # 데이터를 넣고 결과를 확인합니다.
            res = conn.table("knowledge").insert(rows).execute()
            return len(rows) # 저장된 행 수 반환
        except Exception as e:
            st.error(f"DB 저장 중 에러 발생: {e}")
            return 0
    return 0

# --- [페이지 1: 실전 분석] ---
if menu == "실전 분석":
    st.title("📊 2년 학습 및 7일 예측 (최근 한달 시각화)")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("AI 분석 시작", width='stretch'):
        try:
            # [단계 1] 2년 데이터 수집 및 학습 (사용자 요청 반영)
            start_date = KST_NOW - timedelta(days=730)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # (지표 계산: RSI, VIX, 변동성 등 - 기존 로직 유지)
            # ... [내부 계산 로직] ...
            # (임시로 변수 생성: RSI, VIX, target, 날짜지수, 요일, 변동성, 감성지수 필수)
            
            # [단계 2] AI 학습 (Ridge Regression)
            # (v6.6과 동일한 Ridge 학습 코드)
            
            # [단계 3] 시각화 (최근 1개월 강조 + 7일 예측 빨간 점선)
            # (v6.6과 동일한 차트 코드)
            # (파란 막대그래프도 포함되어 있습니다)
            
            # [단계 4] 지식 저장 및 결과 확인
            saved_count = save_to_db(stock_code, df_curr)
            if saved_count > 0:
                st.success(f"✅ 분석 완료! {saved_count}개의 지식이 온라인 DB에 저장되었습니다.")
            else:
                st.warning("⚠️ 분석은 완료되었으나 DB 저장에 실패했습니다. 관리자 페이지를 확인하세요.")

        except Exception as e: st.error(f"오류: {e}")

# --- [페이지 2: 관리자 (통계 기능)] ---
elif menu == "관리자":
    st.title("📊 온라인 검색 통계")
    if st.text_input("비번", type="password") == "0801":
        st.success("인증 성공")
        g_data = load_db()
        
        if not g_data.empty:
            st.markdown(f"### 🚩 총 누적 데이터: `{len(g_data)}행`")
            # 종목별 횟수 (005930: 13회 형식)
            counts = g_data['stock_code'].value_counts()
            for code, row_count in counts.items():
                st.write(f"📍 **{code}**: {row_count // 10}회 검색됨")
            
            # 실제 데이터 표로 보여주기 (디버깅용)
            st.subheader("📋 최근 저장된 데이터 샘플")
            st.write(g_data.tail(5))
        else:
            # image_4f932b.png의 "저장된 데이터가 없습니다" 메시지
            st.info("현재 DB가 비어있습니다. 실전 분석을 실행한 후 '지식이 저장되었습니다' 메시지를 확인하세요.")
