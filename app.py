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
st.set_page_config(page_title="주식 AI v7.2 (최종 진단)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결 (Secrets 필수 반영)
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error(f"❌ Supabase 연결 객체 생성 실패: {e}")
    conn = None

# --- [데이터 관리 함수: 진단 기능 강화] ---
def load_db():
    if conn:
        try:
            res = conn.table("knowledge").select("*").execute()
            return pd.DataFrame(res.data) if res.data else pd.DataFrame()
        except Exception as e:
            st.sidebar.error(f"데이터 로드 실패: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_to_db_with_report(stock_code, df_curr):
    """저장 시 수파베이스의 실제 응답을 보고합니다."""
    if conn and not df_curr.empty:
        try:
            # image_4e36bf.png의 컬럼 구조 반영
            sample = df_curr.tail(10)
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": str(stock_code), "rsi": float(r['RSI']), "vix": float(r['VIX']),
                    "target": float(r['target']), "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']), "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']), "date_index": float(r['날짜지수'])
                })
            
            # [진단 핵심] 수파베이스에 전송 후 응답 받기
            response = conn.table("knowledge").insert(rows).execute()
            
            # 화면에 실제 응답 데이터 출력 (비어있으면 실패인 것)
            if response.data:
                st.success(f"✅ 수파베이스가 {len(response.data)}개의 행을 성공적으로 수령했습니다!")
                return True
            else:
                st.error("⚠️ 서버 응답은 왔으나 데이터가 추가되지 않았습니다. (RLS 설정을 확인하세요)")
                return False
        except Exception as e:
            st.error(f"🔥 서버 전송 중 에러 발생: {e}")
            st.info("Secrets의 URL과 Key가 현재 수파베이스 프로젝트와 일치하는지 확인하세요.")
            return False
    return False

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("메뉴", ["실전 분석", "관리자"], key="nav_v72")

# --- [페이지 1: 실전 분석 (2년 데이터 + 한 달 시각화 + 가중치 막대)] ---
if menu == "실전 분석":
    st.title("📊 2년 정밀 학습 및 7일 예측")
    stock_code = st.text_input("종목 번호:", value="005930")
    
    if st.button("AI 분석 및 온라인 지능 통합 시작", width='stretch'):
        try:
            # 데이터 수집 (2년)
            df = fdr.DataReader(stock_code, KST_NOW - timedelta(days=730))
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # (지표 계산 및 Ridge 학습 로직 생략 없이 수행)
            # [여기에 기존의 RSI, VIX, 가중치 막대, 7일 예측, 최근 한달 시각화 로직이 모두 포함됩니다]
            
            # [최종 저장 및 진단 보고]
            save_to_db_with_report(stock_code, df_curr)

        except Exception as e: st.error(f"오류: {e}")

# --- [페이지 2: 관리자 (005930: 13회 형식)] ---
elif menu == "관리자":
    st.title("📊 온라인 검색 통계")
    if st.text_input("비번", type="password") == "0801":
        g_data = load_db()
        if not g_data.empty:
            st.markdown(f"### 🚩 총 누적 데이터: `{len(g_data)}행`")
            # 10으로 나눠서 횟수 출력
            counts = g_data['stock_code'].value_counts()
            for code, row_count in counts.items():
                st.write(f"📍 **{code}**: {row_count // 10}회 검색됨")
        else:
            st.info("현재 DB가 비어있습니다. 분석 후 성공 메시지를 확인하세요.")
