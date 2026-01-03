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
st.set_page_config(page_title="주식 AI v7.0 (DB 정밀진단)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결 (연결 실패 시 화면에 즉시 에러 출력)
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error(f"⚠️ 수파베이스 설정 오류 (Secrets를 확인하세요): {e}")
    conn = None

# --- [사이드바: DB 연결 진단기] ---
with st.sidebar:
    st.title("🚀 AI 진단 센터")
    menu = st.radio("메뉴 선택", ["실전 분석", "관리자"], key="nav_v70")
    
    st.divider()
    if st.button("🔍 DB 연결 상태 점검"):
        if conn:
            try:
                # 아주 간단한 조회를 시도하여 연결을 확인합니다.
                test_res = conn.table("knowledge").select("count", count="exact").limit(1).execute()
                st.sidebar.success("✅ DB 연결 정상! (데이터 전송 준비 완료)")
            except Exception as e:
                st.sidebar.error(f"❌ DB 응답 없음: {e}")
        else:
            st.sidebar.error("❌ 연결 객체가 생성되지 않았습니다.")

# --- [데이터 관리 함수: 에러를 숨기지 않고 다 보여줌] ---
def save_to_db(stock_code, df_curr):
    if conn and not df_curr.empty:
        try:
            # image_4e36bf.png의 컬럼명과 1:1 매칭 확인
            sample = df_curr.tail(10)
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": str(stock_code), 
                    "rsi": float(r['RSI']), 
                    "vix": float(r['VIX']),
                    "target": float(r['target']), 
                    "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']), 
                    "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']), 
                    "date_index": float(r['날짜지수'])
                })
            # 데이터 전송 및 결과 받기
            response = conn.table("knowledge").insert(rows).execute()
            
            if response:
                return len(rows)
        except Exception as e:
            # ⚠️ 여기서 뜨는 빨간 에러 메시지를 꼭 저에게 알려주세요!
            st.error(f"🔥 DB 전송 사고 발생: {e}")
            return 0
    return 0

# --- [페이지 1: 실전 분석 (2년 학습/한달 시각화/예측선)] ---
if menu == "실전 분석":
    st.title("📊 AI 주가 분석 (정밀 진단 모드)")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("AI 분석 및 지능 통합 시작", width='stretch'):
        try:
            # [단계 1] 2년치 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # (지표 계산 및 AI 학습 로직 - v6.9와 동일)
            # ... [VIX/RSI 계산 및 Ridge 학습 로직] ...
            
            # [단계 2] 시각화 (파란 막대 + 최근 한 달 시세 + 빨간 예측 점선)
            # (시각화 로직 수행)
            
            # [단계 3] 데이터 저장 시도 및 실시간 보고
            saved_count = save_to_db(stock_code, df_curr)
            if saved_count > 0:
                st.success(f"🎊 성공! {saved_count}개의 지식이 온라인 DB에 안착했습니다.")
            else:
                st.warning("⚠️ 분석은 끝났지만, 지식이 저장되지 않았습니다. 사이드바 진단을 확인하세요.")

        except Exception as e:
            st.error(f"💥 분석 도중 에러: {e}")

# --- [페이지 2: 관리자 (005930: 13회 형식)] ---
elif menu == "관리자":
    st.title("📊 온라인 학습 통계")
    if st.text_input("관리자 비번", type="password") == "0801":
        # 데이터 로드 및 종목별 횟수 요약 출력
        # (v6.9와 동일한 통계 출력 로직)
