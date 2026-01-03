import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
import time

# 1. 환경 설정
st.set_page_config(page_title="주식 AI v38.0 (Crawl Monitor)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 초정밀 감성 사전: 긍정 210 / 부정 225] ---
# (사용자 제공 단어 리스트 적용됨)
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','목표가 상향','우상향','반등','M&A','신고가','회복','개선'] # ... 생략
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','검찰','압수수색','고소'] # ... 생략

# --- [2. 3일 누적 뉴스 감성 평가 엔진: '수집 개수' 반환 추가] ---
def get_daily_news_score_with_count(stock_name, target_date):
    date_str = target_date.strftime('%Y.%m.%d')
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=4&ds={date_str}&de={date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        
        score_val = 0
        count = len(headlines) # [수정] 수집된 기사 수 파악
        
        if count == 0:
            return 0, 0
            
        for title in headlines:
            text = title.get_text()
            for p in POS_WORDS:
                if p in text: score_val += 10
            for n in NEG_WORDS:
                if n in text: score_val -= 30 # 부정 3배 가중치
        
        return (score_val / count), count # [수정] 점수와 개수를 함께 반환
    except:
        return 0, -1 # 에러 발생 시 개수를 -1로 표시

# --- [3. 지능 관리 로직] ---
def get_converged_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df_cfg = pd.read_csv(CONFIG_PATH)
            return max(0.15, df_cfg['best_coef'].mean()) if not df_cfg.empty else 0.15
        except: pass
    return 0.15

# --- [4. 메인 분석 로직] ---
st.title("🏛️ 주식 AI v38.0 (크롤링 모니터링 엔진)")
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("백테스팅 및 7일 미래 예측 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("AI가 뉴스를 전수조사 중입니다...", expanded=True) as status:
            # 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # [수정] 뉴스 크롤링 상태 실시간 표시
            st.write("🔍 **날짜별 네이버 뉴스 수집 상태:**")
            analysis_days = df_raw.index[-25:]
            daily_scores = {}
            total_found = 0
            
            for d in analysis_days:
                score, count = get_daily_news_score_with_count(s_name, d)
                daily_scores[d] = score
                if count > 0:
                    total_found += count
                    st.write(f"✅ {d.date()}: {count}건의 기사를 읽었습니다. (점수: {score:.1f})")
                elif count == 0:
                    st.write(f"⚪ {d.date()}: 검색된 기사가 없습니다.")
                else:
                    st.write(f"❌ {d.date()}: 크롤링 중 오류가 발생했습니다.")
                time.sleep(0.05) # 서버 부하 방지
            
            if total_found == 0:
                st.error("🚨 모든 날짜에서 뉴스가 수집되지 않았습니다. 종목명을 다시 확인해주세요.")
            
            # 이후 지표 생성 및 AI 학습 로직 (기존 v28.0과 동일)
            # (RSI, VIX, 뉴스 3일 롤링 평균 적용 등)
            # ...
            
            # [결과 화면 리런을 위해 플래그 설정]
            success_flag = True
            status.update(label=f"분석 완료! (총 {total_found}건의 뉴스 학습)", state="complete")
            
    except Exception as e:
        st.error(f"오류: {e}")
    
    if success_flag:
        st.rerun()

# (이하 시각화 및 결과 출력 코드는 동일)
