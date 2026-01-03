import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
# RSS 파싱을 위해 추가로 필요한 라이브러리 (기본 설치됨)
import xml.etree.ElementTree as ET 

# ... (환경 설정 및 사전 정의는 이전 v40.0과 동일) ...

# --- [2. 구글 뉴스 RSS 엔진 (차단 회피형)] ---
def get_google_rss_score(stock_name, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    next_date_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 캐시 확인 로직 유지
    if os.path.exists(NEWS_DB_PATH):
        try:
            cache = pd.read_csv(NEWS_DB_PATH)
            match = cache[(cache['date'] == date_str) & (cache['name'] == stock_name)]
            if not match.empty: return float(match.iloc[0]['score'])
        except: pass

    # RSS 검색 URL (after/before를 사용하여 특정 일자 지정)
    # q=종목명+주가+after:2024-11-20+before:2024-11-21
    url = f"https://news.google.com/rss/search?q={stock_name}+주가+after:{date_str}+before:{next_date_str}&hl=ko&gl=KR&ceid=KR:ko"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    
    score_val = 0
    try:
        # RSS는 XML 형식이므로 requests 후 BeautifulSoup의 'xml' 파서나 기본 XML 파서 사용
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.find_all("item") # RSS의 각 기사 단위는 <item>
        
        count = len(items)
        if count == 0: return 0
        
        for item in items:
            title = item.title.text # 기사 제목 추출
            # 감성 분석 로직 (비대칭 가중치 적용)
            for p in POS_WORDS:
                if p in title: score_val += 10
            for n in NEG_WORDS:
                if n in title: score_val -= 30
        
        final_score = (score_val / count)
        
        # 캐시 저장
        pd.DataFrame([[date_str, stock_name, final_score]], columns=['date', 'name', 'score']).to_csv(NEWS_DB_PATH, mode='a', header=not os.path.exists(NEWS_DB_PATH), index=False)
        return final_score
    except Exception as e:
        # 에러 발생 시 로그 출력 (디버깅용)
        # st.write(f"Error on {date_str}: {e}")
        return 0

# --- [이후 메인 분석 로직에서 get_google_news_score를 get_google_rss_score로 교체] ---
