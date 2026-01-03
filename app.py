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
st.set_page_config(page_title="주식 AI v40.0 (Stable)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
NEWS_DB_PATH = "news_cache_v40.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 초대형 감성 사전 (긍정 210 / 부정 225)] ---
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','반등','M&A','신고가','회복','개선'] # (중략... 210개)
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','검찰','압수수색','고소'] # (중략... 225개)

# --- [2. 구글 뉴스 엔진 (에러 방지 강화)] ---
def get_google_news_score(stock_name, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    if os.path.exists(NEWS_DB_PATH):
        try:
            cache = pd.read_csv(NEWS_DB_PATH)
            match = cache[(cache['date'] == date_str) & (cache['name'] == stock_name)]
            if not match.empty: return float(match.iloc[0]['score'])
        except: pass

    url = f"https://www.google.com/search?q={stock_name}+주가&tbm=nws&tbs=cdr:1,cd_min:{date_str},cd_max:{date_str}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}
    
    score_val = 0 # 초기화 보장
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select("div.nD7G9e") or soup.select("div.mCBKyf") or soup.select(".vvSy9b")
        
        count = len(headlines)
        if count == 0: return 0
        
        for title in headlines:
            text = title.get_text()
            for p in POS_WORDS: 
                if p in text: score_val += 10
            for n in NEG_WORDS: 
                if n in text: score_val -= 30
        
        final_score = (score_val / count)
        pd.DataFrame([[date_str, stock_name, final_score]], columns=['date', 'name', 'score']).to_csv(NEWS_DB_PATH, mode='a', header=not os.path.exists(NEWS_DB_PATH), index=False)
        return final_score
    except: return 0

# --- [3. 지능 수렴 로직] ---
def get_fluid_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            return max(0.15, df['best_coef'].mean())
        except: pass
    return 0.15

def get_next_trading_days(start_date, n):
    days = []
    curr = start_date
    while len(days) < n:
        curr += timedelta(days=1)
        if curr.weekday() < 5: days.append(curr)
    return days

# --- [4. 메인 분석 엔진] ---
st.title("🏛️ 주식 AI v40.0 (에러 해결 및 그래프 연결)")

with st.sidebar:
    st.title("🧠 AI 분석 센터")
    if 'importance' in st.session_state:
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    st.metric("보편 유지계수", f"{get_fluid_coef():.4f}")

c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("통합 분석 및 미래 예측 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("전 지표 통합 분석 및 뉴스 전수조사 중...", expanded=True) as status:
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 뉴스 분석
            analysis_days = df_raw.index[-25:]
            daily_scores = {d: get_google_news_score(s_name, d) for d in analysis_days}
            
            # 지표 생성
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            
            # RSI 안전 계산 (에러 방지)
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))).fillna(50)
            
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            # 뉴스 감성 적용 (3일 누적)
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 300 
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # AI 학습
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_final[features])
            y = df_final['target']
            model = Ridge(alpha=0.2).fit(X_scaled, y)
            
            # 미래 7일 예측
            f_dates = get_next_trading_days(df_final.index[-1], 7)
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            for d in f_dates:
                last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
                pred = model.predict(scaler.transform(last_f))[0]
                tmp_p *= (1 + pred); f_prices.append(tmp_p)

            # 결과 저장
            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            st.session_state.result = {
                'df': df_final.tail(60), 'f_dates': f_dates, 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], (df_final['종가'] * (1 + model.predict(X_scaled))).shift(1).fillna(df_final['종가'])),
                'AI_복기': (df_final['종가'] * (1 + model.predict(X_scaled))).shift(1).fillna(df_final['종가'])
            }
            success_flag = True
            status.update(label="분석 완료!", state="complete")
            
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [5. 시각화: 그래프 끊김 해결 로직] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 리포트 (오차율: {res['mape']:.2%})")
    
    fig = go.Figure()
    # 1. 실제 시세
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    # 2. 백테스팅 (노란 점선)
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['AI_복기'].tail(60), name="AI 백테스팅", line=dict(color='yellow', dash='dot'), opacity=0.5))
    
    # [수정] 미래 예측 선을 실제 시세의 마지막 데이터와 연결
    connect_x = [res['df'].index[-1]] + res['f_dates']
    connect_y = [res['df']['종가'].iloc[-1]] + res['f_prices']
    
    fig.add_trace(go.Scatter(x=connect_x, y=connect_y, name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)
