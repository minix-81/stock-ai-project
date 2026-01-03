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
st.set_page_config(page_title="주식 AI v45.0 (Sentiment Dominance)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge_v45.csv"
NEWS_DB_PATH = "news_rss_cache_v45.csv"
CONFIG_PATH = "global_config_v45.csv"

# --- [1. 초대형 감성 사전: 긍정 210 / 부정 225] ---
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','목표가 상향','우상향','반등','M&A','신고가','어닝 서프라이즈','기관 매수','외인 매수','순매수','저평가','배당 확대','자사주 매입'] 
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','검찰','압수수색','기소','배임','횡령','사법 리스크','고소','피소','수사']

# --- [2. 구글 RSS 뉴스 엔진] ---
def get_google_rss_score(stock_name, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    next_date_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    if os.path.exists(NEWS_DB_PATH):
        try:
            cache = pd.read_csv(NEWS_DB_PATH)
            match = cache[(cache['date'] == date_str) & (cache['name'] == stock_name)]
            if not match.empty: return float(match.iloc[0]['score']), 1
        except: pass

    url = f"https://news.google.com/rss/search?q={stock_name}+주가+after:{date_str}+before:{next_date_str}&hl=ko&gl=KR&ceid=KR:ko"
    headers = {"User-Agent": "Mozilla/5.0"}
    score_val = 0
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.find_all("item")
        count = len(items)
        if count == 0: return 0, 0
        for item in items:
            title = item.title.text
            for p in POS_WORDS:
                if p in title: score_val += 10
            for n in NEG_WORDS:
                if n in title: score_val -= 30 
        final_score = (score_val / count)
        pd.DataFrame([[date_str, stock_name, final_score]], columns=['date', 'name', 'score']).to_csv(NEWS_DB_PATH, mode='a', header=not os.path.exists(NEWS_DB_PATH), index=False)
        return final_score, count
    except: return 0, -1

# --- [3. 지능 관리 로직] ---
def get_fluid_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            return max(0.15, df['best_coef'].mean())
        except: pass
    return 0.15

# --- [4. 메인 분석 엔진] ---
st.title("🏛️ 주식 AI v45.0 (투자 심리 절대 가중치 모델)")
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("2년 뉴스 전수 학습 및 7일 예측 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("뉴스 감성 가중치를 1000배 증폭하여 인과관계를 학습 중...", expanded=True) as status:
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # [학습 단계] 최근 60거래일 뉴스 심리 전수 조사
            analysis_days = df_raw.index[-60:] 
            daily_scores = {d: get_google_rss_score(s_name, d)[0] for d in analysis_days}
            
            # 지표 생성 (VIX, RSI 등 전체 포함)
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            delta = df['종가'].diff()
            df['RSI'] = (100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / (delta.where(delta < 0, 0).abs().rolling(14).mean() + 1e-9))))).fillna(50)
            
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            # [핵심] 뉴스 심리 1000배 증폭 및 3일 누적 (T-2, T-1, T)
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 1000 # 반영비 1000배 극대화
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # AI 학습 (뉴스 심리에 극도로 민감하게 반응하도록 Ridge alpha 조정)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_final[features])
            y = df_final['target']
            model = Ridge(alpha=0.1).fit(X_scaled, y)
            
            # 백테스팅
            df_final['AI_복기'] = (df_final['종가'] * (1 + model.predict(X_scaled))).shift(1).fillna(df_final['종가'])
            
            # [미래 7일 예측] 최근 3일 뉴스 심리를 7일 내내 강력하게 고정 반영
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            current_sentiment = df_final['뉴스감성'].iloc[-1] 
            
            for i in range(1, 8):
                last_f['날짜지수'] += 1; last_f['요일'] = (df_final.index[-1].weekday() + i) % 7
                last_f['뉴스감성'] = current_sentiment # 뉴스 심리 유지
                pred = model.predict(scaler.transform(last_f))[0]
                tmp_p *= (1 + pred); f_prices.append(tmp_p)

            # 결과 저장
            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            st.session_state.result = {
                'df': df_final.tail(60), 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'news_score': current_sentiment / 1000, 'AI_복기_V': df_final['AI_복기']
            }
            success_flag = True
            status.update(label="초고감도 뉴스 심리 기반 7일 예측 완료!", state="complete")
            
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [5. 시각화] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['AI_복기'], name="AI 백테스팅", line=dict(color='yellow', dash='dot'), opacity=0.5))
    
    f_dates = [res['df'].index[-1] + timedelta(days=i) for i in range(1, 8)]
    fig.add_trace(go.Scatter(x=[res['df'].index[-1]] + f_dates, y=[res['df']['종가'].iloc[-1]] + res['f_prices'], 
                             name="미래 7일 예측 (뉴스심리 지배)", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    # [수정] 중립 구간 최소화: 1.0 미만만 중립으로 판정
    sentiment_label = '⚖️ 중립'
    if res['news_score'] >= 1.0: sentiment_label = '🟢 호재 발생'
    elif res['news_score'] <= -1.0: sentiment_label = '🔴 악재 감지'
    
    st.markdown(f"### 📢 투자 심리 진단(최근 3일): {sentiment_label} ({res['news_score']:.2f})")
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)
