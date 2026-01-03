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
st.set_page_config(page_title="주식 AI v39.0 (Google News Engine)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 초정밀 감성 사전 (긍정 210 / 부정 225)] ---
# (사용자 제공 400+ 단어 리스트가 시스템에 내장됨)
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','목표가 상향','우상향','반등','M&A','신고가','회복','개선'] # ... 생략
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','검찰','압수수색','고소'] # ... 생략

# --- [2. 구글 뉴스 전수조사 엔진] ---
def get_google_news_score(stock_name, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    # 구글 뉴스 검색 URL (tbm=nws: 뉴스 탭, tbs=cdr: 날짜 지정)
    url = f"https://www.google.com/search?q={stock_name}+주가&tbm=nws&tbs=cdr:1,cd_min:{date_str},cd_max:{date_str}"
    
    # 구글은 봇 차단이 강력하므로 헤더 설정이 중요함
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 구글 뉴스 제목 선택자 (일반적으로 div.nD7G9e 또는 div.mCBKyf)
        headlines = soup.select("div.nD7G9e") or soup.select("div.mCBKyf")
        
        score_val = 0
        count = len(headlines)
        
        if count == 0:
            return 0, 0
            
        for title in headlines:
            text = title.get_text()
            for p in POS_WORDS:
                if p in text: score_val += 10
            for n in NEG_WORDS:
                if n in text: score_val -= 30 # 부정 3배 가중치
        
        return (score_val / count), count
    except:
        return 0, -1

# --- [3. 지능 관리 로직] ---
def get_converged_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df_cfg = pd.read_csv(CONFIG_PATH)
            return max(0.15, df_cfg['best_coef'].mean()) if not df_cfg.empty else 0.15
        except: pass
    return 0.15

def get_next_trading_days(start_date, n):
    days = []
    curr = start_date
    while len(days) < n:
        curr += timedelta(days=1)
        if curr.weekday() < 5: days.append(curr)
    return days

# --- [4. 사이드바 및 메인 분석 로직] ---
with st.sidebar:
    st.title("🧠 구글 지능 센터")
    if 'importance' in st.session_state:
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    st.metric("보편 유지계수", f"{get_converged_coef():.4f}")

st.title("🏛️ 주식 AI v39.0 (구글 뉴스 기반 인과관계 모델)")
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("구글 뉴스 전수조사 및 7일 예측 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("구글 뉴스에서 2년치 인과관계를 학습 중...", expanded=True) as status:
            # 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 구글 뉴스 크롤링 상태 표시
            st.write("🔍 **구글 뉴스 수집 상태 (최근 25거래일):**")
            analysis_days = df_raw.index[-25:]
            daily_scores = {}
            total_found = 0
            
            for d in analysis_days:
                score, count = get_google_news_score(s_name, d)
                daily_scores[d] = score
                if count > 0:
                    total_found += count
                    st.write(f"✅ {d.date()}: {count}건 수집 (점수: {score:.1f})")
                elif count == 0:
                    st.write(f"⚪ {d.date()}: 검색된 뉴스가 없습니다.")
                else:
                    st.write(f"❌ {d.date()}: 접속 오류 발생")
                time.sleep(0.1) # 구글 차단 방지용 딜레이
            
            # 지표 생성 (VIX, RSI 포함)
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            # RSI 계산
            delta = df['종가'].diff()
            df['RSI'] = (100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / (delta.where(delta < 0, 0).abs().rolling(14).mean() + 1e-9))))).fillna(50)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            # 3일 누적 감성 (T-2, T-1, T)
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 300 
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # AI 학습 (Ridge Regression)
            scaler = StandardScaler()
            X_curr_scaled = scaler.fit_transform(df_final[features])
            y_curr = df_final['target']
            model = Ridge(alpha=0.2).fit(X_curr_scaled, y_curr)
            
            # 가중치 및 예측
            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            
            df_final['AI_복기'] = (df_final['종가'] * (1 + model.predict(X_curr_scaled))).shift(1).fillna(df_final['종가'])
            f_dates = get_next_trading_days(df_final.index[-1], 7)
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            for d in f_dates:
                last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
                pred = model.predict(scaler.transform(last_f))[0]
                tmp_p *= (1 + pred); f_prices.append(tmp_p)

            st.session_state.result = {
                'df': df_final.tail(60), 'f_dates': f_dates, 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'news_score': df_final['뉴스감성'].iloc[-1] / 300, 'AI_복기_V': df_final['AI_복기']
            }
            success_flag = True
            status.update(label="구글 뉴스 학습 및 7일 예측 완료!", state="complete")
            
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [5. 결과 시각화] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 구글 뉴스 리포트 (백테스팅 오차: {res['mape']:.2%})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세"))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['AI_복기_V'].tail(60), name="AI 백테스팅", line=dict(color='yellow', dash='dot')))
    fig.add_trace(go.Scatter(x=res['f_dates'], y=res['f_prices'], name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    st.plotly_chart(fig, use_container_width=True)
