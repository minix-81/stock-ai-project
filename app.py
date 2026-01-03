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
st.set_page_config(page_title="주식 AI v17.0 (Deep News Backtest)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 과거 날짜별 뉴스 크롤링 함수] ---
def get_past_news_sentiment(stock_name, target_date):
    """특정 날짜(target_date)의 뉴스 헤드라인을 가져와 점수화"""
    date_str = target_date.strftime('%Y.%m.%d')
    # 네이버 뉴스 날짜 지정 검색 URL (ds: 시작일, de: 종료일)
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=4&ds={date_str}&de={date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        
        pos = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천']
        neg = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락']
        
        score, count = 0, 0
        for title in headlines:
            text = title.get_text()
            for p in pos: 
                if p in text: score += 10
            for n in neg: 
                if n in text: score -= 10
            count += 1
        return np.clip(score / count * 5, -100, 100) if count > 0 else 0
    except:
        return 0

# --- [2. 지능 수렴 및 데이터 처리 (영업일 기준)] ---
def get_converged_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            return df['best_coef'].mean() if not df.empty else 0.15
        except: pass
    return 0.15

def update_global_intelligence(new_coef):
    pd.DataFrame([[datetime.now(), new_coef]], columns=['date', 'best_coef']).to_csv(
        CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)

def get_next_trading_days(start_date, n):
    days = []
    curr = start_date
    while len(days) < n:
        curr += timedelta(days=1)
        if curr.weekday() < 5: days.append(curr)
    return days

# --- [3. 메인 분석 로직] ---
st.title("🏛️ 주식 AI v17.0 (인과관계 백테스팅)")
st.markdown("과거 특정일의 뉴스를 직접 크롤링하여 다음 날 주가와의 상관관계를 학습합니다.")

# 사이드바 설정
knowledge_df = pd.read_csv(DB_PATH, dtype={'stock_code': str}) if os.path.exists(DB_PATH) else pd.DataFrame()
converged_coef = get_converged_coef()

with st.sidebar:
    st.header("🧠 지능 수렴 상태")
    st.metric("보편 감수계수", f"{converged_coef:.4f}")
    if 'importance' in st.session_state:
        st.write("### 변수 영향력 (Blue Bar)")
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')

# 입력창
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("과거 뉴스 전수조사 및 백테스팅 시작", use_container_width=True):
    try:
        with st.status("데이터 분석 중...", expanded=True) as status:
            # 1. 주가 데이터 로드
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 2. 뉴스 백테스팅 (최근 20거래일 깊게 조사 - 속도 문제로 샘플링)
            st.write("🔍 과거 일자별 뉴스 감성-주가 인과관계 분석 중...")
            recent_days = df_raw.index[-21:-1] # 최근 20일
            news_scores = []
            for d in recent_days:
                score = get_past_news_sentiment(s_name, d)
                news_scores.append(score)
                time.sleep(0.1) # 차단 방지
            
            # 3. 데이터 전처리
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            df['target'] = df['종가'].pct_change().shift(-1) # 다음 날 주가 변화
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            
            # 뉴스 점수 결합 (최근 데이터는 실제 크롤링 점수, 나머지는 주가 기반 추정치로 학습)
            df['뉴스감성'] = (df['종가'].pct_change() * 1000).clip(-100, 100)
            for i, d in enumerate(recent_days):
                df.loc[d, '뉴스감성'] = news_scores[i]
            
            df = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX']
            
            # 4. AI 학습 및 지능 수렴
            X_curr = df[features]
            y_curr = df['target']
            scaler = StandardScaler()
            X_curr_scaled = scaler.fit_transform(X_curr)
            
            if not knowledge_df.empty:
                X_ext = knowledge_df[features]
                y_ext = knowledge_df['target']
                X_total_scaled = scaler.fit_transform(pd.concat([X_curr, X_ext]))
                y_total = pd.concat([y_curr, y_ext])
                
                # 최적 계수 탐색 (백테스팅 기반)
                min_err = float('inf')
                best_local = converged_coef
                for c in [0.1, 0.2, 0.3]:
                    w = np.concatenate([np.ones(len(X_curr)), np.full(len(X_ext), c)])
                    m = Ridge(alpha=1.0).fit(X_total_scaled, y_total, sample_weight=w)
                    err = mean_absolute_percentage_error(y_curr, m.predict(scaler.transform(X_curr)))
                    if err < min_err: min_err = err; best_local = c
                
                update_global_intelligence(best_local)
                final_coef = get_converged_coef()
                weights = np.concatenate([np.ones(len(X_curr)), np.full(len(X_ext), final_coef)])
                model = Ridge(alpha=1.0).fit(X_total_scaled, y_total, sample_weight=weights)
            else:
                model = Ridge(alpha=1.0).fit(X_curr_scaled, y_current := y_curr)
                final_coef = 0.15

            # 5. 백테스팅 그래프 데이터 (노란 점선)
            df['AI_복기종가'] = df['종가'] * (1 + model.predict(scaler.transform(X_curr)))
            df['AI_복기종가'] = df['AI_복기종가'].shift(1).fillna(df['종가'])
            mape = mean_absolute_percentage_error(df['종가'], df['AI_복기종가'])

            # 6. 미래 7영업일 예측
            f_dates = get_next_trading_days(df.index[-1], 7)
            f_prices, temp_p = [], df['종가'].iloc[-1]
            last_f = df[features].iloc[-1:].copy()
            # 현재 실시간 뉴스 점수
            current_news = get_past_news_sentiment(s_name, KST_NOW)
            last_f['뉴스감성'] = current_news

            for d in f_dates:
                last_f['날짜지수'] += 1
                last_f['요일'] = d.weekday()
                pred = model.predict(scaler.transform(last_f))[0]
                temp_p *= (1 + pred)
                f_prices.append(temp_p)

            # 세션 저장
            st.session_state.result = {
                'name': s_name, 'code': s_code, 'mape': mape, 'coef': final_coef,
                'df': df.tail(66), 'f_dates': f_dates, 'f_prices': f_prices,
                'news_score': current_news
            }
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': model.coef_})
            
            # 지식 저장 (빅데이터 성장)
            df['stock_code'] = s_code
            df[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            
            status.update(label="분석 완료!", state="complete")
            st.rerun()

    except Exception as e: st.error(f"오류: {e}")

# --- [4. 시각화 (백테스팅 그래프 포함)] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 {res['name']} 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    # 백테스팅 노란 점선 복구
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['AI_복기종가'], name="AI 과거 복기(백테스팅)", 
                             line=dict(color='yellow', dash='dot'), opacity=0.5))
    # 미래 예측
    fig.add_trace(go.Scatter(x=[res['df'].index[-1]] + res['f_dates'], y=[res['df']['종가'].iloc[-1]] + res['f_prices'], 
                             name="미래 7영업일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    fig.update_layout(template='plotly_dark', title=f"수렴 계수: {res['coef']:.4f} | 현재 뉴스 감성: {res['news_score']:.1f}", height=600)
    st.plotly_chart(fig, use_container_width=True)
