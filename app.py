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
st.set_page_config(page_title="주식 AI v18.0", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 뉴스 감성 분석: 부정 뉴스 가중치 강화] ---
def get_past_news_sentiment(stock_name, target_date):
    date_str = target_date.strftime('%Y.%m.%d')
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=4&ds={date_str}&de={date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        
        pos = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천']
        neg = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크']
        
        score, count = 0, 0
        for title in headlines:
            text = title.get_text()
            for p in pos: 
                if p in text: score += 10
            for n in neg: 
                # [중요] 부정 뉴스의 영향력을 1.5배 가중 (비대칭성 반영)
                if n in text: score -= 15 
            count += 1
        return np.clip(score / count * 5, -100, 100) if count > 0 else 0
    except: return 0

# --- [2. 지능 수렴 및 데이터 로직 (UI 비표시)] ---
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

def prepare_data(df, start_date, current_news_score):
    vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
    df = df.join(vix).ffill().fillna(20)
    delta = df['종가'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = (100 - (100 / (1 + (gain / loss)))).fillna(50)
    df['target'] = df['종가'].pct_change().shift(-1)
    df['날짜지수'] = np.arange(len(df))
    df['요일'] = df.index.weekday
    df['변동성'] = (df['High'] - df['Low']) / df['종가']
    df['뉴스감성'] = (df['종가'].pct_change() * 1000).clip(-100, 100)
    df.iloc[-1, df.columns.get_loc('뉴스감성')] = current_news_score
    return df.dropna()

# --- [3. 사이드바 구성 (가중치 표 추가)] ---
knowledge_df = pd.read_csv(DB_PATH, dtype={'stock_code': str}) if os.path.exists(DB_PATH) else pd.DataFrame()
converged_coef = get_converged_coef()

with st.sidebar:
    st.title("🧠 AI 지능 저장소")
    if not knowledge_df.empty:
        st.write(f"📚 누적 데이터: `{len(knowledge_df)}`개")
    
    st.divider()
    st.subheader("📊 변수 영향력 분석")
    if 'importance' in st.session_state:
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
        
        # [요청 사항] 변수 정의 표 추가
        st.markdown("""
        | 변수명 | 의미 | 비고 |
        | :--- | :--- | :--- |
        | **날짜지수** | 시간의 흐름 | 장기 추세 반영 |
        | **요일** | 주간 패턴 | 월요병/금요효과 |
        | **거래량** | 시장의 관심도 | 에너지의 세기 |
        | **변동성** | 고가-저가 차이 | 불안정성 지표 |
        | **뉴스감성** | 기사 긍/부정 | **부정 뉴스 가중치 반영** |
        | **VIX** | 공포 지수 | 시장 전체 위기감 |
        | **RSI** | 과매수/과매도 | 기술적 반등 지표 |
        """)
    else:
        st.info("분석을 시작하면 가중치와 지표 설명이 나타납니다.")

# --- [4. 메인 분석 로직] ---
st.title("🏛️ 주식 AI v18.0 (비대칭 감성 학습 모델)")
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("뉴스 인과관계 전수조사 시작", use_container_width=True):
    try:
        with st.status("AI 지능 수렴 및 뉴스 백테스팅 중...", expanded=True) as status:
            # 데이터 수집 및 뉴스 크롤링
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 최근 뉴스 인과관계 분석
            recent_days = df_raw.index[-15:-1]
            news_scores = [get_past_news_sentiment(s_name, d) for d in recent_days]
            
            # 전처리
            current_news = get_past_news_sentiment(s_name, KST_NOW)
            df = prepare_data(df_raw, start_date, current_news)
            for i, d in enumerate(recent_days):
                df.loc[d, '뉴스감성'] = news_scores[i]
            
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            X_curr = df[features]
            y_curr = df['target']
            scaler = StandardScaler()
            X_curr_scaled = scaler.fit_transform(X_curr)
            
            # 지능 수렴 학습 (계수는 백그라운드에서 계산)
            if not knowledge_df.empty:
                X_total_scaled = scaler.fit_transform(pd.concat([X_curr, knowledge_df[features]]))
                y_total = pd.concat([y_curr, knowledge_df['target']])
                
                min_err, best_local = float('inf'), converged_coef
                for c in [0.1, 0.2, 0.3]:
                    w = np.concatenate([np.ones(len(X_curr)), np.full(len(knowledge_df), c)])
                    m = Ridge().fit(X_total_scaled, y_total, sample_weight=w)
                    err = mean_absolute_percentage_error(y_curr, m.predict(X_curr_scaled))
                    if err < min_err: min_err = err; best_local = c
                
                update_global_intelligence(best_local)
                final_coef = get_converged_coef()
                weights = np.concatenate([np.ones(len(X_curr)), np.full(len(knowledge_df), final_coef)])
                model = Ridge().fit(X_total_scaled, y_total, sample_weight=weights)
            else:
                model = Ridge().fit(X_curr_scaled, y_curr)
            
            # 결과 저장
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': model.coef_})
            st.session_state.result = {
                'name': s_name, 'code': s_code, 'mape': mean_absolute_percentage_error(df['종가'], df['종가'] * (1 + model.predict(X_curr_scaled))),
                'df': df.tail(60), 'f_dates': get_next_trading_days(df.index[-1], 7),
                'model': model, 'scaler': scaler, 'features': features, 'news_score': current_news
            }
            # 지식 축적
            df['stock_code'] = s_code
            df[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            status.update(label="분석 완료!", state="complete")
            st.rerun()

    except Exception as e: st.error(f"오류: {e}")

# --- [5. 결과 시각화] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 {res['name']} ({res['code']}) - 백테스팅 오차율: {res['mape']:.2%}")
    
    # 예측값 복기 및 미래 계산
    df_v = res['df']
    pred_past = res['model'].predict(res['scaler'].transform(df_v[res['features']]))
    df_v['AI_복기'] = (df_v['종가'] * (1 + pred_past)).shift(1).fillna(df_v['종가'])
    
    f_prices, temp_p = [], df_v['종가'].iloc[-1]
    last_f = df_v[res['features']].iloc[-1:].copy()
    for d in res['f_dates']:
        last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
        temp_p *= (1 + res['model'].predict(res['scaler'].transform(last_f))[0])
        f_prices.append(temp_p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v['종가'], name="실제 시세", line=dict(color='#00CCFF')))
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v['AI_복기'], name="AI 백테스팅", line=dict(color='yellow', dash='dot'), opacity=0.5))
    fig.add_trace(go.Scatter(x=[df_v.index[-1]] + res['f_dates'], y=[df_v['종가'].iloc[-1]] + f_prices, 
                             name="미래 7영업일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    fig.update_layout(template='plotly_dark', height=600, title=f"현재 뉴스 감성지수: {res['news_score']:.1f}")
    st.plotly_chart(fig, use_container_width=True)
