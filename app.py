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

# 1. 페이지 설정 및 환경 구축
st.set_page_config(page_title="주식 AI v15.0 (News Crawling)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [뉴스 크롤링 및 감성 분석 엔진] ---
def get_news_sentiment(stock_code, stock_name):
    """네이버 뉴스에서 헤드라인 수집 후 감성 점수(-100~100) 계산"""
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        
        # 단순 키워드 사전 (확장 가능)
        pos_words = ['상승', '호재', '돌파', '영업이익', '최대', '최고', '강세', '매수', '성공', '수주', '기대']
        neg_words = ['하락', '악재', '위기', '감소', '적자', '최저', '약세', '매도', '실패', '우려', '손실']
        
        total_score = 0
        count = 0
        
        for title in headlines:
            text = title.get_text()
            score = 0
            for pw in pos_words:
                if pw in text: score += 10
            for nw in neg_words:
                if nw in text: score -= 10
            total_score += score
            count += 1
        
        if count == 0: return 0
        # -100 ~ 100 사이로 클리핑
        final_score = np.clip(total_score / count * 5, -100, 100)
        return final_score
    except:
        return 0

# --- [영업일 및 지능 관리 (기존 유지)] ---
def get_converged_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            if not df.empty: return df['best_coef'].mean()
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

def load_knowledge():
    if os.path.exists(DB_PATH):
        try: return pd.read_csv(DB_PATH, dtype={'stock_code': str})
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_knowledge(df_curr, stock_code):
    # 이제 뉴스 기반 감성지수도 함께 저장
    features_to_save = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI', 'target']
    new_data = df_curr[features_to_save].tail(25).copy()
    new_data['stock_code'] = str(stock_code)
    if os.path.exists(DB_PATH):
        old_data = pd.read_csv(DB_PATH, dtype={'stock_code': str})
        pd.concat([old_data, new_data]).drop_duplicates().tail(5000).to_csv(DB_PATH, index=False)
    else: new_data.to_csv(DB_PATH, index=False)

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
    
    # [핵심] 뉴스 감성지수 적용
    # 과거 데이터는 주가 기반 감성을 뉴스 감성 대용으로 사용하고,
    # 가장 최근 데이터에 실시간 크롤링 점수를 주입
    df['뉴스감성'] = (df['종가'].pct_change() * 1000).clip(-100, 100) 
    df.iloc[-1, df.columns.get_loc('뉴스감성')] = current_news_score
    
    return df.dropna()

# --- [사이드바 및 메인 화면] ---
knowledge_df = load_knowledge()
converged_coef = get_converged_coef()

with st.sidebar:
    st.title("🧠 뉴스 분석 센터")
    side_tab1, side_tab2 = st.tabs(["🏛️ 지능 수렴", "📊 뉴스 가중치"])
    with side_tab1:
        st.metric("수렴 계수", f"{converged_coef:.4f}")
    with side_tab2:
        if 'importance' in st.session_state:
            st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')

st.title("🏛️ 주식 AI v15.0 (실시간 뉴스 감성 엔진)")
stock_code = st.text_input("종목 코드:", value="005930")
stock_name = st.text_input("종목명 (뉴스 검색용):", value="삼성전자")

if st.button("뉴스 크롤링 및 통합 분석 시작", use_container_width=True):
    try:
        with st.spinner(f"'{stock_name}' 관련 최신 뉴스를 분석 중..."):
            # 1. 뉴스 크롤링
            news_score = get_news_sentiment(stock_code, stock_name)
            st.toast(f"실시간 뉴스 감성 점수: {news_score:.1f}점 수집 완료")

            # 2. 데이터 수집 및 전처리
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(stock_code, start_date)
            df_raw = df_raw.rename(columns={'Close': '종가', 'Volume': '거래량'})
            df_curr = prepare_data(df_raw, start_date, news_score)
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # 3. AI 학습 및 수렴 (전수 백테스팅 포함)
            X_current = df_curr[features]
            y_current = df_curr['target']
            scaler = StandardScaler()
            X_curr_scaled = scaler.fit_transform(X_current)

            if not knowledge_df.empty:
                X_ext = knowledge_df[features]
                y_ext = knowledge_df['target']
                X_total_scaled = scaler.fit_transform(pd.concat([X_current, X_ext]))
                y_total = pd.concat([y_current, y_ext])
                
                # 감수계수 최적화
                min_err = float('inf')
                best_local = converged_coef
                for c in np.linspace(0.1, 0.3, 5):
                    w = np.concatenate([np.ones(len(X_current)), np.full(len(X_ext), c)])
                    m = Ridge(alpha=1.0).fit(X_total_scaled, y_total, sample_weight=w)
                    if mean_absolute_percentage_error(y_current, m.predict(X_curr_scaled)) < min_err:
                        best_local = c
                
                update_global_intelligence(best_local)
                final_coef = get_converged_coef()
                weights = np.concatenate([np.ones(len(X_current)), np.full(len(X_ext), final_coef)])
                model = Ridge(alpha=1.0).fit(X_total_scaled, y_total, sample_weight=weights)
            else:
                model = Ridge(alpha=1.0).fit(X_curr_scaled, y_current)
                final_coef = 0.15

            # 4. 결과 저장
            st.session_state.result = {
                'stock_code': stock_code,
                'news_score': news_score,
                'mape': mean_absolute_percentage_error(df_curr['종가'], df_curr['종가'] * (1 + model.predict(X_curr_scaled))),
                'view_df': df_curr.tail(66),
                'f_dates': get_next_trading_days(df_curr.index[-1], 7),
                'model': model, 'scaler': scaler, 'features': features, 'last_p': df_curr['종가'].iloc[-1]
            }
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': model.coef_})
            save_knowledge(df_curr, stock_code)
            st.rerun()

    except Exception as e:
        st.error(f"오류 발생: {e}")

# --- [결과 출력 영역] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 {res['stock_code']} 분석 (뉴스 점수: {res['news_score']:.1f}, 오차율: {res['mape']:.2%})")
    
    # 미래 예측 계산
    f_prices = []
    temp_p, last_f = res['last_p'], res['view_df'][res['features']].iloc[-1:].copy()
    for d in res['f_dates']:
        last_f['날짜지수'] += 1
        last_f['요일'] = d.weekday()
        # 미래 뉴스 점수는 중립(0)으로 가정하거나 현재 점수 유지 가능
        pred = res['model'].predict(res['scaler'].transform(last_f))[0]
        temp_p *= (1 + pred)
        f_prices.append(temp_p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['view_df'].index, y=res['view_df']['종가'], name="실제 시세", line=dict(color='#00CCFF')))
    fig.add_trace(go.Scatter(x=[res['view_df'].index[-1]]+res['f_dates'], y=[res['last_p']]+f_prices, 
                             name="뉴스 기반 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)
