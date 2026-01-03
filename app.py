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

# 1. 환경 설정
st.set_page_config(page_title="주식 AI v52.3 (Fluid Intelligence)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge_v52.csv"
NEWS_DB_PATH = "news_rss_cache_v52.csv"
CONFIG_PATH = "global_config_v52.csv"

# --- [1. 감성 사전 및 지능 로직] ---
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','목표가 상향','우상향','반등','M&A','신고가','어닝 서프라이즈','기관 매수','외인 매수','순매수','저평가','배당 확대','자사주 매입'] 
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','검찰','압수수색','기소','배임','횡령','사법 리스크','고소','피소','수사']

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

def get_progressive_intelligence():
    """과거에 축적된 지능의 평균치를 반환하되, 하한선을 0.25로 보장"""
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            if not df.empty: return max(0.25, df['best_m_coef'].mean())
        except: pass
    return 0.25

def get_learning_volume():
    if os.path.exists(DB_PATH):
        try:
            df = pd.read_csv(DB_PATH)
            return len(df)
        except: pass
    return 0

# --- [2. 메인 분석 엔진] ---
st.title("🏛️ 주식 AI v52.3 (유동적 지능 및 3주 집중 모델)")

with st.sidebar:
    st.title("🧠 지능 센터")
    if 'importance' in st.session_state:
        st.write("### AI 지표 판단 비중 (%)")
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    # 유동적으로 변화하는 축적 지능 표시
    st.metric("데이터 유지 지능 (축적치)", f"{get_progressive_intelligence():.4f}")
    st.metric("누적 학습 데이터량", f"{get_learning_volume():,} pt")

c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("2년 통합 분석 및 지능 축적 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("유지 지능의 하한선을 0.25로 설정하여 유동적 학습 중...", expanded=True) as status:
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 최근 뉴스 조사 (60일)
            analysis_days = df_raw.index[-60:] 
            daily_scores = {d: get_google_rss_score(s_name, d)[0] for d in analysis_days}
            
            # 모든 7대 지표 생성
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            delta = df['종가'].diff()
            df['RSI'] = (100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / (delta.where(delta < 0, 0).abs().rolling(14).mean() + 1e-9))))).fillna(50)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            # 뉴스 심리 530배 반영 및 3일 누적 인과관계
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 530 
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_final[features])
            y = df_final['target']
            knowledge_df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else pd.DataFrame()
            
            # [유동적 지능 탐색] 하한선 0.25 적용
            accumulated_coef = get_progressive_intelligence()
            if not knowledge_df.empty and all(col in knowledge_df.columns for col in features):
                X_total = scaler.fit_transform(pd.concat([df_final[features], knowledge_df[features]]))
                y_total = pd.concat([y, knowledge_df['target']])
                
                min_err, best_c = float('inf'), accumulated_coef
                # 0.25부터 0.6까지 유동적으로 탐색
                for c in np.linspace(0.25, 0.6, 15):
                    w = np.concatenate([np.ones(len(df_final)), np.full(len(knowledge_df), c)])
                    m_temp = Ridge(alpha=0.2).fit(X_total, y_total, sample_weight=w)
                    if mean_absolute_percentage_error(y, m_temp.predict(X_scaled)) < min_err:
                        best_c = c
                
                # 새로운 최적 지능을 저장하여 축적
                pd.DataFrame([[datetime.now(), best_c]], columns=['date', 'best_m_coef']).to_csv(CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)
                model = Ridge(alpha=0.2).fit(X_total, y_total, sample_weight=np.concatenate([np.ones(len(df_final)), np.full(len(knowledge_df), get_progressive_intelligence())]))
            else:
                model = Ridge(alpha=0.2).fit(X_scaled, y)

            # 결과 도출 및 미래 7일 예측
            df_final['AI_복기'] = (df_final['종가'] * (1 + model.predict(X_scaled))).shift(1).fillna(df_final['종가'])
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            current_sentiment = df_final['뉴스감성'].iloc[-1] 
            for i in range(1, 8):
                last_f['날짜지수'] += 1; last_f['요일'] = (df_final.index[-1].weekday() + i) % 7
                last_f['뉴스감성'] = current_sentiment 
                pred = model.predict(scaler.transform(last_f))[0]
                tmp_p *= (1 + pred); f_prices.append(tmp_p)

            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            st.session_state.result = {
                'df': df_final.tail(21), # 가시적 그래프는 최근 3주
                'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'news_score': current_sentiment / 530, 'AI_복기_V': df_final['AI_복기']
            }
            # 학습량 누적을 위한 데이터 저장
            df_final['stock_code'] = s_code
            df_final[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            success_flag = True
            status.update(label="0.25 하한선 기반 유동 지능 수렴 완료!", state="complete")
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [3. 시각화 영역] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 리포트 (최근 3주 집중 및 유동 지능 모델)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['AI_복기_V'].loc[res['df'].index], name="AI 백테스팅", line=dict(color='yellow', dash='dot'), opacity=0.5))
    f_dates = [res['df'].index[-1] + timedelta(days=i) for i in range(1, 8)]
    fig.add_trace(go.Scatter(x=[res['df'].index[-1]] + f_dates, y=[res['df']['종가'].iloc[-1]] + res['f_prices'], 
                             name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    st.markdown(f"### 📢 투자 심리 진단: {'🟢 호재' if res['news_score'] >= 1.0 else ('🔴 악재' if res['news_score'] <= -1.0 else '⚖️ 중립')} ({res['news_score']:.2f})")
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)
