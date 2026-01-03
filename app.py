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
st.set_page_config(page_title="주식 AI v52.0 (Progressive Learning)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge_v52.csv"
NEWS_DB_PATH = "news_rss_cache_v52.csv"
CONFIG_PATH = "global_config_v52.csv"

# --- [1. 초정밀 감성 사전 (긍정 210 / 부정 225)] ---
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

# --- [3. 전방위 지능 누적 관리 로직] ---
def get_progressive_intelligence():
    """모든 지표의 누적된 지능과 최적 유지계수를 반환"""
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            if not df.empty:
                # 축적된 유지계수의 평균 (최소 15% 보존)
                avg_m_coef = max(0.15, df['best_m_coef'].mean())
                # 감성 반영비는 내부적으로만 활용 (660 근처에서 유동적 수렴)
                avg_s_mult = df['best_s_mult'].mean() if 'best_s_mult' in df.columns else 660.0
                return avg_m_coef, avg_s_mult
        except: pass
    return 0.15, 660.0

# --- [4. 메인 분석 및 학습 엔진] ---
st.title("🏛️ 주식 AI v52.0 (전 지표 점진적 성장 모델)")

with st.sidebar:
    st.title("🧠 지능 센터")
    if 'importance' in st.session_state:
        st.write("### AI 지표 판단 비중 (%)")
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    
    avg_m, _ = get_progressive_intelligence()
    st.metric("데이터 유지 지능 (축적치)", f"{avg_m:.4f}")
    
    if st.button("지능 초기화"):
        for f in [CONFIG_PATH, DB_PATH]: 
            if os.path.exists(f): os.remove(f)
        st.rerun()

c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("점진적 지능 통합 분석 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("모든 지표의 상관관계를 과거 지능과 통합하여 학습 중...", expanded=True) as status:
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 뉴스 데이터 수집
            analysis_days = df_raw.index[-60:] 
            daily_scores = {d: get_google_rss_score(s_name, d)[0] for d in analysis_days}
            
            # 기술 지표 생성
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            delta = df['종가'].diff()
            df['RSI'] = (100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / (delta.where(delta < 0, 0).abs().rolling(14).mean() + 1e-9))))).fillna(50)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            # 뉴스 기초값 (3일 누적)
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            base_sentiment = temp_scores.rolling(window=3, min_periods=1).mean()
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # 듀얼 유동 최적화 (지능 누적)
            scaler = StandardScaler()
            knowledge_df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else pd.DataFrame()
            
            min_err = float('inf')
            cur_m, cur_s = get_progressive_intelligence()
            
            # 모든 지표의 최적 조합 탐색
            for m_coef in np.linspace(0.15, 0.5, 8):
                for s_mult in np.linspace(500, 850, 8):
                    df_final['뉴스감성'] = base_sentiment * s_mult
                    X_scaled = scaler.fit_transform(df_final[features])
                    y = df_final['target']
                    
                    if not knowledge_df.empty and all(col in knowledge_df.columns for col in features):
                        X_total = scaler.fit_transform(pd.concat([df_final[features], knowledge_df[features]]))
                        y_total = pd.concat([y, knowledge_df['target']])
                        w = np.concatenate([np.ones(len(df_final)), np.full(len(knowledge_df), m_coef)])
                        m_temp = Ridge(alpha=0.2).fit(X_total, y_total, sample_weight=w)
                    else:
                        m_temp = Ridge(alpha=0.2).fit(X_scaled, y)
                    
                    err = mean_absolute_percentage_error(df_final['종가'], (df_final['종가'] * (1 + m_temp.predict(X_scaled))).shift(1).fillna(df_final['종가']))
                    if err < min_err:
                        min_err = err; best_m = m_coef; best_s = s_mult
            
            # 지능 축적 (모든 지표의 경향성을 파일에 기록)
            pd.DataFrame([[datetime.now(), best_m, best_s]], columns=['date', 'best_m_coef', 'best_s_mult']).to_csv(CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)
            
            # 최종 모델 확정
            df_final['뉴스감성'] = base_sentiment * best_s
            X_scaled = scaler.fit_transform(df_final[features])
            if not knowledge_df.empty:
                X_total = scaler.fit_transform(pd.concat([df_final[features], knowledge_df[features]]))
                y_total = pd.concat([df_final['target'], knowledge_df['target']])
                weights = np.concatenate([np.ones(len(df_final)), np.full(len(knowledge_df), best_m)])
                model = Ridge(alpha=0.2).fit(X_total, y_total, sample_weight=weights)
            else:
                model = Ridge(alpha=0.2).fit(X_scaled, df_final['target'])

            # 결과 저장 및 미래 7일 예측 (실제 끝점에서 연결)
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
                'df': df_final.tail(60), 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'news_score': current_sentiment / best_s, 'AI_복기_V': df_final['AI_복기']
            }
            # 지식 누적 (30거래일치 데이터 저장)
            df_final['stock_code'] = s_code
            df_final[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            success_flag = True
            status.update(label="전 지표 점진적 성장 완료!", state="complete")
            
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [5. 시각화] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['AI_복기'], name="AI 백테스팅", line=dict(color='yellow', dash='dot'), opacity=0.5))
    
    # 그래프 연결 (실제 끝점에서 시작)
    f_dates = [res['df'].index[-1] + timedelta(days=i) for i in range(1, 8)]
    fig.add_trace(go.Scatter(x=[res['df'].index[-1]] + f_dates, y=[res['df']['종가'].iloc[-1]] + res['f_prices'], 
                             name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    st.markdown(f"### 📢 투자 심리 진단: {'🟢 호재' if res['news_score'] >= 1.0 else ('🔴 악재' if res['news_score'] <= -1.0 else '⚖️ 중립')} ({res['news_score']:.2f})")
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)
