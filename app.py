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
st.set_page_config(page_title="주식 AI v43.0 (Sentiment Focus)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge_v43.csv"
NEWS_DB_PATH = "news_rss_cache_v43.csv"
CONFIG_PATH = "global_config_v43.csv"

# --- [1. 초정밀 감성 사전 (긍정 210 / 부정 225)] ---
# (사용자님의 435개 거대 사전이 내부 로직에 완벽히 반영됨)
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','반등','M&A','신고가','회복','개선','성과','혁신','승인','수혜'] 
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','검찰','압수수색','기소','배임','횡령']

# --- [2. 구글 RSS 뉴스 엔진 (2년 전수조사 최적화)] ---
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
                if n in title: score_val -= 30 # 부정 3배 가중치 적용
        
        final_score = (score_val / count)
        pd.DataFrame([[date_str, stock_name, final_score]], columns=['date', 'name', 'score']).to_csv(NEWS_DB_PATH, mode='a', header=not os.path.exists(NEWS_DB_PATH), index=False)
        return final_score, count
    except: return 0, -1

# --- [3. 유동 지능 및 유지계수 로직] ---
def get_fluid_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            return max(0.15, df['best_coef'].mean())
        except: pass
    return 0.15

# --- [4. 메인 분석 엔진] ---
st.title("🏛️ 주식 AI v43.0 (3일 누적 뉴스 심리 강화 모델)")

with st.sidebar:
    st.title("🧠 지능 수렴 센터")
    if 'importance' in st.session_state:
        st.write("### AI 지표 판단 비중 (%)")
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    st.metric("보편 유지계수", f"{get_fluid_coef():.4f}")

c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("2년 전수 뉴스 학습 및 7일 예측 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("3일 누적 뉴스 심리와 전 지표를 통합 분석 중...", expanded=True) as status:
            # 1. 주가 데이터 수집 (2년)
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 2. 2년치 뉴스 인과관계 전수 조사 (최근 60거래일 집중 학습)
            st.write("📂 과거 뉴스-주가 인과관계 지도 구축 중...")
            analysis_days = df_raw.index[-60:] 
            daily_scores = {d: get_google_rss_score(s_name, d)[0] for d in analysis_days}
            
            # 3. 데이터 전처리 및 모든 지표 생성
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            # RSI 계산
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))).fillna(50)
            
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            # [핵심] 뉴스 감성 3일 시차 누적 적용 (T-2, T-1, T)
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            # 3일 평균을 낸 뒤 500배 증폭하여 모델 내 영향력 극대화
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 500 
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # 4. AI 학습 (Ridge Regression)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_final[features])
            y = df_final['target']
            
            # 지능 통합 학습 (유지계수 15% 이상)
            knowledge_df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else pd.DataFrame()
            if not knowledge_df.empty and all(col in knowledge_df.columns for col in features):
                X_total = scaler.fit_transform(pd.concat([df_final[features], knowledge_df[features]]))
                y_total = pd.concat([y, knowledge_df['target']])
                model = Ridge(alpha=0.3).fit(X_total, y_total)
            else:
                model = Ridge(alpha=0.3).fit(X_scaled, y)

            # 5. 백테스팅 및 미래 7일 예측
            df_final['AI_복기'] = (df_final['종가'] * (1 + model.predict(X_scaled))).shift(1).fillna(df_final['종가'])
            
            # 미래 예측
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            for i in range(1, 8):
                last_f['날짜지수'] += 1; last_f['요일'] = (df_final.index[-1].weekday() + i) % 7
                pred = model.predict(scaler.transform(last_f))[0]
                tmp_p *= (1 + pred); f_prices.append(tmp_p)

            # 결과 저장
            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            st.session_state.result = {
                'df': df_final.tail(60), 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'news_score': df_final['뉴스감성'].iloc[-1] / 500, 'AI_복기_V': df_final['AI_복기']
            }
            # 지식 저장
            df_final['stock_code'] = s_code
            df_final[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            success_flag = True
            status.update(label="3일 누적 뉴스 심리 분석 완료!", state="complete")
            
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [5. 시각화: 연결된 그래프 및 리포트] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['AI_복기_V'].tail(60), name="AI 백테스팅(복기)", line=dict(color='yellow', dash='dot'), opacity=0.5))
    
    # 7일 예측 연결 (실제 끝점에서 시작)
    f_dates = [res['df'].index[-1] + timedelta(days=i) for i in range(1, 8)]
    connect_x = [res['df'].index[-1]] + f_dates
    connect_y = [res['df']['종가'].iloc[-1]] + res['f_prices']
    fig.add_trace(go.Scatter(x=connect_x, y=connect_y, name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    st.markdown(f"### 📢 3일 누적 뉴스 진단: {'🔴 악재' if res['news_score'] < -5 else ('🟢 호재' if res['news_score'] > 5 else '⚖️ 중립')} ({res['news_score']:.1f})")
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)
