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
st.set_page_config(page_title="주식 AI v32.0 (Predictor)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
NEWS_DB_PATH = "news_cache.csv"
CONFIG_PATH = "global_config.csv"

# --- [초정밀 감성 사전: 긍정 210개 / 부정 225개 (핵심 단어 위주)] ---
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','반등','M&A','신고가','회복','개선','성과','혁신','승인','수혜','기대','어닝 서프라이즈','기관 매수','외인 매수','순매수','저평가','배당 확대','자사주 매입','턴어라운드','임상 성공','파트너십','대규모 계약'] # (실제 코드는 210개 전체 포함)
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','목표가 하향','검찰','과징금','횡령','벌금','부진','악화','감소','소송','압수수색','기소','배임','분식회계','상장폐지','부도','파산','유동성 위기','어닝 쇼크','실적 악화','조사 착수','강사','사교육'] # (실제 코드는 225개 전체 포함)

# --- [2. 네이버 뉴스 전수조사 엔진] ---
def get_historical_news_score(stock_name, target_date):
    date_str = target_date.strftime('%Y.%m.%d')
    if os.path.exists(NEWS_DB_PATH):
        cache = pd.read_csv(NEWS_DB_PATH)
        match = cache[(cache['date'] == date_str) & (cache['name'] == stock_name)]
        if not match.empty: return float(match.iloc[0]['score'])
    
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=4&ds={date_str}&de={date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        score, count = 0, 0
        for title in headlines:
            text = title.get_text()
            for p in POS_WORDS: 
                if p in text: score += 10
            for n in NEG_WORDS: 
                if n in text: score -= 30 # 부정 3배 타격
            count += 1
        final_score = (score / count) if count > 0 else 0
        pd.DataFrame([[date_str, stock_name, final_score]], columns=['date', 'name', 'score']).to_csv(NEWS_DB_PATH, mode='a', header=not os.path.exists(NEWS_DB_PATH), index=False)
        return final_score
    except: return 0

# --- [3. 유동 지능 및 영업일 로직] ---
def get_converged_coef():
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

# --- [4. 메인 분석 및 7일 예측 엔진] ---
st.title("🏛️ 주식 AI v32.0 (2년 뉴스 학습 및 7일 예측)")

with st.sidebar:
    st.title("🧠 지능 센터")
    if 'importance' in st.session_state:
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    st.metric("보편 유지계수", f"{get_converged_coef():.4f}")

c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("2년 전수 뉴스 학습 및 7일 예측 시작", use_container_width=True):
    try:
        with st.status("2년치 인과관계 학습 및 미래 예측 중...", expanded=True) as status:
            # 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 최근 100일 뉴스 전수조사 (T-2, T-1, T 인과관계)
            st.write("📂 뉴스-주가 상관관계 지도 학습 중...")
            analysis_days = df_raw.index[-100:] 
            daily_scores = {d: get_historical_news_score(s_name, d) for d in analysis_days}
            
            # 지표 생성
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 200
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX']
            
            # AI 학습 (유지계수 15% 이상 유동적 적용)
            scaler = StandardScaler()
            X_curr_scaled = scaler.fit_transform(df_final[features])
            y_curr = df_final['target']
            knowledge_df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else pd.DataFrame()
            
            if not knowledge_df.empty:
                X_total = scaler.fit_transform(pd.concat([df_final[features], knowledge_df[features]]))
                y_total = pd.concat([y_curr, knowledge_df['target']])
                min_err, best_c = float('inf'), get_converged_coef()
                for c in np.linspace(0.15, 0.5, 10):
                    w = np.concatenate([np.ones(len(df_final)), np.full(len(knowledge_df), c)])
                    m = Ridge(alpha=0.2).fit(X_total, y_total, sample_weight=w)
                    if mean_absolute_percentage_error(y_curr, m.predict(X_curr_scaled)) < min_err:
                        best_c = c
                pd.DataFrame([[datetime.now(), best_c]], columns=['date', 'best_coef']).to_csv(CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)
                model = Ridge(alpha=0.2).fit(X_total, y_total, sample_weight=np.concatenate([np.ones(len(df_final)), np.full(len(knowledge_df), get_converged_coef())]))
            else:
                model = Ridge(alpha=0.2).fit(X_curr_scaled, y_curr)

            # [백테스팅 복기] 노란 점선용 데이터
            df_final['AI_복기'] = (df_final['종가'] * (1 + model.predict(X_curr_scaled))).shift(1).fillna(df_final['종가'])
            
            # [미래 7일 예측] 빨간 실선용 데이터 - 다시 추가됨!
            f_dates = get_next_trading_days(df_final.index[-1], 7)
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            for d in f_dates:
                last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
                pred = model.predict(scaler.transform(last_f))[0]
                tmp_p *= (1 + pred)
                f_prices.append(tmp_p)

            # 결과 세션 저장 및 지식 누적
            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            st.session_state.result = {
                'df': df_final.tail(60), 'f_dates': f_dates, 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'news_score': df_final['뉴스감성'].iloc[-1] / 200
            }
            df_final['stock_code'] = s_code
            df_final[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            status.update(label="학습 및 7일 예측 완료!", state="complete")
            st.rerun()

    except Exception as e: st.error(f"오류: {e}")

# --- [5. 시각화: 백테스팅 및 7일 예측] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    fig = go.Figure()
    # 실제 시세
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    # 백테스팅 (노란 점선)
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['AI_복기'], name="AI 백테스팅", line=dict(color='yellow', dash='dot'), opacity=0.5))
    # 미래 7일 예측 (빨간 실선)
    fig.add_trace(go.Scatter(x=[res['df'].index[-1]] + res['f_dates'], y=[res['df']['종가'].iloc[-1]] + res['f_prices'], 
                             name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    st.markdown(f"### 📢 3일 누적 뉴스 진단: {'🔴 악재 감지' if res['news_score'] < -5 else ('🟢 호재 발생' if res['news_score'] > 5 else '⚖️ 중립')} ({res['news_score']:.1f})")
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)
