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

# 1. 환경 설정 (2026-01-04 실시간 동기화)
st.set_page_config(page_title="주식 AI v53.2 (Ultimate Sync)", layout="wide")
KST_NOW = datetime.utcnow() + timedelta(hours=9)
TODAY_STR = KST_NOW.strftime('%Y-%m-%d')
DB_PATH = "stock_knowledge_v53.csv"
NEWS_DB_PATH = "news_rss_cache_v53.csv"
CONFIG_PATH = "global_config_v53.csv"

# --- [1. 최신 정보 수집 엔진] ---
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','목표가 상향','우상향','반등','M&A','신고가','어닝 서프라이즈','기관 매수','외인 매수','순매수','저평가','배당 확대','자사주 매입'] 
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','검찰','압수수색','기소','배임','횡령','사법 리스크','고소','피소','수사']

def get_google_rss_score(stock_name, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    next_date_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    if date_str != TODAY_STR and os.path.exists(NEWS_DB_PATH):
        try:
            cache = pd.read_csv(NEWS_DB_PATH)
            match = cache[(cache['date'] == date_str) & (cache['name'] == stock_name)]
            if not match.empty: return float(match.iloc[0]['score']), 1
        except: pass
    url = f"https://news.google.com/rss/search?q={stock_name}+주가+after:{date_str}+before:{next_date_str}&hl=ko&gl=KR&ceid=KR:ko"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.find_all("item")
        count = len(items); score_val = 0
        if count == 0: return 0, 0
        for item in items:
            title = item.title.text
            for p in POS_WORDS:
                if p in title: score_val += 10
            for n in NEG_WORDS:
                if n in title: score_val -= 30 
        final_score = (score_val / count)
        if date_str != TODAY_STR:
            pd.DataFrame([[date_str, stock_name, final_score]], columns=['date', 'name', 'score']).to_csv(NEWS_DB_PATH, mode='a', header=not os.path.exists(NEWS_DB_PATH), index=False)
        return final_score, count
    except: return 0, -1

def get_progressive_intelligence():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            if not df.empty: return max(0.25, df['best_m_coef'].mean())
        except: pass
    return 0.25

def get_learning_volume():
    if os.path.exists(DB_PATH):
        try: return len(pd.read_csv(DB_PATH))
        except: pass
    return 0

# --- [2. 메인 분석 엔진] ---
st.title(f"🏛️ 주식 AI v53.2 (영업일 자동 추적 분석)")
st.info(f"📅 분석 기준 일시: {KST_NOW.strftime('%Y-%m-%d %H:%M')} (KST)")

with st.sidebar:
    st.title("🧠 지능 센터")
    if 'importance' in st.session_state:
        st.write("### AI 지표 판단 비중 (%)")
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    st.metric("데이터 유지 지능 (축적치)", f"{get_progressive_intelligence():.4f}")
    st.metric("누적 학습 데이터량", f"{get_learning_volume():,} pt")

c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("영업일 정밀 동기화 분석 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("AI 학습 및 최신 인과관계 분석 중...", expanded=True) as status:
            # [핵심 수정] 종료일을 명시하지 않고 가장 최신 거래일까지 강제 수집
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 마지막 데이터 날짜 확인
            last_actual_date = df_raw.index[-1]
            
            # 뉴스 분석 (마지막 데이터일 기준 전수조사)
            analysis_days = df_raw.index[-60:].tolist()
            if pd.Timestamp(TODAY_STR) not in [d.date() for d in analysis_days]:
                analysis_days.append(pd.Timestamp(TODAY_STR))
            
            daily_scores = {d: get_google_rss_score(s_name, d)[0] for d in analysis_days}
            
            # 7대 지표 생성
            vix = fdr.DataReader('^VIX', start_date).ffill().fillna(20).rename(columns={'Close': 'VIX'})[['VIX']]
            df = df_raw.join(vix).ffill().fillna(20)
            delta = df['종가'].diff()
            df['RSI'] = (100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / (delta.where(delta < 0, 0).abs().rolling(14).mean() + 1e-9))))).fillna(50)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            # 뉴스 심리 530배 및 3일 누적
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): 
                if d in temp_scores.index: temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 530 
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_final[features])
            y = df_final['target']
            
            # AI 지능 수렴 (0.25 하한선 및 유동성 확보)
            knowledge_df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else pd.DataFrame()
            acc_coef = get_progressive_intelligence()
            if not knowledge_df.empty:
                k_df_filtered = knowledge_df[knowledge_df['stock_code'] != str(s_code)]
                if not k_df_filtered.empty:
                    X_total = scaler.fit_transform(pd.concat([df_final[features], k_df_filtered[features]]))
                    y_total = pd.concat([y, k_df_filtered['target']])
                    min_err, best_c = float('inf'), acc_coef
                    for c in np.linspace(0.25, 0.85, 20):
                        w = np.concatenate([np.ones(len(df_final)), np.full(len(k_df_filtered), c)])
                        m_temp = Ridge(alpha=0.5).fit(X_total, y_total, sample_weight=w)
                        err = mean_absolute_percentage_error(y, m_temp.predict(X_scaled)) + (c * 0.001)
                        if err < min_err: min_err = err; best_c = c
                    pd.DataFrame([[datetime.now(), best_c]], columns=['date', 'best_m_coef']).to_csv(CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)
                    model = Ridge(alpha=0.5).fit(X_total, y_total, sample_weight=np.concatenate([np.ones(len(df_final)), np.full(len(k_df_filtered), get_progressive_intelligence())]))
                else: model = Ridge(alpha=0.5).fit(X_scaled, y)
            else: model = Ridge(alpha=0.5).fit(X_scaled, y)

            # 결과 도출 및 미래 예측 (심리 동조화)
            df_final['AI_복기'] = (df_final['종가'] * (1 + model.predict(X_scaled))).shift(1).fillna(df_final['종가'])
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            current_sentiment = df_final['뉴스감성'].iloc[-1] 
            for i in range(1, 8):
                last_f['날짜지수'] += 1; last_f['요일'] = (df_final.index[-1].weekday() + i) % 7
                last_f['뉴스감성'] = current_sentiment 
                pred = model.predict(scaler.transform(last_f))[0]
                if (current_sentiment / 530) <= -2.0 and pred > 0: pred *= 0.3
                tmp_p *= (1 + pred); f_prices.append(tmp_p)

            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (np.abs(model.coef_) / np.sum(np.abs(model.coef_)) * 100).round(1)})
            st.session_state.result = {
                'df': df_final.tail(21), 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'news_score': current_sentiment / 530, 'AI_복기_V': df_final['AI_복기'],
                'last_date': last_actual_date.strftime('%Y-%m-%d'), 'last_close': df_final['종가'].iloc[-1]
            }
            # 데이터 축적
            df_final['stock_code'] = s_code
            df_final[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            success_flag = True
            status.update(label="최신 영업일 기준 분석 완료!", state="complete")
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [3. 분석 리포트] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 결과 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    st.markdown(f"**🏷️ 마지막 데이터 기준일:** {res['last_date']} | **종가:** {int(res['last_close']):,}원")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="최근 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['AI_복기_V'].loc[res['df'].index], name="AI 복기", line=dict(color='yellow', dash='dot'), opacity=0.5))
    f_dates = [res['df'].index[-1] + timedelta(days=i) for i in range(1, 8)]
    fig.add_trace(go.Scatter(x=[res['df'].index[-1]] + f_dates, y=[res['df']['종가'].iloc[-1]] + res['f_prices'], 
                             name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    st.markdown(f"### 📢 투자 심리 진단: {'🟢 호재' if res['news_score'] >= 1.0 else ('🔴 악재' if res['news_score'] <= -1.0 else '⚖️ 중립')} ({res['news_score']:.2f})")
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)
