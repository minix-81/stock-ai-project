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

# 1. 환경 설정 (2026-01-04 실시간 대응)
st.set_page_config(page_title="주식 AI v53.5 (Full Report)", layout="wide")
KST_NOW = datetime.utcnow() + timedelta(hours=9)
TODAY_STR = KST_NOW.strftime('%Y-%m-%d')
DB_PATH = "stock_knowledge_v53.csv"
NEWS_DB_PATH = "news_rss_cache_v53.csv"
CONFIG_PATH = "global_config_v53.csv"

# --- [1. 지식 사전 및 뉴스 엔진] ---
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
st.title(f"🏛️ 주식 AI v53.5 (사이드바 및 백테스팅 복구 모델)")

# 사이드바 구성
with st.sidebar:
    st.header("🧠 지능 센터")
    if 'importance' in st.session_state:
        st.write("### AI 지표 판단 비중 (%)")
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    st.metric("데이터 유지 지능 (축적치)", f"{get_progressive_intelligence():.4f}")
    st.metric("누적 학습 데이터량", f"{get_learning_volume():,} pt")
    st.info("실시간 데이터와 과거 지능이 통합되어 분석됩니다.")

c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("실전 데이터 분석 시작", use_container_width=True):
    try:
        with st.status("AI 학습 및 백테스팅 분석 중...", expanded=True) as status:
            # 1. 최신 시세 수집
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 2. 뉴스 및 지표 생성
            analysis_days = df_raw.index[-60:].tolist()
            if pd.Timestamp(TODAY_STR) not in [d.date() for d in analysis_days]: analysis_days.append(pd.Timestamp(TODAY_STR))
            daily_scores = {d: get_google_rss_score(s_name, d)[0] for d in analysis_days}
            
            vix = fdr.DataReader('^VIX', start_date).rename(columns={'Close': 'VIX'})[['VIX']]
            df = df_raw.join(vix).ffill().fillna(20)
            
            # 요일 원-핫 인코딩 (굴곡 복원)
            df['요일'] = df.index.weekday
            weekday_dummies = pd.get_dummies(df['요일'], prefix='day')
            df = pd.concat([df, weekday_dummies], axis=1)
            weekday_cols = [col for col in df.columns if col.startswith('day_')]
            
            delta = df['종가'].diff()
            df['RSI'] = (100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / (delta.where(delta < 0, 0).abs().rolling(14).mean() + 1e-9))))).fillna(50)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df))
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): 
                if d in temp_scores.index: temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 530 
            
            features = ['날짜지수', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI'] + weekday_cols
            df_train = df.dropna(subset=['target'] + features)
            df_latest = df.iloc[-1:]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(df_train[features])
            y_train = df_train['target']
            
            # 3. AI 학습 및 지능 수렴
            knowledge_df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else pd.DataFrame()
            acc_coef = get_progressive_intelligence()
            if not knowledge_df.empty:
                k_df_filtered = knowledge_df[knowledge_df['stock_code'] != str(s_code)]
                if not k_df_filtered.empty:
                    X_total = scaler.fit_transform(pd.concat([df_train[features], k_df_filtered[features]]))
                    y_total = pd.concat([y_train, k_df_filtered['target']])
                    min_err, best_c = float('inf'), acc_coef
                    for c in np.linspace(0.25, 0.85, 20):
                        w = np.concatenate([np.ones(len(df_train)), np.full(len(k_df_filtered), c)])
                        m_temp = Ridge(alpha=0.1).fit(X_total, y_total, sample_weight=w)
                        err = mean_absolute_percentage_error(y_train, m_temp.predict(X_train)) + (c * 0.001)
                        if err < min_err: min_err = err; best_c = c
                    pd.DataFrame([[datetime.now(), best_c]], columns=['date', 'best_m_coef']).to_csv(CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)
                    model = Ridge(alpha=0.1).fit(X_total, y_total, sample_weight=np.concatenate([np.ones(len(df_train)), np.full(len(k_df_filtered), get_progressive_intelligence())]))
                else: model = Ridge(alpha=0.1).fit(X_train, y_train)
            else: model = Ridge(alpha=0.1).fit(X_train, y_train)

            # 4. 백테스팅 및 미래 예측
            # 백테스팅(AI 복기) 데이터 산출
            df_train['AI_복기'] = (df_train['종가'] * (1 + model.predict(X_train))).shift(1).fillna(df_train['종가'])
            
            f_prices, tmp_p = [], df_latest['종가'].iloc[-1]
            last_f = df_latest[features].copy()
            current_sentiment = df_latest['뉴스감성'].iloc[-1] 
            for i in range(1, 8):
                last_f['날짜지수'] += 1
                new_day = (df_latest.index[-1].weekday() + i) % 7
                for col in weekday_cols: last_f[col] = 1 if col == f'day_{new_day}' else 0
                pred = model.predict(scaler.transform(last_f))[0]
                if (current_sentiment / 530) <= -1.0 and pred > 0: pred *= 0.1 # 심리 필터
                tmp_p *= (1 + pred); f_prices.append(tmp_p)

            # 지표 비중 저장
            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            
            st.session_state.result = {
                'df': df.tail(21), 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_train['종가'], df_train['AI_복기']),
                'news_score': current_sentiment / 530, 'AI_복기_V': df_train['AI_복기'],
                'last_date': df_latest.index[-1].strftime('%Y-%m-%d'), 'last_close': df_latest['종가'].iloc[-1]
            }
            # 데이터 축적
            df_train['stock_code'] = s_code
            df_train[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            success_flag = True
            status.update(label="백테스팅 및 분석 완료!", state="complete")
            st.rerun()
    except Exception as e: st.error(f"오류: {e}")

# --- [3. 분석 리포트 영역] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    st.write(f"🏷️ 기준일: **{res['last_date']}** | **현재가:** {int(res['last_close']):,}원")
    
    fig = go.Figure()
    # 1. 실제 시세
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    # 2. 백테스팅(AI 복기) 다시 살려냄
    fig.add_trace(go.Scatter(x=res['AI_복기_V'].index[-21:], y=res['AI_복기_V'].tail(21), name="AI 백테스팅", line=dict(color='yellow', dash='dot'), opacity=0.7))
    # 3. 미래 예측
    f_dates = [pd.to_datetime(res['last_date']) + timedelta(days=i) for i in range(1, 8)]
    fig.add_trace(go.Scatter(x=[pd.to_datetime(res['last_date'])] + f_dates, y=[res['last_close']] + res['f_prices'], 
                             name="미래 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    st.markdown(f"### 📢 투자 심리: {'🟢 호재' if res['news_score'] >= 1.0 else ('🔴 악재' if res['news_score'] <= -1.0 else '⚖️ 중립')} ({res['news_score']:.2f})")
    fig.update_layout(template='plotly_dark', height=550)
    st.plotly_chart(fig, use_container_width=True)
