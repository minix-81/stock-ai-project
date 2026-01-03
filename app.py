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

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI v26.0 (Fluid Intelligence)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 뉴스 감성 분석 엔진 (기존 400+ 사전 유지)] ---
# (POS_WORDS, NEG_WORDS 및 get_daily_news_score 로직은 이전과 동일하게 유지됩니다)
# ... [생략: 이전 v25.0의 단어 리스트와 뉴스 스코어링 함수 동일] ...

def get_daily_news_score(stock_name, target_date):
    # (이전 v25.0 소스 코드의 로직 적용)
    # ... 
    return score # (생략된 부분은 이전과 동일한 로직으로 작동함)

# --- [2. 유동적 지능 수렴 및 유지계수 로직] ---
def get_fluid_converged_coef():
    """누적된 지능을 바탕으로 유동적인 유지계수 반환 (최소 0.15)"""
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            if not df.empty:
                avg_coef = df['best_coef'].mean()
                return max(0.15, avg_coef) # 최소 15% 이상 유지
        except: pass
    return 0.15

def update_fluid_intelligence(new_coef):
    pd.DataFrame([[datetime.now(), new_coef]], columns=['date', 'best_coef']).to_csv(
        CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)

def get_next_trading_days(start_date, n):
    days = []
    curr = start_date
    while len(days) < n:
        curr += timedelta(days=1)
        if curr.weekday() < 5: days.append(curr)
    return days

# --- [3. 사이드바: 동적 비중(%) 표시] ---
knowledge_df = pd.read_csv(DB_PATH, dtype={'stock_code': str}) if os.path.exists(DB_PATH) else pd.DataFrame()

with st.sidebar:
    st.title("🧠 유동 지능 센터")
    stab1, stab2 = st.tabs(["🏛️ 지능 수렴", "📊 실시간 비중(%)"])
    with stab1:
        st.metric("현재 유동 유지계수", f"{get_fluid_converged_coef():.4f}")
        st.info(f"누적 지식량: {len(knowledge_df)}개 데이터")
    with stab2:
        if 'importance_pct' in st.session_state:
            st.write("### AI 지표별 판단 비중")
            st.bar_chart(st.session_state.importance_pct.set_index('지표'), color='#00CCFF')
            # 상세 수치 표시
            st.table(st.session_state.importance_pct)

# --- [4. 메인 분석 로직] ---
st.title("🏛️ 주식 AI v26.0 (동적 비중 및 유동 학습 모델)")
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("유동 지능 통합 분석 시작", use_container_width=True):
    try:
        with st.status("AI가 지표 비중을 최적화하고 지능을 수렴시키는 중...", expanded=True) as status:
            # 데이터 준비
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 3일 누적 뉴스 및 기술 지표 생성
            analysis_days = df_raw.index[-25:]
            daily_scores = {d: get_daily_news_score(s_name, d) for d in analysis_days}
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 100
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX']
            
            # 머신러닝 학습
            scaler = StandardScaler()
            X_curr = df_final[features]
            y_curr = df_final['target']
            X_curr_scaled = scaler.fit_transform(X_curr)
            
            curr_converged = get_fluid_converged_coef()
            if not knowledge_df.empty:
                X_total_scaled = scaler.fit_transform(pd.concat([X_curr, knowledge_df[features]]))
                y_total = pd.concat([y_current := y_curr, knowledge_df['target']])
                
                # [유동적 계수 최적화] 소수점 단위 정밀 탐색 (0.15 ~ 0.50)
                min_err, best_local = float('inf'), curr_converged
                for c in np.linspace(0.15, 0.5, 20): # 20단계 유동적 탐색
                    w = np.concatenate([np.ones(len(X_curr)), np.full(len(knowledge_df), c)])
                    temp_m = Ridge(alpha=0.2).fit(X_total_scaled, y_total, sample_weight=w)
                    err = mean_absolute_percentage_error(y_curr, temp_m.predict(scaler.transform(X_curr)))
                    if err < min_err: min_err = err; best_local = c
                
                update_fluid_intelligence(best_local)
                final_coef = get_fluid_converged_coef()
                weights = np.concatenate([np.ones(len(X_curr)), np.full(len(knowledge_df), final_coef)])
                model = Ridge(alpha=0.2).fit(X_total_scaled, y_total, sample_weight=weights)
            else:
                model = Ridge(alpha=0.2).fit(X_curr_scaled, y_curr)
                final_coef = 0.15

            # [비중 최적화 결과 계산]
            abs_coef = np.abs(model.coef_)
            pct_importance = (abs_coef / np.sum(abs_coef) * 100)
            st.session_state.importance_pct = pd.DataFrame({'지표': features, '비중(%)': pct_importance.round(1)})

            # 결과 데이터 및 미래 예측
            df_final['AI_복기종가'] = df_final['종가'] * (1 + model.predict(scaler.transform(X_curr)))
            df_final['AI_복기종가'] = df_final['AI_복기종가'].shift(1).fillna(df_final['종가'])
            
            f_dates = get_next_trading_days(df_final.index[-1], 7)
            f_prices, temp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            for d in f_dates:
                last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
                pred = model.predict(scaler.transform(last_f))[0]
                temp_p *= (1 + pred)
                f_prices.append(temp_p)

            st.session_state.result = {
                'name': s_name, 'code': s_code, 'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기종가']),
                'df': df_final.tail(66), 'f_dates': f_dates, 'f_prices': f_prices,
                'news_score': df_final['뉴스감성'].iloc[-1] / 100
            }
            
            # 지식 누적 저장
            df_final['stock_code'] = s_code
            df_final[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            status.update(label="지능 수렴 및 비중 최적화 완료!", state="complete")
            st.rerun()

    except Exception as e: st.error(f"오류: {e}")

# 시각화 영역은 v25.0과 동일하게 유지
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 {res['name']} 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['AI_복기종가'], name="AI 백테스팅(복기)", line=dict(color='yellow', dash='dot'), opacity=0.5))
    fig.add_trace(go.Scatter(x=[res['df'].index[-1]] + res['f_dates'], y=[res['df']['종가'].iloc[-1]] + res['f_prices'], name="미래 7영업일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    st.plotly_chart(fig, use_container_width=True)
