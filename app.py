import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
import os
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI v12.1 (Bug Fixed)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"

# --- [지식 관리 함수 수정] ---
def load_knowledge():
    if os.path.exists(DB_PATH):
        try: 
            # stock_code를 읽을 때부터 문자열로 고정 (앞자리 0 보존 및 타입 에러 방지)
            return pd.read_csv(DB_PATH, dtype={'stock_code': str})
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_knowledge(df_curr, stock_code):
    features_to_save = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI', 'target']
    new_data = df_curr[features_to_save].tail(25).copy()
    new_data['stock_code'] = str(stock_code) # 저장 시에도 문자열로 강제 변환
    if os.path.exists(DB_PATH):
        try:
            old_data = pd.read_csv(DB_PATH, dtype={'stock_code': str})
            pd.concat([old_data, new_data]).drop_duplicates().tail(5000).to_csv(DB_PATH, index=False)
        except: new_data.to_csv(DB_PATH, index=False)
    else: new_data.to_csv(DB_PATH, index=False)

# --- [나머지 데이터 처리 함수 (기존과 동일)] ---
def prepare_data(df, start_date):
    try:
        vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
        df = df.join(vix).ffill().fillna(20)
    except: df['VIX'] = 20
    delta = df['종가'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = (100 - (100 / (1 + (gain / loss)))).fillna(50)
    df['target'] = df['종가'].pct_change().shift(-1)
    df['날짜지수'] = np.arange(len(df))
    df['요일'] = df.index.weekday
    df['변동성'] = (df['High'] - df['Low']) / df['종가']
    df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
    return df.dropna()

# --- [메인 로직] ---
st.title("🤖 자율 성장형 AI v12.1")
st.markdown("자율 최적화 감수계수 및 백테스팅 엔진이 탑재된 최종 안정화 버전입니다.")

knowledge_df = load_knowledge()
with st.sidebar:
    st.header("📚 지능 저장소 정보")
    if not knowledge_df.empty:
        st.success(f"누적 데이터: {len(knowledge_df)}개")
        # [에러 수정 포인트] map(str, ...)을 사용하여 안전하게 결합
        codes = map(str, knowledge_df['stock_code'].unique())
        st.info(f"학습 종목: {', '.join(codes)}")
    else:
        st.warning("저장된 지식이 없습니다.")

stock_code = st.text_input("분석할 종목 코드:", value="005930")

if st.button("AI 자율 최적화 분석 시작", use_container_width=True):
    try:
        # 데이터 준비
        start_date = KST_NOW - timedelta(days=730)
        df_raw = fdr.DataReader(stock_code, start_date)
        df_raw = df_raw.rename(columns={'Close': '종가', 'Volume': '거래량'})
        df_curr = prepare_data(df_raw, start_date)
        features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']
        
        X_current = df_curr[features]
        y_current = df_curr['target']
        scaler = StandardScaler()
        X_curr_scaled = scaler.fit_transform(X_current)

        # 감수계수 자율 최적화 및 백테스팅 시뮬레이션
        best_coef = 0.1
        min_error = float('inf')
        
        if not knowledge_df.empty:
            X_ext = knowledge_df[features]
            y_ext = knowledge_df['target']
            X_total_raw = pd.concat([X_current, X_ext])
            y_total = pd.concat([y_current, y_ext])
            X_total_scaled = scaler.fit_transform(X_total_raw)
            
            for coef in [0.1, 0.15, 0.2, 0.25, 0.3]:
                weights = np.concatenate([np.ones(len(X_current)), np.full(len(X_ext), coef)])
                temp_model = Ridge(alpha=1.0).fit(X_total_scaled, y_total, sample_weight=weights)
                
                pred = temp_model.predict(X_curr_scaled)
                error = mean_absolute_percentage_error(y_current, pred)
                
                if error < min_error:
                    min_error = error
                    best_coef = coef
            
            st.write(f"🎯 최적 감수계수 발견: `{best_coef}` (검증 오차: {min_error:.4f})")
            final_weights = np.concatenate([np.ones(len(X_current)), np.full(len(X_ext), best_coef)])
            model = Ridge(alpha=1.0).fit(X_total_scaled, y_total, sample_weight=final_weights)
        else:
            st.write("💡 초기 학습 단계입니다.")
            model = Ridge(alpha=1.0).fit(X_curr_scaled, y_current)

        # 분석 및 시각화
        df_curr['pred_target'] = model.predict(X_curr_scaled)
        df_curr['AI_복기종가'] = df_curr['종가'] * (1 + df_curr['pred_target'].shift(1))
        df_curr['AI_복기종가'] = df_curr['AI_복기종가'].fillna(df_curr['종가'])
        
        last_p, last_d = df_curr['종가'].iloc[-1], df_curr.index[-1]
        f_prices, f_dates = [], []
        temp_p, last_f = last_p, df_curr[features].iloc[-1:].copy()
        for i in range(1, 8):
            last_f['날짜지수'] += 1
            last_f['요일'] = (last_d + timedelta(days=i)).weekday()
            pred = model.predict(scaler.transform(last_f))[0]
            temp_p *= (1 + pred)
            f_prices.append(temp_p)
            f_dates.append(last_d + timedelta(days=i))

        view_df = df_curr.tail(66)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['AI_복기종가'], name=f"AI 백테스팅(coef:{best_coef})", 
                                 line=dict(color='yellow', dash='dot'), opacity=0.4))
        fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="미래 7일 예측", 
                                 line=dict(color='#FF3366', width=4), mode='lines+markers'))
        
        fig.update_layout(template='plotly_dark', title=f"{stock_code} 분석 결과", height=500)
        st.plotly_chart(fig, use_container_width=True)

        save_knowledge(df_curr, stock_code)
        st.success(f"✅ 지식 저장 완료!")

    except Exception as e:
        st.error(f"오류 발생: {e}")
