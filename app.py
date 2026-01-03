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
st.set_page_config(page_title="주식 AI v14.0 (Sidebar Intel)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [기존 핵심 로직: 지능 수렴, 영업일 계산, 데이터 처리] ---
def get_converged_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            config_df = pd.read_csv(CONFIG_PATH)
            if not config_df.empty: return config_df['best_coef'].mean()
        except: pass
    return 0.15

def update_global_intelligence(new_coef):
    new_entry = pd.DataFrame([[datetime.now(), new_coef]], columns=['date', 'best_coef'])
    new_entry.to_csv(CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)

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
    features_to_save = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI', 'target']
    new_data = df_curr[features_to_save].tail(25).copy()
    new_data['stock_code'] = str(stock_code)
    if os.path.exists(DB_PATH):
        old_data = pd.read_csv(DB_PATH, dtype={'stock_code': str})
        pd.concat([old_data, new_data]).drop_duplicates().tail(5000).to_csv(DB_PATH, index=False)
    else: new_data.to_csv(DB_PATH, index=False)

def prepare_data(df, start_date):
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
    df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
    return df.dropna()

# --- [사이드바 구성] ---
knowledge_df = load_knowledge()
converged_coef = get_converged_coef()

with st.sidebar:
    st.title("🧠 AI 분석 센터")
    
    # 사이드바 탭 생성
    side_tab1, side_tab2 = st.tabs(["🏛️ 지능 상태", "📊 변수 영향력"])
    
    with side_tab1:
        st.metric("수렴된 감수계수", f"{converged_coef:.4f}")
        if not knowledge_df.empty:
            st.write(f"📚 누적 지식: `{len(knowledge_df)}`개")
            st.info(f"학습 종목군:\n{', '.join(map(str, knowledge_df['stock_code'].unique()))}")
        else:
            st.warning("데이터가 아직 없습니다.")
            
    with side_tab2:
        st.write("### AI 가중치 (Blue Bar)")
        if 'importance' in st.session_state:
            # 사용자가 요청한 파란색 막대 형태의 변수 표시
            st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
        else:
            st.info("분석을 실행하면 변수 영향력이 여기에 표시됩니다.")

# --- [메인 화면] ---
st.title("🏛️ 주식 AI v14.0 (영업일 최적화 & 사이드바 분석)")
stock_code = st.text_input("분석할 종목 코드:", value="005930")

if st.button("통합 분석 및 지능 수렴 시작", use_container_width=True):
    try:
        # 1. 데이터 준비
        start_date = KST_NOW - timedelta(days=730)
        df_raw = fdr.DataReader(stock_code, start_date)
        df_raw = df_raw.rename(columns={'Close': '종가', 'Volume': '거래량'})
        df_curr = prepare_data(df_raw, start_date)
        features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']
        
        X_current = df_curr[features]
        y_current = df_curr['target']
        scaler = StandardScaler()
        X_curr_scaled = scaler.fit_transform(X_current)

        # 2. 감수계수 자율 최적화 및 수렴
        best_local_coef = converged_coef
        if not knowledge_df.empty:
            X_ext = knowledge_df[features]
            y_ext = knowledge_df['target']
            X_total_raw = pd.concat([X_current, X_ext])
            y_total = pd.concat([y_current, y_ext])
            X_total_scaled = scaler.fit_transform(X_total_raw)
            
            min_err = float('inf')
            for c in np.linspace(0.1, 0.3, 5):
                w = np.concatenate([np.ones(len(X_current)), np.full(len(X_ext), c)])
                m = Ridge(alpha=1.0).fit(X_total_scaled, y_total, sample_weight=w)
                err = mean_absolute_percentage_error(y_current, m.predict(X_curr_scaled))
                if err < min_err:
                    min_err = err
                    best_local_coef = c
            
            update_global_intelligence(best_local_coef)
            final_coef = get_converged_coef() # 업데이트된 평균값 사용
            final_weights = np.concatenate([np.ones(len(X_current)), np.full(len(X_ext), final_coef)])
            model = Ridge(alpha=1.0).fit(X_total_scaled, y_total, sample_weight=final_weights)
        else:
            model = Ridge(alpha=1.0).fit(X_curr_scaled, y_current)
            final_coef = 0.15

        # [중요] 사이드바 표시용 변수 가중치 저장 (Session State)
        st.session_state.importance = pd.DataFrame({'지표': features, '가중치': model.coef_})

        # 3. 백테스팅(전수 검증) 및 미래 영업일 예측
        df_curr['pred_target'] = model.predict(X_curr_scaled)
        df_curr['AI_복기종가'] = df_curr['종가'] * (1 + df_curr['pred_target'].shift(1))
        df_curr['AI_복기종가'] = df_curr['AI_복기종가'].fillna(df_curr['종가'])
        mape = mean_absolute_percentage_error(df_curr['종가'], df_curr['AI_복기종가'])

        f_dates = get_next_trading_days(df_curr.index[-1], 7)
        f_prices = []
        temp_p, last_f = df_curr['종가'].iloc[-1], df_curr[features].iloc[-1:].copy()
        for next_date in f_dates:
            last_f['날짜지수'] += 1
            last_f['요일'] = next_date.weekday()
            pred = model.predict(scaler.transform(last_f))[0]
            temp_p *= (1 + pred)
            f_prices.append(temp_p)

        # 4. 시각화 (메인 화면에는 차트와 오차율만 집중)
        st.subheader(f"📊 {stock_code} 분석 리포트 (백테스팅 오차율: {mape:.2%})")
        
        view_df = df_curr.tail(66)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['AI_복기종가'], name=f"AI 백테스팅(복기)", 
                                 line=dict(color='yellow', dash='dot'), opacity=0.4))
        fig.add_trace(go.Scatter(x=[df_curr.index[-1]]+f_dates, y=[df_curr['종가'].iloc[-1]]+f_prices, 
                                 name="미래 7영업일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
        
        fig.update_layout(template='plotly_dark', title=f"감수 수렴 계수: {final_coef:.4f}", height=600)
        st.plotly_chart(fig, use_container_width=True)

        save_knowledge(df_curr, stock_code)
        st.rerun() # 사이드바 가중치 즉시 업데이트를 위해 리런

    except Exception as e:
        st.error(f"오류 발생: {e}")
