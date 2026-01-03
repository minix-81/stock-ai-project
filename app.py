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
st.set_page_config(page_title="주식 AI v11.0 (Incremental Learning)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"

# --- [지식 관리 함수] ---
def load_knowledge():
    if os.path.exists(DB_PATH):
        return pd.read_csv(DB_PATH)
    return pd.DataFrame()

def save_knowledge(df_curr, stock_code):
    # 핵심 특징량만 추출하여 저장 (데이터 최적화)
    features_to_save = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI', 'target']
    new_data = df_curr[features_to_save].tail(20).copy() # 각 분석당 최근 20일치 지식 추출
    new_data['stock_code'] = stock_code
    
    if os.path.exists(DB_PATH):
        old_data = pd.read_csv(DB_PATH)
        combined = pd.concat([old_data, new_data]).drop_duplicates().tail(5000) # 최대 5000개 지식 유지
        combined.to_csv(DB_PATH, index=False)
    else:
        new_data.to_csv(DB_PATH, index=False)

# --- [데이터 처리 함수] ---
def prepare_data(df, start_date):
    try:
        vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
        df = df.join(vix).ffill().fillna(20)
    except:
        df['VIX'] = 20
    
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

# --- [메인 화면] ---
st.title("🚀 점진적 성장형 AI v11.0")
st.markdown("분석하는 모든 종목의 지식을 로컬에 축적하고, 다음 분석 시 **감수계수**를 적용해 반영합니다.")

# 지식 상태 표시
knowledge_df = load_knowledge()
if not knowledge_df.empty:
    st.sidebar.success(f"📚 현재 축적된 지식: {len(knowledge_df)}개 데이터 포인트")
    st.sidebar.info(f"분석된 종목들: {', '.join(knowledge_df['stock_code'].unique())}")
else:
    st.sidebar.warning("📚 축적된 지식이 없습니다. 첫 분석을 시작하세요.")

stock_code = st.text_input("종목 코드 입력 (6자리):", value="005930")
decay_coef = st.slider("감수계수 (타 종목 지식 반영 비율):", 0.0, 0.5, 0.15, help="높을수록 과거/타 종목의 지식을 더 많이 반영합니다.")

if st.button("AI 지능 통합 분석 시작", use_container_width=True):
    try:
        # 1. 현재 종목 데이터 수집 (2년)
        start_date = KST_NOW - timedelta(days=730)
        df_raw = fdr.DataReader(stock_code, start_date)
        df_raw = df_raw.rename(columns={'Close': '종가', 'Volume': '거래량'})
        df_curr = prepare_data(df_raw, start_date)
        
        features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']
        
        # 2. 지식 통합 (현재 데이터 + 로컬 DB 지식)
        X_current = df_curr[features]
        y_current = df_curr['target']
        weights = np.ones(len(X_current)) # 현재 종목 가중치 1.0
        
        if not knowledge_df.empty:
            X_ext = knowledge_df[features]
            y_ext = knowledge_df['target']
            # 감수계수 적용 (타 종목 데이터는 상대적으로 낮은 가중치 부여)
            ext_weights = np.full(len(X_ext), decay_coef)
            
            X_total = pd.concat([X_current, X_ext])
            y_total = pd.concat([y_current, y_ext])
            weights_total = np.concatenate([weights, ext_weights])
            st.write(f"💡 타 종목의 지식 {len(X_ext)}개를 반영하여 학습 중...")
        else:
            X_total, y_total, weights_total = X_current, y_current, weights

        # 3. 모델 학습 및 전수 백테스팅
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_total)
        model = Ridge(alpha=1.0).fit(X_scaled, y_total, sample_weight=weights_total)
        
        # 현재 종목 복기
        df_curr['pred_target'] = model.predict(scaler.transform(X_current))
        df_curr['AI_복기종가'] = df_curr['종가'] * (1 + df_curr['pred_target'].shift(1))
        df_curr['AI_복기종가'] = df_curr['AI_복기종가'].fillna(df_curr['종가'])
        mape = mean_absolute_percentage_error(df_curr['종가'], df_curr['AI_복기종가'])

        # 4. 미래 7일 예측
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

        # 5. 시각화 (최근 3개월)
        view_df = df_curr.tail(66)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
        fig.add_trace(go.Scatter(x=view_df.index, y=view_df['AI_복기종가'], name="AI 통합 지능 복기", line=dict(color='yellow', dash='dot', opacity=0.5)))
        fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
        
        fig.update_layout(template='plotly_dark', title=f"{stock_code} 분석 및 통합 학습 결과", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # 6. 지식 저장 (빅데이터 성장)
        save_knowledge(df_curr, stock_code)
        st.success(f"✅ 분석 완료! '{stock_code}'의 특징이 로컬 지식 창고에 저장되었습니다.")

        # 지표 가중치 표시
        importance = pd.DataFrame({'지표': features, '가중치': model.coef_}).set_index('지표')
        st.bar_chart(importance, color='#00CCFF')

    except Exception as e:
        st.error(f"오류 발생: {e}")
