import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI v10.0 (Backtest Master)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# --- [데이터 전처리 함수] ---
def prepare_data(df, start_date):
    # VIX 지표 결합
    try:
        vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
        df = df.join(vix).ffill().fillna(20)
    except:
        df['VIX'] = 20
    
    # RSI 지표 계산
    delta = df['종가'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = (100 - (100 / (1 + (gain / loss)))).fillna(50)
    
    # 특징량 생성
    df['target'] = df['종가'].pct_change().shift(-1)
    df['날짜지수'] = np.arange(len(df))
    df['요일'] = df.index.weekday
    df['변동성'] = (df['High'] - df['Low']) / df['종가']
    df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
    
    return df.dropna()

# --- [메인 로직] ---
st.title("📊 2개년 전수 검증 및 최근 3개월 집중 분석")

with st.sidebar:
    st.header("설정")
    stock_code = st.text_input("종목 코드 (6자리)", value="005930")
    analyze_btn = st.button("전수 조사 및 예측 시작", use_container_width=True)

if analyze_btn:
    try:
        # [단계 1] 2년치 데이터 확보
        start_date = KST_NOW - timedelta(days=730)
        df_raw = fdr.DataReader(stock_code, start_date)
        df_raw = df_raw.rename(columns={'Close': '종가', 'Volume': '거래량'})
        
        df = prepare_data(df_raw, start_date)
        features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']
        
        # [단계 2] 전 기간 학습 및 백테스팅 (Fitted Values)
        scaler = StandardScaler()
        X_all = scaler.fit_transform(df[features])
        y_all = df['target']
        
        model = Ridge(alpha=1.0)
        model.fit(X_all, y_all)
        
        # 전체 기간에 대한 AI의 '복기' (1일 후 예측 수익률 기반 종가 재구성)
        df['pred_target'] = model.predict(X_all)
        # 실제 수익률 대신 모델이 예측한 수익률을 적용했을 때의 궤적 계산
        df['AI_복기종가'] = df['종가'] * (1 + df['pred_target'].shift(1))
        df['AI_복기종가'] = df['AI_복기종가'].fillna(df['종가'])

        # 전체 기간 오차율 계산
        total_mape = mean_absolute_percentage_error(df['종가'], df['AI_복기종가'])

        # [단계 3] 향후 7일 미래 예측
        last_p, last_d = df['종가'].iloc[-1], df.index[-1]
        f_prices, f_dates = [], []
        temp_p, last_f = last_p, df[features].iloc[-1:].copy()
        
        for i in range(1, 8):
            last_f['날짜지수'] += 1
            last_f['요일'] = (last_d + timedelta(days=i)).weekday()
            pred = model.predict(scaler.transform(last_f))[0]
            temp_p *= (1 + pred)
            f_prices.append(temp_p)
            f_dates.append(last_d + timedelta(days=i))

        # [단계 4] 그래프용 데이터 필터링 (최근 3개월 = 약 66거래일)
        view_df = df.tail(66)
        
        # [단계 5] 시각화
        st.subheader(f"🔍 모델 성적표: 2년 전수 검증 오차율 {total_mape:.2%}")
        
        fig = go.Figure()
        
        # 1. 실제 가격 (최근 3개월)
        fig.add_trace(go.Scatter(
            x=view_df.index, y=view_df['종가'],
            name="실제 시세", line=dict(color='#00CCFF', width=2)
        ))
        
        # 2. AI의 백테스팅 복기 (최근 3개월)
        fig.add_trace(go.Scatter(
            x=view_df.index, y=view_df['AI_복기종가'],
            name="AI 전수 백테스팅", line=dict(color='rgba(255, 255, 0, 0.5)', dash='dot')
        ))
        
        # 3. 미래 7일 예측
        fig.add_trace(go.Scatter(
            x=[last_d]+f_dates, y=[last_p]+f_prices,
            name="미래 7일 예측", line=dict(color='#FF3366', width=4),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            title=f"{stock_code} 최근 3개월 흐름 및 미래 예측",
            hovermode='x unified',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # 지표 가중치
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 💡 지표별 가중치")
            importance = pd.DataFrame({'지표': features, '영향력': model.coef_})
            st.bar_chart(importance.set_index('지표'), color='#00CCFF')
        with col2:
            st.write("### 📈 예측 상세 (Next 7 Days)")
            pred_res = pd.DataFrame({'날짜': f_dates, '예측가': f_prices})
            st.dataframe(pred_res.style.format({'예측가': '{:,.0f}원'}))

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
