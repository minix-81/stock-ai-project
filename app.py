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
st.set_page_config(page_title="주식 AI v9.0 (Standalone)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 AI 분석 센터")
    st.info("이 버전은 DB 연동 없이 로컬 데이터와 AI 모델만으로 작동합니다.")
    menu = st.radio("메뉴 선택", ["실전 분석 & 백테스팅"], key="nav_v90")

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
    
    # 학습용 특징량 생성
    df['target'] = df['종가'].pct_change().shift(-1)
    df['날짜지수'] = np.arange(len(df))
    df['요일'] = df.index.weekday
    df['변동성'] = (df['High'] - df['Low']) / df['종가']
    df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
    
    return df.dropna()

# --- [메인 페이지: 실전 분석] ---
if menu == "실전 분석 & 백테스팅":
    st.title("📊 2년 학습 및 7일 예측 (백테스팅 포함)")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("AI 분석 시작", width='stretch'):
        try:
            # [단계 1] 2년치 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(stock_code, start_date)
            df_raw = df_raw.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            df_processed = prepare_data(df_raw, start_date)
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']
            
            # [단계 2] 백테스팅 (최근 7일 데이터를 예측해보고 실제와 비교)
            # 최근 7거래일을 제외하고 학습
            train_df = df_processed.iloc[:-7]
            test_df = df_processed.iloc[-7:]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[features])
            y_train = train_df['target']
            
            model_bt = Ridge(alpha=1.0).fit(X_train, y_train)
            
            # 최근 7일 예측 재구성
            bt_prices = []
            current_p = train_df['종가'].iloc[-1]
            last_features = train_df[features].iloc[-1:].copy()
            
            for i in range(len(test_df)):
                last_features['날짜지수'] += 1
                last_features['요일'] = test_df.index[i].weekday()
                # 나머지 지표는 실제값 활용 (백테스팅 정확도 측정용)
                last_features['거래량'] = test_df['거래량'].iloc[i]
                last_features['VIX'] = test_df['VIX'].iloc[i]
                last_features['RSI'] = test_df['RSI'].iloc[i]
                
                pred_return = model_bt.predict(scaler.transform(last_features))[0]
                current_p *= (1 + pred_return)
                bt_prices.append(current_p)

            mape = mean_absolute_percentage_error(test_df['종가'], bt_prices)

            # [단계 3] 실제 미래 7일 예측 (전체 데이터 학습)
            X_all = scaler.fit_transform(df_processed[features])
            y_all = df_processed['target']
            model_final = Ridge(alpha=1.0).fit(X_all, y_all)
            
            last_p, last_d = df_raw['종가'].iloc[-1], df_raw.index[-1]
            f_prices, f_dates = [], []
            temp_p, last_f = last_p, df_processed[features].iloc[-1:].copy()
            
            for i in range(1, 8):
                last_f['날짜지수'] += 1
                last_f['요일'] = (last_d + timedelta(days=i)).weekday()
                pred = model_final.predict(scaler.transform(last_f))[0]
                temp_p *= (1 + pred)
                f_prices.append(temp_p)
                f_dates.append(last_d + timedelta(days=i))

            # [단계 4] 결과 시각화
            st.subheader(f"📈 모델 신뢰도 체크 (최근 7일 백테스팅 오차율: {mape:.2%})")
            
            fig = go.Figure()
            # 1. 실제 시세 (최근 1개월)
            fig.add_trace(go.Scatter(x=df_raw.index[-22:], y=df_raw['종가'].iloc[-22:], 
                                     name="실제 시세", line=dict(color='#00CCFF', width=3)))
            
            # 2. 백테스팅 결과 (최근 7일)
            fig.add_trace(go.Scatter(x=test_df.index, y=bt_prices, 
                                     name="백테스팅(검증)", line=dict(color='yellow', dash='dot', width=2)))
            
            # 3. 향후 7일 예측
            fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, 
                                     name="AI 미래 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            
            fig.update_layout(template='plotly_dark', title=f"{stock_code} 분석 결과 (Backtest & Forecast)", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # 가중치 확인
            with st.expander("💡 AI 모델 지표별 영향력 확인"):
                importance = pd.DataFrame({'변수': features, '가중치': model_final.coef_})
                st.bar_chart(importance.set_index('변수'), color='#00CCFF')

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
