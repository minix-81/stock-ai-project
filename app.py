import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pytrends.request import TrendReq

# 1. 페이지 설정
st.set_page_config(page_title="K-Investment AI Pro", layout="wide")
st.title("실전 투자용 AI 패턴 분석기 (영업일 기준 7일 예측)")

# 2. 메인 입력창
st.write("---")
col1, col2 = st.columns([2, 1])
with col1:
    stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
with col2:
    st.write("") 
    run_button = st.button("분석 및 예측 시작", use_container_width=True)
st.write("---")

if run_button:
    try:
        status = st.empty()
        status.info("최근 2년 데이터를 학습 중입니다...")

        # [1] 데이터 수집
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2) 
        df = fdr.DataReader(stock_code, start_date, end_date)

        if df.empty:
            st.error("데이터 수집 실패!")
            st.stop()

        df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
        
        # [2] 구글 트렌드 (차단 대비 에러 처리)
        try:
            stocks_krx = fdr.StockListing('KRX')
            stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
            pytrends = TrendReq(hl='ko', tz=360)
            pytrends.build_payload([stock_name], cat=0, timeframe='today 2-y', geo='KR')
            trends_df = pytrends.interest_over_time()
            if not trends_df.empty:
                trends_df = trends_df[[stock_name]].resample('D').interpolate(method='linear')
                df = df.join(trends_df).fillna(method='ffill').fillna(0)
                df = df.rename(columns={stock_name: '구글트렌드'})
            else:
                df['구글트렌드'] = 0
        except:
            df['구글트렌드'] = 0
            st.warning("구글 트렌드 API 제한으로 기본 패턴 분석만 진행합니다.")

        # [3] 변수 생성
        df['날짜지수'] = np.arange(len(df))
        df['요일'] = df.index.weekday
        df['변동성'] = (df['High'] - df['Low']) / df['종가']
        df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
        df['target_return'] = df['종가'].pct_change().shift(-1)
        df_train = df.dropna().copy()

        # [4] AI 학습
        features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', '구글트렌드']
        X = df_train[features]
        y = df_train['target_return']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        # [5] 가중치 시각화
        st.subheader("AI 분석 결과: 변수 기여도 (100%)")
        raw_importance = np.abs(model.coef_)
        ai_weights = (raw_importance / np.sum(raw_importance)) * 100.0
        st.bar_chart(pd.DataFrame({'변수': features, '비중(%)': ai_weights}).set_index('변수'), color='#00CCFF')

        # [6] 미래 7거래일 예측
        last_real_price = df['종가'].iloc[-1]
        last_date = df.index[-1]
        future_prices, future_dates = [], []
        current_price = last_real_price
        last_features = df[features].iloc[-1:].copy()

        check_date = last_date
        while len(future_prices) < 7:
            check_date += timedelta(days=1)
            if check_date.weekday() < 5:
                last_features['날짜지수'] += 1
                last_features['요일'] = check_date.weekday()
                pred_return = model.predict(scaler.transform(last_features))[0]
                current_price *= (1 + pred_return)
                future_prices.append(current_price)
                future_dates.append(check_date)

        # [7] 그래프 출력 (1개월)
        st.subheader(f"최근 1개월 흐름 및 향후 예측")
        fig = go.Figure()
        display_df = df.iloc[-30:] 
        fig.add_trace(go.Scatter(x=display_df.index, y=display_df['종가'], name="실제 시세", line=dict(color='#00CCFF', width=3)))
        fig.add_trace(go.Scatter(x=[last_date] + future_dates, y=[last_real_price] + future_prices, name="AI 예측선", line=dict(color='#FF3300', dash='dash', width=4)))
        fig.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig, use_container_width=True)
        status.success("분석 완료!")

    except Exception as e:
        st.error(f"오류 발생: {e}")
