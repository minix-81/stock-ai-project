import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# 구글 트렌드 라이브러리 (설치: pip install pytrends)
from pytrends.request import TrendReq

# 1. 페이지 설정
st.set_page_config(page_title="K-Investment AI Pro", layout="wide")
st.title("실전 투자용 AI 패턴 예측기 (7거래일 영업일 기준)")

# 2. 메인 입력창
st.write("---")
col1, col2 = st.columns([2, 1])
with col1:
    stock_code = st.text_input("종목 번호 6자리 (예: 삼성전자 005930):", value="005930")
with col2:
    st.write("") 
    run_button = st.button("영업일 기준 정밀 분석 시작", use_container_width=True)
st.write("---")

if run_button:
    try:
        status = st.empty()
        status.info("최근 2년치 데이터를 학습하여 주말을 제외한 7거래일을 예측 중입니다...")

        # [1] 주가 데이터 수집 (2년)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2) 
        df = fdr.DataReader(stock_code, start_date, end_date)

        if df.empty:
            st.error("데이터 수집 실패! 종목 코드를 확인하세요.")
            st.stop()

        df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
        
        # [2] 구글 트렌드 수집 (에러 대응)
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
            st.warning("구글 트렌드 일시적 제한. 기본 패턴 위주로 분석합니다.")

        # [3] 변수 생성 (양방향 감성 반영)
        df['날짜지수'] = np.arange(len(df))
        df['요일'] = df.index.weekday
        df['변동성'] = (df['High'] - df['Low']) / df['종가']
        df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
        
        # 시계열 래깅: 오늘의 정보로 내일의 등락률 예측
        df['target_return'] = df['종가'].pct_change().shift(-1)
        df_train = df.dropna().copy()

        # [4] AI 학습 (100% 패턴 기반)
        features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', '구글트렌드']
        X = df_train[features]
        y = df_train['target_return']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        # [5] 가중치 시각화
        st.subheader("AI 분석 결과: 2개년 데이터 변수별 기여도 (100%)")
        raw_importance = np.abs(model.coef_)
        total_raw = np.sum(raw_importance)
        ai_weights = (raw_importance / total_raw) * 100.0
        weight_data = pd.DataFrame({'변수': features, '비중(%)': list(ai_weights)})
        st.bar_chart(weight_data.set_index('변수'), color='#00CCFF')

        # [6] 미래 7거래일 예측 (주말 제외 로직)
        last_real_price = df['종가'].iloc[-1]
        last_date = df.index[-1]
        
        future_prices = []
        future_dates = []
        current_price = last_real_price
        last_features = df[features].iloc[-1:].copy()

        # [핵심 수정] 토/일요일을 빼고 7개의 영업일만 찾음
        current_check_date = last_date
        while len(future_prices) < 7:
            current_check_date += timedelta(days=1)
            # 월(0) ~ 금(4)인 경우만 예측값 생성
            if current_check_date.weekday() < 5:
                last_features['날짜지수'] += 1
                last_features['요일'] = current_check_date.weekday()
                
                scaled_input = scaler.transform(last_features)
                pred_return = model.predict(scaled_input)[0]
                
                next_price = current_price * (1 + pred_return)
                future_prices.append(next_price)
                future_dates.append(current_check_date)
                current_price = next_price

        # [7] 주가 예측 그래프 (최근 한 달 집중 시각화)
        st.subheader(f"{stock_code} 최근 1개월 흐름 및 향후 7거래일 정밀 예측")
        fig = go.Figure()
        
        # 최근 30거래일 시세
        display_df = df.iloc[-30:] 
        fig.add_trace(go.Scatter(x=display_df.index, y=display_df['종가'], 
                                 name="실제 시세", line=dict(color='#00CCFF', width=3)))
        
        # 미래 예측 (영업일만 연결)
        fig.add_trace(go.Scatter(
            x=[last_date] + future_dates, 
            y=[last_real_price] + future_prices, 
            name="AI 예측선(영업일)", 
            line=dict(color='#FF3300', dash='dash', width=4),
            mode='lines+markers'
        ))
        
        fig.update_layout(template='plotly_dark', hovermode='x unified', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        status.success("주말을 제외한 향후 7거래일의 정밀 패턴 분석이 완료되었습니다.")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")