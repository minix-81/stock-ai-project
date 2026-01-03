import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# 1. 페이지 설정 (가장 먼저 와야 함)
st.set_page_config(page_title="주식 AI v10.1", layout="wide")

# 한국 시간 설정
KST_NOW = datetime.now() + timedelta(hours=9)

# --- [데이터 처리 함수] ---
@st.cache_data(ttl=3600)  # 1시간 동안 결과 캐싱 (성능 향상)
def get_stock_data(stock_code, days=730):
    start_date = KST_NOW - timedelta(days=days)
    df = fdr.DataReader(stock_code, start_date)
    if df.empty:
        return None
    df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
    
    # VIX (공포지수) 추가
    try:
        vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
        df = df.join(vix).ffill().fillna(20)
    except:
        df['VIX'] = 20

    # RSI 지표
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

# --- [메인 화면 구성] ---
st.title("📈 주식 AI: 2년 전수 검증 및 7일 예측")
st.markdown("수파베이스 연결을 제거하고 백테스팅 기능을 강화한 단독 실행 버전입니다.")

# 🔍 검색창을 메인 화면 상단으로 배치
col_input, col_info = st.columns([1, 2])
with col_input:
    stock_code = st.text_input("종목 코드 6자리를 입력하세요:", value="005930", help="예: 삼성전자(005930), SK하이닉스(000660)")
    analyze_btn = st.button("🚀 AI 분석 시작", use_container_width=True)

if analyze_btn:
    with st.spinner('AI가 지난 2년의 데이터를 전수 조사 중입니다...'):
        try:
            # 1. 데이터 로드
            df = get_stock_data(stock_code)
            if df is None:
                st.error("종목 데이터를 가져올 수 없습니다. 코드를 확인해 주세요.")
            else:
                features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']
                
                # 2. 모델 학습 (전수 백테스팅)
                scaler = StandardScaler()
                X_all = scaler.fit_transform(df[features])
                y_all = df['target']
                
                model = Ridge(alpha=1.0)
                model.fit(X_all, y_all)
                
                # AI 복기 데이터 생성
                df['pred_target'] = model.predict(X_all)
                df['AI_복기종가'] = df['종가'] * (1 + df['pred_target'].shift(1))
                df['AI_복기종가'] = df['AI_복기종가'].fillna(df['종가'])
                total_mape = mean_absolute_percentage_error(df['종가'], df['AI_복기종가'])

                # 3. 미래 7일 예측
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

                # 4. 시각화 (최근 3개월 집중)
                view_df = df.tail(66)
                
                st.subheader(f"📊 분석 결과 (전 기간 오차율: {total_mape:.2%})")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=view_df.index, y=view_df['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
                fig.add_trace(go.Scatter(x=view_df.index, y=view_df['AI_복기종가'], name="AI 과거 복기", line=dict(color='rgba(255, 255, 0, 0.4)', dash='dot')))
                fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
                
                fig.update_layout(template='plotly_dark', height=500, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # 5. 하단 상세 정보
                c1, c2 = st.columns(2)
                with c1:
                    st.write("### 💡 지표 영향력")
                    imp = pd.DataFrame({'지표': features, '가중치': model.coef_}).set_index('지표')
                    st.bar_chart(imp, color='#00CCFF')
                with c2:
                    st.write("### 📈 7일 예측가")
                    pred_df = pd.DataFrame({'날짜': [d.strftime('%m-%d') for d in f_dates], '예측가': f_prices})
                    st.table(pred_df.style.format({'예측가': '{:,.0f}원'}))

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
