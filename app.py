import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from st_supabase_connection import SupabaseConnection

# 1. 페이지 및 시간 설정
st.set_page_config(page_title="주식 AI 분석기 v5.7", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. Supabase 연결 (Secrets 연동)
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception:
    conn = None

# --- [온라인 지능 관리 함수] ---
def load_db_knowledge():
    if conn:
        try:
            # image_4e3aa1.png에서 확인한 'knowledge' 테이블 로드
            res = conn.table("knowledge").select("*").execute()
            return pd.DataFrame(res.data)
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_to_db(stock_code, df_current):
    if conn and not df_current.empty:
        try:
            # image_4e36bf.png의 컬럼명에 맞춰 데이터 변형
            sample = df_current.tail(15) 
            rows = []
            for _, row in sample.iterrows():
                rows.append({
                    "stock_code": stock_code,
                    "rsi": float(row['RSI']),
                    "vix": float(row['VIX']),
                    "target": float(row['target']),
                    "volume": float(row['거래량']),
                    "day_of_week": int(row['요일']),
                    "volatility": float(row['변동성']),
                    "sentiment": float(row['감성지수']),
                    "date_index": float(row['날짜지수'])
                })
            conn.table("knowledge").insert(rows).execute()
        except: pass

# --- [페이지 1: 실전 종목 분석기] ---
if st.sidebar.radio("메뉴", ["실전 분석", "관리자"]) == "실전 분석":
    st.title("📈 영구 학습형 주식 AI (v5.7)")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("온라인 지능 통합 분석 시작", width='stretch'):
        try:
            # [단계 1] 데이터 수집 및 지표 생성
            start_date = KST_NOW - timedelta(days=1825)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # VIX 및 RSI 계산
            vix_df = fdr.DataReader('^VIX', start_date)
            df = df.join(vix_df[['Close']].rename(columns={'Close': 'VIX'})).ffill().fillna(20)
            
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = (100 - (100 / (1 + (gain / loss)))).fillna(50)
            
            # AI 학습 변수 생성
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            
            df_current = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']

            # [단계 2] 온라인 지능 통합 (감쇄계수 0.15)
            global_df = load_db_knowledge()
            if not global_df.empty:
                # DB 컬럼명을 코드 변수명으로 매칭
                global_df = global_df.rename(columns={
                    "date_index": "날짜지수", "day_of_week": "요일", "volume": "거래량",
                    "volatility": "변동성", "sentiment": "감성지수", "vix": "VIX", "rsi": "RSI"
                })
                X_total = pd.concat([df_current[features], global_df[features]])
                y_total = pd.concat([df_current['target'], global_df['target']])
                weights = np.array([1.0] * len(df_current) + [0.15] * len(global_df))
                st.info(f"💡 온라인 DB에서 {len(global_df)}개의 지식을 전수받았습니다.")
            else:
                X_total, y_total = df_current[features], df_current['target']
                weights = np.array([1.0] * len(X_total))

            # [단계 3] AI 학습 (Ridge Regression)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            model = Ridge(alpha=1.0).fit(X_scaled, y_total, sample_weight=weights)

            # [단계 4] 파란색 막대그래프 (지표 중요도) 출력
            st.subheader("💡 AI 모델이 분석한 지표별 가중치")
            importance = pd.DataFrame({'변수': features, '가중치': model.coef_})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF') # 재현님이 원하신 파란색 막대

            # [단계 5] 7거래일 미래 예측 로직
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df_current[features].iloc[-1:].copy()
            
            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5:
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    pred = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [단계 6] 예측 그래프 (실제 시세 + 빨간 점선) 출력
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['종가'].iloc[-60:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 7일 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, width='stretch')

            # [단계 7] 온라인 지식 저장
            save_to_db(stock_code, df_current)
            st.success("온라인 지능 통합 분석이 완료되었습니다!")

        except Exception as e:
            st.error(f"분석 도중 오류 발생: {e}")

# --- [관리자 대시보드] ---
elif st.sidebar.radio("메뉴", ["실전 분석", "관리자"]) == "관리자":
    st.title("📊 온라인 지능 통계")
    # (종목별 검색 횟수 출력 로직 생략 가능)
