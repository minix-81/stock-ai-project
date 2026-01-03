import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from st_supabase_connection import SupabaseConnection

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI v7.8 (최종)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결 (Secrets 연동)
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error(f"❌ DB 연결 설정 오류: {e}")
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("메뉴 선택", ["실전 분석", "관리자"], key="nav_v78_final")
    
    st.divider()
    if st.button("🔍 DB 연결 상태 점검"):
        if conn:
            try:
                conn.table("knowledge").select("count", count="exact").limit(1).execute()
                st.sidebar.success("✅ DB 연결 정상!")
            except Exception as e:
                st.sidebar.error(f"❌ DB 응답 없음: {e}")

# --- [데이터 관리 함수] ---
def load_db():
    if conn:
        try:
            res = conn.table("knowledge").select("*").execute()
            return pd.DataFrame(res.data) if res.data else pd.DataFrame()
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_to_db(stock_code, df_curr):
    if conn and not df_curr.empty:
        try:
            sample = df_curr.tail(10)
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": str(stock_code), "rsi": float(r['RSI']), "vix": float(r['VIX']),
                    "target": float(r['target']), "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']), "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']), "date_index": float(r['날짜지수'])
                })
            response = conn.table("knowledge").insert(rows).execute()
            return len(response.data) if response.data else 0
        except: return 0
    return 0

# --- [페이지 1: 실전 분석] ---
if menu == "실전 분석":
    st.title("📊 2년 정밀 학습 및 7일 주가 예측")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("분석 및 지능 저장 시작", width='stretch'):
        try:
            # 2년 데이터 수집
            df = fdr.DataReader(stock_code, KST_NOW - timedelta(days=730))
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # 지표 생성 로직 (RSI, VIX 등)
            vix = fdr.DataReader('^VIX', KST_NOW - timedelta(days=730))[['Close']].rename(columns={'Close': 'VIX'})
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
            
            df_curr = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']

            # AI 학습 (Ridge)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_curr[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_curr['target'])

            # 지표 가중치 막대그래프
            st.subheader("💡 AI 지표별 가중치")
            importance = pd.DataFrame({'변수': features, '가중치': model.coef_})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # 7일 예측 및 최근 한 달 시각화
            last_p, last_d = df['종가'].iloc[-1], df.index[-1]
            f_prices, f_dates = [], []
            temp_p, last_f = last_p, df_curr[features].iloc[-1:].copy()
            for i in range(1, 8):
                last_f['날짜지수'] += 1
                pred = model.predict(scaler.transform(last_f))[0]
                temp_p *= (1 + pred)
                f_prices.append(temp_p)
                f_dates.append(last_d + timedelta(days=i))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-22:], y=df['종가'].iloc[-22:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="AI 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', title=f"{stock_code} 최근 흐름 및 7일 예측", height=500)
            st.plotly_chart(fig, width='stretch')

            # DB 저장 및 확인
            if save_to_db(stock_code, df_curr) > 0:
                st.success("✅ 분석 완료! 지식이 온라인 DB에 성공적으로 저장되었습니다.")
            else:
                st.warning("⚠️ 분석은 완료됐으나 저장은 실패했습니다. Secrets 설정을 확인하세요.")

        except Exception as e: st.error(f"오류: {e}")

# --- [페이지 2: 관리자] ---
elif menu == "관리자":
    st.title("📊 온라인 검색 통계")
    if st.text_input("비번", type="password") == "0801":
        g_data = load_db()
        if not g_data.empty:
            total_searches = len(g_data) // 10
            st.markdown(f"### 🚩 총 누적 검색량: `{total_searches}회`")
            # 종목별 횟수 출력
            counts = g_data['stock_code'].value_counts()
            for code, row_count in counts.items():
                st.write(f"📍 **{code}**: {row_count // 10}회 분석됨")
        else:
            st.info("현재 저장된 데이터가 없습니다. 실전 분석을 먼저 실행하세요.")
