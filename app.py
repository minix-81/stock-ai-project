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
st.set_page_config(page_title="주식 AI v7.8 (Final)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결 (Secrets 연동)
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error(f"❌ DB 연결 설정 오류: {e}")
    conn = None

# --- [사이드바 메뉴 및 진단] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("메뉴 선택", ["실전 분석", "관리자"], key="nav_v78")
    
    st.divider()
    if st.button("🔍 DB 연결 상태 점검"):
        if conn:
            try:
                conn.table("knowledge").select("count", count="exact").limit(1).execute()
                st.sidebar.success("✅ DB 연결 정상! (데이터 전송 준비 완료)")
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
    st.title("📊 2개년 데이터 분석 및 7일 예측")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("분석 및 온라인 지능 통합 시작", width='stretch'):
        try:
            # 1. 2년 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # 지표 생성 (VIX, RSI)
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
            
            df_curr = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']

            # 2. 온라인 지능 통합 학습
            g_df = load_db()
            if not g_df.empty:
                g_df = g_df.rename(columns={"date_index":"날짜지수","day_of_week":"요일","volume":"거래량","volatility":"변동성","sentiment":"감성지수","vix":"VIX","rsi":"RSI"})
                X_total = pd.concat([df_curr[features], g_df[features]])
                y_total = pd.concat([df_curr['target'], g_df['target']])
                weights = np.array([1.0]*len(df_curr) + [0.15]*len(g_df))
                st.info(f"💡 온라인 DB에서 {len(g_df)}개의 지식을 통합했습니다.")
            else:
                X_total, y_total, weights = df_curr[features], df_curr['target'], np.array([1.0]*len(df_curr))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            model = Ridge(alpha=1.0).fit(X_scaled, y_total, sample_weight=weights)

            # 3. 파란색 가중치 막대그래프
            st.subheader("💡 AI 모델 지표별 가중치")
            importance = pd.DataFrame({'변수': features, '가중치': model.coef_})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # 4. 7일 예측 및 최근 1개월 시각화
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
            # 최근 1개월(22거래일) 강조
            fig.add_trace(go.Scatter(x=df.index[-22:], y=df['종가'].iloc[-22:], name="최근 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="AI 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', title=f"{stock_code} 최근 흐름 및 7일 예측", height=500)
            st.plotly_chart(fig, width='stretch')

            # 5. 지식 저장
            if save_to_db(stock_code, df_curr) > 0:
                st.success("✅ 분석 완료! 지식이 온라인 DB에 안착했습니다.")
            else:
                st.warning("⚠️ 분석은 끝났으나 DB 저장에 실패했습니다. Secrets를 확인하세요.")

        except Exception as e: st.error(f"오류: {e}")

# --- [페이지 2: 관리자] ---
elif menu == "관리자":
    st.title("📊 온라인 학습 통계 센터")
    if st.text_input("비번", type="password") == "0801":
        g_data = load_db()
        if not g_data.empty:
            total_searches = len(g_data) // 10
            st.markdown(f"### 🚩 총 누적 검색량: `{total_searches}회`")
            st.write("---")
            # 005930: 13회 형식 출력
            counts = g_data['stock_code'].value_counts()
            for code, row_count in counts.items():
                st.write(f"📍 **{code}**: {row_count // 10}회 분석됨")
        else:
            st.info("현재 저장된 데이터가 없습니다. 실전 분석을 먼저 실행하세요.")
