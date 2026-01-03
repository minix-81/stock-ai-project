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
st.set_page_config(page_title="주식 AI 분석기 v6.0", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception:
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("메뉴", ["실전 분석", "관리자"], key="nav_menu_v6")
    st.info(f"KST: {KST_NOW.strftime('%Y-%m-%d %H:%M:%S')}")

# --- [데이터 관리 함수] ---
def load_db():
    if conn:
        try:
            res = conn.table("knowledge").select("*").execute()
            return pd.DataFrame(res.data)
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_db(stock_code, df_curr):
    if conn and not df_curr.empty:
        try:
            # 한 번 분석할 때 10일치 데이터를 지식으로 저장합니다.
            sample = df_curr.tail(10)
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": stock_code, "rsi": float(r['RSI']), "vix": float(r['VIX']),
                    "target": float(r['target']), "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']), "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']), "date_index": float(r['날짜지수'])
                })
            conn.table("knowledge").insert(rows).execute()
        except: pass

# --- [페이지 1: 실전 분석 (그래프 포함)] ---
if menu == "실전 분석":
    st.title("📊 5개년 정밀 AI 분석 및 예측")
    stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    
    if st.button("분석 시작 (온라인 지능 통합)", width='stretch'):
        try:
            # [단계 1] 데이터 수집 및 지표 생성
            df = fdr.DataReader(stock_code, KST_NOW - timedelta(days=1825))
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            vix = fdr.DataReader('^VIX', KST_NOW - timedelta(days=1825))[['Close']].rename(columns={'Close': 'VIX'})
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

            # [단계 2] 온라인 지능 통합 (감쇄계수 0.15)
            g_df = load_db()
            if not g_df.empty:
                g_df = g_df.rename(columns={"date_index":"날짜지수","day_of_week":"요일","volume":"거래량","volatility":"변동성","sentiment":"감성지수","vix":"VIX","rsi":"RSI"})
                X_total = pd.concat([df_curr[features], g_df[features]])
                y_total = pd.concat([df_curr['target'], g_df['target']])
                weights = np.array([1.0]*len(df_curr) + [0.15]*len(g_df))
                st.info(f"💡 온라인 DB에서 {len(g_df)}개의 지식을 반영했습니다.")
            else:
                X_total, y_total = df_curr[features], df_curr['target']
                weights = np.array([1.0]*len(X_total))

            # [단계 3] AI 학습 (Ridge)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            model = Ridge(alpha=1.0).fit(X_scaled, y_total, sample_weight=weights)

            # [단계 4] 파란색 가중치 막대그래프 출력
            st.subheader("💡 AI 모델이 분석한 지표별 중요도")
            importance = pd.DataFrame({'변수': features, '가중치': model.coef_})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 5] 7거래일 미래 예측
            last_p, last_d = df['종가'].iloc[-1], df.index[-1]
            f_prices, f_dates = [], []
            temp_p, last_f = last_p, df_curr[features].iloc[-1:].copy()
            
            curr_d = last_d
            while len(f_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5:
                    last_f['날짜지수'] += 1
                    last_f['요일'] = curr_d.weekday()
                    pred = model.predict(scaler.transform(last_f))[0]
                    temp_p *= (1 + pred)
                    f_prices.append(temp_p); f_dates.append(curr_d)

            # [단계 6] 7일 예측 주가 그래프 출력
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['종가'].iloc[-60:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="AI 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, width='stretch')

            save_db(stock_code, df_curr)
            st.success("분석 완료! 지능이 수파베이스에 저장되었습니다.")

        except Exception as e: st.error(f"오류: {e}")

# --- [페이지 2: 관리자 (통계 기능 복구)] ---
elif menu == "관리자":
    st.title("📊 온라인 지능 및 검색 통계")
    pw = st.text_input("관리자 비밀번호", type="password")
    if pw == st.secrets.get("admin_password", "0801"):
        st.success("인증 성공")
        g_data = load_db()
        if not g_data.empty:
            # 1. 총 데이터량 (검색 1회당 10행씩 쌓임)
            total_entries = len(g_data)
            estimated_searches = total_entries // 10
            st.markdown(f"### 🚩 총 누적 검색량 (추정): `{estimated_searches}회`")
            st.write(f"(총 데이터 포인트: {total_entries}개)")
            st.write("---")
            
            # 2. 종목별 검색 현황 (005930: 13회 형식)
            if 'stock_code' in g_data.columns:
                st.subheader("📈 종목별 분석 횟수 요약")
                # 각 종목코드별로 데이터 행수를 세고 10으로 나눠서 횟수를 계산합니다.
                counts = g_data['stock_code'].value_counts()
                for code, count in counts.items():
                    actual_count = count // 10
                    st.write(f"📍 **{code}**: {actual_count}회")
            else:
                st.info("시트에 'stock_code' 열이 없습니다.")
        else:
            st.info("아직 온라인 DB에 저장된 데이터가 없습니다.")
