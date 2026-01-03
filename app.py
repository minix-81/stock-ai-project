import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from st_supabase_connection import SupabaseConnection

# 1. 페이지 및 시간 설정
st.set_page_config(page_title="주식 AI v6.9 (최종 정밀)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결 시도
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error(f"📡 수파베이스 연결 설비에 문제가 있습니다: {e}")
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 데이터 센터")
    menu = st.radio("메뉴 선택", ["실전 분석", "관리자"], key="nav_v69_final")
    st.info(f"KST: {KST_NOW.strftime('%Y-%m-%d %H:%M:%S')}")

# --- [데이터 관리 함수] ---
def load_db_with_debug():
    if conn:
        try:
            res = conn.table("knowledge").select("*").execute()
            # 데이터 로드 성공 시 행 수 출력
            if res.data:
                return pd.DataFrame(res.data)
        except Exception as e:
            st.sidebar.error(f"❌ 데이터 불러오기 실패: {e}")
    return pd.DataFrame()

def save_to_db_with_debug(stock_code, df_curr):
    """데이터 저장 시 성공/실패 여부를 화면에 즉시 보고합니다."""
    if not conn:
        st.error("🚫 수파베이스 연결이 끊겨 있어 저장할 수 없습니다.")
        return False
    
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
        
        # 실제 전송 시도
        res = conn.table("knowledge").insert(rows).execute()
        
        # 저장 성공 시 구체적인 행 수 보고
        st.success(f"✅ {len(rows)}개의 지식이 수파베이스 금고에 안착했습니다!")
        return True
    except Exception as e:
        # ⚠️ 여기서 나오는 메시지를 저에게 알려주세요!
        st.error(f"🔥 저장 단계에서 사고 발생: {e}")
        return False

# --- [페이지 1: 실전 분석 (그래프 완전 고정)] ---
if menu == "실전 분석":
    st.title("📊 2년 정밀 학습 및 7일 예측")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("AI 분석 및 지능 통합 시작", width='stretch'):
        try:
            # [단계 1] 2년치 데이터 수집 및 지표 생성 (사용자 요청 반영)
            start_date = KST_NOW - timedelta(days=730)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # VIX, RSI 계산 (재현님의 기존 분석 로직)
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

            # [단계 2] AI 학습 (Ridge Regression)
            # $$J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m \theta_j^2$$
            g_df = load_db_with_debug()
            if not g_df.empty:
                g_df = g_df.rename(columns={"date_index":"날짜지수","day_of_week":"요일","volume":"거래량","volatility":"변동성","sentiment":"감성지수","vix":"VIX","rsi":"RSI"})
                X_total = pd.concat([df_curr[features], g_df[features]])
                y_total = pd.concat([df_curr['target'], g_df['target']])
                weights = np.array([1.0]*len(df_curr) + [0.15]*len(g_df))
                st.info(f"💡 온라인 지능 {len(g_df)}행을 통합하여 더 똑똑해졌습니다.")
            else:
                X_total, y_total, weights = df_curr[features], df_curr['target'], np.array([1.0]*len(df_curr))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            model = Ridge(alpha=1.0).fit(X_scaled, y_total, sample_weight=weights)

            # [단계 3] 가중치 파란 막대그래프 (절대 고정)
            st.subheader("💡 AI 모델이 분석한 지표별 가중치")
            importance = pd.DataFrame({'변수': features, '가중치': model.coef_})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 4] 7일 미래 예측 및 시각화 (최근 한 달 강조)
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

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-22:], y=df['종가'].iloc[-22:], name="최근 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="AI 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', title=f"{stock_code} 최근 1개월 시세 및 7일 예측", height=500)
            st.plotly_chart(fig, width='stretch')

            # [단계 5] 지식 기부 및 저장 시도
            save_to_db_with_debug(stock_code, df_curr)

        except Exception as e: st.error(f"💥 분석 도중 에러가 터졌습니다: {e}")

# --- [페이지 2: 관리자] ---
elif menu == "관리자":
    st.title("📊 온라인 학습 통계 센터")
    if st.text_input("관리자 비번", type="password") == "0801":
        g_data = load_db_with_debug()
        if not g_data.empty:
            st.markdown(f"### 🚩 총 누적 데이터: `{len(g_data)}행`")
            counts = g_data['stock_code'].value_counts()
            for code, row_count in counts.items():
                st.write(f"📍 **{code}**: {row_count // 10}회 분석됨")
        else:
            st.info("현재 DB가 비어있습니다. 분석을 먼저 실행하세요.")
