import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI v9.5 (Final)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. 구글 시트 연결
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"❌ 구글 시트 연결 실패: {e}")
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("메뉴 선택", ["실전 분석", "관리자"], key="nav_v95")

# --- [데이터 관리 함수] ---
def load_db():
    if conn:
        try:
            return conn.read(ttl=0) # 실시간 데이터 로드
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_to_db(stock_code, df_curr):
    if conn and not df_curr.empty:
        try:
            # 1. 기존 데이터 가져오기
            existing_df = load_db()
            
            # 2. 저장할 10일치 데이터 생성
            sample = df_curr.tail(10).copy()
            sample['stock_code'] = str(stock_code)
            
            # 3. 구글 시트 컬럼 순서 맞추기
            new_rows = sample[['stock_code', 'RSI', 'VIX', 'target', '거래량', '요일', '변동성', '감성지수', '날짜지수']]
            new_rows.columns = existing_df.columns if not existing_df.empty else new_rows.columns
            
            # 4. 시트 업데이트
            updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
            conn.update(data=updated_df)
            return True
        except Exception as e:
            st.error(f"저장 실패: {e}")
            return False
    return False

# --- [페이지 1: 실전 분석] ---
if menu == "실전 분석":
    st.title("📊 2년 학습 및 7일 주가 예측")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("AI 분석 및 지능 저장 시작", width='stretch'):
        try:
            # [단계 1] 2년 데이터 수집
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
            
            # 학습용 변수 생성
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            
            # ⚠️ 'df_curr' 변수 정의 완료 (오류 해결!)
            df_curr = df.dropna().copy()
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX', 'RSI']

            # [단계 2] Ridge 학습: $$J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m \theta_j^2$$
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_curr[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_curr['target'])

            # [단계 3] 가중치 파란 막대그래프
            st.subheader("💡 AI 모델 지표별 가중치")
            importance = pd.DataFrame({'변수': features, '가중치': model.coef_})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 4] 최근 한 달 흐름 및 7일 예측 (빨간 점선)
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
            # 최근 1개월(22거래일) 시각화
            fig.add_trace(go.Scatter(x=df.index[-22:], y=df['종가'].iloc[-22:], name="최근 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="AI 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', title=f"{stock_code} 최근 1개월 흐름 및 7일 예측", height=500)
            st.plotly_chart(fig, width='stretch')

            # [단계 5] 데이터 저장
            if save_to_db(stock_code, df_curr):
                st.success("✅ 분석 완료! 지능이 구글 시트에 성공적으로 기록되었습니다.")

        except Exception as e: st.error(f"오류: {e}")

# --- [페이지 2: 관리자] ---
elif menu == "관리자":
    st.title("📊 온라인 학습 통계 (G-Sheet)")
    if st.text_input("비번", type="password") == "0801":
        g_data = load_db()
        if not g_data.empty:
            total_searches = len(g_data) // 10
            st.markdown(f"### 🚩 총 누적 검색량: `{total_searches}회`")
            # 종목별 횟수 출력 (005930: 13회 형식)
            counts = g_data['stock_code'].value_counts()
            for code, row_count in counts.items():
                st.write(f"📍 **{code}**: {row_count // 10}회 분석됨")
        else:
            # 데이터가 없을 때의 메시지
            st.info("현재 저장된 데이터가 없습니다. 실전 분석을 실행하세요.")
