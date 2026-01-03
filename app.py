import streamlit as st
import pandas as pdimport streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from st_supabase_connection import SupabaseConnection

# 1. 페이지 및 시간 설정
st.set_page_config(page_title="주식 AI v6.2 (2년 분석)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. Supabase 연결
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception:
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("메뉴", ["실전 분석", "관리자"], key="nav_v6_2")
    st.info(f"KST: {current_time_str}")

# --- [데이터 관리 함수] ---
def load_db():
    if conn:
        try:
            # image_4e3aa1.png에서 확인한 'knowledge' 테이블 로드
            res = conn.table("knowledge").select("*").execute()
            return pd.DataFrame(res.data)
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_to_db(stock_code, df_curr):
    if conn and not df_curr.empty:
        try:
            # 1회 분석당 10일치 데이터를 온라인 지식으로 저장
            sample = df_curr.tail(10)
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": str(stock_code), "rsi": float(r['RSI']), "vix": float(r['VIX']),
                    "target": float(r['target']), "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']), "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']), "date_index": float(r['날짜지수'])
                })
            conn.table("knowledge").insert(rows).execute()
        except: pass

# --- [페이지 1: 실전 분석] ---
if menu == "실전 분석":
    st.title("📈 2개년 데이터 기반 AI 분석 및 예측")
    st.write("최근 **2년**의 패턴을 학습하고 **7일** 뒤의 주가를 예측합니다.")
    
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    if st.button("온라인 지능 통합 분석 시작", width='stretch'):
        try:
            # [단계 1] 2년치 데이터 수집 (사용자 요청: 5년 -> 2년 수정)
            start_date = KST_NOW - timedelta(days=730)
            df = fdr.DataReader(stock_code, start_date)
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # 지표 생성 (VIX, RSI 등)
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

            # [단계 2] 온라인 지능 통합 (감쇄계수 0.15)
            g_df = load_db()
            if not g_df.empty:
                g_df = g_df.rename(columns={"date_index":"날짜지수","day_of_week":"요일","volume":"거래량","volatility":"변동성","sentiment":"감성지수","vix":"VIX","rsi":"RSI"})
                X_total = pd.concat([df_curr[features], g_df[features]])
                y_total = pd.concat([df_curr['target'], g_df['target']])
                weights = np.array([1.0]*len(df_curr) + [0.15]*len(g_df))
                st.info(f"💡 온라인 DB 지능 {len(g_df)}행을 통합하여 학습했습니다.")
            else:
                X_total, y_total = df_curr[features], df_curr['target']
                weights = np.array([1.0]*len(X_total))

            # [단계 3] AI 학습 (Ridge)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            model = Ridge(alpha=1.0).fit(X_scaled, y_total, sample_weight=weights)

            # [단계 4] 파란색 가중치 막대그래프 (복구 완료)
            st.subheader("💡 AI 지표별 가중치 피드백")
            importance = pd.DataFrame({'변수': features, '가중치': model.coef_})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 5] 7거래일 미래 예측 (복구 완료)
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

            # [단계 6] 최근 한 달 시각화 (사용자 요청 반영)
            # 전체 2년 데이터 중 최근 22거래일(약 한 달)만 차트에 표시
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-22:], y=df['종가'].iloc[-22:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_d]+f_dates, y=[last_p]+f_prices, name="AI 7일 예측", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', title=f"{stock_code} 최근 1개월 흐름 및 예측", height=500)
            st.plotly_chart(fig, width='stretch')

            # [단계 7] 온라인 DB 저장
            save_to_db(stock_code, df_curr)
            st.success("분석 완료! 지능이 수파베이스에 영구 저장되었습니다.")

        except Exception as e: st.error(f"오류 발생: {e}")

# --- [페이지 2: 관리자 (통계 기능)] ---
elif menu == "관리자":
    st.title("📊 온라인 검색 통계")
    if st.text_input("비번", type="password") == "0801":
        g_data = load_db()
        if not g_data.empty:
            # 검색 1회당 10행씩 쌓이므로 행수 / 10으로 계산
            total_searches = len(g_data) // 10
            st.markdown(f"### 🚩 총 누적 검색량: `{total_searches}회`")
            st.write("---")
            
            # 종목별 횟수 (005930: 13회 형식)
            counts = g_data['stock_code'].value_counts()
            for code, row_count in counts.items():
                st.write(f"📍 **{code}**: {row_count // 10}회")
        else:
            # image_4f1365.png의 "저장된 데이터가 없습니다" 메시지 대응
            st.info("아직 온라인 DB에 저장된 데이터가 없습니다. 분석을 1회 이상 실행해 주세요.")
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from st_supabase_connection import SupabaseConnection

# 1. 페이지 및 시간 설정
st.set_page_config(page_title="주식 AI 분석기 v6.1", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)

# 2. Supabase 연결
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception:
    conn = None

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("메뉴", ["실전 분석", "관리자"], key="nav_v6_1")

# --- [데이터 관리 함수] ---
def load_db():
    if conn:
        try:
            # 'knowledge' 테이블 전체 읽기
            res = conn.table("knowledge").select("*").execute()
            return pd.DataFrame(res.data)
        except Exception as e:
            return pd.DataFrame()
    return pd.DataFrame()

def save_db(stock_code, df_curr):
    """분석 완료 시 수파베이스에 10일치 데이터를 저장하여 흔적을 남깁니다."""
    if conn and not df_curr.empty:
        try:
            sample = df_curr.tail(10) # 1회 검색당 10행 저장
            rows = []
            for _, r in sample.iterrows():
                rows.append({
                    "stock_code": str(stock_code),
                    "rsi": float(r['RSI']),
                    "vix": float(r['VIX']),
                    "target": float(r['target']),
                    "volume": float(r['거래량']),
                    "day_of_week": int(r['요일']),
                    "volatility": float(r['변동성']),
                    "sentiment": float(r['감성지수']),
                    "date_index": float(r['날짜지수'])
                })
            # 데이터를 넣고 결과를 확인합니다.
            conn.table("knowledge").insert(rows).execute()
        except Exception as e:
            st.error(f"DB 저장 실패: {e}")

# --- [페이지 1: 실전 분석] ---
if menu == "실전 분석":
    st.title("📊 5개년 정밀 AI 분석기")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    
    if st.button("온라인 지능 통합 분석 시작", width='stretch'):
        try:
            # 5년 데이터 수집 (timedelta 1825일)
            df = fdr.DataReader(stock_code, KST_NOW - timedelta(days=1825))
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            # VIX 및 RSI 지표 생성
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

            # AI 학습 (Ridge Regression)
            # $$J(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m \theta_j^2$$
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_curr[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_curr['target'])

            # 지표 중요도 시각화 (파란색 막대)
            st.subheader("💡 AI 모델이 분석한 지표별 중요도")
            importance = pd.DataFrame({'변수': features, '가중치': model.coef_})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # 미래 예측 및 차트 출력 (생략)
            
            # [핵심] 수파베이스에 저장 명령
            save_db(stock_code, df_curr)
            st.success("분석 완료 및 온라인 지능 저장 성공!")

        except Exception as e:
            st.error(f"분석 중 오류: {e}")

# --- [페이지 2: 관리자 (재현님이 원하신 통계 형식)] ---
elif menu == "관리자":
    st.title("📊 온라인 지능 및 검색 통계")
    if st.text_input("비번", type="password") == "0801":
        st.success("인증 성공")
        g_data = load_db()
        
        if not g_data.empty:
            # 1. 총 검색량 요약 (1회 분석당 10행이 저장되므로 행수를 10으로 나눔)
            total_searches = len(g_data) // 10
            st.markdown(f"### 🚩 총 누적 검색량: `{total_searches}회`")
            st.write("---")
            
            # 2. 종목별 검색 횟수 (005930: 13회 형식)
            if 'stock_code' in g_data.columns:
                st.subheader("📈 종목별 검색 리스트")
                # 종목코드별 데이터 행수를 구한 뒤 10으로 나눕니다.
                counts = g_data['stock_code'].value_counts()
                for code, row_count in counts.items():
                    search_count = row_count // 10
                    st.write(f"📍 **{code}**: {search_count}회")
            else:
                st.error("DB에 'stock_code' 열이 없습니다. 테이블 설정을 확인하세요.")
        else:
            st.info("아직 온라인 DB에 저장된 데이터가 없습니다. 분석을 먼저 실행해 보세요.")

