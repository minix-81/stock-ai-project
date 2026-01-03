import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# 1. 페이지 및 한국 시간(KST) 설정
st.set_page_config(page_title="K-Investment AI Pro v4.3", layout="wide", page_icon="📈")
KST_NOW = datetime.now() + timedelta(hours=9)
current_time_str = KST_NOW.strftime('%Y-%m-%d %H:%M:%S')

# 2. 구글 시트 연결 (연결 실패 대비 예외 처리 강화)
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    conn = None
    st.error(f"시트 연결 엔진 오류: {e}")

# 3. 사이드바 내비게이션
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info(f"접속 시간(KST): {current_time_str}")

# --- [정밀 분석용 보조 함수] ---

def get_vix_data(start_date):
    """^VIX 심볼을 사용하여 공포지수를 안정적으로 수집합니다."""
    try:
        vix = fdr.DataReader('^VIX', start_date)
        if not vix.empty:
            return vix[['Close']].rename(columns={'Close': 'VIX'})
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 7거래일 AI 정밀 분석기")
    stock_code = st.text_input("종목 번호 6자리 (예: 005930):", value="005930")
    run_button = st.button("분석 시작", use_container_width=True)

    if run_button:
        try:
            # [데이터 수집 및 전처리]
            start_date = KST_NOW - timedelta(days=730)
            df = fdr.DataReader(stock_code, start_date)
            if df.empty:
                st.error("주가 데이터를 가져올 수 없습니다.")
                st.stop()
            
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            vix_df = get_vix_data(start_date)
            if not vix_df.empty:
                df = df.join(vix_df, how='left').fillna(method='ffill').fillna(20)
            else:
                df['VIX'] = 20

            # 기술적 지표 생성
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = (100 - (100 / (1 + (gain / loss)))).fillna(50)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target'] = df['종가'].pct_change().shift(-1)
            
            df_train = df.dropna().copy()
            # VIX에 대한 설명 추가 (변수명 변경)
            df_train = df_train.rename(columns={'VIX': 'VIX(시장공포지수)'})
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', 'VIX(시장공포지수)', 'RSI']
            
            # [Ridge AI 학습]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_train[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_train['target'])

            # 변수 기례도 막대그래프 (VIX 설명 포함)
            st.subheader("💡 AI 변수별 예측 기여도 분석")
            importance = pd.DataFrame({'변수': features, '비중(%)': (np.abs(model.coef_) / np.sum(np.abs(model.coef_))) * 100})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [7거래일 예측]
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df_train[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5:
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    pred = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [시각화]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="실제 시세", line=dict(color='#00CCFF', width=3)))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 예측(7거래일)", line=dict(dash='dash', color='red', width=4), mode='lines+markers'))
            fig.update_layout(template='plotly_dark', height=500, legend_title="구분")
            st.plotly_chart(fig, use_container_width=True)
            
            # 로그 기록 시도 (오류 발생 시에도 앱 유지)
            if conn:
                try:
                    data = conn.read(worksheet="Sheet1", ttl=0)
                    new_log = pd.DataFrame([{"날짜": current_time_str, "종목코드": stock_code, "종목명": "조회완료"}])
                    conn.update(worksheet="Sheet1", data=pd.concat([data, new_log], ignore_index=True))
                except: pass

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링")
    pw = st.text_input("비밀번호", type="password")
    if pw == st.secrets.get("admin_password", "0000"):
        st.success("인증 성공")
        if conn:
            try:
                # HTTPError 방지를 위한 캐시 무시 로직
                data = conn.read(worksheet="Sheet1", ttl=0)
                st.dataframe(data.iloc[::-1], use_container_width=True)
            except Exception as e:
                st.error("⚠️ 구글 시트 접근 권한 오류가 발생했습니다.")
                st.write(f"상세 오류: {e}")
                st.info("### 🆘 해결 방법")
                st.markdown("""
                1. **Secrets 주소 확인**: URL 끝이 반드시 `/edit#gid=0` 형태인지 확인하세요.
                2. **공유 설정 재확인**: [링크가 있는 모든 사용자]가 [편집자]인지 다시 한 번 저장하세요.
                3. **직접 접근 테스트**: 브라우저 시크릿 모드에서 해당 시트 주소로 접속했을 때 로그인이 뜨지 않고 바로 열려야 합니다.
                """)
