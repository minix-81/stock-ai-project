import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pytrends.request import TrendReq
from streamlit_gsheets import GSheetsConnection
import requests
from bs4 import BeautifulSoup

# 1. 페이지 및 시간 설정 (KST)
st.set_page_config(page_title="K-Investment AI Pro v3.0", layout="wide")
KST = datetime.now() + timedelta(hours=9)
current_time_str = KST.strftime('%Y-%m-%d %H:%M:%S')

# 2. 구글 시트 연결
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    conn = None

# 3. 사이드바 메뉴
with st.sidebar:
    st.title("🚀 AI 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.info(f"접속 시간(KST): {current_time_str}")

# --- [추가 기능: 3대 심리지표 함수] ---

def get_vix_index():
    """VIX(공포지수) 수집: 시장 전체의 불안정성을 측정합니다."""
    try:
        vix = fdr.DataReader('VIX', KST - timedelta(days=730))
        return vix[['Close']].rename(columns={'Close': 'VIX'})
    except:
        return pd.DataFrame()

def calculate_rsi(df, period=14):
    """RSI(상대강도지수): 과매수/과매도를 판단하는 심리 지표입니다."""
    delta = df['종가'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_news_sentiment(stock_name):
    """뉴스 심리 분석: 네이버 뉴스 제목을 분석하여 긍정/부정 점수를 냅니다."""
    try:
        url = f"https://search.naver.com/search.naver?where=news&query={stock_name}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        titles = [t.text for t in soup.select('.news_tit')]
        
        # 간단한 키워드 매칭 감성 분석
        pos_words = ['상승', '호재', '돌파', '급등', '이익', '성공', '수주']
        neg_words = ['하락', '악재', '이탈', '급락', '손실', '실패', '우려']
        
        score = 0
        for title in titles:
            for pw in pos_words:
                if pw in title: score += 1
            for nw in neg_words:
                if nw in title: score -= 1
        return score
    except:
        return 0

# --- [페이지 1: 실전 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 3대 심리지표 통합 AI 분석기")
    stock_code = st.text_input("종목 번호 6자리:", value="005930")
    run_button = st.button("정밀 분석 및 7일 예측 시작", use_container_width=True)

    if run_button:
        try:
            # [단계 1] 데이터 수집
            df = fdr.DataReader(stock_code, KST - timedelta(days=730))
            df = df.rename(columns={'Close': '종가', 'Volume': '거래량'})
            
            try:
                stocks = fdr.StockListing('KRX')
                stock_name = stocks[stocks['Code'] == stock_code]['Name'].values[0]
            except: stock_name = stock_code

            # [단계 2] 3대 지수 생성
            vix_df = get_vix_index()
            df = df.join(vix_df).fillna(method='ffill').fillna(20) # VIX 기본값 20
            df['RSI'] = calculate_rsi(df)
            news_score = get_news_sentiment(stock_name)
            df['뉴스심리'] = news_score # 현재 시점 뉴스 점수 적용

            # 기존 구글트렌드 유지
            df['구글트렌드'] = 0 
            try:
                pytrends = TrendReq(hl='ko')
                pytrends.build_payload([stock_name], timeframe='today 2-y', geo='KR')
                trends = pytrends.interest_over_time()
                if not trends.empty:
                    df['구글트렌드'] = trends[stock_name].reindex(df.index, method='ffill').fillna(0)
            except: pass

            # [단계 3] AI 변수 정의 (확장됨)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            df['감성지수'] = (df['종가'].pct_change() * 1000).clip(-150, 150).fillna(0)
            df['target'] = df['종가'].pct_change().shift(-1)
            
            df_train = df.dropna().copy()
            # 총 9가지 변수 학습
            features = ['날짜지수', '요일', '거래량', '변동성', '감성지수', '구글트렌드', 'VIX', 'RSI', '뉴스심리']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_train[features])
            model = Ridge(alpha=1.0).fit(X_scaled, df_train['target'])

            # [단계 4] 변수 기여도 시각화 (요청하신 막대그래프)
            st.subheader("💡 3대 지수 포함 AI 변수 기여도 분석")
            importance = pd.DataFrame({'변수': features, '비중(%)': (np.abs(model.coef_) / np.sum(np.abs(model.coef_))) * 100})
            st.bar_chart(importance.set_index('변수'), color='#00CCFF')

            # [단계 5] 미래 7거래일 예측
            last_price, last_date = df['종가'].iloc[-1], df.index[-1]
            future_prices, future_dates = [], []
            temp_price, last_feat = last_price, df[features].iloc[-1:].copy()

            curr_d = last_date
            while len(future_prices) < 7:
                curr_d += timedelta(days=1)
                if curr_d.weekday() < 5:
                    last_feat['날짜지수'] += 1
                    last_feat['요일'] = curr_d.weekday()
                    # VIX, RSI 등은 마지막 값 유지하여 예측
                    pred = model.predict(scaler.transform(last_feat))[0]
                    temp_price *= (1 + pred)
                    future_prices.append(temp_price); future_dates.append(curr_d)

            # [단계 6] 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-30:], y=df['종가'].iloc[-30:], name="실제 시세", line=dict(color='#00CCFF')))
            fig.add_trace(go.Scatter(x=[last_date]+future_dates, y=[last_price]+future_prices, 
                                     name="AI 예측(7일)", line=dict(dash='dash', color='red')))
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"KST 기준 {stock_name} 정밀 분석 완료! (뉴스 심리 점수: {news_score})")

        except Exception as e:
            st.error(f"분석 도중 오류 발생: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 모니터링 (KST)")
    pw = st.text_input("비밀번호", type="password")
    if pw == st.secrets.get("admin_password", "0000"):
        if conn:
            data = conn.read(worksheet="Sheet1", ttl=0)
            st.dataframe(data.iloc[::-1], use_container_width=True)
