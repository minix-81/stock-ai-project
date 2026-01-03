import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
import time

# 1. 환경 설정
st.set_page_config(page_title="주식 AI v21.0 (Sentiment Master)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 뉴스 감성 분석 엔진: 부정 뉴스 가중치 2배] ---
def get_past_news_sentiment(stock_name, target_date):
    date_str = target_date.strftime('%Y.%m.%d')
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}+주가&pd=4&ds={date_str}&de={date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    pos = [
        '상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천',
        '목표가 상향','우상향','반등','M&A','신고가','회복','개선','호조','증가','확대',
        '성장','안정','기대','성과','진전','활황','호황','순항','선전','약진',
        '도약','혁신','정상화','상향','강화','지지','신뢰','합의','타결','협력',
        '채택','승인','확정','수혜','유망','경쟁력','잠재력','모멘텀','전환점','긍정',
        '안착','정착','반전','기회','낙관','상승세','회복세','성장세','개선세','탄탄',
        '견조','순증','가속','촉진','확대 적용','신기록','우위','고무적','성과 확대','미래 성장',
        '실적 개선','이익 증가','시장 확대','동력 확보','주도'
    ]

    neg = [
        '하락','악재','적자','위기','실패','최저','우려','약세','매도','급락',
        '손실','쇼크','목표가 하향','검찰','과징금','횡령','벌금','부진','악화','감소','침체','불안',
        '논란','갈등','혼란','타격','충격','위축','비판','반발','제동','적자 전환',
        '추락','둔화','리스크','부담','난항','지연','차질','무산','중단','붕괴',
        '위반','불법','의혹','파문','후폭풍','불신','부실','취약','심각','경고',
        '악영향','압박','혼선','공방','대립','마찰','냉각','후퇴','축소','불투명',
        '진통','피로감','스캔들','실책','오판','급감','정체','미흡','문제','최악',
        '불리','악조건','위기감','하방','리스크 확대'
    ]

    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        
        score, count = 0, 0
        for title in headlines:
            text = title.get_text()
            for p in pos:
                if p in text: score += 10 # 긍정 +10
            for n in neg:
                if n in text: score -= 20 # 부정 -20 (2배 강력한 타격 반영)
            count += 1
        
        if count == 0: return 0
        # 뉴스 영향력을 AI가 확실히 인지하도록 15배 증폭
        return np.clip((score / count) * 15, -100, 100)
    except:
        return 0

# --- [2. 내부 로직: 영업일 계산 및 지능 수렴] ---
def get_converged_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            return df['best_coef'].mean() if not df.empty else 0.15
        except: pass
    return 0.15

def update_global_intelligence(new_coef):
    pd.DataFrame([[datetime.now(), new_coef]], columns=['date', 'best_coef']).to_csv(
        CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)

def get_next_trading_days(start_date, n):
    days = []
    curr = start_date
    while len(days) < n:
        curr += timedelta(days=1)
        if curr.weekday() < 5: days.append(curr)
    return days

# --- [3. 사이드바 UI: 가중치 차트 및 정의 표] ---
knowledge_df = pd.read_csv(DB_PATH, dtype={'stock_code': str}) if os.path.exists(DB_PATH) else pd.DataFrame()
converged_coef = get_converged_coef()

with st.sidebar:
    st.title("🧠 AI 지능 센터")
    st.write(f"📚 누적 데이터: `{len(knowledge_df)}`개")
    st.divider()
    
    if 'importance' in st.session_state:
        st.subheader("📊 변수별 영향력 (가중치)")
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
        
        st.markdown("### 📖 지표 정의 사전")
        st.table(pd.DataFrame({
            "지표": ["날짜지수", "요일", "거래량", "변동성", "뉴스감성", "VIX", "RSI"],
            "의미": ["시간의 흐름", "주간 반복 패턴", "매수/매도 에너지", "장중 가격 흔들림", "기사 긍부정(부정 2배 가중)", "시장 공포 지수", "과매수/과매도"]
        }))

# --- [4. 메인 분석 로직] ---
st.title("🏛️ 주식 AI v21.0 (Sentiment Master)")
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("뉴스 인과관계 전수조사 및 분석 시작", use_container_width=True):
    success = False
    try:
        with st.status("AI가 뉴스를 읽고 지능을 수렴시키는 중...", expanded=True) as status:
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 최근 뉴스 인과관계 분석 (최근 15영업일)
            recent_days = df_raw.index[-15:-1]
            news_scores = [get_past_news_sentiment(s_name, d) for d in recent_days]
            
            # 현재 시점 뉴스 분석
            current_news = get_past_news_sentiment(s_name, KST_NOW)
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            
            # 뉴스 감성 지표 결합
            df['뉴스감성'] = (df['종가'].pct_change() * 2000).clip(-100, 100)
            for i, d in enumerate(recent_days):
                df.loc[d, '뉴스감성'] = news_scores[i]
            
            df = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # 모델링 및 지능 수렴
            X_curr = df[features]
            y_curr = df['target']
            scaler = StandardScaler()
            X_curr_scaled = scaler.fit_transform(X_curr)
            
            if not knowledge_df.empty:
                X_total_scaled = scaler.fit_transform(pd.concat([X_curr, knowledge_df[features]]))
                y_total = pd.concat([y_curr, knowledge_df['target']])
                
                # 최적 감수계수 탐색 (백그라운드 계산)
                min_err, best_local = float('inf'), converged_coef
                for c in [0.1, 0.2, 0.3]:
                    w = np.concatenate([np.ones(len(X_curr)), np.full(len(knowledge_df), c)])
                    m = Ridge(alpha=0.3).fit(X_total_scaled, y_total, sample_weight=w)
                    err = mean_absolute_percentage_error(y_curr, m.predict(X_curr_scaled))
                    if err < min_err: min_err = err; best_local = c
                
                update_global_intelligence(best_local)
                final_coef = get_converged_coef()
                weights = np.concatenate([np.ones(len(X_curr)), np.full(len(knowledge_df), final_coef)])
                model = Ridge(alpha=0.3).fit(X_total_scaled, y_total, sample_weight=weights)
            else:
                model = Ridge(alpha=0.3).fit(X_curr_scaled, y_curr)
            
            # 결과 저장
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': model.coef_})
            st.session_state.result = {
                'name': s_name, 'code': s_code, 
                'mape': mean_absolute_percentage_error(df['종가'], df['종가'] * (1 + model.predict(X_curr_scaled))),
                'df': df.tail(60), 'f_dates': get_next_trading_days(df.index[-1], 7),
                'model': model, 'scaler': scaler, 'features': features, 'news_score': current_news
            }
            # 지식 축적 (빅데이터 성장)
            df['stock_code'] = s_code
            df[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            
            success = True
            status.update(label="전수조사 완료!", state="complete")
            
    except Exception as e:
        st.error(f"오류 발생: {e}")

    # SyntaxError 방지를 위해 st.rerun()을 try 블록 밖으로 뺐습니다.
    if success:
        st.rerun()

# --- [5. 시각화 영역: 백테스팅 그래프] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 {res['name']} ({res['code']}) - 백테스팅 오차율: {res['mape']:.2%}")
    
    df_v = res['df']
    pred_past = res['model'].predict(res['scaler'].transform(df_v[res['features']]))
    df_v['AI_복기'] = (df_v['종가'] * (1 + pred_past)).shift(1).fillna(df_v['종가'])
    
    f_prices, temp_p = [], df_v['종가'].iloc[-1]
    last_f = df_v[res['features']].iloc[-1:].copy()
    for d in res['f_dates']:
        last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
        temp_p *= (1 + res['model'].predict(res['scaler'].transform(last_f))[0])
        f_prices.append(temp_p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    # [백테스팅 노란 점선]
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v['AI_복기'], name="AI 과거 복기(백테스팅)", 
                             line=dict(color='yellow', dash='dot'), opacity=0.5))
    # [미래 7영업일 예측]
    fig.add_trace(go.Scatter(x=[df_v.index[-1]] + res['f_dates'], y=[df_v['종가'].iloc[-1]] + f_prices, 
                             name="미래 7영업일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    news_color = "red" if res['news_score'] < 0 else "lime"
    st.markdown(f"### 현재 뉴스 감성: <span style='color:{news_color}'>{res['news_score']:.1f}</span>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
