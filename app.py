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

# 1. 페이지 설정
st.set_page_config(page_title="주식 AI v23.0 (Deep Sentiment)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 초정밀 감성 사전 (긍정 200+, 부정 200+)] ---
POS_WORDS = [
    '상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','목표가 상향','우상향','반등','M&A','신고가','회복','개선','호조','증가','확대','성장','안정','기대','성과','진전','활황','호황','순항','선전','약진','도약','혁신','정상화','상향','강화','지지','신뢰','합의','타결','협력','채택','승인','확정','수혜','유망','경쟁력','잠재력','모멘텀','전환점','긍정','안착','정착','반전','기회','낙관','상승세','회복세','성장세','개선세','탄탄','견조','순증','가속','촉진','확대 적용','신기록','우위','고무적','성과 확대','미래 성장','실적 개선','이익 증가','시장 확대','동력 확보','주도','수급 개선','외인 매수','기관 매수','순매수','저평가','가치 상승','배당 확대','자사주 매입','소각','재무 건전성','흑자 전환','턴어라운드','독점','점유율 상승','특허 취득','임상 성공','승인 획득','파트너십','조인트벤처','대규모 계약','어닝 서프라이즈','실적 호전','부채 감소','유동성 확보','신사업','성장 동력','구조조정 성공','비용 절감','효율화','생산성 향상','수요 급증','가격 인상','마진 확대','시장 지배력','압도적','독보적','고성장','지속 가능','유입','활기','장밋빛','활력','청신호','훈풍','쾌조','대박','잭팟','블루칩','가속도','선점','혁신적','차별화','우량','건실','탄탄한','활성화','상생','시너지','상호 보완','대세','주도주','대장주','테마 형성','자금 유입','신용 등급 상향','전망 밝음','낙관적','순풍','비상','도약기','전성기','황금기','역사적 신고가','연중 최고치','업황 개선','사이클 상향','저점 통과','바닥 확인','매수 신호','정배열','강한 흐름','상한가 안착','점상','연상','폭등','경신','가속 페달','탄력','가시화','본격화','가시적 성과','뚜렷한','명확한','압도적 1위','세계 최초','국내 최초','기술력 입증','신뢰도 제고','브랜드 가치','이미지 개선','고부가가치','영업외이익','순이익 급증','퀀텀 점프','글로벌 진출','독보적 기술','시장 선점','점유율 1위','호실적','현금 흐름 개선','수익성 개선','안정적','견고한','강력한','긍정적 시그널','기대감 고조'
]

NEG_WORDS = [
    '하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','목표가 하향','검찰','과징금','횡령','벌금','부진','악화','감소','침체','불안','논란','갈등','혼란','타격','충격','위축','비판','반발','제동','적자 전환','추락','둔화','리스크','부담','난항','지연','차질','무산','중단','붕괴','위반','불법','의혹','파문','후폭풍','불신','부실','취약','심각','경고','악영향','압박','혼선','공방','대립','마찰','냉각','후퇴','축소','불투명','진통','피로감','스캔들','실책','오판','급감','정체','미흡','문제','최악','불리','악조건','위기감','하방','리스크 확대','고소','피소','수사','조사','압수수색','기소','재판','판결','처벌','구속','공정위','금감원','제재','징계','취소','정지','영업정지','면허취소','배임','분식회계','감사의견 거절','관리종목','상장폐지','퇴출','부도','파산','회생절차','워크아웃','자금난','유동성 위기','채무불이행','디폴트','신용등급 강등','어닝 쇼크','실적 악화','매출 감소','영업손실','순손실','이익 급감','비용 증가','부채 급증','이자 부담','고금리','인플레이션','경기 불황','소비 위축','수요 감소','재고 급증','공급 과잉','가격 경쟁 심화','치킨 게임','출혈 경쟁','시장 점유율 하락','경쟁력 약화','기술 유출','핵심 인력 이탈','노사 갈등','파업','태업','불매 운동','여론 악화','이미지 실추','신뢰 추락','갑질 논란','환경 오염','안전 사고','법적 분쟁','소송 소용돌이','소환 조사','구속영장','실형','유죄','패소','손해배상','배상금','징벌적 손해배상','환수','거부','반려','불허','제한','금지','규제 강화','가이드라인 위반','시정 명령','외면','냉대','비관','비관론','공포','패닉','투매','손절','반대매매','깡통계좌','폭락세','하락장','약세장','데드 크로스','매도 신호','역배열','저항선 돌파 실패','지지선 붕괴','투기','거품 붕괴','버블','과평가','고평가 논란','거품 낀','과열된','방만한','부도덕한','비윤리적','비리','유착','뇌물','로비','탈세','세무조사','은폐','조작','왜곡','거짓 발표','허위 공시','공시 번복','신뢰 상실','속수무책','참사','대참사','비극','멸망','강사','사교육','학원','고소장','조사 착수','전수조사','압수','수색','구치소','중형','기소 의견'
]

# --- [2. 실시간 3일 누적 뉴스 감성 평가 엔진] ---
@st.cache_data(ttl=600)
def get_daily_news_score(stock_name, target_date):
    date_str = target_date.strftime('%Y.%m.%d')
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=4&ds={date_str}&de={date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        score, count = 0, 0
        for title in headlines:
            text = title.get_text()
            for p in POS_WORDS:
                if p in text: score += 10
            for n in NEG_WORDS:
                if n in text: score -= 25 # 부정 가중치 2.5배
            count += 1
        return (score / count) if count > 0 else 0
    except: return 0

# --- [3. 내부 로직 및 데이터 처리] ---
def get_converged_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            return df['best_coef'].mean() if not df.empty else 0.15
        except: pass
    return 0.15

def get_next_trading_days(start_date, n):
    days = []
    curr = start_date
    while len(days) < n:
        curr += timedelta(days=1)
        if curr.weekday() < 5: days.append(curr)
    return days

# --- [4. 메인 화면 및 사이드바] ---
with st.sidebar:
    st.title("🧠 AI 지능 센터")
    side_tab1, side_tab2 = st.tabs(["🏛️ 지능 상태", "📊 변수 영향력"])
    with side_tab1:
        st.metric("수렴 감수계수", f"{get_converged_coef():.4f}")
    with side_tab2:
        if 'importance' in st.session_state:
            st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')

st.title("🏛️ 주식 AI v23.0 (누적 뉴스 감성 인과관계 모델)")
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("3일 누적 뉴스 분석 및 7일 예측 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("최근 3일간의 뉴스를 교차 분석 중...", expanded=True) as status:
            # 1. 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 2. 3일 누적 뉴스 백테스팅 (최근 20영업일 집중 조사)
            st.write("🔍 과거 일자별 뉴스 흐름(T, T-1, T-2) 전수조사 중...")
            analysis_days = df_raw.index[-25:]
            daily_scores = {}
            for d in analysis_days:
                daily_scores[d] = get_daily_news_score(s_name, d)
            
            # 3. 데이터 가공 (3일 이동평균 뉴스 감성 생성)
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            
            # 뉴스 감성 지표 (3일 누적 효과 반영)
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items():
                temp_scores[d] = s
            # 3일 롤링 평균 적용 (T, T-1, T-2) 후 영향력 50배 증폭
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 50
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX']
            
            # 4. AI 학습 및 최적화
            scaler = StandardScaler()
            X_curr = df_final[features]
            y_curr = df_final['target']
            X_curr_scaled = scaler.fit_transform(X_curr)
            
            # 지능 수렴 및 Ridge 모델링 (alpha 낮춰 뉴스 민감도 극대화)
            model = Ridge(alpha=0.2).fit(X_curr_scaled, y_curr)
            
            # 5. 결과 저장 및 예측
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': model.coef_})
            
            # 미래 7일 예측 (현재부터 소급하여 최근 3일 뉴스 반영)
            current_sentiment = df['뉴스감성'].iloc[-1]
            f_dates = get_next_trading_days(df_final.index[-1], 7)
            f_prices, temp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            
            for d in f_dates:
                last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
                # 미래 뉴스 감성은 현재의 강력한 심리 상태가 유지된다고 가정
                pred = model.predict(scaler.transform(last_f))[0]
                temp_p *= (1 + pred)
                f_prices.append(temp_p)

            st.session_state.result = {
                'name': s_name, 'code': s_code, 
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['종가'] * (1 + model.predict(X_curr_scaled))),
                'df': df_final.tail(60), 'f_dates': f_dates, 'f_prices': f_prices,
                'news_score': current_sentiment / 50 # 원본 점수대 표시
            }
            
            success_flag = True
            status.update(label="분석 완료!", state="complete")
            
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [5. 시각화 영역] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 {res['name']} 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    
    df_v = res['df']
    # 백테스팅 노란 점선 생성
    df_v['AI_복기'] = (df_v['종가'] * (1 + res['model' if 'model' in res else 'result']['model' if 'model' in res else 'result'].predict(res['scaler'].transform(df_v[['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX']])) if 'model' in res else 0)).shift(1).fillna(df_v['종가'])
    
    # 세션 상태가 리런되어 모델 객체 접근이 어려울 수 있으므로 결과 기반으로 다시 그림
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=[df_v.index[-1]] + res['f_dates'], y=[df_v['종가'].iloc[-1]] + res['f_prices'], 
                             name="AI 미래 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    # 뉴스 감성 상태 출력
    news_status = "심각한 악재 감지" if res['news_score'] < -10 else ("호재 발생" if res['news_score'] > 10 else "중립적")
    st.markdown(f"### 📢 3일 누적 뉴스 진단: **{news_status}** (지수: {res['news_score']:.1f})")
    st.plotly_chart(fig, use_container_width=True)
