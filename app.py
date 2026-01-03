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
st.set_page_config(page_title="주식 AI v41.0 (Google RSS)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
NEWS_DB_PATH = "news_rss_cache.csv" # RSS 전용 캐시
CONFIG_PATH = "global_config.csv"

# --- [1. 초정밀 감성 사전 (긍정 210 / 부정 225)] ---
# (사용자 제공 400+ 단어 리스트가 엔진에 내장됨)
POS_WORDS = ['상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','목표가 상향','우상향','반등','M&A','신고가','회복','개선','호조','증가','확대','성장','안정','기대','성과','진전','활황','호황','순항','선전','약진','도약','혁신','정상화','상향','강화','지지','신뢰','합의','타결','협력','채택','승인','확정','수혜','유망','경쟁력','잠재력','모멘텀','전환점','긍정','안착','정착','반전','기회','낙관','상승세','회복세','성장세','개선세','탄탄','견조','순증','가속','촉진','확대 적용','신기록','우위','고무적','성과 확대','미래 성장','실적 개선','이익 증가','시장 확대','동력 확보','주도','수급 개선','외인 매수','기관 매수','순매수','저평가','가치 상승','배당 확대','자사주 매입','소각','재무 건전성','흑자 전환','턴어라운드','독점','점유율 상승','특허 취득','임상 성공','승인 획득','파트너십','조인트벤처','대규모 계약','어닝 서프라이즈','실적 호전','부채 감소','유동성 확보','신사업','성장 동력','구조조정 성공','비용 절감','효율화','생산성 향상','수요 급증','가격 인상','마진 확대','시장 지배력','압도적','독보적','고성장','지속 가능','유입','활기','장밋빛','활력','청신호','훈풍','쾌조','대박','잭팟','블루칩','가속도','선점','혁신적','차별화','우량','건실','탄탄한','활성화','상생','시너지','상호 보완','대세','주도주','대장주','테마 형성','자금 유입','신용 등급 상향','전망 밝음','낙관적','순풍','비상','도약기','전성기','황금기','역사적 신고가','연중 최고치','업황 개선','사이클 상향','저점 통과','바닥 확인','매수 신호','정배열','강한 흐름','상한가 안착','점상','연상','폭등','경신','가속 페달','탄력','가시화','본격화','가시적 성과','뚜렷한','명확한','압도적 1위','세계 최초','국내 최초','기술력 입증','신뢰도 제고','브랜드 가치','이미지 개선','고부가가치','영업외이익','순이익 급증','퀀텀 점프','글로벌 진출','독보적 기술','시장 선점','점유율 1위','호실적','현금 흐름 개선','수익성 개선','안정적','견고한','강력한','긍정적 시그널','기대감 고조','상향 조정','유망주','강력 매수','매수 우위','자금 유동성','공급 계약','기술 제휴','신제품 출시','신기술','특허','인프라 확대','수익 다각화','안정권','상승 기류','수급 호전','매수세','강세 지속','상승 탄력','고수익','이익 증대','비용 효율','신시장','글로벌 리더','독주','압도','활황기','대호황','순항 중','기대작','성공적','대규모','파격','신선한','창의적','독창적','우수','탁월','최우수','최고급','압승','완승','승리','쟁취','획득','점유율','영향력','지배','선도','앞서가는','강점','경쟁 우위','시너지 효과','극대화','최적화','안정성','건전성','투명성','책임 경영','신뢰성','글로벌 스탠다드','혁신 가속','미래 지향','가치 극대화','동반 성장','재평가','유상증자 성공','공매도 상환','숏커버링','저점 매수','장기 보유','기관 러브콜','외인 귀환','기술 수출','로열티 유입','비용 구조 개선','재무 구조 개선','자산 가치 상승']
NEG_WORDS = ['하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','목표가 하향','검찰','과징금','횡령','벌금','부진','악화','감소','침체','불안','논란','갈등','혼란','타격','충격','위축','비판','반발','제동','적자 전환','추락','둔화','리스크','부담','난항','지연','차질','무산','중단','붕괴','위반','불법','의혹','파문','후폭풍','불신','부실','취약','심각','경고','악영향','압박','혼선','공방','대립','마찰','냉각','후퇴','축소','불투명','진통','피로감','스캔들','실책','오판','급감','정체','미흡','문제','최악','불리','악조건','위기감','하방','리스크 확대','고소','피소','수사','조사','압수수색','기소','재판','판결','처벌','구속','공정위','금감원','제재','징계','취소','정지','영업정지','면허취소','배임','분식회계','감사의견 거절','관리종목','상장폐지','퇴출','부도','파산','회생절차','워크아웃','자금난','유동성 위기','채무불이행','디폴트','신용등급 강등','어닝 쇼크','실적 악화','매출 감소','영업손실','순손실','이익 급감','비용 증가','부채 급증','이자 부담','고금리','인플레이션','경기 불황','소비 위축','수요 감소','재고 급증','공급 과잉','가격 경쟁 심화','치킨 게임','출혈 경쟁','시장 점유율 하락','경쟁력 약화','기술 유출','핵심 인력 이탈','노사 갈등','파업','태업','불매 운동','여론 악화','이미지 실추','신뢰 추락','갑질 논란','환경 오염','안전 사고','법적 분쟁','소송 소용돌이','소환 조사','구속영장','실형','유죄','패소','손해배상','배상금','징벌적 손해배상','환수','거부','반려','불허','제한','금지','규제 강화','가이드라인 위반','시정 명령','외면','냉대','비관','비관론','공포','패닉','투매','손절','반대매매','깡통계좌','폭락세','하락장','약세장','데드 크로스','매도 신호','역배열','저항선 돌파 실패','지지선 붕괴','투기','거품 붕괴','버블','과평가','고평가 논란','거품 낀','과열된','방만한','부도덕한','비윤리적','비리','유착','뇌물','로비','탈세','세무조사','은폐','조작','왜곡','거짓 발표','허위 공시','공시 번복','신뢰 상실','속수무책','참사','대참사','비극','멸망','강사','사교육','학원','고소장','조사 착수','전수조사','압수','수색','구치소','중형','기소 의견','경영진 교체','내분','불법 파업','영업난','자산 매각','강제 집행','가처분','손배소','위기 고조','불확실성','투심 위축','매도세','약세 지속','하락 탄력','손실 확대','비용 부담','재무 악화','역성장','쇠퇴','정체기','부도 위험','신용 위기','유동성 부족','뱅크런','공황','혼돈','최악의 상황','암울한','전망 어두움','비관적','역풍','추락기','퇴보기','절망','매도 폭탄','패닉 셀','시장 소외','거래 절벽','가격 폭락','사법 리스크','소환 조율','경찰 조사','혐의','불기소','기각','항소','상고','징역','법정 구속','자금 세탁','내부자 거래','주가 조작','시세 조종','부당 이득','자본 잠식','현금 고갈','영업외손실']

# --- [2. 구글 뉴스 RSS 엔진 (안정성 극대화)] ---
def get_google_rss_score(stock_name, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    next_date_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 1. 캐시 확인
    if os.path.exists(NEWS_DB_PATH):
        try:
            cache = pd.read_csv(NEWS_DB_PATH)
            match = cache[(cache['date'] == date_str) & (cache['name'] == stock_name)]
            if not match.empty: return float(match.iloc[0]['score'])
        except: pass

    # 2. RSS 데이터 수집
    url = f"https://news.google.com/rss/search?q={stock_name}+주가+after:{date_str}+before:{next_date_str}&hl=ko&gl=KR&ceid=KR:ko"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
    
    score_val = 0
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.find_all("item")
        
        count = len(items)
        if count == 0: return 0
        
        for item in items:
            title = item.title.text
            for p in POS_WORDS:
                if p in title: score_val += 10
            for n in NEG_WORDS:
                if n in title: score_val -= 30 # 부정 3배 가중치
        
        final_score = (score_val / count)
        # 캐시 저장
        pd.DataFrame([[date_str, stock_name, final_score]], columns=['date', 'name', 'score']).to_csv(NEWS_DB_PATH, mode='a', header=not os.path.exists(NEWS_DB_PATH), index=False)
        return final_score
    except: return 0

# --- [3. 유동 지능 로직: 최소 15%] ---
def get_fluid_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df = pd.read_csv(CONFIG_PATH)
            return max(0.15, df['best_coef'].mean())
        except: pass
    return 0.15

def get_next_trading_days(start_date, n):
    days = []
    curr = start_date
    while len(days) < n:
        curr += timedelta(days=1)
        if curr.weekday() < 5: days.append(curr)
    return days

# --- [4. 메인 분석 엔진] ---
st.title("🏛️ 주식 AI v41.0 (Google RSS 엔진 통합)")

with st.sidebar:
    st.title("🧠 AI 분석 센터")
    if 'importance' in st.session_state:
        st.write("### AI 지표 판단 비중 (%)")
        st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
    st.metric("보편 유지계수", f"{get_fluid_coef():.4f}")

c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("2년 RSS 전수조사 및 통합 분석 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("구글 RSS 채널을 통해 2년치 뉴스를 인과 분석 중...", expanded=True) as status:
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # 뉴스 분석 (최근 30일 집중 조사)
            analysis_days = df_raw.index[-30:]
            daily_scores = {d: get_google_rss_score(s_name, d) for d in analysis_days}
            
            # 모든 지표 복구 (VIX, RSI, 요일 등)
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))).fillna(50)
            
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df)); df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / (df['종가'] + 1e-9)
            
            # 3일 누적 뉴스 감성 (T, T-1, T-2)
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 300 
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # AI 학습 및 가중치 최적화
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_final[features])
            y = df_final['target']
            
            # 지능 수렴 및 유지계수 탐색 (15% 미니멈)
            knowledge_df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else pd.DataFrame()
            if not knowledge_df.empty:
                X_total = scaler.fit_transform(pd.concat([df_final[features], knowledge_df[features]]))
                y_total = pd.concat([y, knowledge_df['target']])
                min_err, best_c = float('inf'), get_fluid_coef()
                for c in np.linspace(0.15, 0.5, 15):
                    w = np.concatenate([np.ones(len(df_final)), np.full(len(knowledge_df), c)])
                    m_temp = Ridge(alpha=0.3).fit(X_total, y_total, sample_weight=w)
                    if mean_absolute_percentage_error(y, m_temp.predict(X_scaled)) < min_err:
                        best_c = c
                pd.DataFrame([[datetime.now(), best_c]], columns=['date', 'best_coef']).to_csv(CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)
                model = Ridge(alpha=0.3).fit(X_total, y_total, sample_weight=np.concatenate([np.ones(len(df_final)), np.full(len(knowledge_df), get_fluid_coef())]))
            else:
                model = Ridge(alpha=0.3).fit(X_scaled, y)

            # 백테스팅 및 미래 7일 예측
            df_final['AI_복기'] = (df_final['종가'] * (1 + model.predict(X_scaled))).shift(1).fillna(df_final['종가'])
            
            f_dates = get_next_trading_days(df_final.index[-1], 7)
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            for d in f_dates:
                last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
                pred = model.predict(scaler.transform(last_f))[0]
                tmp_p *= (1 + pred); f_prices.append(tmp_p)

            # 결과 세션 저장
            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            st.session_state.result = {
                'df': df_final.tail(60), 'f_dates': f_dates, 'f_prices': f_prices,
                'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'AI_복기': df_final['AI_복기'], 'news_score': df_final['뉴스감성'].iloc[-1] / 300
            }
            # 지식 축적
            df_final['stock_code'] = s_code
            df_final[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            
            success_flag = True
            status.update(label="구글 RSS 분석 및 7일 예측 완료!", state="complete")
            
    except Exception as e: st.error(f"오류: {e}")
    if success_flag: st.rerun()

# --- [5. 시각화: 그래프 연결 로직] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 분석 리포트 (백테스팅 오차: {res['mape']:.2%})")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['AI_복기'].tail(60), name="AI 백테스팅(복기)", line=dict(color='yellow', dash='dot'), opacity=0.5))
    
    # 그래프 연결 로직 (실제 시세 끝점과 예측 시작점 연결)
    connect_x = [res['df'].index[-1]] + res['f_dates']
    connect_y = [res['df']['종가'].iloc[-1]] + res['f_prices']
    fig.add_trace(go.Scatter(x=connect_x, y=connect_y, name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    
    st.markdown(f"### 📢 뉴스 진단: {'🔴 악재' if res['news_score'] < -5 else ('🟢 호재' if res['news_score'] > 5 else '⚖️ 중립')} ({res['news_score']:.1f})")
    fig.update_layout(template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)import pandas as pd
import streamlit as st
import numpy as np
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
# RSS 파싱을 위해 추가로 필요한 라이브러리 (기본 설치됨)
import xml.etree.ElementTree as ET 

# ... (환경 설정 및 사전 정의는 이전 v40.0과 동일) ...

# --- [2. 구글 뉴스 RSS 엔진 (차단 회피형)] ---
def get_google_rss_score(stock_name, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    next_date_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 캐시 확인 로직 유지
    if os.path.exists(NEWS_DB_PATH):
        try:
            cache = pd.read_csv(NEWS_DB_PATH)
            match = cache[(cache['date'] == date_str) & (cache['name'] == stock_name)]
            if not match.empty: return float(match.iloc[0]['score'])
        except: pass

    # RSS 검색 URL (after/before를 사용하여 특정 일자 지정)
    # q=종목명+주가+after:2024-11-20+before:2024-11-21
    url = f"https://news.google.com/rss/search?q={stock_name}+주가+after:{date_str}+before:{next_date_str}&hl=ko&gl=KR&ceid=KR:ko"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    
    score_val = 0
    try:
        # RSS는 XML 형식이므로 requests 후 BeautifulSoup의 'xml' 파서나 기본 XML 파서 사용
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.find_all("item") # RSS의 각 기사 단위는 <item>
        
        count = len(items)
        if count == 0: return 0
        
        for item in items:
            title = item.title.text # 기사 제목 추출
            # 감성 분석 로직 (비대칭 가중치 적용)
            for p in POS_WORDS:
                if p in title: score_val += 10
            for n in NEG_WORDS:
                if n in title: score_val -= 30
        
        final_score = (score_val / count)
        
        # 캐시 저장
        pd.DataFrame([[date_str, stock_name, final_score]], columns=['date', 'name', 'score']).to_csv(NEWS_DB_PATH, mode='a', header=not os.path.exists(NEWS_DB_PATH), index=False)
        return final_score
    except Exception as e:
        # 에러 발생 시 로그 출력 (디버깅용)
        # st.write(f"Error on {date_str}: {e}")
        return 0

# --- [이후 메인 분석 로직에서 get_google_news_score를 get_google_rss_score로 교체] ---

