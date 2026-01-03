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
st.set_page_config(page_title="주식 AI v29.0 (Causal Learning)", layout="wide")
KST_NOW = datetime.now() + timedelta(hours=9)
DB_PATH = "stock_knowledge.csv"
CONFIG_PATH = "global_config.csv"

# --- [1. 초정밀 감성 사전 (각 200개 이상)] ---
POS_WORDS = [
    '상승','호재','수주','흑자','성공','최고','돌파','급등','강세','추천','목표가 상향','우상향','반등','M&A','신고가','회복','개선','호조','증가','확대','성장','안정','기대','성과','진전','활황','호황','순항','선전','약진','도약','혁신','정상화','상향','강화','지지','신뢰','합의','타결','협력','채택','승인','확정','수혜','유망','경쟁력','잠재력','모멘텀','전환점','긍정','안착','정착','반전','기회','낙관','상승세','회복세','성장세','개선세','탄탄','견조','순증','가속','촉진','확대 적용','신기록','우위','고무적','성과 확대','미래 성장','실적 개선','이익 증가','시장 확대','동력 확보','주도','수급 개선','외인 매수','기관 매수','순매수','저평가','가치 상승','배당 확대','자사주 매입','소각','재무 건전성','흑자 전환','턴어라운드','독점','점유율 상승','특허 취득','임상 성공','승인 획득','파트너십','조인트벤처','대규모 계약','어닝 서프라이즈','실적 호전','부채 감소','유동성 확보','신사업','성장 동력','구조조정 성공','비용 절감','효율화','생산성 향상','수요 급증','가격 인상','마진 확대','시장 지배력','압도적','독보적','고성장','지속 가능','유입','활기','장밋빛','활력','청신호','훈풍','쾌조','대박','잭팟','블루칩','가속도','선점','혁신적','차별화','우량','건실','탄탄한','활성화','상생','시너지','상호 보완','대세','주도주','대장주','테마 형성','자금 유입','신용 등급 상향','전망 밝음','낙관적','순풍','비상','도약기','전성기','황금기','역사적 신고가','연중 최고치','업황 개선','사이클 상향','저점 통과','바닥 확인','매수 신호','정배열','강한 흐름','상한가 안착','점상','연상','폭등','경신','가속 페달','탄력','가시화','본격화','가시적 성과','뚜렷한','명확한','압도적 1위','세계 최초','국내 최초','기술력 입증','신뢰도 제고','브랜드 가치','이미지 개선','고부가가치','영업외이익','순이익 급증','퀀텀 점프','글로벌 진출','독보적 기술','시장 선점','점유율 1위','호실적','현금 흐름 개선','수익성 개선','안정적','견고한','강력한','긍정적 시그널','기대감 고조','상향 조정','유망주','강력 매수','매수 우위','자금 유동성','공급 계약','기술 제휴','신제품 출시','신기술','특허','인프라 확대','수익 다각화','안정권','상승 기류','수급 호전','매수세','강세 지속','상승 탄력','고수익','이익 증대','비용 효율','신시장','글로벌 리더','독주','압도','활황기','대호황','순항 중','기대작','성공적','대규모','파격','신선한','창의적','독창적','우수','탁월','최우수','최고급','압승','완승','승리','쟁취','획득','점유율','영향력','지배','선도','앞서가는','강점','경쟁 우위','시너지 효과','극대화','최적화','안정성','건전성','투명성','책임 경영','신뢰성','글로벌 스탠다드','혁신 가속','미래 지향','가치 극대화','동반 성장','재평가','유상증자 성공','공매도 상환','숏커버링','저점 매수','장기 보유','기관 러브콜','외인 귀환','기술 수출','로열티 유입','비용 구조 개선','재무 구조 개선','자산 가치 상승'
]

NEG_WORDS = [
    '하락','악재','적자','위기','실패','최저','우려','약세','매도','급락','손실','쇼크','목표가 하향','검찰','과징금','횡령','벌금','부진','악화','감소','침체','불안','논란','갈등','혼란','타격','충격','위축','비판','반발','제동','적자 전환','추락','둔화','리스크','부담','난항','지연','차질','무산','중단','붕괴','위반','불법','의혹','파문','후폭풍','불신','부실','취약','심각','경고','악영향','압박','혼선','공방','대립','마찰','냉각','후퇴','축소','불투명','진통','피로감','스캔들','실책','오판','급감','정체','미흡','문제','최악','불리','악조건','위기감','하방','리스크 확대','고소','피소','수사','조사','압수수색','기소','재판','판결','처벌','구속','공정위','금감원','제재','징계','취소','정지','영업정지','면허취소','배임','분식회계','감사의견 거절','관리종목','상장폐지','퇴출','부도','파산','회생절차','워크아웃','자금난','유동성 위기','채무불이행','디폴트','신용등급 강등','어닝 쇼크','실적 악화','매출 감소','영업손실','순손실','이익 급감','비용 증가','부채 급증','이자 부담','고금리','인플레이션','경기 불황','소비 위축','수요 감소','재고 급증','공급 과잉','가격 경쟁 심화','치킨 게임','출혈 경쟁','시장 점유율 하락','경쟁력 약화','기술 유출','핵심 인력 이탈','노사 갈등','파업','태업','불매 운동','여론 악화','이미지 실추','신뢰 추락','갑질 논란','환경 오염','안전 사고','법적 분쟁','소송 소용돌이','소환 조사','구속영장','실형','유죄','패소','손해배상','배상금','징벌적 손해배상','환수','거부','반려','불허','제한','금지','규제 강화','가이드라인 위반','시정 명령','외면','냉대','비관','비관론','공포','패닉','투매','손절','반대매매','깡통계좌','폭락세','하락장','약세장','데드 크로스','매도 신호','역배열','저항선 돌파 실패','지지선 붕괴','투기','거품 붕괴','버블','과평가','고평가 논란','거품 낀','과열된','방만한','부도덕한','비윤리적','비리','유착','뇌물','로비','탈세','세무조사','은폐','조작','왜곡','거짓 발표','허위 공시','공시 번복','신뢰 상실','속수무책','참사','대참사','비극','멸망','강사','사교육','학원','고소장','조사 착수','전수조사','압수','수색','구치소','중형','기소 의견','경영진 교체','내분','불법 파업','영업난','자산 매각','강제 집행','가처분','손배소','위기 고조','불확실성','투심 위축','매도세','약세 지속','하락 탄력','손실 확대','비용 부담','재무 악화','역성장','쇠퇴','정체기','부도 위험','신용 위기','유동성 부족','뱅크런','공황','혼돈','최악의 상황','암울한','전망 어두움','비관적','역풍','추락기','퇴보기','절망','매도 폭탄','패닉 셀','시장 소외','거래 절벽','가격 폭락'
]

# --- [2. 일자별 뉴스 감성 점수 산출 함수] ---
def get_daily_score(stock_name, target_date):
    date_str = target_date.strftime('%Y.%m.%d')
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=4&ds={date_str}&de={date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        score_val = 0
        count = 0
        for title in headlines:
            text = title.get_text()
            for p in POS_WORDS:
                if p in text: score_val += 10
            for n in NEG_WORDS:
                if n in text: score_val -= 30 # 부정 가중치 3배
            count += 1
        return (score_val / count) if count > 0 else 0
    except: return 0

# --- [3. 지능 수렴 및 영업일 로직] ---
def get_converged_coef():
    if os.path.exists(CONFIG_PATH):
        try:
            df_cfg = pd.read_csv(CONFIG_PATH)
            return max(0.15, df_cfg['best_coef'].mean()) if not df_cfg.empty else 0.15
        except: pass
    return 0.15

def get_next_trading_days(start_date, n):
    days = []
    curr = start_date
    while len(days) < n:
        curr += timedelta(days=1)
        if curr.weekday() < 5: days.append(curr)
    return days

# --- [4. 사이드바 UI] ---
knowledge_df = pd.DataFrame()
if os.path.exists(DB_PATH):
    try:
        temp_df = pd.read_csv(DB_PATH, dtype={'stock_code': str})
        if all(col in temp_df.columns for col in ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI', 'target']):
            knowledge_df = temp_df
    except: pass

with st.sidebar:
    st.title("🧠 AI 분석 센터")
    side_tab1, side_tab2 = st.tabs(["🏛️ 지능 상태", "📊 변수 영향력"])
    with side_tab1:
        st.metric("수렴 유지계수", f"{get_converged_coef():.4f}")
        if not knowledge_df.empty:
            st.write(f"📚 누적 데이터: `{len(knowledge_df)}`개")
    with side_tab2:
        if 'importance' in st.session_state:
            st.write("### AI 가중치 비중 (%)")
            st.bar_chart(st.session_state.importance.set_index('지표'), color='#00CCFF')
        else:
            st.info("분석 실행 시 가중치가 표시됩니다.")

# --- [5. 메인 분석 로직] ---
st.title("🏛️ 주식 AI v29.0 (과거 인과관계 학습 엔진)")
c1, c2 = st.columns(2)
with c1: s_code = st.text_input("종목 코드", value="005930")
with c2: s_name = st.text_input("종목 이름", value="삼성전자")

if st.button("뉴스-주가 인과관계 백테스팅 및 예측 시작", use_container_width=True):
    success_flag = False
    try:
        with st.status("AI가 과거 30일간의 뉴스를 전수조사하며 인과관계를 학습 중...", expanded=True) as status:
            # 데이터 수집
            start_date = KST_NOW - timedelta(days=730)
            df_raw = fdr.DataReader(s_code, start_date).rename(columns={'Close':'종가','Volume':'거래량'})
            
            # [중요] 과거 뉴스 인과관계 전수조사 (최근 35거래일)
            # 24일 주가를 분석하기 위해 23, 22, 21일 뉴스를 찾는 로직이 여기서 실행됨
            analysis_days = df_raw.index[-35:]
            daily_scores = {}
            for d in analysis_days:
                daily_scores[d] = get_daily_score(s_name, d)
                time.sleep(0.05) # 서버 부하 방지
            
            # 지표 생성
            vix = fdr.DataReader('^VIX', start_date)[['Close']].rename(columns={'Close': 'VIX'})
            df = df_raw.join(vix).ffill().fillna(20)
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = (100 - (100 / (1 + (gain / loss)))).fillna(50)
            df['target'] = df['종가'].pct_change().shift(-1)
            df['날짜지수'] = np.arange(len(df))
            df['요일'] = df.index.weekday
            df['변동성'] = (df['High'] - df['Low']) / df['종가']
            
            # [핵심] 뉴스 감성 3일 시차 롤링 적용
            # T일의 뉴스감성 변수는 T, T-1, T-2일의 점수 평균임
            temp_scores = pd.Series(0.0, index=df.index)
            for d, s in daily_scores.items(): temp_scores[d] = s
            df['뉴스감성'] = temp_scores.rolling(window=3, min_periods=1).mean() * 200 # 가중치 상향
            
            df_final = df.dropna()
            features = ['날짜지수', '요일', '거래량', '변동성', '뉴스감성', 'VIX', 'RSI']
            
            # AI 학습
            scaler = StandardScaler()
            X_curr = df_final[features]
            y_curr = df_final['target']
            X_curr_scaled = scaler.fit_transform(X_curr)
            
            cur_coef = get_converged_coef()
            if not knowledge_df.empty:
                X_total_scaled = scaler.fit_transform(pd.concat([X_curr, knowledge_df[features]]))
                y_total = pd.concat([y_curr, knowledge_df['target']])
                
                min_err, best_l = float('inf'), cur_coef
                for c in np.linspace(0.15, 0.5, 15):
                    w = np.concatenate([np.ones(len(X_curr)), np.full(len(knowledge_df), c)])
                    m_temp = Ridge(alpha=0.3).fit(X_total_scaled, y_total, sample_weight=w)
                    err = mean_absolute_percentage_error(y_curr, m_temp.predict(scaler.transform(X_curr)))
                    if err < min_err: min_err = err; best_l = c
                
                pd.DataFrame([[datetime.now(), best_l]], columns=['date', 'best_coef']).to_csv(CONFIG_PATH, mode='a', header=not os.path.exists(CONFIG_PATH), index=False)
                weights = np.concatenate([np.ones(len(X_curr)), np.full(len(knowledge_df), get_converged_coef())])
                model = Ridge(alpha=0.3).fit(X_total_scaled, y_total, sample_weight=weights)
            else:
                model = Ridge(alpha=0.3).fit(X_curr_scaled, y_curr)

            # 백테스팅 복기 및 미래 예측
            df_final['AI_복기'] = (df_final['종가'] * (1 + model.predict(scaler.transform(X_curr)))).shift(1).fillna(df_final['종가'])
            f_dates = get_next_trading_days(df_final.index[-1], 7)
            f_prices, tmp_p = [], df_final['종가'].iloc[-1]
            last_f = df_final[features].iloc[-1:].copy()
            for d in f_dates:
                last_f['날짜지수'] += 1; last_f['요일'] = d.weekday()
                pred = model.predict(scaler.transform(last_f))[0]
                tmp_p *= (1 + pred)
                f_prices.append(tmp_p)

            # 비중(%) 저장
            abs_coef = np.abs(model.coef_)
            st.session_state.importance = pd.DataFrame({'지표': features, '가중치': (abs_coef / np.sum(abs_coef) * 100).round(1)})
            st.session_state.result = {
                'name': s_name, 'code': s_code, 'mape': mean_absolute_percentage_error(df_final['종가'], df_final['AI_복기']),
                'df': df_final.tail(60), 'f_dates': f_dates, 'f_prices': f_prices,
                'news_score': df_final['뉴스감성'].iloc[-1] / 200
            }
            
            df_final['stock_code'] = s_code
            df_final[features + ['target', 'stock_code']].tail(30).to_csv(DB_PATH, mode='a', header=not os.path.exists(DB_PATH), index=False)
            success_flag = True
            status.update(label="인과관계 학습 및 예측 완료!", state="complete")
            
    except Exception as e: st.error(f"분석 중 오류 발생: {e}")
    if success_flag: st.rerun()

# --- [6. 시각화] ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.subheader(f"📊 {res['name']} 리포트 (백테스팅 오차율: {res['mape']:.2%})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['종가'], name="실제 시세", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['AI_복기'], name="AI 백테스팅(복기)", line=dict(color='yellow', dash='dot'), opacity=0.5))
    fig.add_trace(go.Scatter(x=[res['df'].index[-1]] + res['f_dates'], y=[res['df']['종가'].iloc[-1]] + res['f_prices'], name="미래 7일 예측", line=dict(color='#FF3366', width=4), mode='lines+markers'))
    st.markdown(f"### 📢 3일 누적 뉴스 감성: {'🔴 악재' if res['news_score'] < -5 else ('🟢 호재' if res['news_score'] > 5 else '⚖️ 중립')} ({res['news_score']:.1f})")
    st.plotly_chart(fig, use_container_width=True)
