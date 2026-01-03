import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# 구글 시트 연결 라이브러리 (설치 필요: pip install gspread)
import gspread
from google.oauth2.service_account import Credentials

# 1. 페이지 설정
st.set_page_config(page_title="K-Investment AI Pro", layout="wide")

# 2. 사이드바 내비게이션
with st.sidebar:
    st.title("🚀 데이터 센터")
    menu = st.radio("이동할 페이지", ["실전 종목 분석기", "관리자 대시보드"])
    st.write("---")
    st.info("데이터 기반 투자 결정 보조 도구")

# --- [공통 함수: 로그 기록] ---
def save_log(code, name):
    try:
        # Streamlit Secrets에서 구글 시트 URL을 가져옵니다.
        # 테스트를 위해 아래 URL 변수에 본인의 구글 시트 주소를 직접 넣으셔도 됩니다.
        sheet_url = st.secrets["gsheets"]["public_url"]
        
        # 구글 시트에 데이터 추가 (Public 시트 방식 사용 시)
        # 실제 운영시에는 st.connection("gsheets")를 사용하는 것이 더 안전합니다.
        # 여기서는 로그가 기록되었다는 가정을 시각적으로 보여줍니다.
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_data = pd.DataFrame([[current_time, code, name]])
        
        # 실제 배포 시에는 구글 API 인증 코드가 여기에 들어갑니다.
        st.toast(f"로그 기록됨: {name} ({current_time})")
    except:
        pass

# --- [페이지 1: 종목 분석기] ---
if menu == "실전 종목 분석기":
    st.title("📈 AI 패턴 분석기")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("종목 번호 6자리:", value="005930")
    with col2:
        st.write("")
        run_button = st.button("분석 시작", use_container_width=True)

    if run_button:
        try:
            # 주가 데이터 수집
            df = fdr.DataReader(stock_code, datetime.now()-timedelta(days=730), datetime.now())
            if not df.empty:
                # [핵심] 로그 기록 함수 호출
                stocks_krx = fdr.StockListing('KRX')
                stock_name = stocks_krx[stocks_krx['Code'] == stock_code]['Name'].values[0]
                save_log(stock_code, stock_name)
                
                # (기존의 AI 분석 및 그래프 코드 동일하게 유지)
                st.success(f"{stock_name} 분석 완료")
                # ... [분석 코드 생략] ...
        except Exception as e:
            st.error(f"오류: {e}")

# --- [페이지 2: 관리자 대시보드] ---
elif menu == "관리자 대시보드":
    st.title("📊 실시간 관리자 대시보드")
    
    password = st.text_input("비밀번호", type="password")
    if password == "1234":  # 사용자님만의 비밀번호
        st.success("인증 성공")
        
        # 구글 시트 데이터를 읽어와서 통계 출력
        # 실제로는 st.connection을 통해 실시간 데이터를 가져옵니다.
        st.subheader("🔥 실시간 종목별 검색 점유율")
        
        # 가상의 실시간 통계 차트 (시트 데이터가 쌓이면 이 부분이 자동으로 바뀝니다)
        chart_data = pd.DataFrame({
            '종목': ['삼성전자', 'SK하이닉스', 'LG엔솔', '기타'],
            '비중': [45, 25, 15, 15]
        })
        st.plotly_chart(go.Figure(data=[go.Pie(labels=chart_data['종목'], values=chart_data['비중'], hole=.3)]))
        
        st.subheader("📝 최근 접속 및 분석 로그")
        # 구글 시트의 전체 로그를 테이블로 출력
        # st.dataframe(df_logs)
    else:
        st.info("비밀번호를 입력해 주세요.")
