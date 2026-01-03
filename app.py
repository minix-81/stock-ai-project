def get_daily_news_score(stock_name, target_date):
    date_str = target_date.strftime('%Y.%m.%d')
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=4&ds={date_str}&de={date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.select(".news_tit")
        
        score_val = 0
        count = len(headlines) # 검색된 기사 개수 파악
        
        if count == 0:
            return 0, 0 # 기사가 없으면 점수 0, 개수 0 반환
            
        for title in headlines:
            text = title.get_text()
            for p in POS_WORDS:
                if p in text: score_val += 10
            for n in NEG_WORDS:
                if n in text: score_val -= 30
        
        return (score_val / count), count # 점수와 개수를 함께 반환
    except:
        return 0, -1 # 에러 발생 시 개수를 -1로 반환하여 구분
