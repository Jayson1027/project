from core.data_pipeline import ETFDataPipeline
from core.clawer import FinancialNewsCrawler
import asyncio
import datetime

NEWS_SOURCES = {
    # 1. 聯準會官方新聞（專注利率、貨幣政策，對 TLT / SPY 極度重要，零雜訊）
    "Federal_Reserve": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    
    # 2. CNBC 市場頻道（過濾掉政治與生活新聞，只留美股、美債、黃金、油價）
    "CNBC_Markets": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    
    "WSJ_Tech": "https://feeds.content.dowjones.io/public/rss/RSSWSJD"
}

def main():
    target_assets=["SPY","QQQ","TLT","GLD"]
    today=datetime.date.today()
    one_week_ago=today-datetime.timedelta(days=7)

    start_date=one_week_ago.strftime('%Y-%m-%d')
    end_date=today.strftime('%Y-%m-%d')
    print(f"{start_date}->{end_date}")

    pipeline=ETFDataPipeline(asset=target_assets,start_date=start_date,end_date=end_date)
    price=pipeline.fetch_data()

    print(price)


    crawler = FinancialNewsCrawler(
        sources=NEWS_SOURCES, 
        interval_seconds=300, 
        output_file="data/news.txt"
    )
    
    try:
        asyncio.run(crawler.start_polling())
    except KeyboardInterrupt:
        # 中斷指令:終端機按下Ctrl+C
        print("\n新聞監控已手動停止。")

if __name__ == "__main__":
    main()
