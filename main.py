from core.data_pipeline import ETFDataPipeline
from core.clawer import FinancialNewsCrawler
import asyncio
import datetime

NEWS_SOURCES = {
    "Yahoo_Finance": "https://finance.yahoo.com/news/rssindex",
    "CNBC_Top_News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?profile=120000000",
    "WSJ_Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
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
