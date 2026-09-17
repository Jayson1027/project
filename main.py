from core.data_pipeline import ETFDataPipeline
import datetime

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

main()
