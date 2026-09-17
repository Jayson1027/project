import yfinance as yf
import pandas as pd

class ETFDataPipeline:
    def __init__(self,asset:list,start_date:str,end_date:str):
        self.asset=asset
        self.start_date=start_date
        self.end_date=end_date
        self.raw_data=None

    def fetch_data(self) -> pd.DataFrame:
        print(f"開始抓取數據{self.asset}")
        self.raw_data=yf.download(self.asset,start=self.start_date,end=self.end_date)['Close']
        self.raw_data=self.raw_data.dropna()
        print("數據清理完成")
        return self.raw_data
