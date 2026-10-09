# core/data_pipeline.py
import yfinance as yf
import pandas as pd

class ETFDataPipeline:
    # 建構子：記住要抓的股票代號和時間範圍
    def __init__(self, asset: list, start_date: str, end_date: str):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None  # 用來存抓下來的股價

    # 抓資料方法：去網路上撈資料並洗乾淨
    def fetch_data(self) -> pd.DataFrame:
        print(f"開始抓取數據: {self.asset}")
        
        # 1. 去 Yahoo 抓資料，並只留下『Close(收盤價)』這一欄
        self.raw_data = yf.download(self.asset, start=self.start_date, end=self.end_date)['Close']
        
        # 2. 刪除所有空白格（把週末沒開盤、缺漏的日期整列刪掉）
        self.raw_data = self.raw_data.dropna()
        return self.raw_data
