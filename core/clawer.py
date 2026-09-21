import asyncio
import urllib.request
from urllib.error import URLError
import xml.etree.ElementTree as ET
from datetime import datetime
import re   


class FinancialNewsCrawler:

    def __init__(self, sources: dict[str, str], interval_seconds: int = 300, output_file: str = "news_data.json"):
        self.sources = sources
        self.interval_seconds = interval_seconds
        self.output_file = output_file   
        self.seen_urls: set[str] = set() 
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    def clean_text(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        text = re.sub(r'<[^>]+>', '', raw_text)
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&quot;', '"')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def fetch_full_article_sync(self, url: str) -> str:
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10.0) as response:
                html_data = response.read().decode('utf-8', errors='ignore')
            
            paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_data, re.IGNORECASE | re.DOTALL)
            
            if not paragraphs:
                return ""
                
            full_text = " ".join(paragraphs)
            return self.clean_text(full_text)
            
        except Exception as e:
            return ""

    def fetch_feed_sync(self, source_name: str, url: str) -> list[dict]:
        cleaned_articles = [] 
        req = urllib.request.Request(url, headers=self.headers)
        
        try:
            with urllib.request.urlopen(req, timeout=15.0) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            for item in root.findall('./channel/item')[:5]:
                link_elem = item.find('link')
                link = link_elem.text.strip() if (link_elem is not None and link_elem.text) else ""

                if not link or link in self.seen_urls:
                    continue

                title_elem = item.find('title')
                raw_title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else "No Title"
                
                pub_elem = item.find('pubDate')
                pub_date = pub_elem.text.strip() if (pub_elem is not None and pub_elem.text) else "Unknown"

                clean_title = self.clean_text(raw_title)

                full_content = self.fetch_full_article_sync(link)
                if not full_content:
                    desc_elem = item.find('description')
                    raw_desc = desc_elem.text if desc_elem is not None else ""
                    full_content = self.clean_text(raw_desc)
                    
                if clean_title:
                    cleaned_articles.append({
                        "source": source_name,
                        "published_at": pub_date,
                        "title": clean_title,
                        "content": full_content
                    })
                
                self.seen_urls.add(link)

        except Exception as e:
            print(f"[{source_name}] 抓取或解析發生錯誤: {e}")

        return cleaned_articles

    def save_to_txt(self, new_articles: list[dict]):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.output_file, 'a', encoding='utf-8') as f:
            for article in new_articles:
                text_line = (
                    f"抓取時間: {timestamp} | "
                    f"發布時間: {article['published_at']} | "
                    f"來源: [{article['source']}] | "
                    f"標題: {article['title']} | "
                    f"內文: {article['content']}\n"
                )
                f.write(text_line)
            
        print(f"✅ 已將 {len(new_articles)} 筆包含日期、標題與內文的文本寫入 {self.output_file}")

    async def run_once(self):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 開始抓取與清洗焦點新聞...")
        tasks = [
            asyncio.to_thread(self.fetch_feed_sync, source, url)
            for source, url in self.sources.items()
        ]

        results = await asyncio.gather(*tasks)
        
        new_articles = [article for sublist in results for article in sublist]

        if not new_articles:
            print("目前沒有新的新聞。")
            return            
        self.save_to_txt(new_articles)
        for i, article in enumerate(new_articles[:3], 1):
            print(f"{i}. [{article['source']}] {article['title'][:50]}...")

    async def start_polling(self):
        print(f"非同步爬蟲啟動，每 {self.interval_seconds} 秒抓取並輸出非結構化文字...")
        while True:
            await self.run_once()
            await asyncio.sleep(self.interval_seconds)