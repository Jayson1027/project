import asyncio
import urllib.request
from urllib.error import URLError
import xml.etree.ElementTree as ET
from datetime import datetime
import re 
import os    

NEWS_SOURCES = {
    
}

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

    def fetch_feed_sync(self, source_name: str, url: str) -> list[dict]:
        cleaned_articles = [] 
        req = urllib.request.Request(url, headers=self.headers)
        
        try:
            with urllib.request.urlopen(req, timeout=15.0) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            for item in root.findall('./channel/item'):
                link_elem = item.find('link')
                link = link_elem.text.strip() if (link_elem is not None and link_elem.text) else ""

                if not link or link in self.seen_urls:
                    continue

                title_elem = item.find('title')
                raw_title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else "No Title"
                
                pub_elem = item.find('pubDate')
                pub_date = pub_elem.text.strip() if (pub_elem is not None and pub_elem.text) else "Unknown"

                desc_elem = item.find('description')
                raw_desc = desc_elem.text if desc_elem is not None else ""

                clean_title = self.clean_text(raw_title)
                clean_desc = self.clean_text(raw_desc)

                if clean_title:
                    cleaned_articles.append({
                        "source": source_name,
                        "publish_date": pub_date,
                        "title": clean_title,
                        "content": clean_desc
                    })
                
                self.seen_urls.add(link)

        except Exception as e:
            print(f"[{source_name}] 抓取或解析發生錯誤: {e}")

        return cleaned_articles