#!/usr/bin/env python3
import argparse
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except Exception:
    FEEDPARSER_AVAILABLE = False

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except Exception:
    NEWSPAPER_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
BASE_DIR = Path("assets/rag_assets")
REQUEST_TIMEOUT = 12
DEFAULT_LIMIT = 12


# ---------------- Utility ----------------
def sanitize_filename(s: str, maxlen: int = 140) -> str:
    if not s:
        return "untitled"
    s = s.strip()
    out = []
    for c in s:
        if c.isalnum() or c in (" ", "_", "-"):
            out.append(c)
        else:
            out.append("_")
    name = "_".join("".join(out).split())
    return name[:maxlen] or "untitled"


def unique_path(folder: Path, base: str, ext: str = ".txt") -> Path:
    name = sanitize_filename(base)[:110]
    p = folder / f"{name}{ext}"
    if not p.exists():
        return p
    h = hashlib.sha1((base + str(time.time())).encode()).hexdigest()[:8]
    return folder / f"{name}_{h}{ext}"


def ensure_dirs(ticker: str) -> Tuple[Path, Path]:
    t = ticker.strip().upper()
    company = BASE_DIR / t
    news = company / "news"
    company.mkdir(parents=True, exist_ok=True)
    news.mkdir(parents=True, exist_ok=True)
    return company, news


# ---------------- News Fetchers ----------------
def google_news_rss(query: str, limit: int = DEFAULT_LIMIT):
    if not FEEDPARSER_AVAILABLE:
        return []
    q_enc = requests.utils.quote(query, safe="")
    url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        entries = []
        for e in (feed.entries or [])[:limit]:
            entries.append({
                "title": e.get("title", "").strip(),
                "link": e.get("link", "").strip(),
                "summary": e.get("summary", ""),
            })
        return entries
    except Exception:
        return []


def bing_news_html(query: str, limit: int = DEFAULT_LIMIT):
    url = "https://www.bing.com/news/search"
    headers = {"User-Agent": USER_AGENT}
    params = {"q": query, "form": "QBNH"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen = set()
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"].strip()
        if not text or not href or len(text) < 20:
            continue
        if href.startswith("/") or "bing.com" in href:
            continue
        if href in seen:
            continue
        seen.add(href)
        results.append({"title": text, "link": href, "summary": ""})
        if len(results) >= limit:
            break
    return results


def collect_headlines(ticker: str, limit: int = DEFAULT_LIMIT):
    queries = [ticker, f"{ticker}.NS", f"{ticker} stock", f"{ticker} news"]
    collected = []
    seen = set()
    for q in queries:
        if FEEDPARSER_AVAILABLE:
            entries = google_news_rss(q, limit)
            for e in entries:
                if e["link"] not in seen:
                    collected.append(e)
                    seen.add(e["link"])
                    if len(collected) >= limit:
                        return collected
        entries = bing_news_html(q, limit)
        for e in entries:
            if e["link"] not in seen:
                collected.append(e)
                seen.add(e["link"])
                if len(collected) >= limit:
                    return collected
        time.sleep(0.2)
    return collected


# ---------------- Article Extraction ----------------
def extract_full_article(url: str):
    headers = {"User-Agent": USER_AGENT}
    if NEWSPAPER_AVAILABLE:
        try:
            art = Article(url)
            art.download()
            art.parse()
            text = art.text.strip()
            if len(text) > 100:
                return text
        except Exception:
            pass
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(paras).strip()
        return text if len(text) > 80 else ""
    except Exception:
        return ""


# ---------------- Fundamentals ----------------
def save_fundamentals(company_folder: Path, ticker: str):
    if not YFINANCE_AVAILABLE:
        return
    try:
        t = yf.Ticker(f"{ticker}.NS")
        info = t.info or {}
        bs = None
        try:
            bs = t.balance_sheet
        except Exception:
            bs = None
        path = company_folder / "fundamentals.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Ticker: {ticker}\nFetched: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for k in ("longName", "shortName", "sector", "industry", "marketCap", "regularMarketPrice"):
                if k in info:
                    f.write(f"{k}: {info[k]}\n")
            f.write("\nAll Info (truncated):\n")
            for k in sorted(info.keys()):
                v = repr(info[k])[:300]
                f.write(f"{k}: {v}\n")
            f.write("\nBalance Sheet (repr):\n")
            f.write(str(bs))
        print(f"Saved fundamentals: {path}")
    except Exception as e:
        print(f"Failed fundamentals for {ticker}: {e}")


# ---------------- Save Articles ----------------
def save_article(news_folder: Path, ticker: str, entry: dict):
    title = entry.get("title") or entry.get("link") or f"{ticker}_news"
    link = entry.get("link", "")
    summary = entry.get("summary", "")
    text = extract_full_article(link)
    if not text:
        text = BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)
    if not text:
        text = "Could not extract article content."

    filename = unique_path(news_folder, title, ".txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\nSource: {link}\n\n{text}")
    print(f"Saved article: {filename.name}")


# ---------------- Main ----------------
def fetch_for_ticker(ticker: str, limit: int = DEFAULT_LIMIT):
    company_folder, news_folder = ensure_dirs(ticker)
    headlines = collect_headlines(ticker, limit)
    if not headlines:
        stub = news_folder / "no_headlines_found.txt"
        with open(stub, "w", encoding="utf-8") as f:
            f.write(f"No headlines found for {ticker} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"No headlines for {ticker}.")
    else:
        for h in headlines:
            try:
                save_article(news_folder, ticker, h)
            except Exception as e:
                print(f"Error saving article: {e}")
    save_fundamentals(company_folder, ticker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*", help="Tickers (e.g. RELIANCE SBIN TCS)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    tickers = args.tickers or input("Enter tickers: ").split()
    if not tickers:
        print("No tickers provided.")
        return

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        print(f"\n=== Processing {t} ===")
        fetch_for_ticker(t, args.limit)
    print("\nAll done. Check assets/rag_assets/<TICKER>/news/ and fundamentals.txt")


if __name__ == "__main__":
    main()
