from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import yfinance as yf
import os

import pandas as pd
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from typing import Optional
import logging
from pathlib import Path
import glob
from rag_setup import fetch_for_ticker
import traceback
import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ------------------------
# LLMs
llm = LLM(model="openai/gpt-4o-mini", stop=["END"], seed=42)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
openai_llm = OpenAI(model="gpt-4o-mini")
client = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0,
)


# ------------------------
# Tools
# ------------------------


class SemanticChromaRAG:
    def __init__(self, docs_path: str, persist_directory: str = "./chroma_db"):
        loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")

        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
        )
        semantic_chunks = semantic_chunker.create_documents(
            [d.page_content for d in documents]
        )
        print(f"Created {len(semantic_chunks)} semantic chunks.")

        self.vectordb = Chroma.from_documents(
            semantic_chunks, embedding=embeddings, persist_directory=persist_directory
        )
        self.vectordb.persist()

        self.retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "search_type": "mmr",
                "fetch_k": 10,
                "lambda_mult": 0.5,
            },
        )
        self.qa_chain = self.retriever | openai_llm

    def query(self, text: str):
        """Run a query over the semantic chunks using the QA chain."""
        return self.qa_chain.invoke(text)


_chromarag: Optional[SemanticChromaRAG] = None

def get_chromarag(force_reinit: bool = False):
    """Get or create the ChromaRAG instance (lazy initialization)"""
    global _chromarag
    
    docs_path = Path("assets/rag_assets/")
    if not docs_path.exists():
        print("No documents directory found")
        return None
    
    pdf_files = list(docs_path.glob("**/*.pdf"))
    if not pdf_files:
        print("No PDF documents found in assets/rag_assets/")
        return None
    
    if _chromarag is None or force_reinit:
        print("Initializing ChromaRAG...")
        _chromarag = SemanticChromaRAG(docs_path="assets/rag_assets/")
        print("ChromaRAG initialization complete")
    return _chromarag


@tool
def CustomRagTool(query: str) -> str:
    """
    Custom RAG tool to solve fundamental questions using uploaded documents.
    Use this for non balance sheet related questions.
    
    Args:
        query: The query to search in the vector store
    """
    chromarag = get_chromarag()
    if chromarag is None:
        return "No documents available for analysis. Please upload documents first."
    
    try:
        results = chromarag.retriever.get_relevant_documents(query)
        ans = ""
        for result in results:
            ans += result.page_content
        return ans if ans else "No relevant information found in documents."
    except Exception as e:
        return f"Error searching documents: {str(e)}"

@tool
def getNewsBodyTool(*args, **kwargs) -> list:
    """
    Get the news body for a company by reading all text files under
    assets/rag_assets/{stock}/news/.
    """
    stock = InvestmentCrew.stock
    print(f"[getNewsBodyTool] Using stock ticker: {stock}")

    base_dir = os.path.join("assets", "rag_assets", str(stock), "news")    
    final_news_content = []

    if not os.path.isdir(base_dir):
        print(f"[getNewsBodyTool] Directory not found: {base_dir}")
    else:
        patterns = ["*.txt", "*.md", "*.text"]
        files = []
        for pattern in patterns:
            files.extend(sorted(glob.glob(os.path.join(base_dir, pattern))))

        for fp in files:
            try:
                if not os.path.isfile(fp):
                    continue
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                    if content:
                        final_news_content.append(content)
            except Exception as e:
                print(f"[getNewsBodyTool] Failed to read {fp}: {e}")

    if not final_news_content:
        print(f"[getNewsBodyTool] ⚠️ No news found for {stock}.")
    else:
        print(f"[getNewsBodyTool] ✅ Found {len(final_news_content)} news articles for {stock}.")

    return final_news_content

@tool
def findnewsTool(*args, **kwargs) -> str:
    """
    Fetch news articles for a company and save them under
    assets/rag_assets/{stock}/news/.
    """
    stock = InvestmentCrew.stock
    print(f"[findnewsTool] Using stock ticker: {stock}")

    try:
        fetch_for_ticker(stock, limit=12)
        return f"News articles fetched and saved for {stock}."
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"[findnewsTool] Error fetching news for {stock}: {e}\n{traceback_str}")
        return f"Failed to fetch news for {stock}: {e}"

@tool
def getAnnualisedVolatilityTool(*args, **kwargs) -> str:
    """
    Get the annualised volatility for a company
    Args:
        stock (str): Stock ticker
    """
    # Access the stock from the class variable
    stock = InvestmentCrew.stock
    print(f"[getAnnualisedVolatilityTool] Using stock ticker: {stock}")
    dat = yf.Ticker(f"{stock}")
    df = dat.history(period="3mo")
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    volatility = log_returns.std() * (252**0.5)
    return volatility


@tool
def getAnnualisedReturnTool(*args, **kwargs) -> float:
    """
    Get the annualised return for a company
    Args:
        stock (str): Stock ticker
    """
    # Access the stock from the class variable
    stock = InvestmentCrew.stock
    print(f"[getAnnualisedReturnTool] Using stock ticker: {stock}")
    dat = yf.Ticker(f"{stock}")
    df = dat.history(period="3mo")
    cummulative_return = (
        float(df["Close"].iloc[-1])
        / float(df["Close"].iloc[0])
    ) - 1
    annualised_return = (1 + cummulative_return) ** (252 / len(df)) - 1
    return annualised_return


@tool
def fundamental_analysis_tool(*args, **kwargs):
    """Tool to analyze the BalanceSheet of a company and provide a summary"""
    # Access the stock from the class variable
    stock = InvestmentCrew.stock
    print(f"[fundamental_analysis_tool] Using stock ticker: {stock}")
    # Get the stock balance sheet
    dat = yf.Ticker(f"{stock}")
    balance_sheet_data = dat.balance_sheet

    # Create messages
    messages = [
        SystemMessage(
            content="You are a financial analyst. Provide correct answers only. If you don't know, say so."
        ),
        HumanMessage(
            content=f"Here is a Pandas DataFrame:\n{balance_sheet_data}\n\nSummarize the financial results in INR. Don't ask for anything else."
        ),
    ]

    # Get response
    response = client(messages)

    # Access content
    summary = response.content
    return summary

@tool
def getMovingAveragesTool(*args, **kwargs) -> dict:
    """
    Compute short-term and long-term moving averages for the company's stock.
    Returns both SMA and EMA for 20 and 50 days.
    """
    stock = InvestmentCrew.stock
    print(f"[getMovingAveragesTool] Using stock ticker: {stock}")
    df = yf.Ticker(stock).history(period="6mo")
    if df.empty:
        return {"error": "No data available"}

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    latest = df.iloc[-1]
    return {
        "SMA_20": round(latest["SMA_20"], 2),
        "SMA_50": round(latest["SMA_50"], 2),
        "EMA_20": round(latest["EMA_20"], 2),
        "EMA_50": round(latest["EMA_50"], 2),
    }


@tool
def getRSITool(*args, **kwargs) -> float:
    """
    Calculate the 14-day Relative Strength Index (RSI).
    Returns RSI as a float.
    """
    stock = InvestmentCrew.stock
    print(f"[getRSITool] Using stock ticker: {stock}")
    df = yf.Ticker(stock).history(period="6mo")
    if df.empty:
        return 0.0

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = round(rsi.iloc[-1], 2)
    return latest_rsi


@tool
def getMACDTool(*args, **kwargs) -> dict:
    """
    Compute MACD (12, 26, 9) and return the latest MACD line, signal line, and histogram.
    """
    stock = InvestmentCrew.stock
    print(f"[getMACDTool] Using stock ticker: {stock}")
    df = yf.Ticker(stock).history(period="6mo")
    if df.empty:
        return {"error": "No data available"}

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line

    return {
        "MACD": round(macd_line.iloc[-1], 3),
        "Signal": round(signal_line.iloc[-1], 3),
        "Histogram": round(hist.iloc[-1], 3),
    }


@tool
def getTechnicalSignalTool(*args, **kwargs) -> str:
    """
    Generate an overall technical trading signal (BUY / SELL / HOLD)
    based on moving averages, RSI, and MACD.
    """
    stock = InvestmentCrew.stock
    print(f"[getTechnicalSignalTool] Using stock ticker: {stock}")
    df = yf.Ticker(stock).history(period="6mo")
    if df.empty:
        return "Insufficient data for signal generation."

    # --- Moving Average Crossover ---
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    ma_signal = "BUY" if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1] else "SELL"

    # --- RSI ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1]
    rsi_signal = "BUY" if latest_rsi < 30 else "SELL" if latest_rsi > 70 else "HOLD"

    # --- MACD ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_signal = "BUY" if macd.iloc[-1] > signal.iloc[-1] else "SELL"

    # --- Combine ---
    signals = [ma_signal, rsi_signal, macd_signal]
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")

    if buy_count >= 2:
        final_signal = "STRONG BUY"
    elif sell_count >= 2:
        final_signal = "STRONG SELL"
    else:
        final_signal = "HOLD"

    return f"Technical consensus: {final_signal} (MA: {ma_signal}, RSI: {rsi_signal}, MACD: {macd_signal})"

# ------------------------
# CrewBase Class
# ------------------------
@CrewBase
class InvestmentCrew:
    """Investment Crew for Stock Analysis & Debate"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    llm = llm
    stock = None  # This will be set dynamically

    # -------- Agents --------
    @agent
    def fundamental_analyst(self) -> Agent:
        # Pass the tools as function references, not instances
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            tools=[CustomRagTool, fundamental_analysis_tool],
            llm=self.llm,
        )

    @agent
    def valuation_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["valuation_analyst"],
            tools=[getAnnualisedVolatilityTool, getAnnualisedReturnTool],
            llm=self.llm,
            max_retry_limit= 3,
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_analyst"],
            tools=[findnewsTool,getNewsBodyTool],
            llm=self.llm,
        )
    
    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_analyst"],
            tools=[
                getMovingAveragesTool,
                getRSITool,
                getMACDTool,
                getTechnicalSignalTool,
            ],
            llm=self.llm,
        )


    @agent
    def moderator(self) -> Agent:
        return Agent(config=self.agents_config["moderator"], llm=self.llm)

    @agent
    def conclusion_agent(self) -> Agent:
        return Agent(config=self.agents_config["conclusion_agent"], llm=self.llm)

    # -------- Tasks --------
    
    @task
    def fundamental_task(self) -> Task:
        return Task(config=self.tasks_config["fundamental_task"])

    @task
    def sentiment_task(self) -> Task:
        return Task(config=self.tasks_config["sentiment_task"])
    
    @task
    def valuation_task(self) -> Task:
        return Task(config=self.tasks_config["valuation_task"])
    
    @task
    def technical_task(self) -> Task:
        return Task(config=self.tasks_config["technical_task"])
    
    @task
    def investment_debate_task(self) -> Task:
        return Task(config=self.tasks_config["investment_debate_task"])

    @task
    def investment_conclusion_task(self) -> Task:
        return Task(config=self.tasks_config["investment_conclusion_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.fundamental_analyst(),
                self.sentiment_analyst(),
                self.valuation_analyst(),
                self.technical_analyst(),
                self.moderator(),
                self.conclusion_agent()
            ],
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )