from dataclasses import dataclass
from typing import Optional
import os

os.environ.setdefault('FINNHUB_API_KEY', 'YOUR KEY')
os.environ.setdefault('FRED_API_KEY', 'YOUR KEY')

@dataclass
class LLMConfig:
    api_base_url: str = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    api_key: str = ""
    model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 2000
    max_retries: int = 3
    timeout: int = 60


@dataclass
class DataSourceConfig:
    yfinance_enabled: bool = True
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    fred_api_key: str = os.getenv("FRED_API_KEY", "")

@dataclass
class PortfolioConfig:
    initial_cash: float = 100000.0
    max_position_pct: float = 0.80
    min_cash_reserve_pct: float = 0.05
    allow_shorting: bool = True

@dataclass
class ModelConfig:
    model_type: str = "hmm"
    model_path: str = "models/"
    design_doc_path: str = "config/model_design.md"
    xgboost_design_doc_path: str = "config/model_design_xgboost.md"
    retrain_frequency_days: int = 30
    n_states: int = 3
    prediction_horizon: int = 5

@dataclass
class RAGConfig:
    vector_db_path: str = "knowledge/vector_store"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    top_k_results: int = 5
    news_top_k: int = 5
    knowledge_top_k: int = 3
    historical_events_top_k: int = 2

@dataclass
class SystemConfig:
    llm: LLMConfig = None
    data_sources: DataSourceConfig = None
    portfolio: PortfolioConfig = None
    model: ModelConfig = None
    rag: RAGConfig = None
    tickers: list = None
    trading_frequency_days: int = 5

    def __post_init__(self):
        self.llm = self.llm or LLMConfig()
        self.data_sources = self.data_sources or DataSourceConfig()
        self.portfolio = self.portfolio or PortfolioConfig()
        self.model = self.model or ModelConfig()
        self.rag = self.rag or RAGConfig()
        self.tickers = self.tickers or ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]

config = SystemConfig()
