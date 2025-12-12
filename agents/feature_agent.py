from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
from .base_agent import BaseAgent, AgentOutput
from sentence_transformers import SentenceTransformer


class FeatureEngineeringAgent(BaseAgent):
    
    def __init__(self, llm_client, config):
        super().__init__("FeatureEngineeringAgent", llm_client, config)
        self.vector_store = None
        
    @property
    def system_prompt(self) -> str:
        return """You are a Feature Engineering Agent for a quantitative trading system.

Your role is to:
1. Compute technical indicators from price data
2. Calculate quantitative factors (momentum, value, volatility, quality)
3. Process news articles into embeddings for retrieval
4. Retrieve historically similar news events

Important:
- Factors must remain as EXACT numerical values for the Quant Model
- Embeddings are used ONLY for retrieval, not passed to other agents
- When retrieving news, return the actual TEXT, not embeddings

You have access to the following tools:
- compute_technical_indicators: Calculate RSI, MACD, Bollinger Bands, etc.
- compute_factors: Calculate momentum, value, volatility factors
- embed_text: Convert text to embedding vector
- store_embedding: Store embedding in vector database
- retrieve_similar_news: Find similar historical news"""

    def _register_tools(self):
        
        self.register_tool(
            name="compute_technical_indicators",
            func=self.compute_technical_indicators,
            description="Compute technical indicators from price data",
            parameters={
                "type": "object",
                "properties": {
                    "prices": {"type": "object", "description": "Price DataFrame (OHLCV)"},
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of indicators: 'RSI', 'MACD', 'BB', 'SMA', 'EMA'"
                    }
                },
                "required": ["prices"]
            }
        )
        
        self.register_tool(
            name="compute_factors",
            func=self.compute_factors,
            description="Compute quantitative factors",
            parameters={
                "type": "object",
                "properties": {
                    "prices": {"type": "object", "description": "Price DataFrame"},
                    "fundamentals": {"type": "object", "description": "Fundamental data dict"}
                },
                "required": ["prices"]
            }
        )
        
        self.register_tool(
            name="embed_text",
            func=self.embed_text,
            description="Convert text to embedding vector",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to embed"}
                },
                "required": ["text"]
            }
        )
        
        self.register_tool(
            name="store_embedding",
            func=self.store_embedding,
            description="Store embedding in vector database with metadata",
            parameters={
                "type": "object",
                "properties": {
                    "embedding": {"type": "array", "description": "Embedding vector"},
                    "text": {"type": "string", "description": "Original text"},
                    "metadata": {"type": "object", "description": "Metadata (date, ticker, source)"}
                },
                "required": ["embedding", "text", "metadata"]
            }
        )
        
        self.register_tool(
            name="retrieve_similar_news",
            func=self.retrieve_similar_news,
            description="Retrieve similar historical news based on current context",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query text or current news"},
                    "top_k": {"type": "integer", "description": "Number of results to return"}
                },
                "required": ["query"]
            }
        )

    def compute_embedding_factors(self, news_texts: List[str]) -> Dict[str, float]:
        if not news_texts:
            return {
                "news_sentiment": 0.0,
                "news_magnitude": 0.0,
                "news_novelty": 0.5
            }

        try:
            bullish_ref = self.embed_text("stock price surge strong earnings beat bullish momentum growth")
            bearish_ref = self.embed_text("stock crash decline losses bearish downturn sell-off warning")

            if not bullish_ref or not bearish_ref:
                return {"news_sentiment": 0.0, "news_magnitude": 0.0, "news_novelty": 0.5}

            bullish_ref = np.array(bullish_ref)
            bearish_ref = np.array(bearish_ref)

            news_embeddings = []
            for text in news_texts:
                emb = self.embed_text(text)
                if emb:
                    news_embeddings.append(np.array(emb))

            if not news_embeddings:
                return {"news_sentiment": 0.0, "news_magnitude": 0.0, "news_novelty": 0.5}

            avg_news_emb = np.mean(news_embeddings, axis=0)

            def cosine_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

            bull_sim = cosine_sim(avg_news_emb, bullish_ref)
            bear_sim = cosine_sim(avg_news_emb, bearish_ref)
            sentiment_score = float(bull_sim - bear_sim)
            magnitude = float((bull_sim + bear_sim) / 2)

            novelty = 0.5
            if self.vector_store:
                similarities = []
                for item in self.vector_store[-20:]:
                    sim = cosine_sim(avg_news_emb, item["embedding"])
                    similarities.append(sim)
                if similarities:
                    novelty = float(1 - np.mean(similarities))

            return {
                "news_sentiment": sentiment_score,
                "news_magnitude": magnitude,
                "news_novelty": novelty
            }

        except Exception as e:
            logger.error(f"Error computing embedding factors: {e}")
            return {"news_sentiment": 0.0, "news_magnitude": 0.0, "news_novelty": 0.5}

    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        price_data = input_data.get("price_data", {})
        fundamentals = input_data.get("fundamentals", {})
        news = input_data.get("news", {})

        logs = []
        features = {}
        retrieved_news = {}

        for ticker, prices in price_data.items():
            logs.append(f"Processing features for {ticker}...")

            ticker_features = {}
            if not prices.empty:
                tech_indicators = self.compute_technical_indicators(prices)
                if not tech_indicators.empty:
                    latest = tech_indicators.iloc[-1]
                    for col in tech_indicators.columns:
                        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            ticker_features[col] = latest[col] if pd.notna(latest[col]) else None

            ticker_fundamentals = fundamentals.get(ticker, {})
            factors = self.compute_factors(prices, ticker_fundamentals)
            ticker_features.update(factors)
            ticker_news = news.get(ticker, [])
            ticker_news_texts = [f"{n.get('headline', '')} {n.get('summary', '')}"
                                 for n in ticker_news if n.get('headline')]
            embedding_factors = self.compute_embedding_factors(ticker_news_texts)
            ticker_features.update(embedding_factors)

            features[ticker] = ticker_features
            for article in ticker_news:
                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                if text.strip():
                    embedding = self.embed_text(text)
                    if embedding:
                        self.store_embedding(
                            embedding=embedding,
                            text=text,
                            metadata={
                                "headline": article.get("headline", ""),
                                "datetime": article.get("datetime", ""),
                                "ticker": ticker,
                                "source": article.get("source", "")
                            }
                        )

            if ticker_news:
                latest_news = ticker_news[0] if ticker_news else {}
                query = f"{latest_news.get('headline', '')} {latest_news.get('summary', '')}"
                if query.strip():
                    retrieved_news[ticker] = self.retrieve_similar_news(query, top_k=5)

            logs.append(f"Computed {len(ticker_features)} features for {ticker}")

        feature_dictionary = self.generate_feature_dictionary(features)

        return AgentOutput(
            success=True,
            data={
                "features": features,
                "retrieved_news": retrieved_news,
                "feature_dictionary": feature_dictionary
            },
            message=f"Processed features for {len(price_data)} tickers",
            logs=logs
        )

    def compute_technical_indicators(self, prices: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        if prices.empty:
            return prices

        if indicators is None:
            indicators = ['RSI', 'MACD', 'BB', 'SMA_20', 'SMA_50']

        df = prices.copy()
        close = df['Close']

        try:
            if 'RSI' in indicators:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI_14'] = 100 - (100 / (1 + rs))

            if 'MACD' in indicators:
                ema_12 = close.ewm(span=12, adjust=False).mean()
                ema_26 = close.ewm(span=26, adjust=False).mean()
                df['MACD'] = ema_12 - ema_26
                df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_hist'] = df['MACD'] - df['MACD_signal']

            if 'BB' in indicators:
                sma_20 = close.rolling(window=20).mean()
                std_20 = close.rolling(window=20).std()
                df['BB_upper'] = sma_20 + (std_20 * 2)
                df['BB_middle'] = sma_20
                df['BB_lower'] = sma_20 - (std_20 * 2)
                df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

            if 'SMA_20' in indicators:
                df['SMA_20'] = close.rolling(window=20).mean()

            if 'SMA_50' in indicators:
                df['SMA_50'] = close.rolling(window=50).mean()

            if 'EMA_12' in indicators:
                df['EMA_12'] = close.ewm(span=12, adjust=False).mean()

            if 'EMA_26' in indicators:
                df['EMA_26'] = close.ewm(span=26, adjust=False).mean()

            logger.info(f"Computed {len(indicators)} technical indicators")
            return df

        except Exception as e:
            logger.error(f"Error computing technical indicators: {e}")
            return prices

    def compute_factors(self, prices: pd.DataFrame, fundamentals: Optional[Dict] = None) -> Dict[str, float]:
        factors = {}

        if prices.empty:
            return factors

        try:
            close = prices['Close']

            if len(close) >= 5:
                factors['return_5d'] = (close.iloc[-1] / close.iloc[-5] - 1) if close.iloc[-5] != 0 else 0
            if len(close) >= 21:
                factors['momentum_1m'] = (close.iloc[-1] / close.iloc[-21] - 1) if close.iloc[-21] != 0 else 0
            if len(close) >= 63:
                factors['momentum_3m'] = (close.iloc[-1] / close.iloc[-63] - 1) if close.iloc[-63] != 0 else 0
            if len(close) >= 252:
                factors['momentum_12m'] = (close.iloc[-1] / close.iloc[-252] - 1) if close.iloc[-252] != 0 else 0
            returns = close.pct_change().dropna()
            if len(returns) >= 30:
                factors['volatility_30d'] = returns.tail(30).std() * np.sqrt(252)
            if 'Volume' in prices.columns:
                volume = prices['Volume']
                if len(volume) >= 20:
                    factors['volume_ratio'] = volume.iloc[-1] / volume.tail(20).mean() if volume.tail(
                        20).mean() != 0 else 1
            if len(close) >= 52:
                high_52w = close.tail(252).max() if len(close) >= 252 else close.max()
                low_52w = close.tail(252).min() if len(close) >= 252 else close.min()
                if high_52w != low_52w:
                    factors['price_position_52w'] = (close.iloc[-1] - low_52w) / (high_52w - low_52w)

            if fundamentals:
                if fundamentals.get('pe_ratio'):
                    factors['pe_ratio'] = fundamentals['pe_ratio']
                if fundamentals.get('pb_ratio'):
                    factors['pb_ratio'] = fundamentals['pb_ratio']
                if fundamentals.get('dividend_yield'):
                    factors['dividend_yield'] = fundamentals['dividend_yield']
                if fundamentals.get('beta'):
                    factors['beta'] = fundamentals['beta']

            logger.info(f"Computed {len(factors)} factors")
            return factors

        except Exception as e:
            logger.error(f"Error computing factors: {e}")
            return factors

    def embed_text(self, text: str) -> List[float]:

        try:
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded embedding model: all-MiniLM-L6-v2")

            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            embedding = self._embedding_model.encode(text).tolist()
            return embedding

        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            return []
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return []

    def store_embedding(self, embedding: List[float], text: str, metadata: Dict[str, Any]) -> bool:
        try:
            if self.vector_store is None:
                self.vector_store = []

            self.vector_store.append({
                "embedding": np.array(embedding),
                "text": text,
                "metadata": metadata
            })

            logger.debug(f"Stored embedding for: {metadata.get('headline', text[:50])}")
            return True

        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False

    def retrieve_similar_news(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        if not self.vector_store:
            logger.warning("Vector store is empty, no news to retrieve")
            return []

        try:
            query_embedding = np.array(self.embed_text(query))
            if len(query_embedding) == 0:
                return []
            similarities = []
            for item in self.vector_store:
                stored_embedding = item["embedding"]
                similarity = np.dot(query_embedding, stored_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append({
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "similarity": float(similarity)
                })

            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            top_results = similarities[:top_k]
            results = []
            for item in top_results:
                results.append({
                    "text": item["text"],
                    "headline": item["metadata"].get("headline", ""),
                    "datetime": item["metadata"].get("datetime", ""),
                    "ticker": item["metadata"].get("ticker", ""),
                    "source": item["metadata"].get("source", ""),
                    "similarity": item["similarity"]
                })

            logger.info(f"Retrieved {len(results)} similar news articles")
            return results

        except Exception as e:
            logger.error(f"Error retrieving similar news: {e}")
            return []

    def generate_feature_dictionary(self, features: Dict[str, Dict]) -> Dict[str, Any]:

        dictionary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": []
        }
        all_features = set()
        for ticker_features in features.values():
            all_features.update(ticker_features.keys())
        feature_definitions = {
            "RSI_14": {"category": "technical", "description": "14-day Relative Strength Index",
                       "interpretation": ">70 overbought, <30 oversold"},
            "MACD": {"category": "technical", "description": "MACD line (12-day EMA - 26-day EMA)",
                     "interpretation": "Positive = bullish momentum"},
            "MACD_signal": {"category": "technical", "description": "MACD signal line (9-day EMA of MACD)",
                            "interpretation": "MACD crossing above = buy signal"},
            "MACD_hist": {"category": "technical", "description": "MACD histogram",
                          "interpretation": "Positive = bullish"},
            "BB_upper": {"category": "technical", "description": "Bollinger Band upper (20-day SMA + 2*std)",
                         "interpretation": "Resistance level"},
            "BB_middle": {"category": "technical", "description": "Bollinger Band middle (20-day SMA)",
                          "interpretation": "Mean price level"},
            "BB_lower": {"category": "technical", "description": "Bollinger Band lower (20-day SMA - 2*std)",
                         "interpretation": "Support level"},
            "BB_width": {"category": "technical", "description": "Bollinger Band width",
                         "interpretation": "High = high volatility"},
            "SMA_20": {"category": "technical", "description": "20-day Simple Moving Average",
                       "interpretation": "Short-term trend"},
            "SMA_50": {"category": "technical", "description": "50-day Simple Moving Average",
                       "interpretation": "Medium-term trend"},
            "EMA_12": {"category": "technical", "description": "12-day Exponential Moving Average",
                       "interpretation": "Short-term trend"},
            "EMA_26": {"category": "technical", "description": "26-day Exponential Moving Average",
                       "interpretation": "Medium-term trend"},
            "return_5d": {"category": "momentum", "description": "5-day return",
                          "interpretation": "Short-term momentum"},
            "momentum_1m": {"category": "momentum", "description": "1-month return",
                            "interpretation": "Positive = upward trend"},
            "momentum_3m": {"category": "momentum", "description": "3-month return",
                            "interpretation": "Medium-term momentum"},
            "momentum_12m": {"category": "momentum", "description": "12-month return",
                             "interpretation": "Long-term momentum"},
            "volatility_30d": {"category": "volatility", "description": "30-day annualized volatility",
                               "interpretation": "Higher = more risk"},
            "volume_ratio": {"category": "volume", "description": "Current volume / 20-day avg volume",
                             "interpretation": ">1 = above average activity"},
            "price_position_52w": {"category": "price", "description": "Position in 52-week range (0-1)",
                                   "interpretation": "1 = at 52-week high"},
            "pe_ratio": {"category": "value", "description": "Price-to-Earnings ratio",
                         "interpretation": "Lower = cheaper"},
            "pb_ratio": {"category": "value", "description": "Price-to-Book ratio",
                         "interpretation": "Lower = cheaper"},
            "dividend_yield": {"category": "value", "description": "Dividend yield",
                               "interpretation": "Higher = more income"},
            "beta": {"category": "risk", "description": "Market beta",
                     "interpretation": ">1 = more volatile than market"},
        }

        for feature_name in sorted(all_features):
            feature_info = feature_definitions.get(feature_name, {
                "category": "other",
                "description": feature_name,
                "interpretation": "N/A"
            })

            dictionary["features"].append({
                "name": feature_name,
                "category": feature_info["category"],
                "description": feature_info["description"],
                "interpretation": feature_info["interpretation"]
            })

        return dictionary
