import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List
from config.config import config, SystemConfig
from llm.llm_client import LLMClient
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureEngineeringAgent
from agents.quant_model_agent import QuantModelingAgent
from agents.market_sense_agent import MarketSenseAgent
from agents.coordinator_agent import CoordinatorAgent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSystem:

    def __init__(self, config: SystemConfig, system_mode: str = "full"):
        self.config = config
        self.system_mode = system_mode
        self.llm_client = LLMClient(config.llm)
        self.data_agent = DataAgent(self.llm_client, config)
        self.feature_agent = FeatureEngineeringAgent(self.llm_client, config)
        self.quant_model_agent = QuantModelingAgent(self.llm_client, config)
        self.market_sense_agent = MarketSenseAgent(self.llm_client, config)
        self.coordinator_agent = CoordinatorAgent(self.llm_client, config)
        
        logger.info("Trading system initialized")

    def run_single_prediction(self, ticker: str) -> Dict[str, Any]:

        logger.info(f"Running prediction for {ticker}")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now().replace(year=datetime.now().year - 2)).strftime("%Y-%m-%d")

        logger.info("[1/5] Data Agent: Collecting data...")
        data_result = self.data_agent.run({
            "tickers": [ticker],
            "start_date": start_date,
            "end_date": end_date,
            "collect_news": True,
            "collect_fundamentals": True,
            "economic_indicators": ["FEDFUNDS", "CPIAUCSL"]
        })
        if not data_result.success:
            logger.error(f"Data collection failed: {data_result.message}")
            return {"success": False, "message": data_result.message}

        collected_data = data_result.data.get("collected_data", {})
        data_dictionary = data_result.data.get("data_dictionary", {})


        logger.info("[2/5] Feature Engineering Agent: Computing features...")
        feature_result = self.feature_agent.run({
            "price_data": collected_data.get("price_data", {}),
            "fundamentals": collected_data.get("fundamentals", {}),
            "news": collected_data.get("news", {}),
            "data_dictionary": data_dictionary
        })
        if not feature_result.success:
            logger.error(f"Feature engineering failed: {feature_result.message}")
            return {"success": False, "message": feature_result.message}

        features = feature_result.data.get("features", {})
        retrieved_news = feature_result.data.get("retrieved_news", {})
        feature_dictionary = feature_result.data.get("feature_dictionary", {})


        logger.info("[3/5] Quant Modeling Agent: Generating signal...")
        quant_result = self.quant_model_agent.run({
            "mode": "predict",
            "price_data": collected_data.get("price_data", {})
        })
        predictions = quant_result.data.get("predictions", {})
        quant_signal = predictions.get(ticker, {})


        logger.info("[4/5] Market-Sense Agent: Analyzing market...")
        ticker_features = features.get(ticker, {})
        ticker_fundamentals = collected_data.get("fundamentals", {}).get(ticker, {})
        ticker_news = collected_data.get("news", {}).get(ticker, [])
        price_df = collected_data.get("price_data", {}).get(ticker)
        current_price = price_df['Close'].iloc[-1] if price_df is not None and not price_df.empty else 0
        market_state = {
            "ticker_data": {
                "ticker": ticker,
                "price": current_price,
                "pe_ratio": ticker_fundamentals.get("pe_ratio"),
                "momentum_1m": ticker_features.get("momentum_1m"),
                "volatility_30d": ticker_features.get("volatility_30d")
            }
        }
        news_texts = [n.get("headline", "") for n in ticker_news[:5]]
        market_result = self.market_sense_agent.run({
            "market_state": market_state,
            "news": news_texts,
            "quant_signal": quant_signal,
            "ticker": ticker
        })
        market_insight = market_result.data.get("insight", {})


        logger.info("[5/5] Coordinator Agent: Making decision...")
        decision_result = self.coordinator_agent.run({
            "ticker": ticker,
            "current_price": current_price,
            "quant_signal": quant_signal,
            "market_insight": market_insight
        })

        logger.info(f"Decision: {decision_result.message}")

        return {
            "success": True,
            "ticker": ticker,
            "current_price": current_price,
            "quant_signal": quant_signal,
            "market_insight": market_insight,
            "decision": decision_result.data,
            "logs": decision_result.logs
        }

    def run_backtest(
            self,
            tickers: List[str],
            start_date: str,
            end_date: str,
            system_mode: str = None
    ) -> Dict[str, Any]:

        mode = system_mode or self.system_mode
        logger.info(f"Running backtest from {start_date} to {end_date} [MODE: {mode}]")
        logger.info(f"Tickers: {tickers}")


        logger.info("Collecting historical data...")
        data_result = self.data_agent.run({
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "collect_news": False,
            "collect_fundamentals": True,
            "economic_indicators": []
        })

        if not data_result.success:
            return {"success": False, "message": data_result.message}

        collected_data = data_result.data.get("collected_data", {})

        decisions = []
        portfolio_values = []

        for ticker in tickers:
            price_df = collected_data.get("price_data", {}).get(ticker)
            if price_df is None or price_df.empty:
                continue

            trading_days = price_df.index[::5]
            logger.info(f"Backtesting {ticker}: {len(trading_days)} trading periods")

            for i, date in enumerate(trading_days):
                if i < 10:
                    continue

                historical_prices = price_df.loc[:date]
                current_price = historical_prices['Close'].iloc[-1]
                ticker_fundamentals = collected_data.get("fundamentals", {}).get(ticker, {})

                if mode != "llm_only":
                    feature_result = self.feature_agent.run({
                        "price_data": {ticker: historical_prices},
                        "fundamentals": {ticker: ticker_fundamentals},
                        "news": collected_data.get("news", {})
                    })
                    ticker_features = feature_result.data.get("features", {}).get(ticker, {})

                    train_result = self.quant_model_agent.run({
                        "mode": "train",
                        "price_data": {ticker: historical_prices},
                        "features": {ticker: ticker_features}
                    })

                    quant_result = self.quant_model_agent.run({
                        "mode": "predict",
                        "price_data": {ticker: historical_prices},
                        "features": {ticker: ticker_features}
                    })
                    predictions = quant_result.data.get("predictions", {})
                    quant_signal = predictions.get(ticker, {})
                else:
                    quant_signal = {
                        "expected_return": 0.0,
                        "confidence": 0.0,
                        "regime": "DISABLED",
                        "regime_probabilities": {},
                        "model_type": "none"
                    }

                features_df = self.feature_agent.compute_technical_indicators(historical_prices)

                if mode != "quant_only":
                    market_state = {
                        "ticker_data": {
                            "ticker": ticker,
                            "price": current_price,
                            "pe_ratio": ticker_fundamentals.get("pe_ratio"),
                            "momentum_1m": features_df['Close'].pct_change(20).iloc[-1] if len(features_df) > 20 else 0,
                            "volatility_30d": features_df['Close'].pct_change().std() * (252 ** 0.5) if len(
                                features_df) > 1 else 0
                        }
                    }

                    market_result = self.market_sense_agent.run({
                        "market_state": market_state,
                        "news": [],
                        "quant_signal": quant_signal,
                        "ticker": ticker
                    })
                    market_insight = market_result.data.get("insight", {})
                else:
                    market_insight = {
                        "outlook": "NEUTRAL",
                        "confidence": 0.0,
                        "reasoning": "Market-Sense disabled in quant_only mode",
                        "risk_flags": []
                    }

                decision_result = self.coordinator_agent.run({
                    "ticker": ticker,
                    "current_price": current_price,
                    "quant_signal": quant_signal,
                    "market_insight": market_insight
                })

                aggregated_signal = decision_result.data.get("aggregated_signal", {})

                decisions.append({
                    "date": str(date),
                    "ticker": ticker,
                    "price": current_price,
                    "decision": decision_result.data.get("trade_result", {}).get("message", ""),
                    "regime": quant_signal.get("regime", "UNKNOWN"),
                    "reasoning": aggregated_signal.get("reasoning", ""),
                    "market_outlook": market_insight.get("outlook", "UNKNOWN"),
                    "market_confidence": market_insight.get("confidence", 0),
                    "signal_direction": aggregated_signal.get("direction", "UNKNOWN"),
                    "signal_strength": aggregated_signal.get("strength", 0)
                })

                if mode == "llm_only":
                    logger.info(f"  LLM-ONLY Debug: outlook={market_insight.get('outlook')}, "
                                f"confidence={market_insight.get('confidence')}, "
                                f"direction={aggregated_signal.get('direction')}, "
                                f"strength={aggregated_signal.get('strength', 0):.3f}")

                portfolio_values.append({
                    "date": str(date),
                    "value": self.coordinator_agent.portfolio.total_value
                })

        for ticker in tickers:
            price_df = collected_data.get("price_data", {}).get(ticker)
            if price_df is not None and not price_df.empty:
                final_price = float(price_df['Close'].iloc[-1])
                if ticker in self.coordinator_agent.portfolio.positions:
                    self.coordinator_agent.portfolio.positions[ticker].current_price = final_price

        initial_value = self.config.portfolio.initial_cash
        final_value = self.coordinator_agent.portfolio.total_value
        total_return = (final_value - initial_value) / initial_value

        metrics = {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": f"{total_return * 100:.2f}%",
            "num_trades": len(self.coordinator_agent.trade_history),
            "num_decisions": len(decisions),
            "system_mode": mode
        }

        logger.info(f"Backtest complete. Return: {metrics['total_return_pct']}")

        return {
            "success": True,
            "decisions": decisions,
            "portfolio_values": portfolio_values,
            "metrics": metrics,
            "final_portfolio": self.coordinator_agent.portfolio.to_dict()
        }

    def train_model(self, design_doc_path: str) -> Dict[str, Any]:

        logger.info(f"Training model from {design_doc_path}")
        model_design = self.quant_model_agent.read_document(design_doc_path)
        if not model_design:
            return {"success": False, "message": f"Could not read design doc: {design_doc_path}"}

        logger.info("Model design loaded")
        tickers = self.config.tickers
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now().replace(year=datetime.now().year - 2)).strftime("%Y-%m-%d")
        logger.info(f"Collecting training data for {tickers}...")
        data_result = self.data_agent.run({
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "collect_news": False,
            "collect_fundamentals": False,
            "economic_indicators": []
        })

        if not data_result.success:
            return {"success": False, "message": f"Data collection failed: {data_result.message}"}

        collected_data = data_result.data.get("collected_data", {})

        logger.info("Training HMM model...")
        train_result = self.quant_model_agent.run({
            "mode": "train",
            "price_data": collected_data.get("price_data", {})
        })

        if not train_result.success:
            return {"success": False, "message": f"Training failed: {train_result.message}"}

        training_results = train_result.data.get("training_results", {})

        logger.info("Training complete!")
        for ticker, result in training_results.items():
            logger.info(f"  {ticker}: log_likelihood={result.get('log_likelihood', 'N/A'):.2f}")

        return {
            "success": True,
            "design_doc": design_doc_path,
            "tickers": tickers,
            "training_results": training_results,
            "date_range": {"start": start_date, "end": end_date}
        }



def main():

    parser = argparse.ArgumentParser(
        description="Quant Trading Agent System"
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "backtest", "train"],
        default="predict",
        help="Running mode"
    )
    parser.add_argument(
        "--system-mode",
        choices=["full", "quant_only", "llm_only"],
        default="full",
        help="System mode: full (both agents), quant_only, or llm_only"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker (for predict mode)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        help="Stock tickers (for backtest mode)"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model design config (for train mode)"
    )

    args = parser.parse_args()

    print("Quant Trading Agent System")
    print()

    system = TradingSystem(config, system_mode=args.system_mode)

    if args.mode == "predict":
        result = system.run_single_prediction(args.ticker)
        print_prediction_result(result)

    elif args.mode == "backtest":
        if not args.start or not args.end:
            parser.error("Backtest mode requires --start and --end dates")
        tickers = args.tickers or [args.ticker]
        result = system.run_backtest(tickers, args.start, args.end)
        print_backtest_result(result)

    elif args.mode == "train":
        if not args.config:
            parser.error("Train mode requires --config path")
        result = system.train_model(args.config)
        print_training_result(result)


def print_prediction_result(result: Dict[str, Any]):

    print("TRADING DECISION")

    if not result.get("success"):
        print(f"ERROR: {result.get('message', 'Unknown error')}")
        return

    ticker = result.get("ticker", "N/A")
    price = result.get("current_price", 0)

    print(f"Ticker: {ticker}")
    print(f"Current Price: ${price:.2f}")
    print()
    quant = result.get("quant_signal", {})
    print("Quant Model Signal")
    print(f"Expected Return: {quant.get('expected_return', 0) * 100:.2f}%")
    print(f"Regime: {quant.get('regime', 'N/A')}")
    print(f"Confidence: {quant.get('confidence', 0):.2f}")
    print()
    insight = result.get("market_insight", {})
    print("Market-Sense Insight")
    print(f"Outlook: {insight.get('outlook', 'N/A')}")
    print(f"Confidence: {insight.get('confidence', 0):.2f}")
    print(f"Reasoning: {insight.get('reasoning', 'N/A')[:100]}...")
    if insight.get("risk_flags"):
        print(f"Risk Flags: {', '.join(insight['risk_flags'])}")
    print()

    decision = result.get("decision", {})
    trade_result = decision.get("trade_result", {})
    portfolio = decision.get("portfolio_state", {})

    print("FINAL DECISION")
    print(f"Action: {trade_result.get('message', 'N/A')}")
    if trade_result.get("order"):
        order = trade_result["order"]
        print(f"Order: {order['action']} {order['shares']} shares @ ${order['price']:.2f}")
    print()

    print("Portfolio State")
    print(f"Cash: ${portfolio.get('cash', 0):,.2f}")
    print(f"Total Value: ${portfolio.get('total_value', 0):,.2f}")
    if portfolio.get("positions"):
        print("Positions:")
        for ticker, pos in portfolio["positions"].items():
            print(f"{ticker}: {pos['shares']} shares @ ${pos['current_price']:.2f} = ${pos['market_value']:,.2f}")

    print("=" * 60)


def print_backtest_result(result: Dict[str, Any]):

    print("BACKTEST RESULTS")

    if not result.get("success"):
        print(f"ERROR: {result.get('message', 'Unknown error')}")
        return

    metrics = result.get("metrics", {})

    print("Performance Metrics")
    print(f"Initial Portfolio Value: ${metrics.get('initial_value', 0):,.2f}")
    print(f"Final Portfolio Value: ${metrics.get('final_value', 0):,.2f}")
    print(f"Total Return: {metrics.get('total_return_pct', 'N/A')}")
    print(f"Number of Trades: {metrics.get('num_trades', 0)}")
    print(f"Number of Decisions: {metrics.get('num_decisions', 0)}")

    portfolio = result.get("final_portfolio", {})
    print("Final Portfolio")
    print(f"Cash: ${portfolio.get('cash', 0):,.2f}")
    if portfolio.get("positions"):
        print("Positions:")
        for ticker, pos in portfolio["positions"].items():
            print(f"{ticker}: {pos['shares']} shares @ ${pos['current_price']:.2f} = ${pos['market_value']:,.2f}")
    else:
        print("Positions: None")

    decisions = result.get("decisions", [])
    if decisions:
        print("Recent Decisions (last 10)")
        for d in decisions[-10:]:
            print(f"{d['date']}: {d['ticker']} @ ${d['price']:.2f} - {d['decision']} (Regime: {d['regime']})")


def print_training_result(result: Dict[str, Any]):

    print("TRAINING RESULTS")

    if not result.get("success"):
        print(f"ERROR: {result.get('message', 'Unknown error')}")
        return

    print(f"Design Document: {result.get('design_doc', 'N/A')}")
    print(f"Tickers: {', '.join(result.get('tickers', []))}")

    date_range = result.get("date_range", {})
    print(f"Training Period: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")

    print("Model Training Results")
    training_results = result.get("training_results", {})
    for ticker, res in training_results.items():
        print(f"\n  {ticker}:")
        print(f"Model Type: {res.get('model_type', 'N/A')}")
        print(f"States: {res.get('n_states', 'N/A')}")
        print(f"Log-Likelihood: {res.get('log_likelihood', 'N/A'):.2f}")
        print(f"Training Samples: {res.get('n_samples', 'N/A')}")
        print(f"Model Path: {res.get('model_path', 'N/A')}")

        if res.get("emission_means"):
            print("Emission Means (Expected Returns):")
            for regime, mean in res["emission_means"].items():
                print(f"{regime}: {mean * 100:.2f}%")

    print("Training complete Models saved to models/ directory.")

if __name__ == "__main__":
    main()