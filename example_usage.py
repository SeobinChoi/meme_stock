#!/usr/bin/env python3
"""
Example script demonstrating the meme stock ML upgrade pipeline.

This script shows how to use the system with sample data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for demonstration."""
    logger.info("Creating sample data...")
    
    # Create sample Reddit data
    dates = pd.date_range('2021-01-01', periods=1000, freq='D')
    tickers = ['GME', 'AMC', 'BB']
    
    reddit_data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            # Simulate Reddit activity with some patterns
            base_activity = np.random.poisson(100)
            
            # Add some spikes
            if i % 50 == 0:  # Periodic spikes
                base_activity *= 5
            
            # Add ticker-specific patterns
            if ticker == 'GME' and i > 200 and i < 300:  # GME spike period
                base_activity *= 3
            
            reddit_data.append({
                'date': date,
                'ticker': ticker,
                'text': f"Sample Reddit post about {ticker} - {'🚀' * np.random.randint(1, 4)}",
                'score': base_activity,
                'comments': base_activity // 2,
                'mentions': base_activity
            })
    
    reddit_df = pd.DataFrame(reddit_data)
    
    # Create sample price data
    price_data = {}
    for ticker in tickers:
        prices = []
        base_price = 100 if ticker == 'GME' else (50 if ticker == 'AMC' else 10)
        
        for i, date in enumerate(dates):
            # Simulate price movements
            daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% volatility
            
            # Add contrarian effect: high Reddit activity -> negative returns
            reddit_activity = reddit_df[(reddit_df['date'] == date) & (reddit_df['ticker'] == ticker)]['mentions'].iloc[0]
            contrarian_effect = -0.0001 * (reddit_activity - 100) / 100  # Small contrarian effect
            daily_return += contrarian_effect
            
            if i == 0:
                price = base_price
            else:
                price = prices[-1]['close'] * (1 + daily_return)
            
            prices.append({
                'date': date,
                'open': price * 0.99,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'volume': np.random.lognormal(10, 1)
            })
        
        price_data[ticker] = pd.DataFrame(prices)
    
    return reddit_df, price_data

def run_example():
    """Run the example pipeline."""
    logger.info("Starting meme stock ML upgrade example...")
    
    # Create sample data
    reddit_df, price_data = create_sample_data()
    
    # Save sample data
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    reddit_path = data_dir / "sample_reddit_data.csv"
    reddit_df.to_csv(reddit_path, index=False)
    logger.info(f"Saved Reddit data: {reddit_path}")
    
    price_data_paths = {}
    for ticker, df in price_data.items():
        price_path = data_dir / f"{ticker}_sample_price_data.csv"
        df.to_csv(price_path, index=False)
        price_data_paths[ticker] = str(price_path)
        logger.info(f"Saved {ticker} price data: {price_path}")
    
    # Import and run the pipeline
    try:
        from run_ml_upgrade import MemeStockMLUpgrade
        
        # Initialize pipeline
        pipeline = MemeStockMLUpgrade(
            target_tickers=['GME', 'AMC', 'BB'],
            date_range=('2021-01-01', '2023-12-31'),
            output_dir="example_results"
        )
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(
            reddit_data_path=str(reddit_path),
            price_data_paths=price_data_paths,
            use_finbert=False,  # Skip FinBERT for speed
            use_sequence_models=False,  # Skip sequence models for speed
            use_ensemble=True
        )
        
        logger.info("Example pipeline completed successfully!")
        logger.info("Check 'example_results' directory for outputs")
        
        # Print summary
        if 'baseline_results' in results:
            baseline_results = results['baseline_results']['results']
            logger.info("\nModel Performance Summary:")
            for model_name, metrics in baseline_results.items():
                logger.info(f"{model_name}: Spearman Corr = {metrics.get('spearman_corr', 0):.4f}")
        
        if 'strategy_results' in results:
            strategy_metrics = results['strategy_results']['best_strategy']['metrics']
            logger.info(f"\nBest Strategy Performance:")
            logger.info(f"Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Hit Rate: {strategy_metrics.get('hit_rate', 0):.2%}")
            logger.info(f"Max Drawdown: {strategy_metrics.get('max_drawdown', 0):.2%}")
        
    except ImportError as e:
        logger.error(f"Could not import pipeline: {e}")
        logger.error("Make sure all dependencies are installed")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def run_quick_test():
    """Run a quick test of individual components."""
    logger.info("Running quick component tests...")
    
    # Test feature engineering
    try:
        from src.features.text_confidence import calculate_confidence_score
        
        test_texts = [
            "GME to the moon! 🚀💎🙌 Diamond hands forever!",
            "Maybe AMC will go up, but I'm not sure...",
            "BB is guaranteed to squeeze! 100% certain!"
        ]
        
        logger.info("Testing text confidence scoring:")
        for text in test_texts:
            score = calculate_confidence_score(text)
            logger.info(f"Text: {text[:30]}... -> Confidence: {score:.2f}")
        
    except ImportError as e:
        logger.warning(f"Could not test text confidence: {e}")
    
    # Test sentiment analysis
    try:
        from src.features.sentiment import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(use_finbert=False)
        test_text = "GME is going to explode! 🚀🚀🚀"
        
        sentiment_result = analyzer.analyze_sentiment(test_text)
        logger.info(f"Sentiment analysis: {sentiment_result['vader_compound']:.3f}")
        
    except ImportError as e:
        logger.warning(f"Could not test sentiment analysis: {e}")
    
    # Test trading strategy
    try:
        from src.evaluation.trading_strategy import backtest_strategy
        
        np.random.seed(42)
        signals = np.random.randn(100) * 0.02
        returns = np.random.randn(100) * 0.03
        
        strategy_result = backtest_strategy(signals, returns, strategy_type='contrarian')
        logger.info(f"Strategy test - Sharpe: {strategy_result['metrics']['sharpe_ratio']:.3f}")
        
    except ImportError as e:
        logger.warning(f"Could not test trading strategy: {e}")
    
    logger.info("Quick component tests completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Meme Stock ML Upgrade Example')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick component tests only')
    
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_test()
    else:
        run_example()
