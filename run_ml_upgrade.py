#!/usr/bin/env python3
"""
Main execution script for meme stock contrarian effect prediction.

Implements the complete ML upgrade pipeline as described in the manual.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import warnings
import argparse
from pathlib import Path
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.features.feature_pipeline import create_meme_stock_features
from src.modeling.validation import TimeSeriesValidator, PurgedKFold, WalkForwardValidator
from src.modeling.baseline_models import train_baseline_models, RidgeModel, LightGBMModel, XGBoostModel
from src.modeling.sequence_models import train_sequence_models
from src.modeling.ensemble import create_meta_ensemble
from src.evaluation.trading_strategy import backtest_strategy, compare_strategies
from src.evaluation.interpretability import InterpretabilityReport
from src.evaluation.robustness import RobustnessReport
from src.evaluation.reporting import ResultsReporter, generate_all_visualizations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meme_stock_ml_upgrade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemeStockMLUpgrade:
    """Main class for meme stock ML upgrade pipeline."""
    
    def __init__(self, 
                 target_tickers: List[str] = ['GME', 'AMC', 'BB'],
                 date_range: Tuple[str, str] = ('2021-01-01', '2023-12-31'),
                 output_dir: str = "results"):
        """
        Initialize ML upgrade pipeline.
        
        Args:
            target_tickers: List of tickers to analyze
            date_range: Date range for analysis
            output_dir: Output directory for results
        """
        self.target_tickers = target_tickers
        self.date_range = date_range
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized ML upgrade pipeline for tickers: {target_tickers}")
        logger.info(f"Date range: {date_range}")
        logger.info(f"Output directory: {output_dir}")
    
    def run_complete_pipeline(self, 
                            reddit_data_path: str,
                            price_data_paths: Dict[str, str],
                            use_finbert: bool = False,
                            use_sequence_models: bool = True,
                            use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Run the complete ML upgrade pipeline.
        
        Args:
            reddit_data_path: Path to Reddit data
            price_data_paths: Dictionary mapping ticker to price data path
            use_finbert: Whether to use FinBERT for sentiment
            use_sequence_models: Whether to train sequence models
            use_ensemble: Whether to create ensemble
            
        Returns:
            Dictionary with all results
        """
        logger.info("Starting complete ML upgrade pipeline...")
        
        all_results = {}
        
        try:
            # Step 1: Feature Engineering
            logger.info("Step 1: Feature Engineering")
            features_df = self._run_feature_engineering(
                reddit_data_path, price_data_paths, use_finbert
            )
            all_results['features'] = features_df
            
            # Step 2: Data Preparation and Validation
            logger.info("Step 2: Data Preparation and Validation")
            train_data, val_data, test_data = self._prepare_data(features_df)
            all_results['data_splits'] = {
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data)
            }
            
            # Step 3: Baseline Models
            logger.info("Step 3: Training Baseline Models")
            baseline_results = self._train_baseline_models(train_data, val_data)
            all_results['baseline_results'] = baseline_results
            
            # Step 4: Sequence Models
            if use_sequence_models:
                logger.info("Step 4: Training Sequence Models")
                sequence_results = self._train_sequence_models(train_data, val_data)
                all_results['sequence_results'] = sequence_results
            
            # Step 5: Ensemble
            if use_ensemble:
                logger.info("Step 5: Creating Ensemble")
                ensemble_results = self._create_ensemble(baseline_results, train_data, val_data)
                all_results['ensemble_results'] = ensemble_results
            
            # Step 6: Trading Strategy
            logger.info("Step 6: Implementing Trading Strategy")
            strategy_results = self._implement_trading_strategy(test_data, all_results)
            all_results['strategy_results'] = strategy_results
            
            # Step 7: Interpretability Analysis
            logger.info("Step 7: Interpretability Analysis")
            interpretability_results = self._run_interpretability_analysis(
                train_data, val_data, all_results
            )
            all_results['interpretability_results'] = interpretability_results
            
            # Step 8: Robustness Testing
            logger.info("Step 8: Robustness Testing")
            robustness_results = self._run_robustness_tests(
                train_data, val_data, all_results
            )
            all_results['robustness_results'] = robustness_results
            
            # Step 9: Generate Reports and Visualizations
            logger.info("Step 9: Generating Reports and Visualizations")
            self._generate_reports_and_visualizations(all_results)
            
            logger.info("Complete ML upgrade pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return all_results
    
    def _run_feature_engineering(self, reddit_data_path: str, 
                               price_data_paths: Dict[str, str], 
                               use_finbert: bool) -> pd.DataFrame:
        """Run feature engineering pipeline."""
        logger.info("Running feature engineering...")
        
        features_df = create_meme_stock_features(
            reddit_data_path=reddit_data_path,
            price_data_paths=price_data_paths,
            target_tickers=self.target_tickers,
            date_range=self.date_range,
            use_finbert=use_finbert,
            output_path=str(self.output_dir / "features_complete.csv")
        )
        
        logger.info(f"Feature engineering completed. Shape: {features_df.shape}")
        return features_df
    
    def _prepare_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare data splits for training."""
        logger.info("Preparing data splits...")
        
        # Ensure date column is datetime
        features_df = features_df.copy()
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        # Create time series validator
        validator = TimeSeriesValidator(
            train_start='2021-01-01',
            train_end='2022-12-31',
            val_start='2023-01-01',
            val_end='2023-06-30',
            test_start='2023-07-01',
            test_end='2023-12-31'
        )
        
        # Create splits
        splits = validator.create_splits(features_df, date_col='date')
        
        # Validate no leakage
        validator.validate_no_leakage(splits, date_col='date')
        
        # Prepare feature matrices
        feature_cols = [col for col in features_df.columns 
                       if col not in ['date', 'ticker', 'next_return']]
        
        train_data = splits['train'][feature_cols + ['next_return']].dropna()
        val_data = splits['val'][feature_cols + ['next_return']].dropna()
        test_data = splits['test'][feature_cols + ['next_return']].dropna()
        
        logger.info(f"Data splits prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _train_baseline_models(self, train_data: pd.DataFrame, 
                             val_data: pd.DataFrame) -> Dict[str, Any]:
        """Train baseline models."""
        logger.info("Training baseline models...")
        
        # Prepare data
        feature_cols = [col for col in train_data.columns if col != 'next_return']
        X_train = train_data[feature_cols].values
        y_train = train_data['next_return'].values
        X_val = val_data[feature_cols].values
        y_val = val_data['next_return'].values
        
        # Train models
        baseline_results = train_baseline_models(
            X_train, y_train, X_val, y_val, feature_cols
        )
        
        logger.info("Baseline models training completed")
        return baseline_results
    
    def _train_sequence_models(self, train_data: pd.DataFrame, 
                              val_data: pd.DataFrame) -> Dict[str, Any]:
        """Train sequence models."""
        logger.info("Training sequence models...")
        
        # Prepare data
        feature_cols = [col for col in train_data.columns if col != 'next_return']
        X_train = train_data[feature_cols].values
        y_train = train_data['next_return'].values
        X_val = val_data[feature_cols].values
        y_val = val_data['next_return'].values
        
        # Train sequence models
        sequence_results = train_sequence_models(
            X_train, y_train, X_val, y_val, sequence_length=20, epochs=50
        )
        
        logger.info("Sequence models training completed")
        return sequence_results
    
    def _create_ensemble(self, baseline_results: Dict[str, Any], 
                        train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """Create ensemble models."""
        logger.info("Creating ensemble...")
        
        # Prepare data
        feature_cols = [col for col in train_data.columns if col != 'next_return']
        X_train = train_data[feature_cols].values
        y_train = train_data['next_return'].values
        X_val = val_data[feature_cols].values
        y_val = val_data['next_return'].values
        
        # Combine models
        models = baseline_results['models'].copy()
        
        # Add sequence models if available
        if hasattr(self, '_sequence_results'):
            models.update(self._sequence_results['models'])
        
        # Create ensemble
        ensemble = create_meta_ensemble(
            models, X_train, y_train, X_val, y_val,
            use_simple_blending=True, use_stacking=True
        )
        
        # Evaluate ensemble
        ensemble_performance = ensemble.get_ensemble_performance(X_val, y_val)
        
        ensemble_results = {
            'ensemble': ensemble,
            'performance': ensemble_performance
        }
        
        logger.info("Ensemble creation completed")
        return ensemble_results
    
    def _implement_trading_strategy(self, test_data: pd.DataFrame, 
                                  all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Implement trading strategy."""
        logger.info("Implementing trading strategy...")
        
        # Prepare test data
        feature_cols = [col for col in test_data.columns if col != 'next_return']
        X_test = test_data[feature_cols].values
        y_test = test_data['next_return'].values
        
        # Get best model predictions
        best_model_name = None
        best_corr = -1
        
        if 'baseline_results' in all_results:
            for model_name, metrics in all_results['baseline_results']['results'].items():
                corr = metrics.get('spearman_corr', 0)
                if corr > best_corr:
                    best_corr = corr
                    best_model_name = model_name
        
        if best_model_name is None:
            logger.warning("No baseline model found, using Ridge as default")
            best_model_name = 'ridge'
        
        # Get predictions
        model = all_results['baseline_results']['models'][best_model_name]
        predictions = model.predict(X_test)
        
        # Test different strategies
        strategy_comparison = compare_strategies(
            predictions, y_test,
            thresholds=[0.0, 0.001, 0.002],
            costs=[0.001, 0.002],
            strategy_types=['contrarian', 'momentum']
        )
        
        # Get best strategy
        best_strategy_idx = strategy_comparison['sharpe_ratio'].idxmax()
        best_strategy_params = strategy_comparison.iloc[best_strategy_idx]
        
        # Run best strategy
        best_strategy_result = backtest_strategy(
            predictions, y_test,
            strategy_type=best_strategy_params['strategy_type'],
            threshold=best_strategy_params['threshold'],
            cost_half=best_strategy_params['cost_half']
        )
        
        strategy_results = {
            'best_model': best_model_name,
            'best_strategy': best_strategy_result,
            'strategy_comparison': strategy_comparison
        }
        
        logger.info("Trading strategy implementation completed")
        return strategy_results
    
    def _run_interpretability_analysis(self, train_data: pd.DataFrame, 
                                     val_data: pd.DataFrame, 
                                     all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run interpretability analysis."""
        logger.info("Running interpretability analysis...")
        
        # Prepare data
        feature_cols = [col for col in train_data.columns if col != 'next_return']
        X_train = train_data[feature_cols].values
        y_train = train_data['next_return'].values
        X_val = val_data[feature_cols].values
        y_val = val_data['next_return'].values
        
        # Define feature groups
        feature_groups = {
            'reddit': [i for i, col in enumerate(feature_cols) 
                      if any(x in col.lower() for x in ['reddit', 'mention', 'sentiment', 'confidence'])],
            'technical': [i for i, col in enumerate(feature_cols) 
                         if any(x in col.lower() for x in ['rsi', 'macd', 'bb', 'sma', 'ema'])],
            'dynamics': [i for i, col in enumerate(feature_cols) 
                        if any(x in col.lower() for x in ['momentum', 'volatility', 'trend'])]
        }
        
        # Get models
        models = {}
        if 'baseline_results' in all_results:
            models.update(all_results['baseline_results']['models'])
        
        # Run interpretability analysis
        interpretability_report = InterpretabilityReport()
        interpretability_results = interpretability_report.generate_report(
            models, X_val, y_val, feature_cols, feature_groups
        )
        
        # Validate contrarian hypothesis
        hypothesis_validation = interpretability_report.validate_contrarian_hypothesis(
            interpretability_results
        )
        interpretability_results['hypothesis_validation'] = hypothesis_validation
        
        logger.info("Interpretability analysis completed")
        return interpretability_results
    
    def _run_robustness_tests(self, train_data: pd.DataFrame, 
                            val_data: pd.DataFrame, 
                            all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run robustness tests."""
        logger.info("Running robustness tests...")
        
        # Prepare data
        feature_cols = [col for col in train_data.columns if col != 'next_return']
        X_train = train_data[feature_cols].values
        y_train = train_data['next_return'].values
        X_val = val_data[feature_cols].values
        y_val = val_data['next_return'].values
        
        # Combine train and val for robustness testing
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        
        # Get models
        models = {}
        if 'baseline_results' in all_results:
            models.update(all_results['baseline_results']['models'])
        
        # Run robustness analysis
        robustness_report = RobustnessReport(n_permutations=50)  # Reduced for speed
        
        # Create ticker groups (simplified)
        ticker_groups = {
            'GME': list(range(0, len(X_combined)//3)),
            'AMC': list(range(len(X_combined)//3, 2*len(X_combined)//3)),
            'BB': list(range(2*len(X_combined)//3, len(X_combined)))
        }
        
        robustness_results = robustness_report.generate_report(
            models, X_combined, y_combined, ticker_groups=ticker_groups
        )
        
        # Validate robustness
        robustness_validation = robustness_report.validate_robustness(robustness_results)
        robustness_results['validation'] = robustness_validation
        
        logger.info("Robustness testing completed")
        return robustness_results
    
    def _generate_reports_and_visualizations(self, all_results: Dict[str, Any]):
        """Generate reports and visualizations."""
        logger.info("Generating reports and visualizations...")
        
        # Create reporter
        reporter = ResultsReporter(str(self.output_dir))
        
        # Generate individual reports
        if 'baseline_results' in all_results:
            model_results = all_results['baseline_results']['results']
            feature_importance = {}
            for model_name, model in all_results['baseline_results']['models'].items():
                try:
                    feature_importance[model_name] = model.get_feature_importance()
                except:
                    feature_importance[model_name] = {}
            
            reporter.generate_model_report(
                model_results, feature_importance,
                str(self.output_dir / "model_report.md")
            )
        
        if 'strategy_results' in all_results:
            reporter.generate_strategy_report(
                all_results['strategy_results']['best_strategy'],
                str(self.output_dir / "strategy_report.md")
            )
        
        if 'robustness_results' in all_results:
            reporter.generate_robustness_report(
                all_results['robustness_results'],
                str(self.output_dir / "robustness_report.md")
            )
        
        # Generate comprehensive report
        comprehensive_results = {
            'model_results': all_results.get('baseline_results', {}).get('results', {}),
            'strategy_results': all_results.get('strategy_results', {}).get('best_strategy', {}),
            'robustness_results': all_results.get('robustness_results', {}),
            'interpretability_results': all_results.get('interpretability_results', {})
        }
        
        reporter.generate_comprehensive_report(
            comprehensive_results,
            str(self.output_dir)
        )
        
        # Generate visualizations
        try:
            figure_paths = generate_all_visualizations(
                comprehensive_results,
                str(self.output_dir / "figures")
            )
            logger.info(f"Generated {len(figure_paths)} visualizations")
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        # Save results as JSON
        results_file = self.output_dir / "all_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_for_json(all_results)
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info("Reports and visualizations generated successfully")

    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Meme Stock ML Upgrade Pipeline')
    parser.add_argument('--reddit-data', required=True, help='Path to Reddit data file')
    parser.add_argument('--price-data-dir', required=True, help='Directory containing price data files')
    parser.add_argument('--tickers', nargs='+', default=['GME', 'AMC', 'BB'], 
                       help='Tickers to analyze')
    parser.add_argument('--date-range', nargs=2, default=['2021-01-01', '2023-12-31'],
                       help='Date range for analysis')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--use-finbert', action='store_true', help='Use FinBERT for sentiment')
    parser.add_argument('--no-sequence-models', action='store_true', 
                       help='Skip sequence models training')
    parser.add_argument('--no-ensemble', action='store_true', 
                       help='Skip ensemble creation')
    
    args = parser.parse_args()
    
    # Create price data paths
    price_data_paths = {}
    for ticker in args.tickers:
        price_file = Path(args.price_data_dir) / f"{ticker}_extended_stock_data.csv"
        if price_file.exists():
            price_data_paths[ticker] = str(price_file)
        else:
            logger.warning(f"Price data file not found for {ticker}: {price_file}")
    
    if not price_data_paths:
        logger.error("No price data files found!")
        return
    
    # Initialize and run pipeline
    pipeline = MemeStockMLUpgrade(
        target_tickers=args.tickers,
        date_range=tuple(args.date_range),
        output_dir=args.output_dir
    )
    
    try:
        results = pipeline.run_complete_pipeline(
            reddit_data_path=args.reddit_data,
            price_data_paths=price_data_paths,
            use_finbert=args.use_finbert,
            use_sequence_models=not args.no_sequence_models,
            use_ensemble=not args.no_ensemble
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

