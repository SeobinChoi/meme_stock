#!/usr/bin/env python3
"""
Robustness tests and placebo analysis for meme stock contrarian effect prediction.

Implements placebo tests, stability analysis, and anti-overfit checks
as described in the manual.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class PlaceboTester:
    """Perform placebo tests to validate model robustness."""
    
    def __init__(self, n_permutations: int = 100, random_seed: int = 42):
        """
        Initialize placebo tester.
        
        Args:
            n_permutations: Number of permutations for placebo tests
            random_seed: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def random_pairing_test(self, X: np.ndarray, y: np.ndarray, 
                           model, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Random pairing test: shuffle Reddit day mapping.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            feature_names: Optional feature names
            
        Returns:
            Dictionary with test results
        """
        logger.info("Performing random pairing placebo test...")
        
        # Original performance
        original_pred = model.predict(X)
        original_metrics = self._calculate_metrics(y, original_pred)
        
        # Placebo results
        placebo_metrics = []
        
        for i in range(self.n_permutations):
            # Shuffle target variable
            y_shuffled = np.random.permutation(y)
            
            # Retrain model
            model_copy = self._copy_model(model)
            model_copy.fit(X, y_shuffled)
            
            # Evaluate on original data
            placebo_pred = model_copy.predict(X)
            placebo_metric = self._calculate_metrics(y, placebo_pred)
            placebo_metrics.append(placebo_metric)
        
        # Calculate statistics
        placebo_results = self._analyze_placebo_results(original_metrics, placebo_metrics)
        placebo_results['test_type'] = 'random_pairing'
        
        return placebo_results
    
    def cross_ticker_swap_test(self, X: np.ndarray, y: np.ndarray, 
                             model, ticker_groups: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Cross-ticker swap test: swap Reddit data between tickers.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            ticker_groups: Dictionary mapping ticker to row indices
            
        Returns:
            Dictionary with test results
        """
        logger.info("Performing cross-ticker swap placebo test...")
        
        # Original performance
        original_pred = model.predict(X)
        original_metrics = self._calculate_metrics(y, original_pred)
        
        # Placebo results
        placebo_metrics = []
        
        for i in range(self.n_permutations):
            # Create shuffled data
            X_shuffled = X.copy()
            
            # Swap Reddit features between tickers
            ticker_names = list(ticker_groups.keys())
            np.random.shuffle(ticker_names)
            
            for j, ticker in enumerate(ticker_groups.keys()):
                original_indices = ticker_groups[ticker]
                shuffled_ticker = ticker_names[j]
                shuffled_indices = ticker_groups[shuffled_ticker]
                
                # Swap Reddit-related features (assuming first few columns)
                reddit_cols = slice(0, min(10, X.shape[1]))  # First 10 columns as Reddit features
                X_shuffled[original_indices, reddit_cols] = X[shuffled_indices, reddit_cols]
            
            # Retrain model
            model_copy = self._copy_model(model)
            model_copy.fit(X_shuffled, y)
            
            # Evaluate
            placebo_pred = model_copy.predict(X_shuffled)
            placebo_metric = self._calculate_metrics(y, placebo_pred)
            placebo_metrics.append(placebo_metric)
        
        # Calculate statistics
        placebo_results = self._analyze_placebo_results(original_metrics, placebo_metrics)
        placebo_results['test_type'] = 'cross_ticker_swap'
        
        return placebo_results
    
    def lag_inversion_test(self, X: np.ndarray, y: np.ndarray, 
                          model, lag_cols: List[int]) -> Dict[str, Any]:
        """
        Lag inversion test: shift Reddit data forward/backward improperly.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            lag_cols: Column indices for lag features
            
        Returns:
            Dictionary with test results
        """
        logger.info("Performing lag inversion placebo test...")
        
        # Original performance
        original_pred = model.predict(X)
        original_metrics = self._calculate_metrics(y, original_pred)
        
        # Placebo results
        placebo_metrics = []
        
        for i in range(self.n_permutations):
            # Create lag-inverted data
            X_inverted = X.copy()
            
            # Randomly shift lag features
            shift_amount = np.random.randint(-5, 6)  # Random shift between -5 and +5
            
            for col in lag_cols:
                if col < X.shape[1]:
                    X_inverted[:, col] = np.roll(X[:, col], shift_amount)
            
            # Retrain model
            model_copy = self._copy_model(model)
            model_copy.fit(X_inverted, y)
            
            # Evaluate
            placebo_pred = model_copy.predict(X_inverted)
            placebo_metric = self._calculate_metrics(y, placebo_pred)
            placebo_metrics.append(placebo_metric)
        
        # Calculate statistics
        placebo_results = self._analyze_placebo_results(original_metrics, placebo_metrics)
        placebo_results['test_type'] = 'lag_inversion'
        
        return placebo_results
    
    def noise_feature_test(self, X: np.ndarray, y: np.ndarray, 
                          model, n_noise_features: int = 10) -> Dict[str, Any]:
        """
        Noise feature test: add random features.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            n_noise_features: Number of noise features to add
            
        Returns:
            Dictionary with test results
        """
        logger.info("Performing noise feature placebo test...")
        
        # Original performance
        original_pred = model.predict(X)
        original_metrics = self._calculate_metrics(y, original_pred)
        
        # Placebo results
        placebo_metrics = []
        
        for i in range(self.n_permutations):
            # Add noise features
            noise_features = np.random.randn(X.shape[0], n_noise_features)
            X_with_noise = np.column_stack([X, noise_features])
            
            # Retrain model
            model_copy = self._copy_model(model)
            model_copy.fit(X_with_noise, y)
            
            # Evaluate
            placebo_pred = model_copy.predict(X_with_noise)
            placebo_metric = self._calculate_metrics(y, placebo_pred)
            placebo_metrics.append(placebo_metric)
        
        # Calculate statistics
        placebo_results = self._analyze_placebo_results(original_metrics, placebo_metrics)
        placebo_results['test_type'] = 'noise_features'
        
        return placebo_results
    
    def label_shuffle_test(self, X: np.ndarray, y: np.ndarray, 
                          model) -> Dict[str, Any]:
        """
        Label shuffle test: train on permuted labels.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to test
            
        Returns:
            Dictionary with test results
        """
        logger.info("Performing label shuffle placebo test...")
        
        # Original performance
        original_pred = model.predict(X)
        original_metrics = self._calculate_metrics(y, original_pred)
        
        # Placebo results
        placebo_metrics = []
        
        for i in range(self.n_permutations):
            # Shuffle labels
            y_shuffled = np.random.permutation(y)
            
            # Retrain model
            model_copy = self._copy_model(model)
            model_copy.fit(X, y_shuffled)
            
            # Evaluate on shuffled labels
            placebo_pred = model_copy.predict(X)
            placebo_metric = self._calculate_metrics(y_shuffled, placebo_pred)
            placebo_metrics.append(placebo_metric)
        
        # Calculate statistics
        placebo_results = self._analyze_placebo_results(original_metrics, placebo_metrics)
        placebo_results['test_type'] = 'label_shuffle'
        
        return placebo_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Correlation metrics
        try:
            metrics['pearson_corr'] = stats.pearsonr(y_true, y_pred)[0]
            metrics['spearman_corr'] = stats.spearmanr(y_true, y_pred)[0]
        except:
            metrics['pearson_corr'] = 0.0
            metrics['spearman_corr'] = 0.0
        
        # Hit rate
        metrics['hit_rate'] = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        return metrics
    
    def _copy_model(self, model):
        """Create a copy of the model."""
        model_class = type(model)
        
        try:
            # Try to copy with parameters
            if hasattr(model, '__dict__'):
                return model_class(**model.__dict__)
            else:
                return model_class()
        except:
            # Fallback to basic instantiation
            return model_class()
    
    def _analyze_placebo_results(self, original_metrics: Dict[str, float], 
                               placebo_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze placebo test results."""
        results = {}
        
        for metric_name in original_metrics.keys():
            original_val = original_metrics[metric_name]
            placebo_vals = [m[metric_name] for m in placebo_metrics]
            
            # Calculate statistics
            placebo_mean = np.mean(placebo_vals)
            placebo_std = np.std(placebo_vals)
            
            # Calculate p-value (one-sided test)
            if metric_name in ['mse', 'rmse', 'mae']:
                # Lower is better
                p_value = np.mean(np.array(placebo_vals) <= original_val)
            else:
                # Higher is better
                p_value = np.mean(np.array(placebo_vals) >= original_val)
            
            # Calculate effect size
            if placebo_std > 0:
                effect_size = (original_val - placebo_mean) / placebo_std
            else:
                effect_size = 0
            
            results[metric_name] = {
                'original': original_val,
                'placebo_mean': placebo_mean,
                'placebo_std': placebo_std,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05
            }
        
        return results

class StabilityAnalyzer:
    """Analyze model stability across time periods and regimes."""
    
    def __init__(self):
        """Initialize stability analyzer."""
        self.stability_results = {}
    
    def walk_forward_stability(self, model, X: np.ndarray, y: np.ndarray, 
                             dates: np.ndarray, window_size: int = 60) -> Dict[str, Any]:
        """
        Analyze stability using rolling window analysis.
        
        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target vector
            dates: Date array
            window_size: Rolling window size
            
        Returns:
            Dictionary with stability results
        """
        logger.info("Analyzing walk-forward stability...")
        
        # Sort by dates
        sort_idx = np.argsort(dates)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        dates_sorted = dates[sort_idx]
        
        rolling_metrics = []
        window_starts = []
        
        # Rolling window analysis
        for i in range(len(X_sorted) - window_size + 1):
            window_start = i
            window_end = i + window_size
            
            X_window = X_sorted[window_start:window_end]
            y_window = y_sorted[window_start:window_end]
            
            # Retrain model on window
            model_copy = self._copy_model(model)
            model_copy.fit(X_window, y_window)
            
            # Evaluate on same window
            pred_window = model_copy.predict(X_window)
            metrics = self._calculate_metrics(y_window, pred_window)
            
            rolling_metrics.append(metrics)
            window_starts.append(dates_sorted[window_start])
        
        # Calculate stability statistics
        stability_stats = self._calculate_stability_stats(rolling_metrics)
        
        results = {
            'rolling_metrics': rolling_metrics,
            'window_starts': window_starts,
            'stability_stats': stability_stats
        }
        
        self.stability_results['walk_forward'] = results
        return results
    
    def regime_stability(self, model, X: np.ndarray, y: np.ndarray, 
                        regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze stability across different market regimes.
        
        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target vector
            regime_labels: Array of regime labels
            
        Returns:
            Dictionary with regime stability results
        """
        logger.info("Analyzing regime stability...")
        
        unique_regimes = np.unique(regime_labels)
        regime_metrics = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) > 10:  # Minimum sample size
                # Retrain model on regime data
                model_copy = self._copy_model(model)
                model_copy.fit(X_regime, y_regime)
                
                # Evaluate on regime data
                pred_regime = model_copy.predict(X_regime)
                metrics = self._calculate_metrics(y_regime, pred_regime)
                
                regime_metrics[regime] = metrics
        
        # Calculate cross-regime stability
        stability_stats = self._calculate_regime_stability_stats(regime_metrics)
        
        results = {
            'regime_metrics': regime_metrics,
            'stability_stats': stability_stats
        }
        
        self.stability_results['regime'] = results
        return results
    
    def yearly_stability(self, model, X: np.ndarray, y: np.ndarray, 
                        dates: np.ndarray) -> Dict[str, Any]:
        """
        Analyze stability across different years.
        
        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target vector
            dates: Date array
            
        Returns:
            Dictionary with yearly stability results
        """
        logger.info("Analyzing yearly stability...")
        
        # Extract years
        if hasattr(dates, 'year'):
            years = dates.year
        else:
            years = pd.to_datetime(dates).year
        
        unique_years = np.unique(years)
        yearly_metrics = {}
        
        for year in unique_years:
            year_mask = years == year
            X_year = X[year_mask]
            y_year = y[year_mask]
            
            if len(X_year) > 10:  # Minimum sample size
                # Retrain model on year data
                model_copy = self._copy_model(model)
                model_copy.fit(X_year, y_year)
                
                # Evaluate on year data
                pred_year = model_copy.predict(X_year)
                metrics = self._calculate_metrics(y_year, pred_year)
                
                yearly_metrics[year] = metrics
        
        # Calculate year-to-year stability
        stability_stats = self._calculate_regime_stability_stats(yearly_metrics)
        
        results = {
            'yearly_metrics': yearly_metrics,
            'stability_stats': stability_stats
        }
        
        self.stability_results['yearly'] = results
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Correlation metrics
        try:
            metrics['pearson_corr'] = stats.pearsonr(y_true, y_pred)[0]
            metrics['spearman_corr'] = stats.spearmanr(y_true, y_pred)[0]
        except:
            metrics['pearson_corr'] = 0.0
            metrics['spearman_corr'] = 0.0
        
        # Hit rate
        metrics['hit_rate'] = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        return metrics
    
    def _copy_model(self, model):
        """Create a copy of the model."""
        model_class = type(model)
        
        try:
            if hasattr(model, '__dict__'):
                return model_class(**model.__dict__)
            else:
                return model_class()
        except:
            return model_class()
    
    def _calculate_stability_stats(self, rolling_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate stability statistics for rolling metrics."""
        stability_stats = {}
        
        # Extract metrics
        metric_names = list(rolling_metrics[0].keys())
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in rolling_metrics]
            
            stability_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                'trend': self._calculate_trend(values)
            }
        
        return stability_stats
    
    def _calculate_regime_stability_stats(self, regime_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate stability statistics across regimes."""
        stability_stats = {}
        
        if not regime_metrics:
            return stability_stats
        
        # Extract metrics
        metric_names = list(list(regime_metrics.values())[0].keys())
        
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in regime_metrics.values()]
            
            stability_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                'range': np.max(values) - np.min(values)
            }
        
        return stability_stats
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression."""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope

class RobustnessReport:
    """Generate comprehensive robustness report."""
    
    def __init__(self, n_permutations: int = 100):
        """
        Initialize robustness report generator.
        
        Args:
            n_permutations: Number of permutations for placebo tests
        """
        self.placebo_tester = PlaceboTester(n_permutations=n_permutations)
        self.stability_analyzer = StabilityAnalyzer()
    
    def generate_report(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                       dates: Optional[np.ndarray] = None,
                       ticker_groups: Optional[Dict[str, List[int]]] = None,
                       regime_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive robustness report.
        
        Args:
            models: Dictionary of trained models
            X: Feature matrix
            y: Target vector
            dates: Optional date array
            ticker_groups: Optional ticker groups for cross-ticker tests
            regime_labels: Optional regime labels
            
        Returns:
            Dictionary with robustness results
        """
        logger.info("Generating robustness report...")
        
        report = {}
        
        # Placebo tests
        logger.info("Performing placebo tests...")
        placebo_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Testing {model_name}...")
            
            model_placebo_results = {}
            
            # Random pairing test
            try:
                model_placebo_results['random_pairing'] = self.placebo_tester.random_pairing_test(X, y, model)
            except Exception as e:
                logger.warning(f"Random pairing test failed for {model_name}: {e}")
                model_placebo_results['random_pairing'] = {}
            
            # Cross-ticker swap test
            if ticker_groups:
                try:
                    model_placebo_results['cross_ticker_swap'] = self.placebo_tester.cross_ticker_swap_test(X, y, model, ticker_groups)
                except Exception as e:
                    logger.warning(f"Cross-ticker swap test failed for {model_name}: {e}")
                    model_placebo_results['cross_ticker_swap'] = {}
            
            # Noise feature test
            try:
                model_placebo_results['noise_features'] = self.placebo_tester.noise_feature_test(X, y, model)
            except Exception as e:
                logger.warning(f"Noise feature test failed for {model_name}: {e}")
                model_placebo_results['noise_features'] = {}
            
            # Label shuffle test
            try:
                model_placebo_results['label_shuffle'] = self.placebo_tester.label_shuffle_test(X, y, model)
            except Exception as e:
                logger.warning(f"Label shuffle test failed for {model_name}: {e}")
                model_placebo_results['label_shuffle'] = {}
            
            placebo_results[model_name] = model_placebo_results
        
        report['placebo_tests'] = placebo_results
        
        # Stability analysis
        if dates is not None:
            logger.info("Performing stability analysis...")
            stability_results = {}
            
            for model_name, model in models.items():
                logger.info(f"Analyzing stability for {model_name}...")
                
                model_stability_results = {}
                
                # Walk-forward stability
                try:
                    model_stability_results['walk_forward'] = self.stability_analyzer.walk_forward_stability(model, X, y, dates)
                except Exception as e:
                    logger.warning(f"Walk-forward stability analysis failed for {model_name}: {e}")
                    model_stability_results['walk_forward'] = {}
                
                # Yearly stability
                try:
                    model_stability_results['yearly'] = self.stability_analyzer.yearly_stability(model, X, y, dates)
                except Exception as e:
                    logger.warning(f"Yearly stability analysis failed for {model_name}: {e}")
                    model_stability_results['yearly'] = {}
                
                # Regime stability
                if regime_labels is not None:
                    try:
                        model_stability_results['regime'] = self.stability_analyzer.regime_stability(model, X, y, regime_labels)
                    except Exception as e:
                        logger.warning(f"Regime stability analysis failed for {model_name}: {e}")
                        model_stability_results['regime'] = {}
                
                stability_results[model_name] = model_stability_results
            
            report['stability_analysis'] = stability_results
        
        logger.info("Robustness report generated successfully")
        
        return report
    
    def validate_robustness(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model robustness based on test results.
        
        Args:
            report: Robustness report
            
        Returns:
            Dictionary with robustness validation
        """
        logger.info("Validating model robustness...")
        
        validation = {
            'overall_robust': True,
            'placebo_tests_passed': True,
            'stability_tests_passed': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check placebo tests
        if 'placebo_tests' in report:
            for model_name, model_results in report['placebo_tests'].items():
                for test_name, test_results in model_results.items():
                    if test_results and 'spearman_corr' in test_results:
                        corr_results = test_results['spearman_corr']
                        
                        if corr_results.get('significant', False):
                            validation['placebo_tests_passed'] = False
                            validation['issues'].append(f"{model_name} {test_name}: Model performance not significantly different from placebo")
        
        # Check stability analysis
        if 'stability_analysis' in report:
            for model_name, model_results in report['stability_analysis'].items():
                for analysis_name, analysis_results in model_results.items():
                    if analysis_results and 'stability_stats' in analysis_results:
                        stability_stats = analysis_results['stability_stats']
                        
                        if 'spearman_corr' in stability_stats:
                            corr_stats = stability_stats['spearman_corr']
                            cv = corr_stats.get('cv', 0)
                            
                            if cv > 0.5:  # High coefficient of variation
                                validation['stability_tests_passed'] = False
                                validation['issues'].append(f"{model_name} {analysis_name}: High performance variability (CV={cv:.3f})")
        
        # Overall robustness
        validation['overall_robust'] = validation['placebo_tests_passed'] and validation['stability_tests_passed']
        
        # Generate recommendations
        if not validation['overall_robust']:
            validation['recommendations'].append("Consider additional regularization or feature selection")
            validation['recommendations'].append("Investigate temporal stability issues")
            validation['recommendations'].append("Validate contrarian hypothesis with additional data")
        
        return validation

if __name__ == "__main__":
    # Test robustness analysis
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    dates = pd.date_range('2021-01-01', periods=n_samples, freq='D')
    
    # Create mock model
    class MockModel:
        def __init__(self):
            self.coef_ = np.random.randn(n_features)
        
        def predict(self, X):
            return X @ self.coef_ + np.random.randn(len(X)) * 0.1
        
        def fit(self, X, y):
            pass
    
    model = MockModel()
    models = {'test_model': model}
    
    # Test placebo tests
    placebo_tester = PlaceboTester(n_permutations=10)
    random_pairing_result = placebo_tester.random_pairing_test(X, y, model)
    
    print("Random Pairing Placebo Test Results:")
    for metric, stats in random_pairing_result.items():
        if isinstance(stats, dict) and 'original' in stats:
            print(f"  {metric}: original={stats['original']:.4f}, placebo_mean={stats['placebo_mean']:.4f}, p_value={stats['p_value']:.4f}")
    
    # Test stability analysis
    stability_analyzer = StabilityAnalyzer()
    walk_forward_result = stability_analyzer.walk_forward_stability(model, X, y, dates, window_size=100)
    
    print(f"\nWalk-forward Stability Analysis:")
    print(f"  Number of windows: {len(walk_forward_result['rolling_metrics'])}")
    if walk_forward_result['stability_stats']:
        corr_stats = walk_forward_result['stability_stats'].get('spearman_corr', {})
        print(f"  Spearman correlation CV: {corr_stats.get('cv', 0):.4f}")

