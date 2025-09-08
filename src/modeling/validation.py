#!/usr/bin/env python3
"""
Validation framework for meme stock contrarian effect prediction.

Implements purged K-fold, walk-forward, and other validation schemes
as described in the manual to prevent data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Generator
import logging
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
import warnings

logger = logging.getLogger(__name__)

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validation for time series data.
    
    Prevents data leakage by purging overlapping samples and adding embargo periods.
    """
    
    def __init__(self, n_splits: int = 5, purge_days: int = 1, embargo_days: int = 5):
        """
        Initialize Purged K-Fold.
        
        Args:
            n_splits: Number of folds
            purge_days: Days to purge around validation period
            embargo_days: Days to embargo after validation period
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/validation splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (e.g., dates)
            
        Yields:
            Tuple of (train_indices, val_indices)
        """
        if groups is None:
            raise ValueError("Groups (dates) must be provided for purged CV")
        
        X, y, groups = indexable(X, y, groups)
        
        # Sort by groups (dates)
        sorted_indices = np.argsort(groups)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices] if y is not None else None
        groups_sorted = groups[sorted_indices]
        
        # Create fold boundaries
        unique_groups = np.unique(groups_sorted)
        n_groups = len(unique_groups)
        
        fold_size = n_groups // self.n_splits
        
        for i in range(self.n_splits):
            # Define validation period
            val_start_idx = i * fold_size
            val_end_idx = min((i + 1) * fold_size, n_groups)
            
            val_groups = unique_groups[val_start_idx:val_end_idx]
            
            # Find indices for validation set
            val_mask = np.isin(groups_sorted, val_groups)
            val_indices = np.where(val_mask)[0]
            
            # Purge overlapping samples
            if len(val_indices) > 0:
                val_start_date = groups_sorted[val_indices[0]]
                val_end_date = groups_sorted[val_indices[-1]]
                
                # Add purge and embargo periods
                purge_start = val_start_date - pd.Timedelta(days=self.purge_days)
                purge_end = val_end_date + pd.Timedelta(days=self.embargo_days)
                
                # Find indices to exclude
                exclude_mask = (groups_sorted >= purge_start) & (groups_sorted <= purge_end)
                exclude_indices = np.where(exclude_mask)[0]
                
                # Training indices are all indices not in exclude set
                train_indices = np.setdiff1d(np.arange(len(groups_sorted)), exclude_indices)
            else:
                train_indices = np.arange(len(groups_sorted))
            
            # Convert back to original indices
            train_indices_orig = sorted_indices[train_indices]
            val_indices_orig = sorted_indices[val_indices]
            
            yield train_indices_orig, val_indices_orig
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits

class WalkForwardValidator:
    """
    Walk-forward validation for time series data.
    
    Implements expanding and rolling window validation schemes.
    """
    
    def __init__(self, 
                 initial_train_size: int = 252,  # 1 year
                 step_size: int = 63,  # 3 months
                 validation_size: int = 21,  # 1 month
                 expanding_window: bool = True):
        """
        Initialize walk-forward validator.
        
        Args:
            initial_train_size: Initial training window size (days)
            step_size: Step size for moving window (days)
            validation_size: Validation window size (days)
            expanding_window: Whether to use expanding (True) or rolling (False) window
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.validation_size = validation_size
        self.expanding_window = expanding_window
    
    def split(self, X, y=None, groups=None):
        """
        Generate walk-forward train/validation splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (dates)
            
        Yields:
            Tuple of (train_indices, val_indices)
        """
        if groups is None:
            raise ValueError("Groups (dates) must be provided for walk-forward validation")
        
        X, y, groups = indexable(X, y, groups)
        
        # Sort by groups (dates)
        sorted_indices = np.argsort(groups)
        groups_sorted = groups[sorted_indices]
        unique_groups = np.unique(groups_sorted)
        
        n_groups = len(unique_groups)
        
        # Calculate number of steps
        n_steps = (n_groups - self.initial_train_size - self.validation_size) // self.step_size + 1
        
        for step in range(n_steps):
            # Define training period
            train_start_idx = 0 if self.expanding_window else step * self.step_size
            train_end_idx = self.initial_train_size + step * self.step_size
            
            # Define validation period
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + self.validation_size
            
            # Check bounds
            if val_end_idx >= n_groups:
                break
            
            train_groups = unique_groups[train_start_idx:train_end_idx]
            val_groups = unique_groups[val_start_idx:val_end_idx]
            
            # Find indices
            train_mask = np.isin(groups_sorted, train_groups)
            val_mask = np.isin(groups_sorted, val_groups)
            
            train_indices = sorted_indices[np.where(train_mask)[0]]
            val_indices = sorted_indices[np.where(val_mask)[0]]
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        if groups is None:
            return 0
        
        groups_sorted = np.sort(groups)
        n_groups = len(np.unique(groups_sorted))
        
        n_steps = (n_groups - self.initial_train_size - self.validation_size) // self.step_size + 1
        return max(0, n_steps)

class TimeSeriesValidator:
    """
    Comprehensive time series validation framework.
    """
    
    def __init__(self, 
                 train_start: str = '2021-01-01',
                 train_end: str = '2022-12-31',
                 val_start: str = '2023-01-01',
                 val_end: str = '2023-06-30',
                 test_start: str = '2023-07-01',
                 test_end: str = '2023-12-31'):
        """
        Initialize time series validator.
        
        Args:
            train_start: Training period start
            train_end: Training period end
            val_start: Validation period start
            val_end: Validation period end
            test_start: Test period start
            test_end: Test period end
        """
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.val_start = pd.to_datetime(val_start)
        self.val_end = pd.to_datetime(val_end)
        self.test_start = pd.to_datetime(test_start)
        self.test_end = pd.to_datetime(test_end)
    
    def create_splits(self, df: pd.DataFrame, date_col: str = 'date') -> Dict[str, pd.DataFrame]:
        """
        Create train/validation/test splits.
        
        Args:
            df: Input dataframe
            date_col: Date column name
            
        Returns:
            Dictionary with 'train', 'val', 'test' dataframes
        """
        logger.info("Creating time series splits...")
        
        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create masks
        train_mask = (df[date_col] >= self.train_start) & (df[date_col] <= self.train_end)
        val_mask = (df[date_col] >= self.val_start) & (df[date_col] <= self.val_end)
        test_mask = (df[date_col] >= self.test_start) & (df[date_col] <= self.test_end)
        
        # Create splits
        splits = {
            'train': df[train_mask].copy(),
            'val': df[val_mask].copy(),
            'test': df[test_mask].copy()
        }
        
        # Log split sizes
        for split_name, split_df in splits.items():
            logger.info(f"{split_name} split: {len(split_df)} records")
        
        return splits
    
    def validate_no_leakage(self, splits: Dict[str, pd.DataFrame], date_col: str = 'date') -> bool:
        """
        Validate that there's no data leakage between splits.
        
        Args:
            splits: Dictionary of dataframes
            date_col: Date column name
            
        Returns:
            True if no leakage detected
        """
        logger.info("Validating no data leakage...")
        
        train_dates = splits['train'][date_col]
        val_dates = splits['val'][date_col]
        test_dates = splits['test'][date_col]
        
        # Check for overlap
        train_max = train_dates.max()
        val_min = val_dates.min()
        val_max = val_dates.max()
        test_min = test_dates.min()
        
        leakage_detected = False
        
        if train_max >= val_min:
            logger.error(f"Data leakage detected: train max ({train_max}) >= val min ({val_min})")
            leakage_detected = True
        
        if val_max >= test_min:
            logger.error(f"Data leakage detected: val max ({val_max}) >= test min ({test_min})")
            leakage_detected = True
        
        if not leakage_detected:
            logger.info("No data leakage detected")
        
        return not leakage_detected

class CrossValidationResults:
    """Container for cross-validation results."""
    
    def __init__(self):
        self.results = []
        self.metrics = {}
    
    def add_fold_result(self, fold_idx: int, train_metrics: Dict, val_metrics: Dict):
        """Add results from a single fold."""
        self.results.append({
            'fold': fold_idx,
            'train': train_metrics,
            'val': val_metrics
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across folds."""
        if not self.results:
            return {}
        
        summary = {}
        
        # Extract metrics
        train_metrics = [r['train'] for r in self.results]
        val_metrics = [r['val'] for r in self.results]
        
        # Calculate statistics for each metric
        for metric_name in train_metrics[0].keys():
            train_values = [m[metric_name] for m in train_metrics]
            val_values = [m[metric_name] for m in val_metrics]
            
            summary[f'train_{metric_name}_mean'] = np.mean(train_values)
            summary[f'train_{metric_name}_std'] = np.std(train_values)
            summary[f'val_{metric_name}_mean'] = np.mean(val_values)
            summary[f'val_{metric_name}_std'] = np.std(val_values)
        
        return summary

def run_cross_validation(model, X, y, groups, cv_method='purged', **cv_params):
    """
    Run cross-validation with specified method.
    
    Args:
        model: Model to validate
        X: Feature matrix
        y: Target vector
        groups: Group labels (dates)
        cv_method: 'purged', 'walk_forward', or 'time_series'
        **cv_params: Parameters for CV method
        
    Returns:
        CrossValidationResults object
    """
    logger.info(f"Running {cv_method} cross-validation...")
    
    # Select CV method
    if cv_method == 'purged':
        cv = PurgedKFold(**cv_params)
    elif cv_method == 'walk_forward':
        cv = WalkForwardValidator(**cv_params)
    elif cv_method == 'time_series':
        cv = TimeSeriesValidator(**cv_params)
    else:
        raise ValueError(f"Unknown CV method: {cv_method}")
    
    results = CrossValidationResults()
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        logger.info(f"Processing fold {fold_idx + 1}...")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, train_pred)
        val_metrics = calculate_metrics(y_val, val_pred)
        
        # Store results
        results.add_fold_result(fold_idx, train_metrics, val_metrics)
    
    logger.info(f"Cross-validation completed. {len(results.results)} folds processed.")
    
    return results

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr, pearsonr
    
    metrics = {}
    
    # Regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Correlation metrics
    try:
        metrics['pearson_corr'] = pearsonr(y_true, y_pred)[0]
        metrics['spearman_corr'] = spearmanr(y_true, y_pred)[0]
    except:
        metrics['pearson_corr'] = 0.0
        metrics['spearman_corr'] = 0.0
    
    # Hit rate (sign accuracy)
    metrics['hit_rate'] = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    return metrics

if __name__ == "__main__":
    # Test validation framework
    np.random.seed(42)
    
    # Create sample data
    dates = pd.date_range('2021-01-01', periods=500, freq='D')
    n_samples = len(dates)
    
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)
    groups = dates
    
    # Test purged K-fold
    cv = PurgedKFold(n_splits=5, purge_days=1, embargo_days=5)
    
    print("Purged K-Fold splits:")
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    # Test walk-forward
    wf = WalkForwardValidator(initial_train_size=100, step_size=50, validation_size=20)
    
    print("\nWalk-forward splits:")
    for i, (train_idx, val_idx) in enumerate(wf.split(X, y, groups)):
        print(f"Step {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        if i >= 2:  # Limit output
            break

