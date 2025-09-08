#!/usr/bin/env python3
"""
Trading strategy implementation for meme stock contrarian effect prediction.

Implements positioning rules, costs, slippage, and portfolio metrics
as described in the manual.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Base trading strategy class."""
    
    def __init__(self, 
                 threshold: float = 0.0,
                 cost_half: float = 0.001,  # 10 bps enter/exit
                 max_position: float = 1.0,
                 stateful: bool = True):
        """
        Initialize trading strategy.
        
        Args:
            threshold: Signal threshold for position changes
            cost_half: Half-round-trip cost (enter or exit)
            max_position: Maximum position size
            stateful: Whether to maintain stateful positions
        """
        self.threshold = threshold
        self.cost_half = cost_half
        self.max_position = max_position
        self.stateful = stateful
        
        self.positions = []
        self.returns = []
        self.costs = []
        self.signals = []
    
    def calculate_position(self, signal: float, current_position: float = 0.0) -> float:
        """
        Calculate position based on signal.
        
        Args:
            signal: Model prediction signal
            current_position: Current position
            
        Returns:
            New position
        """
        if self.stateful:
            # Stateful: keep position unless threshold crossed
            if signal >= self.threshold and current_position <= 0:
                return self.max_position
            elif signal <= -self.threshold and current_position >= 0:
                return -self.max_position
            else:
                return current_position
        else:
            # Non-stateful: pure signal-based
            if signal >= self.threshold:
                return self.max_position
            elif signal <= -self.threshold:
                return -self.max_position
            else:
                return 0.0
    
    def calculate_costs(self, old_position: float, new_position: float) -> float:
        """
        Calculate trading costs.
        
        Args:
            old_position: Previous position
            new_position: New position
            
        Returns:
            Trading cost
        """
        position_change = abs(new_position - old_position)
        return self.cost_half * position_change
    
    def execute_strategy(self, signals: np.ndarray, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute trading strategy.
        
        Args:
            signals: Model predictions
            returns: Actual returns
            
        Returns:
            Dictionary with strategy results
        """
        logger.info(f"Executing strategy with threshold={self.threshold}, cost={self.cost_half}")
        
        n_periods = len(signals)
        positions = np.zeros(n_periods)
        strategy_returns = np.zeros(n_periods)
        costs = np.zeros(n_periods)
        
        current_position = 0.0
        
        for t in range(n_periods):
            # Calculate new position
            new_position = self.calculate_position(signals[t], current_position)
            positions[t] = new_position
            
            # Calculate costs
            cost = self.calculate_costs(current_position, new_position)
            costs[t] = cost
            
            # Calculate strategy return
            strategy_returns[t] = new_position * returns[t] - cost
            
            # Update position
            current_position = new_position
        
        # Store results
        self.positions = positions
        self.returns = strategy_returns
        self.costs = costs
        self.signals = signals
        
        return {
            'positions': positions,
            'returns': strategy_returns,
            'costs': costs,
            'signals': signals
        }

class ContrarianStrategy(TradingStrategy):
    """Contrarian trading strategy that bets against Reddit sentiment."""
    
    def __init__(self, 
                 threshold: float = 0.001,  # 0.1% threshold
                 cost_half: float = 0.001,
                 max_position: float = 1.0,
                 contrarian_multiplier: float = -1.0):
        """
        Initialize contrarian strategy.
        
        Args:
            threshold: Signal threshold
            cost_half: Half-round-trip cost
            max_position: Maximum position size
            contrarian_multiplier: Multiplier for contrarian effect (-1 for inverse)
        """
        super().__init__(threshold, cost_half, max_position)
        self.contrarian_multiplier = contrarian_multiplier
    
    def calculate_position(self, signal: float, current_position: float = 0.0) -> float:
        """Calculate contrarian position."""
        # Apply contrarian multiplier
        contrarian_signal = signal * self.contrarian_multiplier
        
        if self.stateful:
            if contrarian_signal >= self.threshold and current_position <= 0:
                return self.max_position
            elif contrarian_signal <= -self.threshold and current_position >= 0:
                return -self.max_position
            else:
                return current_position
        else:
            if contrarian_signal >= self.threshold:
                return self.max_position
            elif contrarian_signal <= -self.threshold:
                return -self.max_position
            else:
                return 0.0

class PortfolioMetrics:
    """Calculate portfolio performance metrics."""
    
    def __init__(self, annual_factor: float = 252):
        """
        Initialize portfolio metrics calculator.
        
        Args:
            annual_factor: Factor to annualize metrics (252 for daily data)
        """
        self.annual_factor = annual_factor
    
    def calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Dictionary of metrics
        """
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (self.annual_factor / len(returns)) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(self.annual_factor)
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0
        
        # Hit rate
        hit_rate = np.mean(returns > 0)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        if abs(max_drawdown) > 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        # Turnover (average absolute position changes)
        # This would need position data, so we'll calculate it separately
        
        # Information Coefficient (correlation with actual returns)
        # This would need actual returns, so we'll calculate it separately
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def calculate_turnover(self, positions: np.ndarray) -> float:
        """Calculate average turnover."""
        if len(positions) < 2:
            return 0.0
        
        position_changes = np.abs(np.diff(positions))
        return np.mean(position_changes)
    
    def calculate_information_coefficient(self, predictions: np.ndarray, actual_returns: np.ndarray) -> Dict[str, float]:
        """Calculate Information Coefficient metrics."""
        from scipy.stats import spearmanr, pearsonr
        
        try:
            pearson_corr = pearsonr(predictions, actual_returns)[0]
            spearman_corr = spearmanr(predictions, actual_returns)[0]
        except:
            pearson_corr = 0.0
            spearman_corr = 0.0
        
        return {
            'ic': pearson_corr,
            'rank_ic': spearman_corr
        }

class MultiAssetPortfolio:
    """Multi-asset portfolio with equal-weight or score-weighted allocation."""
    
    def __init__(self, 
                 allocation_method: str = 'equal_weight',
                 volatility_target: Optional[float] = None,
                 rebalance_frequency: int = 1):
        """
        Initialize multi-asset portfolio.
        
        Args:
            allocation_method: 'equal_weight' or 'score_weighted'
            volatility_target: Target annual volatility (e.g., 0.10 for 10%)
            rebalance_frequency: Rebalancing frequency in periods
        """
        self.allocation_method = allocation_method
        self.volatility_target = volatility_target
        self.rebalance_frequency = rebalance_frequency
    
    def calculate_weights(self, signals: Dict[str, np.ndarray], 
                        volatilities: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Calculate portfolio weights for each asset.
        
        Args:
            signals: Dictionary mapping ticker to signals
            volatilities: Optional dictionary mapping ticker to volatilities
            
        Returns:
            Dictionary mapping ticker to weights
        """
        tickers = list(signals.keys())
        n_periods = len(list(signals.values())[0])
        
        weights = {}
        
        for t in range(n_periods):
            if t % self.rebalance_frequency != 0:
                # Use previous weights
                if t > 0:
                    for ticker in tickers:
                        weights[ticker][t] = weights[ticker][t-1]
                continue
            
            # Calculate weights for this period
            period_weights = {}
            
            if self.allocation_method == 'equal_weight':
                # Equal weight across all assets
                n_assets = len(tickers)
                for ticker in tickers:
                    period_weights[ticker] = 1.0 / n_assets
            
            elif self.allocation_method == 'score_weighted':
                # Weight by signal strength
                period_signals = {ticker: signals[ticker][t] for ticker in tickers}
                signal_sum = sum(abs(sig) for sig in period_signals.values())
                
                if signal_sum > 0:
                    for ticker in tickers:
                        period_weights[ticker] = abs(period_signals[ticker]) / signal_sum
                else:
                    # Equal weight if no signals
                    n_assets = len(tickers)
                    for ticker in tickers:
                        period_weights[ticker] = 1.0 / n_assets
            
            # Apply volatility targeting if specified
            if self.volatility_target is not None and volatilities is not None:
                # Scale weights to achieve target volatility
                portfolio_vol = sum(
                    period_weights[ticker] * volatilities[ticker][t] 
                    for ticker in tickers
                )
                
                if portfolio_vol > 0:
                    vol_scalar = self.volatility_target / portfolio_vol
                    for ticker in tickers:
                        period_weights[ticker] *= vol_scalar
            
            # Store weights
            for ticker in tickers:
                if ticker not in weights:
                    weights[ticker] = np.zeros(n_periods)
                weights[ticker][t] = period_weights[ticker]
        
        return weights
    
    def calculate_portfolio_returns(self, 
                                  weights: Dict[str, np.ndarray],
                                  returns: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate portfolio returns.
        
        Args:
            weights: Dictionary mapping ticker to weights
            returns: Dictionary mapping ticker to returns
            
        Returns:
            Portfolio returns
        """
        tickers = list(weights.keys())
        n_periods = len(list(weights.values())[0])
        
        portfolio_returns = np.zeros(n_periods)
        
        for t in range(n_periods):
            period_return = 0.0
            for ticker in tickers:
                period_return += weights[ticker][t] * returns[ticker][t]
            portfolio_returns[t] = period_return
        
        return portfolio_returns

def backtest_strategy(signals: np.ndarray, 
                     actual_returns: np.ndarray,
                     strategy_type: str = 'contrarian',
                     threshold: float = 0.001,
                     cost_half: float = 0.001,
                     **strategy_params) -> Dict[str, Any]:
    """
    Backtest a trading strategy.
    
    Args:
        signals: Model predictions
        actual_returns: Actual returns
        strategy_type: Type of strategy ('contrarian', 'momentum', 'base')
        threshold: Signal threshold
        cost_half: Half-round-trip cost
        **strategy_params: Additional strategy parameters
        
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"Backtesting {strategy_type} strategy...")
    
    # Create strategy
    if strategy_type == 'contrarian':
        strategy = ContrarianStrategy(threshold=threshold, cost_half=cost_half, **strategy_params)
    else:
        strategy = TradingStrategy(threshold=threshold, cost_half=cost_half, **strategy_params)
    
    # Execute strategy
    results = strategy.execute_strategy(signals, actual_returns)
    
    # Calculate metrics
    metrics_calc = PortfolioMetrics()
    performance_metrics = metrics_calc.calculate_metrics(results['returns'])
    performance_metrics['turnover'] = metrics_calc.calculate_turnover(results['positions'])
    performance_metrics['ic'] = metrics_calc.calculate_information_coefficient(signals, actual_returns)
    
    # Combine results
    backtest_results = {
        'strategy_type': strategy_type,
        'parameters': {
            'threshold': threshold,
            'cost_half': cost_half,
            **strategy_params
        },
        'results': results,
        'metrics': performance_metrics
    }
    
    logger.info(f"Backtest completed. Sharpe: {performance_metrics['sharpe_ratio']:.3f}")
    
    return backtest_results

def compare_strategies(signals: np.ndarray, 
                      actual_returns: np.ndarray,
                      thresholds: List[float] = [0.0, 0.001, 0.002],
                      costs: List[float] = [0.0005, 0.001, 0.002],
                      strategy_types: List[str] = ['contrarian', 'momentum']) -> pd.DataFrame:
    """
    Compare multiple strategies across different parameters.
    
    Args:
        signals: Model predictions
        actual_returns: Actual returns
        thresholds: List of thresholds to test
        costs: List of costs to test
        strategy_types: List of strategy types to test
        
    Returns:
        DataFrame with comparison results
    """
    logger.info("Comparing strategies across parameters...")
    
    results = []
    
    for strategy_type in strategy_types:
        for threshold in thresholds:
            for cost in costs:
                try:
                    backtest = backtest_strategy(
                        signals, actual_returns,
                        strategy_type=strategy_type,
                        threshold=threshold,
                        cost_half=cost
                    )
                    
                    result_row = {
                        'strategy_type': strategy_type,
                        'threshold': threshold,
                        'cost_half': cost,
                        **backtest['metrics']
                    }
                    
                    results.append(result_row)
                    
                except Exception as e:
                    logger.warning(f"Error testing {strategy_type} with threshold={threshold}, cost={cost}: {e}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test trading strategy
    np.random.seed(42)
    
    # Create sample data
    n_periods = 252  # 1 year of daily data
    signals = np.random.randn(n_periods) * 0.02  # 2% daily signal volatility
    actual_returns = np.random.randn(n_periods) * 0.03  # 3% daily return volatility
    
    # Test contrarian strategy
    backtest = backtest_strategy(
        signals, actual_returns,
        strategy_type='contrarian',
        threshold=0.001,
        cost_half=0.001
    )
    
    print("Contrarian Strategy Results:")
    for metric, value in backtest['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Compare strategies
    comparison = compare_strategies(
        signals, actual_returns,
        thresholds=[0.0, 0.001],
        costs=[0.001],
        strategy_types=['contrarian', 'momentum']
    )
    
    print("\nStrategy Comparison:")
    print(comparison[['strategy_type', 'threshold', 'sharpe_ratio', 'max_drawdown', 'hit_rate']])

