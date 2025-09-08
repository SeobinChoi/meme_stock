#!/usr/bin/env python3
"""
Technical analysis features for meme stock price prediction.

Implements technical indicators as described in the manual for contrarian effect prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class TechnicalFeaturesEngine:
    """Engine for calculating technical analysis features."""
    
    def __init__(self,
                 sma_windows: List[int] = [10, 20, 50],
                 ema_windows: List[int] = [12, 26],
                 rsi_window: int = 14,
                 bb_window: int = 20,
                 bb_std: float = 2.0,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 volatility_windows: List[int] = [5, 10, 20]):
        """
        Initialize technical features engine.
        
        Args:
            sma_windows: Windows for Simple Moving Averages
            ema_windows: Windows for Exponential Moving Averages
            rsi_window: Window for RSI calculation
            bb_window: Window for Bollinger Bands
            bb_std: Standard deviation multiplier for Bollinger Bands
            macd_fast: Fast EMA for MACD
            macd_slow: Slow EMA for MACD
            macd_signal: Signal line EMA for MACD
            volatility_windows: Windows for volatility calculations
        """
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.volatility_windows = volatility_windows
    
    def calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=window, min_periods=1).mean()
    
    def calculate_ema(self, series: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """Calculate Exponential Moving Average."""
        if alpha is None:
            alpha = 2.0 / (window + 1)
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, window: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series (typically close prices)
            window: RSI window (default: self.rsi_window)
            
        Returns:
            RSI values between 0 and 100
        """
        if window is None:
            window = self.rsi_window
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI when no data
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = None, std_mult: float = None) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Moving average window
            std_mult: Standard deviation multiplier
            
        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        if window is None:
            window = self.bb_window
        if std_mult is None:
            std_mult = self.bb_std
        
        middle = self.calculate_sma(prices, window)
        std = prices.rolling(window=window, min_periods=1).std()
        
        upper = middle + (std * std_mult)
        lower = middle - (std * std_mult)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_macd(self, prices: pd.Series, 
                      fast: int = None, slow: int = None, signal: int = None) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal line EMA window
            
        Returns:
            Dictionary with 'macd', 'signal', 'histogram'
        """
        if fast is None:
            fast = self.macd_fast
        if slow is None:
            slow = self.macd_slow
        if signal is None:
            signal = self.macd_signal
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_returns(self, prices: pd.Series, periods: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, pd.Series]:
        """
        Calculate returns for different periods.
        
        Args:
            prices: Price series
            periods: List of periods for return calculation
            
        Returns:
            Dictionary with return series for each period
        """
        returns = {}
        
        for period in periods:
            returns[f'return_{period}d'] = prices.pct_change(period)
        
        return returns
    
    def calculate_cumulative_returns(self, prices: pd.Series, periods: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, pd.Series]:
        """
        Calculate cumulative returns for different periods.
        
        Args:
            prices: Price series
            periods: List of periods for cumulative return calculation
            
        Returns:
            Dictionary with cumulative return series for each period
        """
        returns = {}
        
        for period in periods:
            returns[f'cumret_{period}d'] = (1 + prices.pct_change()).rolling(window=period).apply(lambda x: x.prod() - 1, raw=False)
        
        return returns
    
    def calculate_volatility(self, returns: pd.Series, windows: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Return series
            windows: List of windows for volatility calculation
            
        Returns:
            Dictionary with volatility series for each window
        """
        if windows is None:
            windows = self.volatility_windows
        
        volatility = {}
        
        for window in windows:
            volatility[f'volatility_{window}d'] = returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
        
        return volatility
    
    def calculate_price_ratios(self, prices: pd.Series, sma_windows: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate price ratios relative to moving averages.
        
        Args:
            prices: Price series
            sma_windows: List of SMA windows
            
        Returns:
            Dictionary with price ratio series for each window
        """
        if sma_windows is None:
            sma_windows = self.sma_windows
        
        ratios = {}
        
        for window in sma_windows:
            sma = self.calculate_sma(prices, window)
            ratios[f'price_ratio_sma{window}'] = prices / sma
        
        return ratios
    
    def calculate_volume_features(self, volume: pd.Series, prices: pd.Series, windows: List[int] = [10, 20]) -> Dict[str, pd.Series]:
        """
        Calculate volume-based features.
        
        Args:
            volume: Volume series
            prices: Price series
            windows: List of windows for volume calculations
            
        Returns:
            Dictionary with volume feature series
        """
        features = {}
        
        # Volume moving averages
        for window in windows:
            features[f'volume_sma{window}'] = self.calculate_sma(volume, window)
            features[f'volume_ratio{window}'] = volume / features[f'volume_sma{window}'].replace(0, np.nan)
            features[f'volume_ratio{window}'] = features[f'volume_ratio{window}'].fillna(1)
        
        # Volume spike indicator
        volume_ma = self.calculate_sma(volume, 20)
        features['volume_spike'] = (volume > volume_ma * 2).astype(int)
        
        # Price-volume correlation
        features['price_volume_corr'] = prices.rolling(20).corr(volume)
        
        return features
    
    def calculate_calendar_features(self, dates: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate calendar-based features.
        
        Args:
            dates: Date series
            
        Returns:
            Dictionary with calendar feature series
        """
        features = {}
        
        dt = pd.to_datetime(dates)
        
        # Day of week (0=Monday, 6=Sunday)
        features['day_of_week'] = dt.dt.dayofweek
        
        # Month
        features['month'] = dt.dt.month
        
        # Quarter
        features['quarter'] = dt.dt.quarter
        
        # Is Monday (often has different behavior)
        features['is_monday'] = (dt.dt.dayofweek == 0).astype(int)
        
        # Is Friday (often has different behavior)
        features['is_friday'] = (dt.dt.dayofweek == 4).astype(int)
        
        # Is weekend (for non-trading days)
        features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        
        # Days since start of month
        features['days_since_month_start'] = dt.dt.day
        
        # Days until end of month
        features['days_until_month_end'] = dt.dt.days_in_month - dt.dt.day
        
        return features
    
    def calculate_trend_features(self, prices: pd.Series, windows: List[int] = [5, 10, 20]) -> Dict[str, pd.Series]:
        """
        Calculate trend-based features.
        
        Args:
            prices: Price series
            windows: List of windows for trend calculations
            
        Returns:
            Dictionary with trend feature series
        """
        features = {}
        
        for window in windows:
            # Linear regression slope
            features[f'trend_slope_{window}'] = prices.rolling(window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0
            )
            
            # Trend strength (R-squared)
            features[f'trend_strength_{window}'] = prices.rolling(window).apply(
                lambda x: stats.linregress(range(len(x)), x)[2]**2 if len(x) > 1 else 0
            )
        
        return features
    
    def calculate_all_technical_features(self, df: pd.DataFrame,
                                        price_col: str = 'close',
                                        volume_col: str = 'volume',
                                        date_col: str = 'date',
                                        group_col: str = 'ticker') -> pd.DataFrame:
        """
        Calculate all technical features for a dataframe.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for close prices
            volume_col: Column name for volume
            date_col: Column name for dates
            group_col: Column to group by (e.g., ticker)
            
        Returns:
            DataFrame with all technical features
        """
        logger.info(f"Calculating technical features for {len(df)} records")
        
        result_df = df.copy()
        
        # Sort by group and date
        result_df = result_df.sort_values([group_col, date_col])
        
        # Group by ticker and calculate features
        features_list = []
        
        for ticker, group in result_df.groupby(group_col):
            ticker_features = group.copy()
            
            prices = group[price_col]
            volume = group[volume_col] if volume_col in group.columns else None
            dates = group[date_col]
            
            # Returns
            returns_dict = self.calculate_returns(prices)
            ticker_features.update(returns_dict)
            
            # Cumulative returns
            cumret_dict = self.calculate_cumulative_returns(prices)
            ticker_features.update(cumret_dict)
            
            # Volatility
            if 'return_1d' in ticker_features.columns:
                vol_dict = self.calculate_volatility(ticker_features['return_1d'])
                ticker_features.update(vol_dict)
            
            # SMAs
            for window in self.sma_windows:
                sma = self.calculate_sma(prices, window)
                ticker_features[f'sma_{window}'] = sma
            
            # EMAs
            for window in self.ema_windows:
                ema = self.calculate_ema(prices, window)
                ticker_features[f'ema_{window}'] = ema
            
            # RSI
            ticker_features['rsi'] = self.calculate_rsi(prices)
            
            # Bollinger Bands
            bb_dict = self.calculate_bollinger_bands(prices)
            ticker_features.update({f'bb_{k}': v for k, v in bb_dict.items()})
            ticker_features['bb_width'] = (bb_dict['upper'] - bb_dict['lower']) / bb_dict['middle']
            ticker_features['bb_position'] = (prices - bb_dict['lower']) / (bb_dict['upper'] - bb_dict['lower'])
            
            # MACD
            macd_dict = self.calculate_macd(prices)
            ticker_features.update(macd_dict)
            
            # Price ratios
            price_ratio_dict = self.calculate_price_ratios(prices)
            ticker_features.update(price_ratio_dict)
            
            # Volume features
            if volume is not None:
                volume_dict = self.calculate_volume_features(volume, prices)
                ticker_features.update(volume_dict)
            
            # Calendar features
            calendar_dict = self.calculate_calendar_features(dates)
            ticker_features.update(calendar_dict)
            
            # Trend features
            trend_dict = self.calculate_trend_features(prices)
            ticker_features.update(trend_dict)
            
            features_list.append(ticker_features)
        
        # Combine all ticker features
        result_df = pd.concat(features_list, ignore_index=True)
        
        logger.info(f"Technical features calculated. Features added: {len(result_df.columns) - len(df.columns)}")
        
        return result_df

def calculate_technical_features(df: pd.DataFrame,
                                price_col: str = 'close',
                                volume_col: str = 'volume',
                                date_col: str = 'date',
                                group_col: str = 'ticker',
                                **kwargs) -> pd.DataFrame:
    """
    Calculate technical features for a dataframe.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for close prices
        volume_col: Column name for volume
        date_col: Column name for dates
        group_col: Column to group by
        **kwargs: Additional arguments for TechnicalFeaturesEngine
        
    Returns:
        DataFrame with additional technical features
    """
    engine = TechnicalFeaturesEngine(**kwargs)
    return engine.calculate_all_technical_features(df, price_col, volume_col, date_col, group_col)

if __name__ == "__main__":
    # Test technical features calculation
    np.random.seed(42)
    
    # Create sample data
    dates = pd.date_range('2021-01-01', periods=100, freq='D')
    tickers = ['GME', 'AMC', 'BB']
    
    data = []
    for ticker in tickers:
        # Simulate price data with trend and volatility
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% daily return, 2% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        volumes = np.random.lognormal(10, 1, 100)
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'ticker': ticker,
                'close': prices[i],
                'volume': volumes[i]
            })
    
    df = pd.DataFrame(data)
    
    # Calculate technical features
    engine = TechnicalFeaturesEngine()
    result_df = engine.calculate_all_technical_features(df)
    
    print("Sample technical features:")
    print(result_df[['date', 'ticker', 'close', 'rsi', 'bb_position', 'macd']].head(10))

