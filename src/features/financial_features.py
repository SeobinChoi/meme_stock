"""
Financial Market Feature Engineering (35 features per stock)
Creates financial features from stock price data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Technical analysis imports
try:
    import talib  # type: ignore
except Exception:  # [FIX] fallback to 'ta' if TA-Lib not available
    talib = None
    from ta.momentum import RSIIndicator  # type: ignore
    from ta.trend import MACD  # type: ignore
    from ta.volatility import BollingerBands  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialFeatureEngineer:
    """
    Financial market feature engineering (35 features per stock)
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_features(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """
        Generate financial features for all stocks (35 features per stock)
        """
        logger.info("📈 Generating financial market features...")
        
        financial_features = {}
        unified_data = data['unified']
        stock_data = data['stocks']
        
        for symbol in ['GME', 'AMC', 'BB']:
            logger.info(f"  Processing {symbol} financial features...")
            
            # Use stock data if available, otherwise use unified data
            if symbol in stock_data and stock_data[symbol] is not None:
                stock_df = self._prepare_stock_data(stock_data[symbol])
            else:
                # Extract from unified data
                price_col = f'{symbol}_close'
                volume_col = f'{symbol}_volume'
                
                if price_col in unified_data.columns and volume_col in unified_data.columns:
                    # Create a stock-like DataFrame from unified data
                    stock_df = pd.DataFrame({
                        'Date': unified_data.index,
                        'Close': unified_data[price_col],
                        'Volume': unified_data[volume_col],
                        'Open': unified_data[price_col],  # Use close as proxy
                        'High': unified_data[price_col],  # Use close as proxy
                        'Low': unified_data[price_col]    # Use close as proxy
                    }).set_index('Date')
                else:
                    logger.warning(f"    ⚠️ No data for {symbol}")
                    continue
            
            # Generate features
            features_df = self._generate_stock_features(stock_df, symbol)
            financial_features[symbol] = features_df
            
            logger.info(f"    {symbol}: {features_df.shape[1]} features generated")
        
        logger.info(f"✅ Financial features generated for {len(financial_features)} stocks")
        return financial_features
    
    def _prepare_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare stock data for feature engineering
        """
        stock_df = df.copy()
        
        # Check if Date is already the index
        if stock_df.index.name == 'Date' or isinstance(stock_df.index, pd.DatetimeIndex):
            # Data is already properly indexed
            pass
        else:
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in stock_df.columns:
                    raise ValueError(f"Required column {col} not found in stock data")
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
            
            # Sort by date and set as index
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            stock_df = stock_df.sort_values('Date').reset_index(drop=True)
            stock_df = stock_df.set_index('Date')
        
        return stock_df
    
    def _generate_stock_features(self, stock_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate all financial features for a single stock (35 features)
        """
        features_df = pd.DataFrame(index=stock_df.index)
        
        # 1. Price-based features (15 features)
        features_df = self._add_price_based_features(features_df, stock_df)
        
        # 2. Volume-based features (10 features)
        features_df = self._add_volume_based_features(features_df, stock_df)
        
        # 3. Market microstructure features (10 features)
        features_df = self._add_microstructure_features(features_df, stock_df)
        # [FIX] Add TA indicators (RSI/MACD/BBANDS) if available
        try:
            features_df = self._add_ta_indicators(features_df, stock_df)
        except Exception:
            pass
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        return features_df
    
    def _add_price_based_features(self, features_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features (15 features)
        """
        logger.info("    Generating price-based features...")
        
        # 1. Returns (1-day, 3-day, 7-day, 14-day)
        features_df['returns_1d'] = stock_df['Close'].pct_change(1)
        features_df['returns_3d'] = stock_df['Close'].pct_change(3)
        features_df['returns_7d'] = stock_df['Close'].pct_change(7)
        features_df['returns_14d'] = stock_df['Close'].pct_change(14)
        
        # 2. Volatility (rolling standard deviation)
        features_df['volatility_5d'] = features_df['returns_1d'].rolling(5).std()
        features_df['volatility_10d'] = features_df['returns_1d'].rolling(10).std()
        features_df['volatility_20d'] = features_df['returns_1d'].rolling(20).std()
        
        # 3. Price momentum
        features_df['price_momentum_5d'] = stock_df['Close'] / stock_df['Close'].shift(5) - 1
        features_df['price_momentum_10d'] = stock_df['Close'] / stock_df['Close'].shift(10) - 1
        features_df['price_momentum_20d'] = stock_df['Close'] / stock_df['Close'].shift(20) - 1
        
        # 4. Price acceleration (change in momentum)
        features_df['price_acceleration_5d'] = features_df['price_momentum_5d'].diff()
        features_df['price_acceleration_10d'] = features_df['price_momentum_10d'].diff()
        
        # 5. Relative performance (vs. moving averages)
        features_df['relative_to_ma5'] = stock_df['Close'] / stock_df['Close'].rolling(5).mean() - 1
        features_df['relative_to_ma10'] = stock_df['Close'] / stock_df['Close'].rolling(10).mean() - 1
        features_df['relative_to_ma20'] = stock_df['Close'] / stock_df['Close'].rolling(20).mean() - 1
        
        # 6. Distance from highs/lows
        features_df['distance_from_high_20d'] = (stock_df['High'].rolling(20).max() - stock_df['Close']) / stock_df['Close']
        features_df['distance_from_low_20d'] = (stock_df['Close'] - stock_df['Low'].rolling(20).min()) / stock_df['Close']
        
        return features_df
    
    def _add_volume_based_features(self, features_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features (10 features)
        """
        logger.info("    Generating volume-based features...")
        
        # 1. Volume patterns
        features_df['volume_ma5'] = stock_df['Volume'].rolling(5).mean()
        features_df['volume_ma10'] = stock_df['Volume'].rolling(10).mean()
        features_df['volume_ma20'] = stock_df['Volume'].rolling(20).mean()
        
        # 2. Volume ratios
        features_df['volume_ratio_5d'] = stock_df['Volume'] / features_df['volume_ma5']
        features_df['volume_ratio_10d'] = stock_df['Volume'] / features_df['volume_ma10']
        
        # 3. Volume-weighted average price (VWAP) deviations
        vwap_5d = (stock_df['Close'] * stock_df['Volume']).rolling(5).sum() / stock_df['Volume'].rolling(5).sum()
        features_df['vwap_deviation_5d'] = (stock_df['Close'] - vwap_5d) / vwap_5d
        
        # 4. Unusual volume activity
        volume_std = stock_df['Volume'].rolling(20).std()
        volume_mean = stock_df['Volume'].rolling(20).mean()
        features_df['volume_zscore'] = (stock_df['Volume'] - volume_mean) / (volume_std + 1)
        
        # 5. Volume-price relationship
        features_df['volume_price_trend'] = (
            (stock_df['Close'] - stock_df['Close'].shift(1)) * stock_df['Volume']
        ).rolling(5).sum()
        
        # 6. Volume momentum
        features_df['volume_momentum_5d'] = stock_df['Volume'] / stock_df['Volume'].shift(5) - 1
        features_df['volume_momentum_10d'] = stock_df['Volume'] / stock_df['Volume'].shift(10) - 1
        
        # 7. Volume volatility
        features_df['volume_volatility_10d'] = stock_df['Volume'].rolling(10).std() / stock_df['Volume'].rolling(10).mean()
        
        return features_df
    
    def _add_microstructure_features(self, features_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features (10 features)
        """
        logger.info("    Generating microstructure features...")
        
        # 1. Price range features
        features_df['daily_range'] = (stock_df['High'] - stock_df['Low']) / stock_df['Close']
        features_df['daily_range_ma5'] = features_df['daily_range'].rolling(5).mean()
        
        # 2. Gap analysis
        features_df['gap_up'] = (stock_df['Open'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1)
        features_df['gap_down'] = (stock_df['Close'].shift(1) - stock_df['Open']) / stock_df['Close'].shift(1)
        
        # 3. Intraday momentum
        features_df['intraday_momentum'] = (stock_df['Close'] - stock_df['Open']) / stock_df['Open']
        features_df['intraday_momentum_ma5'] = features_df['intraday_momentum'].rolling(5).mean()
        
        # 4. Price efficiency (random walk test)
        features_df['price_efficiency'] = self._calculate_price_efficiency(stock_df['Close'])
        
        # 5. Volatility clustering
        features_df['volatility_clustering'] = features_df['volatility_5d'].rolling(10).std()
        
        # 6. Jump detection
        features_df['price_jumps'] = self._detect_price_jumps(stock_df['Close'])
        
        # 7. Market regime indicators
        features_df['trend_strength'] = self._calculate_trend_strength(stock_df['Close'])
        
        # 8. Mean reversion indicators
        features_df['mean_reversion_signal'] = self._calculate_mean_reversion(stock_df['Close'])
        
        # 9. Price momentum vs volume momentum
        features_df['momentum_divergence'] = (
            features_df['price_momentum_5d'] - features_df['volume_momentum_5d']
        )
        
        # 10. Volatility of volatility
        features_df['vol_of_vol'] = features_df['volatility_5d'].rolling(10).std()
        
        return features_df

    # [FIX] Additional TA indicators using talib/ta where available
    def _add_ta_indicators(self, features_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
        close = stock_df['Close']
        if talib is not None:
            try:
                features_df['RSI_14'] = talib.RSI(close, timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                features_df['MACD'] = macd
                features_df['MACD_signal'] = macd_signal
                features_df['MACD_hist'] = macd_hist
                upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                features_df['BB_upper'] = upper
                features_df['BB_middle'] = middle
                features_df['BB_lower'] = lower
            except Exception:
                pass
        else:
            try:
                features_df['RSI_14'] = RSIIndicator(close=close, window=14).rsi()
                macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
                features_df['MACD'] = macd_ind.macd()
                features_df['MACD_signal'] = macd_ind.macd_signal()
                features_df['MACD_hist'] = macd_ind.macd_diff()
                bb = BollingerBands(close=close, window=20, window_dev=2)
                features_df['BB_upper'] = bb.bollinger_hband()
                features_df['BB_middle'] = bb.bollinger_mavg()
                features_df['BB_lower'] = bb.bollinger_lband()
            except Exception:
                pass
        return features_df
    
    def _calculate_price_efficiency(self, prices: pd.Series) -> pd.Series:
        """
        Calculate price efficiency (random walk test)
        """
        # Simplified efficiency measure based on autocorrelation
        efficiency = prices.rolling(20).apply(
            lambda x: 1 - abs(x.autocorr()) if len(x) > 1 else 0
        )
        return efficiency
    
    def _detect_price_jumps(self, prices: pd.Series) -> pd.Series:
        """
        Detect unusual price movements (jumps)
        """
        returns = prices.pct_change()
        jump_threshold = returns.rolling(20).std() * 3
        jumps = (abs(returns) > jump_threshold).astype(int)
        return jumps
    
    def _calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """
        Calculate trend strength indicator
        """
        # Linear regression slope over rolling window
        def linear_trend(x):
            if len(x) < 5:
                return 0
            y = np.arange(len(x))
            slope = np.polyfit(y, x, 1)[0]
            return slope / x.mean() if x.mean() != 0 else 0
        
        trend_strength = prices.rolling(10).apply(linear_trend)
        return trend_strength
    
    def _calculate_mean_reversion(self, prices: pd.Series) -> pd.Series:
        """
        Calculate mean reversion signal
        """
        # Distance from moving average
        ma_20 = prices.rolling(20).mean()
        mean_reversion = (prices - ma_20) / ma_20
        return mean_reversion


def main():
    """
    Test financial feature engineering
    """
    logger.info("🚀 Testing Financial Feature Engineering...")
    
    # Load sample data
    from pathlib import Path
    data_dir = Path("data")
    
    # Load stock data
    stock_symbols = ["GME", "AMC", "BB"]
    stock_data = {}
    
    for symbol in stock_symbols:
        stock_file = data_dir / "raw" / f"{symbol}_enhanced_stock_data.csv"
        if stock_file.exists():
            stock_df = pd.read_csv(stock_file)
            stock_data[symbol] = stock_df
            logger.info(f"  Loaded {symbol} data: {len(stock_df)} records")
        else:
            logger.warning(f"  {symbol} enhanced data not found")
            # Try original data
            original_file = data_dir / "raw" / f"{symbol}_stock_data.csv"
            if original_file.exists():
                stock_df = pd.read_csv(original_file)
                stock_data[symbol] = stock_df
                logger.info(f"  Loaded {symbol} original data: {len(stock_df)} records")
    
    if stock_data:
        # Initialize feature engineer
        engineer = FinancialFeatureEngineer()
        
        # Generate features
        features = engineer.generate_features(stock_data)
        
        logger.info(f"✅ Financial features generated for {len(features)} stocks")
        for symbol, feature_df in features.items():
            logger.info(f"  {symbol}: {feature_df.shape[1]} features, shape: {feature_df.shape}")
        
        return features
    else:
        logger.error("❌ No stock data found for testing")
        return None


if __name__ == "__main__":
    main() 