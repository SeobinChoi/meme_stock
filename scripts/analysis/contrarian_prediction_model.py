#!/usr/bin/env python3
"""
Contrarian-Effect Based Prediction Model
Uses discovered negative correlation between reddit_surprise and returns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data efficiently"""
    print("Loading data...")
    
    cols = ['date', 'ticker', 'log_mentions', 'returns_1d', 'returns_5d', 
            'reddit_surprise', 'reddit_momentum_3', 'reddit_momentum_7',
            'rsi_14', 'volume_ratio', 'vol_5d', 'market_sentiment']
    
    dtypes = {
        'ticker': 'category',
        'log_mentions': 'float32',
        'returns_1d': 'float32', 
        'returns_5d': 'float32',
        'reddit_surprise': 'float32',
        'reddit_momentum_3': 'float32',
        'reddit_momentum_7': 'float32',
        'rsi_14': 'float32',
        'volume_ratio': 'float32',
        'vol_5d': 'float32',
        'market_sentiment': 'float32'
    }
    
    files = [
        'data/colab_datasets/tabular_train_20250814_031335.csv',
        'data/colab_datasets/tabular_val_20250814_031335.csv', 
        'data/colab_datasets/tabular_test_20250814_031335.csv'
    ]
    
    dfs = []
    for file in files:
        chunk = pd.read_csv(file, usecols=cols, dtype=dtypes)
        dfs.append(chunk)
    
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} samples")
    return df

def create_contrarian_features(df):
    """Create features based on contrarian effect"""
    print("Creating contrarian features...")
    
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Contrarian signals (higher surprise = sell signal)
    df['contrarian_signal'] = -df['reddit_surprise']  # Flip the signal
    df['contrarian_momentum'] = -df['reddit_momentum_3']  # Flip momentum
    
    # Interaction features
    df['surprise_rsi_interaction'] = df['reddit_surprise'] * df['rsi_14']
    df['surprise_vol_interaction'] = df['reddit_surprise'] * df['vol_5d']
    
    # Moving averages of contrarian signals
    for window in [3, 7, 14]:
        df[f'contrarian_ma_{window}'] = df.groupby('ticker')['contrarian_signal'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # Regime indicators (high/low surprise periods)
    df['high_surprise_regime'] = (df['reddit_surprise'] > df['reddit_surprise'].quantile(0.8)).astype(int)
    df['low_surprise_regime'] = (df['reddit_surprise'] < df['reddit_surprise'].quantile(0.2)).astype(int)
    
    return df

def create_prediction_targets(df):
    """Create prediction targets"""
    print("Creating prediction targets...")
    
    # Future returns (what we want to predict)
    df['target_1d'] = df.groupby('ticker')['returns_1d'].shift(-1)
    df['target_5d'] = df.groupby('ticker')['returns_5d'].shift(-1)
    
    # Binary classification targets (up/down)
    df['target_direction_1d'] = (df['target_1d'] > 0).astype(int)
    df['target_direction_5d'] = (df['target_5d'] > 0).astype(int)
    
    return df

def prepare_features(df):
    """Prepare feature matrix"""
    feature_cols = [
        'contrarian_signal',
        'contrarian_momentum', 
        'surprise_rsi_interaction',
        'surprise_vol_interaction',
        'contrarian_ma_3',
        'contrarian_ma_7',
        'contrarian_ma_14',
        'high_surprise_regime',
        'low_surprise_regime',
        'rsi_14',
        'volume_ratio',
        'vol_5d',
        'market_sentiment',
        'log_mentions'
    ]
    
    # Remove rows with NaN targets and features
    valid_data = df.dropna(subset=['target_1d', 'target_5d'] + feature_cols)
    
    X = valid_data[feature_cols]
    y_reg_1d = valid_data['target_1d']
    y_reg_5d = valid_data['target_5d']
    y_class_1d = valid_data['target_direction_1d']
    y_class_5d = valid_data['target_direction_5d']
    
    dates = valid_data['date']
    tickers = valid_data['ticker']
    
    return X, y_reg_1d, y_reg_5d, y_class_1d, y_class_5d, dates, tickers

def train_contrarian_models(X, y_reg_1d, y_reg_5d, dates, tickers):
    """Train models using time series split"""
    print("Training contrarian models...")
    
    # Time series split (respect temporal order)
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'lightgbm': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbose=-1)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        mse_scores_1d = []
        mae_scores_1d = []
        mse_scores_5d = []
        mae_scores_5d = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_1d, y_test_1d = y_reg_1d.iloc[train_idx], y_reg_1d.iloc[test_idx]
            y_train_5d, y_test_5d = y_reg_5d.iloc[train_idx], y_reg_5d.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train 1-day model
            model_1d = model.__class__(**model.get_params())
            model_1d.fit(X_train_scaled, y_train_1d)
            pred_1d = model_1d.predict(X_test_scaled)
            
            mse_scores_1d.append(mean_squared_error(y_test_1d, pred_1d))
            mae_scores_1d.append(mean_absolute_error(y_test_1d, pred_1d))
            
            # Train 5-day model
            model_5d = model.__class__(**model.get_params())
            model_5d.fit(X_train_scaled, y_train_5d)
            pred_5d = model_5d.predict(X_test_scaled)
            
            mse_scores_5d.append(mean_squared_error(y_test_5d, pred_5d))
            mae_scores_5d.append(mean_absolute_error(y_test_5d, pred_5d))
        
        results[model_name] = {
            'mse_1d': np.mean(mse_scores_1d),
            'mae_1d': np.mean(mae_scores_1d),
            'mse_5d': np.mean(mse_scores_5d),
            'mae_5d': np.mean(mae_scores_5d)
        }
        
        print(f"{model_name}: MAE_1d={np.mean(mae_scores_1d):.4f}, MAE_5d={np.mean(mae_scores_5d):.4f}")
    
    return results

def create_trading_strategy(df, X, y_reg_1d, dates, tickers):
    """Create and backtest trading strategy"""
    print("Creating trading strategy...")
    
    # Simple strategy: use contrarian signal directly
    df_strategy = df.copy()
    
    # Generate signals
    df_strategy['signal'] = 0
    df_strategy.loc[df_strategy['reddit_surprise'] > df_strategy['reddit_surprise'].quantile(0.8), 'signal'] = -1  # Sell
    df_strategy.loc[df_strategy['reddit_surprise'] < df_strategy['reddit_surprise'].quantile(0.2), 'signal'] = 1   # Buy
    
    # Calculate strategy returns
    df_strategy['strategy_return'] = df_strategy['signal'] * df_strategy['target_1d']
    
    # Focus on main meme stocks
    main_stocks = ['GME', 'AMC', 'BB']
    strategy_results = {}
    
    for stock in main_stocks:
        stock_data = df_strategy[df_strategy['ticker'] == stock].copy()
        
        if len(stock_data) > 100:
            # Calculate performance metrics
            total_return = stock_data['strategy_return'].sum()
            hit_rate = (stock_data['strategy_return'] > 0).mean()
            sharpe = stock_data['strategy_return'].mean() / stock_data['strategy_return'].std() * np.sqrt(252) if stock_data['strategy_return'].std() > 0 else 0
            
            # Benchmark (buy and hold)
            benchmark_return = stock_data['target_1d'].sum()
            
            strategy_results[stock] = {
                'total_return': total_return,
                'benchmark_return': benchmark_return,
                'excess_return': total_return - benchmark_return,
                'hit_rate': hit_rate,
                'sharpe_ratio': sharpe,
                'num_trades': (stock_data['signal'] != 0).sum()
            }
            
            print(f"{stock}: Strategy={total_return:.4f}, Benchmark={benchmark_return:.4f}, "
                  f"Excess={total_return-benchmark_return:.4f}")
    
    return strategy_results

def visualize_results(model_results, strategy_results):
    """Create visualization of results"""
    print("Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model performance comparison
    models = list(model_results.keys())
    mae_1d = [model_results[m]['mae_1d'] for m in models]
    mae_5d = [model_results[m]['mae_5d'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0,0].bar(x - width/2, mae_1d, width, label='1-day MAE', alpha=0.7)
    axes[0,0].bar(x + width/2, mae_5d, width, label='5-day MAE', alpha=0.7)
    axes[0,0].set_title('Model Performance (Lower is Better)')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(models)
    axes[0,0].legend()
    
    # Strategy returns
    stocks = list(strategy_results.keys())
    strategy_rets = [strategy_results[s]['total_return'] for s in stocks]
    benchmark_rets = [strategy_results[s]['benchmark_return'] for s in stocks]
    
    x = np.arange(len(stocks))
    axes[0,1].bar(x - width/2, strategy_rets, width, label='Strategy', alpha=0.7)
    axes[0,1].bar(x + width/2, benchmark_rets, width, label='Benchmark', alpha=0.7)
    axes[0,1].set_title('Strategy vs Benchmark Returns')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(stocks)
    axes[0,1].legend()
    
    # Hit rates
    hit_rates = [strategy_results[s]['hit_rate'] for s in stocks]
    axes[1,0].bar(stocks, hit_rates, alpha=0.7)
    axes[1,0].set_title('Strategy Hit Rates')
    axes[1,0].set_ylabel('Hit Rate')
    axes[1,0].axhline(y=0.5, color='red', linestyle='--', label='Random')
    axes[1,0].legend()
    
    # Sharpe ratios
    sharpe_ratios = [strategy_results[s]['sharpe_ratio'] for s in stocks]
    axes[1,1].bar(stocks, sharpe_ratios, alpha=0.7)
    axes[1,1].set_title('Strategy Sharpe Ratios')
    axes[1,1].set_ylabel('Sharpe Ratio')
    axes[1,1].axhline(y=0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('contrarian_model_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main execution"""
    print("=== CONTRARIAN EFFECT PREDICTION MODEL ===")
    
    # 1. Load and prepare data
    df = load_data()
    df = create_contrarian_features(df)
    df = create_prediction_targets(df)
    
    # 2. Prepare features
    X, y_reg_1d, y_reg_5d, y_class_1d, y_class_5d, dates, tickers = prepare_features(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # 3. Train models
    model_results = train_contrarian_models(X, y_reg_1d, y_reg_5d, dates, tickers)
    
    # 4. Create trading strategy
    strategy_results = create_trading_strategy(df, X, y_reg_1d, dates, tickers)
    
    # 5. Visualize results
    visualize_results(model_results, strategy_results)
    
    # 6. Summary
    print("\n=== SUMMARY ===")
    print("Model Performance (MAE):")
    for model, metrics in model_results.items():
        print(f"  {model}: 1d={metrics['mae_1d']:.4f}, 5d={metrics['mae_5d']:.4f}")
    
    print("\nStrategy Performance:")
    for stock, metrics in strategy_results.items():
        print(f"  {stock}: Return={metrics['total_return']:.4f}, "
              f"Excess={metrics['excess_return']:.4f}, "
              f"Sharpe={metrics['sharpe_ratio']:.2f}")
    
    # Paper readiness assessment
    best_model_mae = min([m['mae_1d'] for m in model_results.values()])
    avg_excess_return = np.mean([s['excess_return'] for s in strategy_results.values()])
    avg_sharpe = np.mean([s['sharpe_ratio'] for s in strategy_results.values()])
    
    print(f"\n=== PAPER READINESS ASSESSMENT ===")
    print(f"Best model MAE: {best_model_mae:.4f}")
    print(f"Average excess return: {avg_excess_return:.4f}")
    print(f"Average Sharpe ratio: {avg_sharpe:.2f}")
    
    if avg_excess_return > 0.01 and avg_sharpe > 0.5:
        print("🎉 READY FOR PAPER: Strong predictive power and economic significance!")
    elif avg_excess_return > 0.005:
        print("✅ PROMISING: Good economic significance, needs refinement")
    else:
        print("⚠️ NEEDS WORK: Weak economic significance")

if __name__ == "__main__":
    main()
