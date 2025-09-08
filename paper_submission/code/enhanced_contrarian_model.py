#!/usr/bin/env python3
"""
Enhanced Contrarian Model with Sharpe Optimization & Robustness Testing
Paper-grade implementation with risk management and transaction costs
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_data():
    """Load all available data"""
    print("Loading enhanced dataset...")
    
    cols = ['date', 'ticker', 'log_mentions', 'returns_1d', 'returns_5d', 
            'reddit_surprise', 'reddit_momentum_3', 'reddit_momentum_7',
            'rsi_14', 'volume_ratio', 'vol_5d', 'market_sentiment',
            'reddit_ema_3', 'price_ratio_sma20']
    
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
        'market_sentiment': 'float32',
        'reddit_ema_3': 'float32',
        'price_ratio_sma20': 'float32'
    }
    
    files = [
        'data/colab_datasets/tabular_train_20250814_031335.csv',
        'data/colab_datasets/tabular_val_20250814_031335.csv', 
        'data/colab_datasets/tabular_test_20250814_031335.csv'
    ]
    
    dfs = []
    for file in files:
        try:
            chunk = pd.read_csv(file, usecols=cols, dtype=dtypes)
            dfs.append(chunk)
        except:
            # Fallback if some columns missing
            available_cols = pd.read_csv(file, nrows=1).columns
            use_cols = [c for c in cols if c in available_cols]
            chunk = pd.read_csv(file, usecols=use_cols)
            dfs.append(chunk)
    
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} samples across {df['ticker'].nunique()} tickers")
    print(f"Tickers: {sorted(df['ticker'].unique())}")
    return df

def create_enhanced_features(df):
    """Create enhanced contrarian features with risk management"""
    print("Creating enhanced features...")
    
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Core contrarian signals
    df['contrarian_signal'] = -df['reddit_surprise']
    df['contrarian_momentum'] = -df['reddit_momentum_3']
    
    # Enhanced interaction features
    df['surprise_rsi_interaction'] = df['reddit_surprise'] * df['rsi_14']
    df['surprise_vol_interaction'] = df['reddit_surprise'] * df['vol_5d']
    
    # Moving averages with different windows
    for window in [3, 7, 14, 21]:
        df[f'contrarian_ma_{window}'] = df.groupby('ticker')['contrarian_signal'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # Volatility-adjusted signals (key for Sharpe improvement)
    df['vol_adj_signal'] = df['contrarian_signal'] / (df['vol_5d'] + 0.01)  # Avoid div by zero
    
    # Regime indicators (more granular)
    for q in [0.1, 0.2, 0.8, 0.9]:
        df[f'surprise_q{int(q*100)}'] = (df['reddit_surprise'] > df['reddit_surprise'].quantile(q)).astype(int)
    
    # Momentum consistency (trend following within contrarian)
    df['momentum_consistency'] = df.groupby('ticker')['contrarian_signal'].transform(
        lambda x: x.rolling(5, min_periods=1).apply(lambda y: 1 if (y > 0).all() or (y < 0).all() else 0)
    )
    
    # Risk-based position sizing features
    df['risk_score'] = df['vol_5d'] * abs(df['reddit_surprise'])  # Higher = riskier
    df['position_size'] = 1 / (1 + df['risk_score'])  # Smaller positions for risky situations
    
    return df

def create_enhanced_targets(df):
    """Create targets with forward-looking validation"""
    print("Creating enhanced targets...")
    
    # Multiple horizon targets (use available columns)
    available_returns = [c for c in df.columns if c.startswith('returns_') and c.endswith('d')]
    print(f"Available return columns: {available_returns}")
    
    for horizon in [1, 5]:  # Use only available horizons
        if f'returns_{horizon}d' in df.columns:
            df[f'target_{horizon}d'] = df.groupby('ticker')[f'returns_{horizon}d'].shift(-horizon)
            df[f'target_direction_{horizon}d'] = (df[f'target_{horizon}d'] > 0).astype(int)
    
    # Volatility-adjusted returns (for Sharpe improvement)
    df['target_vol_adj_1d'] = df['target_1d'] / (df['vol_5d'] + 0.01)
    
    return df

def enhanced_strategy_with_risk_mgmt(df, transaction_cost=0.001):
    """Enhanced strategy with position sizing and risk management"""
    print("Creating enhanced strategy with risk management...")
    
    df_strategy = df.copy()
    
    # Generate signals with multiple conditions
    df_strategy['signal_strength'] = 0.0
    
    # Primary signal: contrarian effect
    high_surprise = df_strategy['reddit_surprise'] > df_strategy['reddit_surprise'].quantile(0.8)
    low_surprise = df_strategy['reddit_surprise'] < df_strategy['reddit_surprise'].quantile(0.2)
    
    # Secondary filters
    not_too_volatile = df_strategy['vol_5d'] < df_strategy['vol_5d'].quantile(0.9)
    reasonable_volume = df_strategy['volume_ratio'] > 0.5
    
    # Combined signals with position sizing
    sell_condition = high_surprise & not_too_volatile & reasonable_volume
    buy_condition = low_surprise & not_too_volatile & reasonable_volume
    
    df_strategy.loc[sell_condition, 'signal_strength'] = -df_strategy.loc[sell_condition, 'position_size']
    df_strategy.loc[buy_condition, 'signal_strength'] = df_strategy.loc[buy_condition, 'position_size']
    
    # Calculate returns with transaction costs
    df_strategy['position_change'] = df_strategy.groupby('ticker')['signal_strength'].diff().fillna(0)
    df_strategy['transaction_cost'] = abs(df_strategy['position_change']) * transaction_cost
    
    df_strategy['gross_return'] = df_strategy['signal_strength'] * df_strategy['target_1d']
    df_strategy['net_return'] = df_strategy['gross_return'] - df_strategy['transaction_cost']
    
    return df_strategy

def robustness_test_by_periods(df):
    """Test strategy across different time periods"""
    print("Running robustness tests across time periods...")
    
    df['year'] = df['date'].dt.year
    years = sorted(df['year'].unique())
    
    # Test on different periods
    periods = [
        ('2021', [2021]),
        ('2022', [2022]), 
        ('2023', [2023]),
        ('2021-2022', [2021, 2022]),
        ('2022-2023', [2022, 2023]),
        ('All', years)
    ]
    
    period_results = {}
    
    for period_name, period_years in periods:
        period_data = df[df['year'].isin(period_years)].copy()
        
        if len(period_data) < 100:
            continue
            
        # Run enhanced strategy on this period
        strategy_data = enhanced_strategy_with_risk_mgmt(period_data)
        
        # Calculate metrics by ticker
        ticker_results = {}
        for ticker in strategy_data['ticker'].unique():
            ticker_data = strategy_data[strategy_data['ticker'] == ticker]
            
            if len(ticker_data) > 20:
                total_return = ticker_data['net_return'].sum()
                gross_return = ticker_data['gross_return'].sum()
                total_cost = ticker_data['transaction_cost'].sum()
                
                returns_series = ticker_data['net_return']
                sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
                
                hit_rate = (returns_series > 0).mean()
                max_drawdown = calculate_max_drawdown(returns_series.cumsum())
                
                ticker_results[ticker] = {
                    'total_return': total_return,
                    'gross_return': gross_return,
                    'transaction_costs': total_cost,
                    'sharpe_ratio': sharpe,
                    'hit_rate': hit_rate,
                    'max_drawdown': max_drawdown,
                    'num_trades': (ticker_data['signal_strength'] != 0).sum()
                }
        
        period_results[period_name] = ticker_results
        
        # Print summary for this period
        if ticker_results:
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in ticker_results.values()])
            avg_return = np.mean([r['total_return'] for r in ticker_results.values()])
            print(f"{period_name}: Avg Sharpe={avg_sharpe:.3f}, Avg Return={avg_return:.4f}")
    
    return period_results

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown"""
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def expanded_ticker_analysis(df):
    """Analyze all available tickers"""
    print("Running expanded ticker analysis...")
    
    strategy_data = enhanced_strategy_with_risk_mgmt(df)
    
    ticker_performance = {}
    
    for ticker in strategy_data['ticker'].unique():
        ticker_data = strategy_data[strategy_data['ticker'] == ticker]
        
        if len(ticker_data) > 50:  # Minimum data requirement
            # Calculate comprehensive metrics
            net_returns = ticker_data['net_return']
            gross_returns = ticker_data['gross_return']
            
            total_net = net_returns.sum()
            total_gross = gross_returns.sum()
            total_costs = ticker_data['transaction_cost'].sum()
            
            sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
            hit_rate = (net_returns > 0).mean()
            max_dd = calculate_max_drawdown(net_returns.cumsum())
            
            # Risk metrics
            volatility = net_returns.std() * np.sqrt(252)
            skewness = stats.skew(net_returns)
            kurtosis = stats.kurtosis(net_returns)
            
            # Benchmark comparison
            benchmark_return = ticker_data['target_1d'].sum()
            excess_return = total_net - benchmark_return
            
            ticker_performance[ticker] = {
                'total_net_return': total_net,
                'total_gross_return': total_gross,
                'transaction_costs': total_costs,
                'benchmark_return': benchmark_return,
                'excess_return': excess_return,
                'sharpe_ratio': sharpe,
                'hit_rate': hit_rate,
                'max_drawdown': max_dd,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'num_trades': (ticker_data['signal_strength'] != 0).sum(),
                'sample_size': len(ticker_data)
            }
    
    return ticker_performance

def create_comprehensive_visualization(ticker_performance, period_results):
    """Create comprehensive visualization"""
    print("Creating comprehensive visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. Sharpe ratios by ticker
    tickers = list(ticker_performance.keys())
    sharpes = [ticker_performance[t]['sharpe_ratio'] for t in tickers]
    
    axes[0,0].bar(tickers, sharpes, color=['green' if x > 0 else 'red' for x in sharpes])
    axes[0,0].set_title('Sharpe Ratios by Ticker')
    axes[0,0].set_ylabel('Sharpe Ratio')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Excess returns vs benchmark
    excess_returns = [ticker_performance[t]['excess_return'] for t in tickers]
    axes[0,1].bar(tickers, excess_returns, color=['green' if x > 0 else 'red' for x in excess_returns])
    axes[0,1].set_title('Excess Returns vs Benchmark')
    axes[0,1].set_ylabel('Excess Return')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Risk-Return scatter
    returns = [ticker_performance[t]['total_net_return'] for t in tickers]
    volatilities = [ticker_performance[t]['volatility'] for t in tickers]
    
    scatter = axes[0,2].scatter(volatilities, returns, alpha=0.7, s=60)
    axes[0,2].set_xlabel('Volatility (Annualized)')
    axes[0,2].set_ylabel('Total Return')
    axes[0,2].set_title('Risk-Return Profile')
    
    # Add ticker labels
    for i, ticker in enumerate(tickers):
        axes[0,2].annotate(ticker, (volatilities[i], returns[i]), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    
    # 4. Hit rates
    hit_rates = [ticker_performance[t]['hit_rate'] for t in tickers]
    axes[1,0].bar(tickers, hit_rates, alpha=0.7)
    axes[1,0].set_title('Hit Rates')
    axes[1,0].set_ylabel('Hit Rate')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].axhline(y=0.5, color='red', linestyle='--', label='Random')
    axes[1,0].legend()
    
    # 5. Transaction costs impact
    gross_returns = [ticker_performance[t]['total_gross_return'] for t in tickers]
    net_returns = [ticker_performance[t]['total_net_return'] for t in tickers]
    
    x = np.arange(len(tickers))
    width = 0.35
    axes[1,1].bar(x - width/2, gross_returns, width, label='Gross', alpha=0.7)
    axes[1,1].bar(x + width/2, net_returns, width, label='Net', alpha=0.7)
    axes[1,1].set_title('Gross vs Net Returns')
    axes[1,1].set_ylabel('Return')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(tickers, rotation=45)
    axes[1,1].legend()
    
    # 6. Max drawdowns
    max_drawdowns = [ticker_performance[t]['max_drawdown'] for t in tickers]
    axes[1,2].bar(tickers, max_drawdowns, color='red', alpha=0.7)
    axes[1,2].set_title('Maximum Drawdowns')
    axes[1,2].set_ylabel('Max Drawdown')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    # 7. Period robustness - Sharpe ratios
    period_names = list(period_results.keys())
    period_sharpes = []
    
    for period in period_names:
        if period_results[period]:
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in period_results[period].values()])
            period_sharpes.append(avg_sharpe)
        else:
            period_sharpes.append(0)
    
    axes[2,0].bar(period_names, period_sharpes, alpha=0.7)
    axes[2,0].set_title('Robustness: Sharpe by Period')
    axes[2,0].set_ylabel('Average Sharpe')
    axes[2,0].tick_params(axis='x', rotation=45)
    axes[2,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 8. Period robustness - Returns
    period_returns = []
    for period in period_names:
        if period_results[period]:
            avg_return = np.mean([r['total_return'] for r in period_results[period].values()])
            period_returns.append(avg_return)
        else:
            period_returns.append(0)
    
    axes[2,1].bar(period_names, period_returns, alpha=0.7)
    axes[2,1].set_title('Robustness: Returns by Period')
    axes[2,1].set_ylabel('Average Return')
    axes[2,1].tick_params(axis='x', rotation=45)
    
    # 9. Summary statistics table
    axes[2,2].axis('off')
    
    # Calculate overall statistics
    avg_sharpe = np.mean(sharpes)
    avg_excess = np.mean(excess_returns)
    positive_sharpe_pct = (np.array(sharpes) > 0).mean() * 100
    positive_excess_pct = (np.array(excess_returns) > 0).mean() * 100
    
    summary_text = f"""ENHANCED MODEL SUMMARY
    
Total Tickers Analyzed: {len(tickers)}
Average Sharpe Ratio: {avg_sharpe:.3f}
Average Excess Return: {avg_excess:.4f}

Positive Sharpe: {positive_sharpe_pct:.1f}%
Positive Excess: {positive_excess_pct:.1f}%

Best Performer: {tickers[np.argmax(sharpes)]}
Best Sharpe: {max(sharpes):.3f}

Worst Performer: {tickers[np.argmin(sharpes)]}
Worst Sharpe: {min(sharpes):.3f}

Transaction Cost Impact:
Avg Gross: {np.mean(gross_returns):.4f}
Avg Net: {np.mean(net_returns):.4f}
Avg Cost: {np.mean([ticker_performance[t]['transaction_costs'] for t in tickers]):.4f}"""
    
    axes[2,2].text(0.1, 0.5, summary_text, fontsize=11, transform=axes[2,2].transAxes,
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    plt.tight_layout()
    plt.savefig('enhanced_contrarian_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def paper_readiness_final_assessment(ticker_performance, period_results):
    """Final assessment for paper readiness"""
    print("\n" + "="*60)
    print("FINAL PAPER READINESS ASSESSMENT")
    print("="*60)
    
    tickers = list(ticker_performance.keys())
    sharpes = [ticker_performance[t]['sharpe_ratio'] for t in tickers]
    excess_returns = [ticker_performance[t]['excess_return'] for t in tickers]
    
    # Key metrics
    avg_sharpe = np.mean(sharpes)
    median_sharpe = np.median(sharpes)
    positive_sharpe_pct = (np.array(sharpes) > 0).mean() * 100
    significant_sharpe_pct = (np.array(sharpes) > 0.5).mean() * 100
    
    avg_excess = np.mean(excess_returns)
    median_excess = np.median(excess_returns)
    positive_excess_pct = (np.array(excess_returns) > 0).mean() * 100
    
    # Robustness across periods
    period_consistency = 0
    for period_data in period_results.values():
        if period_data:
            period_sharpes = [r['sharpe_ratio'] for r in period_data.values()]
            if np.mean(period_sharpes) > 0:
                period_consistency += 1
    
    robustness_score = period_consistency / len(period_results) * 100
    
    print(f"📊 PERFORMANCE METRICS:")
    print(f"   Average Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"   Median Sharpe Ratio: {median_sharpe:.3f}")
    print(f"   Positive Sharpe: {positive_sharpe_pct:.1f}%")
    print(f"   Significant Sharpe (>0.5): {significant_sharpe_pct:.1f}%")
    
    print(f"\n💰 ECONOMIC SIGNIFICANCE:")
    print(f"   Average Excess Return: {avg_excess:.4f}")
    print(f"   Median Excess Return: {median_excess:.4f}")
    print(f"   Positive Excess Return: {positive_excess_pct:.1f}%")
    
    print(f"\n🔒 ROBUSTNESS:")
    print(f"   Period Consistency: {robustness_score:.1f}%")
    print(f"   Number of Tickers: {len(tickers)}")
    
    # Final score calculation
    sharpe_score = min(100, max(0, (avg_sharpe + 1) * 50))  # -1 to 1 mapped to 0-100
    economic_score = min(100, max(0, avg_excess * 1000))  # Scale excess returns
    robustness_score = robustness_score  # Already 0-100
    
    final_score = (sharpe_score + economic_score + robustness_score) / 3
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Sharpe Score: {sharpe_score:.1f}/100")
    print(f"   Economic Score: {economic_score:.1f}/100")
    print(f"   Robustness Score: {robustness_score:.1f}/100")
    print(f"   OVERALL SCORE: {final_score:.1f}/100")
    
    if final_score >= 70:
        print("\n🎉 EXCELLENT: Ready for top-tier finance journal!")
        print("   - Strong risk-adjusted returns")
        print("   - Significant economic impact")
        print("   - Robust across periods")
        recommendation = "Submit to Journal of Finance or similar"
    elif final_score >= 50:
        print("\n✅ GOOD: Ready for solid finance journal")
        print("   - Decent risk-adjusted returns")
        print("   - Some economic significance")
        print("   - Reasonable robustness")
        recommendation = "Submit to mid-tier finance journal"
    elif final_score >= 30:
        print("\n⚠️ PROMISING: Needs minor improvements")
        print("   - Some positive results")
        print("   - Limited economic significance")
        print("   - Mixed robustness")
        recommendation = "Refine and resubmit"
    else:
        print("\n❌ NEEDS WORK: Major improvements required")
        recommendation = "Substantial revision needed"
    
    print(f"\n📝 RECOMMENDATION: {recommendation}")
    
    return final_score

def main():
    """Main execution with full enhancement"""
    print("=== ENHANCED CONTRARIAN MODEL - PAPER GRADE ===")
    
    # 1. Load enhanced data
    df = load_enhanced_data()
    df = create_enhanced_features(df)
    df = create_enhanced_targets(df)
    
    print(f"Enhanced dataset: {len(df)} samples, {df['ticker'].nunique()} tickers")
    
    # 2. Expanded ticker analysis
    ticker_performance = expanded_ticker_analysis(df)
    
    # 3. Robustness testing
    period_results = robustness_test_by_periods(df)
    
    # 4. Comprehensive visualization
    create_comprehensive_visualization(ticker_performance, period_results)
    
    # 5. Final paper assessment
    final_score = paper_readiness_final_assessment(ticker_performance, period_results)
    
    return final_score

if __name__ == "__main__":
    main()
