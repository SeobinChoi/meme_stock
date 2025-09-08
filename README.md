# Meme Stock Contrarian Effect Prediction - ML Upgrade

This repository implements a comprehensive machine learning upgrade for predicting meme stock contrarian effects using Reddit sentiment data, as described in the manual. The system analyzes Reddit posts about meme stocks (GME, AMC, BB) and predicts future price movements with a contrarian approach.

## 🎯 Overview

The ML upgrade implements a complete pipeline that:

- **Engineers features** from Reddit text (confidence, sentiment, dynamics) and technical indicators
- **Trains multiple models** including Ridge, LightGBM, TCN, and TFT
- **Creates ensembles** with simple blending and stacking
- **Implements trading strategies** with realistic costs and slippage
- **Validates robustness** through placebo tests and stability analysis
- **Provides interpretability** via SHAP analysis and ablation studies
- **Generates comprehensive reports** and visualizations

## 📁 Project Structure

```
meme_stock/
├── src/
│   ├── features/
│   │   ├── text_confidence.py      # Text confidence scoring
│   │   ├── sentiment.py            # Sentiment analysis (VADER, FinBERT)
│   │   ├── reddit_dynamics.py      # Reddit Surprise, EMAs, momentum
│   │   ├── technical_features.py   # Technical indicators
│   │   └── feature_pipeline.py     # Main feature engineering pipeline
│   ├── modeling/
│   │   ├── validation.py           # Purged K-fold, walk-forward validation
│   │   ├── baseline_models.py      # Ridge, LightGBM, XGBoost
│   │   ├── sequence_models.py      # TCN, TFT
│   │   └── ensemble.py             # Meta-ensemble with stacking
│   └── evaluation/
│       ├── trading_strategy.py     # Trading logic with costs
│       ├── interpretability.py     # SHAP analysis, ablation studies
│       ├── robustness.py           # Placebo tests, stability analysis
│       └── reporting.py             # Reports and visualizations
├── run_ml_upgrade.py               # Main execution script
├── manual.txt                      # Original manual specification
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost tensorflow shap matplotlib seaborn scipy
```

### 2. Prepare Data

Ensure you have:
- Reddit data CSV file with columns: `date`, `ticker`, `text`, `score`, `comments`
- Price data CSV files for each ticker with columns: `date`, `open`, `high`, `low`, `close`, `volume`

### 3. Run the Pipeline

```bash
python run_ml_upgrade.py \
    --reddit-data data/processed/reddit/reddit_archive_daily_2021_2023.csv \
    --price-data-dir data/raw/stocks/ \
    --tickers GME AMC BB \
    --date-range 2021-01-01 2023-12-31 \
    --output-dir results \
    --use-finbert
```

## 📊 Key Features

### Feature Engineering

- **Text Confidence**: Over-assertive vs hedging language detection
- **Sentiment Analysis**: VADER and FinBERT sentiment scoring
- **Reddit Dynamics**: Reddit Surprise (RS), EMAs, momentum, volatility
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages

### Models

- **Baseline Models**: Ridge regression, LightGBM, XGBoost
- **Sequence Models**: TCN (Temporal Convolutional Network), TFT (Temporal Fusion Transformer)
- **Ensemble**: Simple blending and stacking with meta-learners

### Validation

- **Purged K-Fold**: Prevents data leakage with embargo periods
- **Walk-Forward**: Rolling window validation for time series
- **Static Splits**: Train (2021-2022), Val (2023 H1), Test (2023 H2)

### Trading Strategy

- **Contrarian Logic**: Bets against Reddit sentiment
- **Realistic Costs**: 20-50 bps round-trip costs
- **Position Management**: Stateful positions to reduce turnover
- **Risk Metrics**: Sharpe ratio, max drawdown, hit rate

### Robustness Testing

- **Placebo Tests**: Random pairing, cross-ticker swap, lag inversion
- **Stability Analysis**: Walk-forward performance, regime stability
- **Anti-Overfit Checks**: Noise features, label shuffle

## 📈 Expected Results

The system should demonstrate:

1. **Contrarian Effect**: Higher Reddit confidence/sentiment predicts negative future returns
2. **Model Performance**: Spearman correlation > 0.05, hit rate > 50%
3. **Trading Alpha**: Positive Sharpe ratio after costs
4. **Robustness**: Passes placebo tests and shows stability across regimes

## 🔧 Configuration

### Model Parameters

```python
# LightGBM (default)
max_depth=5, num_leaves=31, learning_rate=0.05
n_estimators=600, feature_fraction=0.8, bagging_fraction=0.8

# TCN
sequence_length=20, filters=64, kernel_size=3
dilation_rates=[1, 2, 4], dropout_rate=0.2

# TFT
d_model=64, num_heads=4, num_layers=2
dff=128, dropout_rate=0.1
```

### Trading Parameters

```python
# Strategy
threshold=0.001  # 0.1% signal threshold
cost_half=0.001  # 10 bps enter/exit cost
max_position=1.0  # Maximum position size
```

## 📋 Output Files

The pipeline generates:

- `features_complete.csv`: Complete feature dataset
- `model_report.md`: Model performance analysis
- `strategy_report.md`: Trading strategy results
- `robustness_report.md`: Robustness test results
- `comprehensive_report.md`: Complete analysis summary
- `figures/`: Visualization plots
- `all_results.json`: Complete results in JSON format

## 🧪 Validation Checklist

Before considering the analysis complete, verify:

- [ ] No future data leakage in features
- [ ] Purged/embargoed validation implemented
- [ ] Walk-forward evaluation completed
- [ ] Test set used only once
- [ ] Cost sensitivity analysis (20/35/50 bps)
- [ ] Yearly breakdown (2021/2022/2023)
- [ ] SHAP or attention-based interpretability
- [ ] Ablations and placebos completed
- [ ] Strategy curve vs Buy & Hold plotted

## 🔍 Interpretability

The system provides multiple interpretability tools:

- **Feature Importance**: Model-specific importance rankings
- **SHAP Analysis**: Instance-level explanations
- **Ablation Studies**: Feature group contribution analysis
- **Attention Weights**: TFT attention visualization (if available)

## ⚠️ Limitations

- **Sample Size**: Limited to available Reddit data
- **Platform Bias**: Only Reddit data (no Twitter/StockTwits)
- **Regime Fragility**: Performance may vary across market conditions
- **Data Quality**: Dependent on Reddit data quality and completeness

## 🚀 Extensions

Potential improvements:

- **Cross-Sectional Ranking**: Rank stocks within each day
- **Additional Platforms**: Include Twitter, Stocktwits data
- **Alternative Assets**: Extend to crypto, other meme stocks
- **Real-Time**: Implement live prediction system
- **Advanced Models**: Transformer architectures, graph neural networks

## 📚 References

This implementation follows the specifications in `manual.txt` which describes:
- Data ingestion and time alignment
- Feature engineering methodology
- Validation design principles
- Model architectures
- Trading strategy implementation
- Interpretability and robustness testing

## 🤝 Contributing

To contribute to this project:

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure all validation checks pass
5. Follow the manual specifications

## 📄 License

This project is for research and educational purposes. Please ensure compliance with data usage terms for Reddit and financial data sources.

---

**Note**: This is a research implementation for academic purposes. Always validate results and consider transaction costs, market impact, and regulatory requirements before any real trading applications.

