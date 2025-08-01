# 📊 Meme Stock Prediction Project - Week 1 Implementation

## 🎯 **Project Overview**
Academic competition-winning meme stock prediction system with comprehensive data preprocessing, feature engineering, and baseline model construction. This Week 1 implementation provides a solid foundation for predicting meme stock movements using Reddit sentiment, technical indicators, and social media dynamics.

## 🏆 **Academic Competition Features**
- **40+ engineered features** combining Reddit sentiment, technical indicators, and cross-features
- **3 baseline models**: LightGBM (short-term), XGBoost (long-term), LSTM (sequential patterns)
- **Time series cross-validation** with walk-forward validation
- **Comprehensive evaluation framework** with academic-grade metrics
- **Reproducible research** with fixed random states and documentation

## 📁 **Project Structure**
```
meme_stock/
├── week1_baseline/           # Week 1 implementation
│   ├── data_preprocessing.py # Data loading, cleaning, merging
│   ├── feature_engineering.py # 40+ feature generation
│   ├── models.py             # LightGBM, XGBoost, LSTM models
│   ├── evaluation.py         # Comprehensive evaluation framework
│   └── main.py              # Complete pipeline orchestration
├── data/                     # Data storage
│   ├── reddit_wsb.csv       # Reddit WSB posts
│   ├── meme_stocks.csv      # GME, AMC, BB price/volume data
│   ├── wsb_mention_counts.csv # Daily stock mention counts
│   └── processed_data.csv   # Preprocessed merged data
├── models/                   # Trained models
├── results/                  # Evaluation results and reports
├── requirements.txt          # Python dependencies
└── auto_push.sh             # Auto-push to GitHub
```

## 🚀 **Quick Start**

### **1. Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install tensorflow-gpu
```

### **2. Run Complete Pipeline**
```bash
cd week1_baseline
python main.py
```

### **3. Individual Execution**
```bash
# Data preprocessing only
python data_preprocessing.py

# Feature engineering only
python feature_engineering.py

# Model training only
python models.py

# Evaluation only
python evaluation.py
```

## 📊 **Feature Engineering (40+ Features)**

### **A. Reddit Features (15 features)**
- **Sentiment Analysis**: BERT-based positive/negative/neutral scores
- **Viral Indicators**: Mention surge rates (1, 3, 7-day rolling)
- **Community Dynamics**: Sentiment consensus, weekend patterns
- **Engagement Metrics**: Score/comment aggregation

### **B. Technical Features (15 features)**
- **Price Indicators**: Returns, moving averages, volatility
- **Volume Analysis**: Volume ratios, VWAP deviations
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Multi-timeframe**: 1-day, 3-day, 7-day horizons

### **C. Cross Features (10 features)**
- **Social-Price Correlations**: Sentiment vs price relationships
- **Mention-Volume Sync**: Social activity vs trading volume
- **Cross-Stock Correlations**: GME-AMC-BB relationships
- **Weekend Effects**: Monday impact predictions

## 🤖 **Baseline Models**

### **Model 1: LightGBM (Short-term Specialized)**
- **Target**: 1-3 day price direction prediction
- **Features**: Optimized for short-term patterns
- **Validation**: Time series cross-validation
- **Performance Target**: >65% directional accuracy

### **Model 2: XGBoost (Long-term Trend Specialized)**
- **Target**: 3-7 day price change magnitude
- **Features**: Focus on trend identification
- **Interpretability**: SHAP analysis included
- **Performance Target**: RMSE < 0.15

### **Model 3: LSTM (Sequential Patterns)**
- **Target**: Sequential pattern learning
- **Architecture**: 60-day lookback window
- **Regularization**: Dropout + Early stopping
- **Use Case**: Complex temporal dependencies

## 📈 **Evaluation Framework**

### **Metrics**
- **Classification**: Accuracy, F1-score, AUC-ROC
- **Regression**: MAE, RMSE, Directional accuracy
- **Trading**: Sharpe ratio, Maximum drawdown

### **Validation Strategy**
- **Time Series Split**: 5-fold walk-forward validation
- **Out-of-Sample Testing**: Recent 3 months
- **Feature Importance**: Model interpretability analysis

## 🎯 **Performance Targets**

| Model | Target | Metric | Goal |
|-------|--------|--------|------|
| LightGBM | Direction (1-3d) | Accuracy | >65% |
| XGBoost | Magnitude (3-7d) | RMSE | <0.15 |
| LSTM | Sequential | Pattern Capture | Confirmed |

## 📋 **Deliverables**

### **Data Products**
- ✅ `processed_data.csv` - Clean, merged dataset
- ✅ `features_data.csv` - 40+ engineered features
- ✅ `baseline_performance.csv` - Model comparison

### **Models**
- ✅ Trained LightGBM models (`.pkl`)
- ✅ Trained XGBoost models (`.pkl`)
- ✅ Trained LSTM models (`.h5`)

### **Visualizations**
- ✅ `feature_importance.png` - Top features analysis
- ✅ `model_comparison.png` - Performance comparison
- ✅ `predictions_plot.png` - Predictions vs actual

### **Reports**
- ✅ `comprehensive_evaluation.csv` - Detailed metrics
- ✅ `week1_final_report.txt` - Executive summary

## 🔧 **Technical Implementation**

### **Data Preprocessing Pipeline**
1. **Loading**: Reddit posts, stock prices, mention counts
2. **Cleaning**: Missing values, outliers, date alignment
3. **Merging**: Time-based joins with forward filling
4. **Validation**: Data quality checks and documentation

### **Feature Engineering Pipeline**
1. **Reddit Features**: Sentiment, viral indicators, community dynamics
2. **Technical Features**: Price/volume indicators, volatility measures
3. **Cross Features**: Social-price relationships, correlations
4. **Scaling**: StandardScaler for numerical features

### **Model Training Pipeline**
1. **Data Preparation**: Feature selection, target creation
2. **Cross-Validation**: Time series split with walk-forward
3. **Hyperparameter Tuning**: Optuna optimization (optional)
4. **Model Persistence**: Save/load functionality

## 🚨 **Important Notes**

### **Data Leakage Prevention**
- ✅ No future information in features
- ✅ Proper time series validation
- ✅ Walk-forward testing strategy

### **Reproducibility**
- ✅ Fixed random states (42)
- ✅ Version-controlled dependencies
- ✅ Comprehensive logging

### **Memory Management**
- ✅ Chunked processing for large datasets
- ✅ Efficient feature computation
- ✅ Model serialization

## 🎓 **Academic Competition Advantages**

### **Technical Excellence**
- **Comprehensive Feature Set**: 40+ engineered features
- **Multiple Model Types**: Ensemble approach with different strengths
- **Robust Validation**: Time series cross-validation
- **Interpretability**: Feature importance and SHAP analysis

### **Research Quality**
- **Reproducible Code**: Clean, documented implementation
- **Performance Metrics**: Academic-grade evaluation framework
- **Visualization**: Professional-quality plots and reports
- **Documentation**: Comprehensive README and inline comments

### **Competitive Edge**
- **Meme-Specific Features**: Reddit sentiment and viral indicators
- **Cross-Asset Analysis**: GME-AMC-BB correlations
- **Temporal Patterns**: Weekend effects and sequential learning
- **Practical Metrics**: Sharpe ratio and drawdown analysis

## 🔮 **Week 2 Roadmap**

### **Meme-Specific Enhancements**
- Advanced sentiment analysis with meme detection
- Social media trend analysis (Twitter, TikTok)
- Options flow analysis and gamma exposure
- Short interest and squeeze indicators

### **Advanced Models**
- Transformer-based sequence models
- Graph neural networks for social relationships
- Reinforcement learning for trading strategies
- Ensemble methods with dynamic weighting

## 📞 **Support & Contact**

For questions about the implementation or academic competition preparation:

- **Repository**: https://github.com/SeobinChoi/meme_stock
- **Documentation**: Comprehensive inline comments
- **Issues**: GitHub issues for bug reports

## 📄 **License**

MIT License - Academic and research use encouraged.

---

**🎯 Ready for Academic Competition Success! 🚀**

This Week 1 implementation provides a solid, competitive foundation for meme stock prediction with proven ML techniques and comprehensive evaluation frameworks. 