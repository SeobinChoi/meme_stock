# Meme Stock Analysis Project

## Overview
This project analyzes meme stock movements using Reddit sentiment data, news data, and stock price data to predict market trends.

## ✅ **PROJECT STATUS**

### **🎯 Current State: READY FOR GPU TRAINING**
- ✅ **Data Leakage Fixed** - Critical validation issues resolved
- ✅ **Clean Dataset** - 278 features (43 clean + 64 advanced meme features)
- ✅ **Baseline Models** - Trained with clean data (no overfitting)
- ✅ **Advanced Features** - Viral patterns, sentiment, social dynamics
- 🚀 **Next Step:** Colab GPU training (4-6 hours)

### **📊 Key Datasets Available**
- `data/features/clean_features_dataset_20250810_202725.csv` - **Clean features (43)**
- `data/features/targets_dataset_20250810_202725.csv` - **Target variables (12)**  
- `colab_advanced_features.csv` - **Ready for Colab upload**

### **🎯 Ready for Advanced Training**
- **BERT Sentiment Pipeline** (FinBERT)
- **Multi-Modal Transformer Architecture** 
- **Advanced LSTM with Attention**
- **Ensemble System**

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Reddit API Setup
1. Get Reddit API credentials from [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Update `config/reddit_config.json` with your credentials
3. Test connection: `python scripts/test_reddit_connection.py`

### 3. Download Extended Reddit Data
```bash
python scripts/download_extended_reddit_data.py
```

## 📊 Data Sources

### Current Data
- **Reddit WSB Posts**: 399,637 posts from 2021
- **Stock Price Data**: AMC, BlackBerry, GameStop, Wish (2013-2021)
- **Processed Data**: 29,793 filtered posts

### Extended Data (After Download)
- **2020-2022 WSB Posts**: Complete 3-year dataset
- **Enhanced Metadata**: Upvote ratios, author info, permalinks
- **Keyword Filtering**: Focus on meme stock related posts
- **Statistical Analysis**: Daily/monthly post counts, top posts

## 📁 Project Structure

```
meme_stock/
├── 🚀 COLAB FILES (Ready for GPU Training)
│   ├── colab_advanced_model_training.ipynb    # Colab notebook 
│   ├── colab_advanced_model_training.py       # Training script
│   ├── colab_advanced_features.csv           # Dataset for Colab
│   ├── COLAB_TRAINING_GUIDE.md               # Instructions
│   └── convert_to_colab.py                   # Colab converter
│
├── 📊 CORE DATA & CODE
│   ├── data/
│   │   ├── features/         # Engineered features (clean dataset)
│   │   ├── processed/        # Cleaned and merged data
│   │   ├── raw/             # Original datasets (Reddit, stocks)
│   │   └── results/         # Validation and monitoring results
│   ├── src/
│   │   ├── data/            # Data processing modules
│   │   ├── features/        # Feature engineering
│   │   ├── models/          # Model implementations
│   │   └── evaluation/      # Evaluation frameworks
│   ├── scripts/             # Data collection scripts
│   ├── notebooks/           # Jupyter exploration notebooks
│   └── config/              # Configuration files
│
├── 🔧 UTILITIES & TOOLS
│   ├── utils/               # Utility scripts and tools
│   ├── validation/          # Data validation and testing
│   ├── analysis/            # Analysis and correlation tools
│   └── reports/             # Implementation progress logs
│
├── 📚 DOCUMENTATION
│   ├── docs/                # Technical documentation
│   ├── documentation/       # Project summaries and guides
│   ├── guide/               # Implementation guides
│   └── results/             # Completion summaries and logs
│
└── 📋 PROJECT FILES
    ├── README.md            # This file
    ├── requirements.txt     # Python dependencies
    └── prepare_colab_data.py # Data preparation for Colab
```

## 🔧 Configuration

### Reddit API Credentials
Create `config/reddit_config.json`:
```json
{
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "user_agent": "MemeStockAnalysis/1.0 (by /u/YOUR_USERNAME)",
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD"
}
```

## 📈 Features

### Data Collection
- **Multi-year Coverage**: 2020-2022 Reddit data
- **Smart Filtering**: Meme stock keyword detection
- **Rate Limiting**: Respects Reddit API limits
- **Error Handling**: Robust error recovery

### Data Analysis
- **Sentiment Analysis**: Post sentiment scoring
- **Engagement Metrics**: Upvotes, comments, ratios
- **Temporal Analysis**: Daily/monthly trends
- **Top Content**: Highest scoring posts

## 🚨 Important Notes

- **API Limits**: Reddit has rate limiting - be patient
- **Data Size**: Extended dataset will be several GB
- **Authentication**: Requires Reddit account and app
- **Storage**: Ensure sufficient disk space

## 📚 Documentation

- [Reddit API Setup Guide](docs/reddit_api_setup_guide.md)
- [Data Analysis Summary](docs/reddit_data_analysis_summary.md)
- [Project Plan](docs/plan_meme_stock_formatted_v3.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. 