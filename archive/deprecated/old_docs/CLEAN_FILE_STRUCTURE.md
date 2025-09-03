# 🧹 Clean File Structure - Meme Stock Analysis Project

## 📁 **Root Directory**
```
meme_stock/
├── README.md                           # Main project overview
├── README_DATA.md                      # Data documentation
├── requirements.txt                     # Python dependencies
├── Makefile                           # Build automation
├── .gitignore                         # Git ignore rules
├── schema_contract.yaml               # Data schema definition
├── metadata_schema.yaml               # Metadata schema
├── feature_scaler.pkl                 # ML feature scaler
├── training_results.json              # Training results
├── validate_and_align.py              # Data validation script
├── collect_prices.py                  # Stock price collection
├── collect_prices_v2.py              # Enhanced price collection
├── collect_reddit.py                  # Reddit data collection
└── backfill_metadata.py               # Metadata backfill
```

## 🔍 **Analysis Directory** (`analysis/`)
```
analysis/
├── reddit_mechanism/                  # Reddit mechanism analysis
│   ├── reddit_mechanism_complete_robust.py
│   └── comprehensive_mechanism_analysis.png
├── price_correlation/                  # Price correlation analysis
│   ├── reddit_price_correlation_analysis.py
│   ├── price_reddit_analysis_20250814_152030.png
│   ├── price_reddit_analysis_20250814_152311.png
│   └── mentions_distribution_analysis.png
├── experiments/                        # ML experiments
│   ├── financial_ml_comparison_experiment.py
│   ├── financial_ml_experiment_clean.py
│   ├── test_experiment.py
│   ├── quick_experiment.py
│   ├── debug_ic_values.py
│   └── verify_ic_calculation.py
└── colab_guides/                       # Colab optimization guides
    ├── COLAB_A100_DEBUG_GUIDE.md
    ├── A100_COLAB_EXECUTION_GUIDE.md
    ├── COLAB_READY_SUMMARY.md
    └── fix_lstm_data_loading.py
```

## 📊 **Data Directory** (`data/`)
```
data/
├── colab_datasets/                     # Colab-ready datasets
│   ├── tabular_train_20250814_031335.csv
│   ├── tabular_val_20250814_031335.csv
│   └── tabular_test_20250814_031335.csv
├── features/                           # Feature engineering outputs
├── models/                             # Trained ML models
├── processed/                          # Processed data
├── raw/                                # Raw data sources
└── results/                            # Analysis results
```

## 📚 **Documentation Directory** (`docs/`)
```
docs/
├── CLEAN_FILE_STRUCTURE.md            # This file
├── reddit_data_analysis_report.md     # Reddit analysis report
├── project_status_and_roadmap_20250813.md
├── data_directory_structure.md
├── ORGANIZATION_SUMMARY.md
├── COLAB_TRAINING_GUIDE.md
├── reddit_api_setup_guide.md
└── [other documentation files]
```

## 🛠️ **Scripts Directory** (`scripts/`)
```
scripts/
├── plot_price_reddit.py               # Price-Reddit plotting
├── plot_price_reddit_en.py           # English version
├── py_to_ipynb.py                    # Python to notebook converter
├── py_to_ipynb_improved.py           # Improved converter
├── process_archive_reddit_data.py     # Archive data processor
├── historical_reddit_collector.py     # Historical Reddit collector
├── build_reddit_from_archive.py       # Archive builder
└── [other utility scripts]
```

## 🏗️ **Source Directory** (`src/`)
```
src/
├── data/                              # Data processing modules
├── features/                           # Feature engineering
├── models/                             # ML model definitions
├── utils/                              # Utility functions
└── evaluation/                         # Model evaluation
```

## 🔧 **Configuration Directory** (`config/`)
```
config/
├── asset_reddit_map.yaml              # Asset-Reddit mapping
└── reddit_config.json                 # Reddit API credentials
```

## 📋 **Notebooks Directory** (`notebooks/`)
```
notebooks/
├── meme_stock_deep_learning_colab.ipynb
├── meme_stock_deep_learning_colab_fixed.ipynb
├── meme_stock_deep_learning_colab_fixed_improved.ipynb
├── colab_advanced_model_training.ipynb
└── [other Jupyter notebooks]
```

## 🧪 **Validation Directory** (`validation/`)
```
validation/
├── comprehensive_validation.py         # Data validation
├── fix_data_leakage.py                # Leakage fixes
├── simple_leakage_test.py             # Leakage testing
└── [other validation scripts]
```

## 📈 **Results Directory** (`results/`)
```
results/
├── README_RESULTS.md                   # Results overview
├── week1_comprehensive_summary.json   # Week 1 summary
├── week1_summary_report.txt           # Week 1 report
├── [other result files]
└── [day completion summaries]
```

## 🎯 **Key Analysis Files**

### **Reddit Mechanism Analysis** (Most Important)
- **File**: `analysis/reddit_mechanism/reddit_mechanism_complete_robust.py`
- **Output**: `comprehensive_mechanism_analysis.png`
- **Purpose**: Complete analysis of Reddit mentions vs price correlation
- **Key Finding**: Strong contrarian effect (-0.0510 correlation)

### **Price Correlation Analysis**
- **File**: `analysis/price_correlation/reddit_price_correlation_analysis.py`
- **Purpose**: Detailed correlation analysis between Reddit mentions and stock prices

### **ML Experiments**
- **Directory**: `analysis/experiments/`
- **Purpose**: Various ML model experiments and IC calculations

## 🚀 **How to Use**

1. **For Reddit Analysis**: Run `analysis/reddit_mechanism/reddit_mechanism_complete_robust.py`
2. **For Price Correlation**: Run `analysis/price_correlation/reddit_price_correlation_analysis.py`
3. **For ML Experiments**: Use scripts in `analysis/experiments/`
4. **For Colab Optimization**: Check `analysis/colab_guides/`

## 📝 **Maintenance Notes**

- **Keep only the final, working versions** of analysis scripts
- **PNG files** are automatically generated - can be regenerated
- **Temporary/experimental files** should be cleaned up regularly
- **Use descriptive names** for new files
- **Organize by functionality** not by creation date

---
*Last Updated: 2025-01-14*
*Status: Clean and Organized* ✨
