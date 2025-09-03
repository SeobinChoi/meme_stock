# Meme Stock Data Pipeline

A comprehensive data collection and validation pipeline for analyzing **meme stocks/crypto** price movements with **Reddit community activity**. Collects financial data via yfinance/Polygon APIs and Reddit daily engagement via BigQuery, then validates and aligns them for ML modeling.

## Project Purpose

This pipeline automates the collection, validation, and alignment of:
- **Financial data**: Stock prices (GME, AMC, BB, KOSS, BBBY) and crypto prices (DOGE, SHIB, PEPE, BONK)
- **Social data**: Reddit daily activity (posts, comments, scores) from relevant subreddits
- **Aligned datasets**: Price and social data merged with proper UTC timestamps for modeling

All data follows strict schemas with UTC standardization, versioning, metadata tracking, and comprehensive validation gates.

## Prerequisites

### Python Environment
- **Python ≥ 3.10**
- Install dependencies: `pip install -r requirements.txt`
  - Key packages: `yfinance`, `google-cloud-bigquery`, `pandas`, `pyyaml`, `python-dateutil`

### BigQuery Access (for Reddit data)
- **Required**: Google Cloud Project with BigQuery API enabled
- **Authentication**: Choose one:
  ```bash
  # Option 1: Application Default Credentials
  gcloud auth application-default login
  
  # Option 2: Service Account Key
  export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
  ```
- **Project ID**: Set `GCP_PROJECT` environment variable

### Optional APIs
- **Polygon.io API** (for delisted stocks like BBBY): Set `POLYGON_API_KEY`

## Directory Layout

```
meme_stock/
├── data/
│   ├── raw/                    # Raw collected data (never overwritten)
│   │   ├── stocks/            # Stock price CSVs + metadata
│   │   ├── crypto/            # Crypto price CSVs + metadata  
│   │   └── reddit/            # Reddit daily aggregates + metadata
│   ├── processed/             # Aligned and validated datasets
│   │   ├── stocks/            # Individual stock aligned datasets
│   │   ├── crypto/            # Individual crypto aligned datasets
│   │   └── panel/             # Panel datasets (all stocks/crypto combined)
│   └── INDEX.jsonl            # Complete dataset index with versions
├── config/
│   ├── schema_contract.yaml   # Data schema specifications
│   └── asset_reddit_map.yaml  # Asset-to-subreddit mappings
├── sql/                       # BigQuery SQL templates
├── common/                    # Shared utilities (paths, validation, etc.)
├── reports/                   # Data quality reports (markdown)
├── logs/                      # Structured execution logs
└── Makefile                   # Pipeline orchestration
```

## Quick Start

### 1. Environment Setup
```bash
# Set required environment variables
export GCP_PROJECT="your-gcp-project-id"

# Optional: Polygon API for delisted stocks
export POLYGON_API_KEY="your-polygon-key"

# Install dependencies
pip install -r requirements.txt
```

### 2. Backfill Metadata (for existing files)
```bash
make backfill
```

### 3. Full Pipeline (default: 2020-12-01 to 2023-12-31)
```bash
make all
```

This runs:
1. **Price collection** → `data/raw/stocks/` and `data/raw/crypto/`
2. **Reddit collection** → `data/raw/reddit/`  
3. **Validation & alignment** → `data/processed/` + `reports/`

## Smoke Test (Fast Development)

For quick validation with a 3-month window:

```bash
export GCP_PROJECT="your-project"
make smoke
```

**Expected outputs:**
- `data/raw/stocks/GME_stock_data*.csv` + `.meta.json`
- `data/raw/crypto/DOGE_crypto_data*.csv` + `.meta.json`
- `data/raw/reddit/reddit_wallstreetbets*.csv` + `.meta.json`
- `data/processed/stocks/GME_aligned.*` / `crypto/DOGE_aligned.*`
- `data/processed/panel/stocks_panel.csv` / `cryptos_panel.csv`
- `reports/data_quality_YYYYMMDD.md`
- `logs/YYYYMMDD.log`
- Updated `data/INDEX.jsonl`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `START` | 2020-12-01 | Start date (YYYY-MM-DD) |
| `END` | 2023-12-31 | End date (YYYY-MM-DD) |
| `STOCKS` | GME AMC BB KOSS BBBY | Stock symbols to collect |
| `CRYPTO` | DOGE SHIB PEPE BONK | Crypto symbols to collect |
| `SUBREDDITS` | wallstreetbets stocks... | Reddit communities to monitor |
| `USE_KEYWORDS` | true | Filter Reddit posts by asset keywords |
| `GCP_PROJECT` | *required* | Google Cloud Project ID |
| `POLYGON_API_KEY` | *optional* | Polygon.io API key for delisted stocks |
| `VS` | usd | Base currency for crypto prices |

### Asset-Subreddit Mapping

Edit `config/asset_reddit_map.yaml` to configure which subreddits to monitor for each asset:

```yaml
stocks:
  GME: [wallstreetbets, stocks, GME, Superstonk]
  AMC: [wallstreetbets, amcstock, stocks]
  BBBY: [wallstreetbets, stocks, BBBY]

crypto:
  DOGE: [cryptocurrency, dogecoin]
  SHIB: [cryptocurrency, SHIBArmy]

defaults:
  stocks: [wallstreetbets, stocks]
  crypto: [cryptocurrency]
```

## Data Contracts

### Schema Standards
All data follows specifications in `schema_contract.yaml`:
- **UTC timestamps**: ISO 8601 format with timezone (`2021-01-01T00:00:00+00:00`)
- **Column order**: Standardized across all datasets
- **Validation rules**: Required fields, data types, bounds checking

### Versioning & Metadata
- **Versioned writes**: New files get `_vYYYYMMDDHHMMSS` suffix
- **Sidecar metadata**: Every CSV has corresponding `.meta.json` with:
  ```json
  {
    "symbol": "GME",
    "asset_type": "stock", 
    "source": "yfinance",
    "total_records": 1250,
    "collection_timestamp": "2025-08-12T15:30:00+00:00",
    "version": "20250812153000",
    "date_range": "2020-12-01 to 2023-12-31",
    "checksum_sha256": "a1b2c3d4...",
    "notes": "use_polygon_fallback=true"
  }
  ```

### Dataset Index
`data/INDEX.jsonl` tracks all datasets with one JSON record per line:
```json
{"symbol": "GME", "asset_type": "stock", "file_path": "data/raw/stocks/GME_stock_data_v20250812153000.csv", "version": "20250812153000", "checksum": "a1b2c3d4"}
```

## Individual Pipeline Steps

### Step 1: Price Collection
```bash
make prices
```
- **Stocks**: yfinance → Polygon.io fallback for delisted
- **Crypto**: yfinance with automatic start date detection
- **Output**: `data/raw/stocks/` and `data/raw/crypto/`

### Step 2: Reddit Collection  
```bash
make reddit
```
- **Source**: BigQuery public Reddit dataset
- **Aggregation**: Daily post/comment counts, scores, engagement
- **Filtering**: Optional keyword matching (asset symbols)
- **Output**: `data/raw/reddit/`

### Step 3: Validation & Alignment
```bash
make validate
```
- **Alignment**: Merge price and Reddit data on UTC dates
- **Validation**: Check duplicates, missing dates, outliers, bounds
- **Correlation**: Compute engagement-return correlations (lag 0/1/2)
- **Output**: `data/processed/` + `reports/data_quality_*.md`

## Troubleshooting

### Common Issues

**BigQuery authentication failed**
```bash
❌ ERROR: BigQuery client initialization failed
```
**Solution**: 
- Verify `GCP_PROJECT` is set: `echo $GCP_PROJECT`
- Check authentication: `gcloud auth application-default login`
- Or set service account: `export GOOGLE_APPLICATION_CREDENTIALS="/path/key.json"`

**API rate limiting (429 errors)**
```bash
⚠️ WARNING: Rate limit hit, retrying in 60s...
```
**Solution**: Built-in retry/backoff handles this automatically. Just wait or rerun.

**Empty BBBY data (delisted stock)**
```bash
❌ ERROR: No data returned for BBBY from yfinance
```
**Solution**: 
- Set `POLYGON_API_KEY` for delisted stock fallback
- Or expect empty data with metadata note: `delisted_no_polygon_key`

**Validation gate failures**
```bash
❌ ERROR: Found 5 duplicate dates in GME data
```
**Solution**: Check detailed logs in `logs/YYYYMMDD.log` and data quality report in `reports/`

**Missing Python packages**
```bash
ModuleNotFoundError: No module named 'yfinance'
```
**Solution**: `pip install -r requirements.txt`

### Data Quality Issues

**High missing dates count**
- **Stock data**: Normal for weekends/holidays (business days only)
- **Crypto data**: Should be continuous (calendar days)
- **Check**: `reports/data_quality_*.md` for details

**Zero engagement correlation**
- **Cause**: No Reddit data found for asset or date range mismatch
- **Check**: Verify asset is in `config/asset_reddit_map.yaml`
- **Check**: Date ranges overlap between price and Reddit data

**Outlier detection alerts**
- **Volume outliers**: Z-score > 6 for trading volume
- **Engagement outliers**: Z-score > 6 for Reddit activity
- **Action**: Review in data quality report; often legitimate extreme days

## Reproducibility & Logging

### Execution Logs
All pipeline runs create structured logs in `logs/YYYYMMDD.log`:
```
2025-08-12 15:30:00 - collect_prices - INFO - [RUN] collector=stocks tickers=GME,AMC start=2021-01-01 end=2021-12-31
2025-08-12 15:30:15 - collect_prices - INFO - [WRITE] path=data/raw/stocks/GME_stock_data.csv rows=252 checksum=a1b2c3d4 version=20250812153000
2025-08-12 15:30:15 - collect_prices - INFO - [VALIDATION] symbol=GME no_duplicates=True monotonic_dates=True non_negative_volume=True
```

### Dataset Tracking
Every dataset write appends to `data/INDEX.jsonl` for complete audit trail.

### Reproducibility
- **Idempotent**: Rerunning targets with same parameters produces identical outputs
- **Versioned**: Raw files never overwritten; processed files versioned by timestamp
- **Atomic writes**: Partial files avoided via temporary write + rename

## Safety Features

### Data Preservation
- **Raw data**: Never deleted or overwritten (`make clean` preserves `data/raw/`)
- **Versioning**: Multiple collections create new versioned files
- **Atomic operations**: Failed writes don't corrupt existing data

### Validation Gates
- **Schema compliance**: Column presence, types, ordering
- **Data integrity**: Duplicates, missing dates, negative values
- **Business logic**: Price bounds (close between low/high)
- **Continuity**: Expected trading days for stocks, calendar days for crypto

### Error Handling
- **API failures**: Retry with exponential backoff
- **Data validation**: Fail fast with clear error messages  
- **Partial failures**: Continue processing other assets, report failures

## Make Targets Reference

```bash
make help          # Show all available targets
make check-env     # Validate environment setup
make all           # Full pipeline: prices -> reddit -> validate
make prices        # Collect stock & crypto prices only
make reddit        # Collect Reddit data only  
make validate      # Validate & align data only
make backfill      # Add metadata to existing raw files
make smoke         # Fast 3-month test run
make clean         # Remove processed files (preserve raw)
make lint          # Python code linting
make fmt           # Python code formatting
```

## Next Steps

After successful pipeline execution:

1. **Explore data quality reports** in `reports/data_quality_*.md`
2. **Analyze processed datasets** in `data/processed/panel/`
3. **Build ML models** using aligned price-social features
4. **Schedule regular updates** via cron/GitHub Actions
5. **Monitor logs** for ongoing data quality

For advanced modeling, consider the aligned panel datasets with features:
- `return_pct`, `log_return` (price movements)
- `total_engagement`, `posts`, `comments` (social activity)  
- `is_weekend`, `is_market_open` (temporal features)
- Lag correlations for predictive modeling