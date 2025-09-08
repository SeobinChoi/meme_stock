.PHONY: check-env reddit

check-env:
	@python -c "from google.cloud import bigquery; print('✅ BigQuery client import OK')" && \
	python -c "import os; print('GCP_PROJECT=', os.environ.get('GCP_PROJECT','<unset>'))"

# Usage:
# make reddit START=2021-01-01 END=2023-12-31 SUBREDDITS="wallstreetbets stocks GME amcstock"
reddit:
	@mkdir -p data/raw/reddit
	python scripts/collect_reddit_bq.py \
		--project $${GCP_PROJECT:?set GCP_PROJECT} \
		--start $(START) --end $(END) \
		--subreddits $(SUBREDDITS) \
		--keyword-pattern '(?i)\b(GME|AMC|BB|KOSS|BBBY|DOGE|SHIB|PEPE|BONK)\b' \
		--output-dir data/raw/reddit \
		--prefix keywords

# Meme Stock Data Pipeline Orchestration
# Automates Steps 1-8: collect prices/reddit -> validate -> processed datasets + reports

# Shell configuration
SHELL := /bin/bash
.ONESHELL:
.SILENT:

# Python environment
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1

# Default date range (configurable)
START ?= 2020-12-01
END ?= 2023-12-31

# Asset configuration
STOCKS ?= GME AMC BB KOSS BBBY
CRYPTO ?= DOGE SHIB PEPE BONK
SUBREDDITS ?= wallstreetbets stocks BBBY GME amcstock cryptocurrency dogecoin SHIBArmy pepecoin bonk

# API configuration
USE_KEYWORDS ?= true
GCP_PROJECT ?= $(shell echo $$GCP_PROJECT)
VS ?= usd

# Required but optional env vars (warn if missing)
POLYGON_API_KEY ?= $(shell echo $$POLYGON_API_KEY)
GOOGLE_APPLICATION_CREDENTIALS ?= $(shell echo $$GOOGLE_APPLICATION_CREDENTIALS)

# Phony targets
.PHONY: all prices reddit validate backfill smoke clean lint fmt help check-env

# Default target
all: check-env prices reddit validate

help:
	echo "Meme Stock Data Pipeline - Make Targets:"
	echo ""
	echo "  📊 Main Pipeline:"
	echo "    all        - Run complete pipeline: prices -> reddit -> validate"
	echo "    prices     - Collect stock & crypto prices (yfinance + Polygon fallback)"
	echo "    reddit     - Collect Reddit daily aggregates via BigQuery"
	echo "    validate   - Align price & Reddit data, generate processed datasets + reports"
	echo ""
	echo "  🔧 Utilities:"
	echo "    backfill   - Backfill metadata for existing raw CSV files"
	echo "    smoke      - End-to-end smoke test (3-month window, subset of assets)"
	echo "    clean      - Remove processed files and reports (raw data preserved)"
	echo "    check-env  - Validate required environment variables"
	echo ""
	echo "  📋 Configuration (environment variables):"
	echo "    START=2020-12-01       - Start date (YYYY-MM-DD)"
	echo "    END=2023-12-31         - End date (YYYY-MM-DD)" 
	echo "    STOCKS='GME AMC BB KOSS BBBY'  - Stock symbols"
	echo "    CRYPTO='DOGE SHIB PEPE BONK'   - Crypto symbols"
	echo "    USE_KEYWORDS=true      - Use keyword filtering for Reddit"
	echo "    GCP_PROJECT=project-id - Google Cloud Project ID (required)"
	echo "    POLYGON_API_KEY=...    - Polygon.io API key (optional, for delisted)"
	echo ""
	echo "  📖 Quick Start:"
	echo "    export GCP_PROJECT=your-project-id"
	echo "    make smoke    # Fast test (3 months)"
	echo "    make all      # Full pipeline"

check-env:
	echo "🔍 Checking environment configuration..."
	if ! command -v python >/dev/null 2>&1; then \
		echo "❌ ERROR: Python not found in PATH"; \
		exit 1; \
	fi
	if ! python -c "import yfinance, pandas, yaml" >/dev/null 2>&1; then \
		echo "❌ ERROR: Missing required Python packages. Run: pip install -r requirements.txt"; \
		exit 1; \
	fi
	if [ -z "$(GCP_PROJECT)" ]; then \
		echo "⚠️  WARNING: GCP_PROJECT not set. Reddit collection will fail."; \
		echo "   Set with: export GCP_PROJECT=your-project-id"; \
	else \
		echo "✅ GCP_PROJECT: $(GCP_PROJECT)"; \
	fi
	if [ -z "$(GOOGLE_APPLICATION_CREDENTIALS)" ] && ! gcloud auth application-default print-access-token >/dev/null 2>&1; then \
		echo "⚠️  WARNING: BigQuery authentication not configured."; \
		echo "   Either set GOOGLE_APPLICATION_CREDENTIALS or run: gcloud auth application-default login"; \
	else \
		echo "✅ BigQuery authentication configured"; \
	fi
	if [ -z "$(POLYGON_API_KEY)" ]; then \
		echo "⚠️  INFO: POLYGON_API_KEY not set. Delisted stocks (BBBY) may fail."; \
	else \
		echo "✅ Polygon API key configured"; \
	fi
	echo "✅ Environment check complete"

prices:
	echo "📈 Collecting price data..."
	echo "  Stocks: $(STOCKS)"
	echo "  Date range: $(START) to $(END)"
	echo ""
	
	# Collect stock prices
	python collect_prices.py stocks \
		--tickers $(STOCKS) \
		--start $(START) \
		--end $(END) \
		--use-polygon-fallback true
	
	# TODO: Crypto collection not yet implemented
	# python collect_prices.py crypto \
	# 	--symbols $(CRYPTO) \
	# 	--vs-currency $(VS) \
	# 	--start auto \
	# 	--end $(END)
	
	echo "✅ Price collection complete (stocks only - crypto TODO)"

reddit:
	echo "💬 Collecting Reddit data..."
	echo "  Subreddits: $(SUBREDDITS)"
	echo "  Date range: $(START) to $(END)"
	echo "  Keywords: $(USE_KEYWORDS)"
	echo "  GCP Project: $(GCP_PROJECT)"
	echo ""
	
	if [ -z "$(GCP_PROJECT)" ]; then
		echo "❌ ERROR: GCP_PROJECT must be set for Reddit collection"
		exit 1
	fi
	
	python collect_reddit.py \
		--subreddits $(SUBREDDITS) \
		--start $(START) \
		--end $(END) \
		--use-keywords $(USE_KEYWORDS) \
		--project $(GCP_PROJECT) \
		--sql-dir sql \
		--output-dir data/raw/reddit \
		--dry-run false
	
	echo "✅ Reddit collection complete"

validate:
	echo "🔍 Validating and aligning data..."
	echo "  Stocks: $(STOCKS)"
	echo "  Crypto: $(CRYPTO)"
	echo "  Date range: $(START) to $(END)"
	echo ""
	
	python validate_and_align.py \
		--stocks $(STOCKS) \
		--crypto $(CRYPTO) \
		--subreddit-map config/asset_reddit_map.yaml \
		--reddit-use-keywords $(USE_KEYWORDS) \
		--start $(START) \
		--end $(END) \
		--output-dir data/processed \
		--report-dir reports \
		--write-parquet false \
		--write-csv true \
		--compute-correlations true \
		--rolling 7 14 30 \
		--merge-policy left_on_price
	
	echo "✅ Validation and alignment complete"
	echo "📊 Check reports/ for data quality reports"

backfill:
	echo "🔄 Backfilling metadata for existing files..."
	python backfill_metadata.py
	echo "✅ Metadata backfill complete"

smoke: 
	echo "🚀 Running end-to-end smoke test..."
	echo "  Testing subset: GME, BBBY (stocks only - crypto TODO)"
	echo "  Date range: 2021-01-01 to 2021-03-31 (3 months)"
	echo "  Subreddits: wallstreetbets, GME"
	echo ""
	
	# Create smoke test targets with narrow scope
	$(MAKE) prices \
		START=2021-01-01 \
		END=2021-03-31 \
		STOCKS="GME BBBY"
	
	# Skip Reddit collection (requires BigQuery setup)
	echo "⚠️  Skipping Reddit collection (requires GCP_PROJECT setup)"
	# $(MAKE) reddit \
	# 	START=2021-01-01 \
	# 	END=2021-03-31 \
	# 	SUBREDDITS="wallstreetbets GME" \
	# 	USE_KEYWORDS=true
	
	# Test validation with existing mock data
	echo "🔍 Testing validation with mock Reddit data..."
	$(MAKE) validate \
		START=2021-01-01 \
		END=2021-03-31 \
		STOCKS="GME" \
		CRYPTO=""
	
	echo ""
	echo "🎉 Smoke test complete! Check outputs:"
	echo "  📁 data/raw/stocks/     - Price data CSVs + metadata"
	echo "  📁 data/processed/      - Aligned datasets (stocks/, panel/)"
	echo "  📁 reports/             - Data quality reports (markdown)"
	echo "  📁 logs/                - Structured execution logs"
	echo "  📄 data/INDEX.jsonl     - Complete dataset index (if backfilled)"

clean:
	echo "🧹 Cleaning processed files and reports..."
	echo "  (Raw data in data/raw/ will be preserved)"
	
	# Remove processed files but preserve directory structure
	rm -rf data/processed/* 2>/dev/null || true
	rm -rf reports/* 2>/dev/null || true
	rm -rf logs/* 2>/dev/null || true
	
	# Recreate directories
	mkdir -p data/processed/stocks data/processed/crypto data/processed/panel
	mkdir -p reports logs
	
	echo "✅ Clean complete - ready for fresh run"

# Development targets (optional)
lint:
	echo "🔍 Running Python linting..."
	if command -v ruff >/dev/null 2>&1; then
		ruff check .
	elif command -v flake8 >/dev/null 2>&1; then
		flake8 --max-line-length=120 *.py common/
	else
		echo "⚠️  No linter found. Install ruff or flake8."
	fi

fmt:
	echo "🎨 Formatting Python code..."
	if command -v ruff >/dev/null 2>&1; then
		ruff format .
	elif command -v black >/dev/null 2>&1; then
		black --line-length=120 *.py common/
	else
		echo "⚠️  No formatter found. Install ruff or black."
	fi