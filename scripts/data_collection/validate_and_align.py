#!/usr/bin/env python3
"""
Validation & Alignment Pipeline

Aligns price data (stocks/crypto) with Reddit data, validates quality,
and produces processed datasets with comprehensive quality reporting.

Usage:
    python validate_and_align.py \
      --stocks GME AMC BB KOSS BBBY \
      --crypto DOGE SHIB PEPE BONK \
      --subreddit-map config/asset_reddit_map.yaml \
      --reddit-use-keywords true \
      --start 2020-12-01 --end 2023-12-31 \
      --output-dir data/processed \
      --report-dir reports \
      --write-parquet true \
      --write-csv true \
      --compute-correlations true \
      --rolling 7 14 30 \
      --merge-policy left_on_price
"""

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import warnings

# Add common package to path
sys.path.append(str(Path(__file__).parent))

from common.logging_utils import get_logger, log_run_start, log_validation, log_write
from common.paths import ensure_dirs_exist, dir_raw_stocks, dir_raw_crypto, dir_raw_reddit
from common.time_utils import now_utc_iso, parse_any_ts, count_missing_business_days, count_missing_calendar_days
from common.validation import validate_stock_data, validate_crypto_data, validate_reddit_data, ValidationError
from common.metadata import build_metadata, write_metadata
from common.io_utils import safe_write_versioned, latest_version_path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class ValidationAndAlignmentPipeline:
    """Pipeline for validating and aligning price and Reddit data."""
    
    def __init__(self, subreddit_map_path: str, reddit_use_keywords: bool = False,
                 output_dir: str = "data/processed", report_dir: str = "reports",
                 write_parquet: bool = True, write_csv: bool = True,
                 compute_correlations: bool = True, merge_policy: str = "left_on_price",
                 log_level: str = "INFO"):
        
        self.logger = get_logger(__name__, log_level)
        self.reddit_use_keywords = reddit_use_keywords
        self.output_dir = Path(output_dir)
        self.report_dir = Path(report_dir)
        self.write_parquet = write_parquet
        self.write_csv = write_csv
        self.compute_correlations = compute_correlations
        self.merge_policy = merge_policy
        
        # Ensure directories exist
        ensure_dirs_exist()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "stocks").mkdir(exist_ok=True)
        (self.output_dir / "crypto").mkdir(exist_ok=True)
        (self.output_dir / "panel").mkdir(exist_ok=True)
        
        # Load subreddit mapping
        self.subreddit_map = self._load_subreddit_map(subreddit_map_path)
        
        # Initialize report data
        self.report_data = {
            'run_info': {},
            'assets': {},
            'warnings': [],
            'errors': []
        }
    
    def _load_subreddit_map(self, map_path: str) -> Dict[str, Any]:
        """Load asset to subreddit mapping from YAML file."""
        try:
            with open(map_path, 'r') as f:
                mapping = yaml.safe_load(f)
                self.logger.info(f"Loaded subreddit mapping from {map_path}")
                return mapping
        except Exception as e:
            self.logger.error(f"Failed to load subreddit mapping: {e}")
            return {"stocks": {}, "crypto": {}, "defaults": {"stocks": [], "crypto": []}}
    
    def get_asset_subreddits(self, asset: str, asset_type: str) -> List[str]:
        """Get list of subreddits for an asset."""
        asset_map = self.subreddit_map.get(asset_type, {})
        
        # Try exact match first
        if asset in asset_map:
            return asset_map[asset]
        
        # Try case-insensitive match
        for key, value in asset_map.items():
            if key.upper() == asset.upper():
                return value
        
        # Fall back to defaults
        defaults = self.subreddit_map.get("defaults", {})
        return defaults.get(asset_type, [])
    
    def load_price_data(self, asset: str, asset_type: str) -> Optional[pd.DataFrame]:
        """Load price data for an asset."""
        try:
            if asset_type == "stock":
                base_dir = dir_raw_stocks()
                pattern = f"{asset}_stock_data.csv"
            else:  # crypto
                base_dir = dir_raw_crypto()
                pattern = f"{asset}_crypto_data.csv"
            
            # Find latest version of the file
            base_path = base_dir / pattern
            latest_path = latest_version_path(base_path)
            
            if not latest_path or not latest_path.exists():
                self.logger.warning(f"Price data not found for {asset}: {base_path}")
                return None
            
            df = pd.read_csv(latest_path)
            self.logger.debug(f"Loaded {len(df)} price records for {asset}")
            
            # Normalize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Parse dates and ensure UTC
            df['date'] = pd.to_datetime(df['date'], utc=True)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load price data for {asset}: {e}")
            return None
    
    def load_reddit_data(self, subreddit: str) -> Optional[pd.DataFrame]:
        """Load Reddit data for a subreddit."""
        try:
            suffix = "_keywords" if self.reddit_use_keywords else ""
            filename = f"reddit_{subreddit.lower()}{suffix}.csv"
            file_path = dir_raw_reddit() / filename
            
            # Try to find latest version
            latest_path = latest_version_path(file_path)
            
            if not latest_path or not latest_path.exists():
                self.logger.warning(f"Reddit data not found for r/{subreddit}: {file_path}")
                return None
            
            df = pd.read_csv(latest_path)
            self.logger.debug(f"Loaded {len(df)} Reddit records for r/{subreddit}")
            
            # Parse dates and ensure UTC (keep as date strings for consistency)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load Reddit data for r/{subreddit}: {e}")
            return None
    
    def aggregate_reddit_for_asset(self, asset: str, asset_type: str) -> Optional[pd.DataFrame]:
        """Aggregate Reddit data across subreddits for an asset."""
        subreddits = self.get_asset_subreddits(asset, asset_type)
        
        if not subreddits:
            self.logger.warning(f"No subreddits configured for {asset}")
            return None
        
        reddit_dfs = []
        loaded_subreddits = []
        
        # Load data from all configured subreddits
        for subreddit in subreddits:
            df = self.load_reddit_data(subreddit)
            if df is not None:
                df['subreddit'] = subreddit  # Track source
                reddit_dfs.append(df)
                loaded_subreddits.append(subreddit)
        
        if not reddit_dfs:
            self.logger.warning(f"No Reddit data found for {asset} in subreddits {subreddits}")
            return None
        
        self.logger.info(f"Aggregating Reddit data for {asset} from {len(loaded_subreddits)} subreddits: {loaded_subreddits}")
        
        # Combine all subreddit data
        combined = pd.concat(reddit_dfs, ignore_index=True)
        
        # Aggregate by date - sum across subreddits
        reddit_cols = ['posts', 'comments', 'score', 'total_engagement']
        agg_dict = {col: 'sum' for col in reddit_cols if col in combined.columns}
        agg_dict['is_weekend'] = 'first'  # Take first value (should be same across subreddits)
        
        aggregated = combined.groupby('date').agg(agg_dict).reset_index()
        
        # Fill missing columns with 0 if needed
        for col in reddit_cols:
            if col not in aggregated.columns:
                aggregated[col] = 0
        
        # Ensure column order
        final_cols = ['date'] + reddit_cols
        if 'is_weekend' in aggregated.columns:
            final_cols.append('is_weekend')
        
        return aggregated[final_cols].sort_values('date').reset_index(drop=True)
    
    def align_and_merge(self, price_df: pd.DataFrame, reddit_df: pd.DataFrame, 
                       asset: str, asset_type: str) -> pd.DataFrame:
        """Align and merge price and Reddit data."""
        
        # Convert price dates to string format for joining
        price_df = price_df.copy()
        price_df['date_str'] = price_df['date'].dt.strftime('%Y-%m-%d')
        
        if self.merge_policy == "left_on_price":
            # Keep only price trading days
            merged = price_df.merge(reddit_df, left_on='date_str', right_on='date', how='left', suffixes=('', '_reddit'))
            
            # Fill missing Reddit data with 0
            reddit_cols = ['posts', 'comments', 'score', 'total_engagement']
            for col in reddit_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0).astype('int64')
            
            # Handle is_weekend
            if 'is_weekend' in merged.columns:
                merged['is_weekend'] = merged['is_weekend'].fillna(0).astype('int64')
            
        elif self.merge_policy == "full_outer":
            # Keep all days from either source
            merged = price_df.merge(reddit_df, left_on='date_str', right_on='date', how='outer', suffixes=('', '_reddit'))
            
            # Fill Reddit fields with 0 where missing
            reddit_cols = ['posts', 'comments', 'score', 'total_engagement']
            for col in reddit_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0).astype('int64')
        
        else:
            raise ValueError(f"Unknown merge policy: {self.merge_policy}")
        
        # Add derived fields
        if 'close' in merged.columns:
            merged['return_pct'] = merged['close'].pct_change() * 100
            merged['log_return'] = np.log(merged['close']) - np.log(merged['close'].shift(1))
        
        # Add market status
        if asset_type == "stock":
            merged['is_market_open'] = 1  # All stock data represents trading days
        
        # Clean up columns
        cols_to_drop = ['date_str', 'date_reddit'] 
        merged = merged.drop(columns=[col for col in cols_to_drop if col in merged.columns])
        
        # Ensure proper column order: price fields first, then reddit, then derived
        price_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        reddit_cols = ['posts', 'comments', 'score', 'total_engagement', 'is_weekend']
        derived_cols = ['return_pct', 'log_return']
        
        if asset_type == "stock":
            derived_cols.append('is_market_open')
        
        final_cols = []
        for col_group in [price_cols, reddit_cols, derived_cols]:
            for col in col_group:
                if col in merged.columns:
                    final_cols.append(col)
        
        merged = merged[final_cols].sort_values('date').reset_index(drop=True)
        
        self.logger.info(f"Merged {asset}: {len(merged)} rows, merge_policy={self.merge_policy}")
        return merged
    
    def validate_aligned_data(self, df: pd.DataFrame, asset: str, asset_type: str) -> Dict[str, Any]:
        """Validate aligned data and compute quality metrics."""
        validation_result = {
            'asset': asset,
            'asset_type': asset_type,
            'total_rows': len(df),
            'date_min': df['date'].min().strftime('%Y-%m-%d') if len(df) > 0 else 'N/A',
            'date_max': df['date'].max().strftime('%Y-%m-%d') if len(df) > 0 else 'N/A',
            'duplicates': 0,
            'missing_dates': 0,
            'outliers_volume': 0,
            'outliers_engagement': 0,
            'validation_passed': True,
            'errors': []
        }
        
        if len(df) == 0:
            validation_result['validation_passed'] = False
            validation_result['errors'].append("Empty dataset")
            return validation_result
        
        try:
            # Check duplicates
            duplicates = df['date'].duplicated().sum()
            validation_result['duplicates'] = int(duplicates)
            
            if duplicates > 0:
                validation_result['validation_passed'] = False
                validation_result['errors'].append(f"Found {duplicates} duplicate dates")
            
            # Check date continuity
            dates = pd.to_datetime(df['date'])
            if asset_type == "stock":
                # Business days continuity
                expected_days = len(pd.bdate_range(dates.min(), dates.max()))
                missing = expected_days - len(df)
            else:
                # Calendar days continuity
                expected_days = (dates.max() - dates.min()).days + 1
                missing = expected_days - len(df)
            
            validation_result['missing_dates'] = max(0, missing)
            
            # Outlier detection
            if 'volume' in df.columns:
                volume_outliers = self.detect_outliers(df['volume'], method='z_score', threshold=6)
                validation_result['outliers_volume'] = int(volume_outliers.sum())
            
            if 'total_engagement' in df.columns:
                engagement_outliers = self.detect_outliers(df['total_engagement'], method='z_score', threshold=6)
                validation_result['outliers_engagement'] = int(engagement_outliers.sum())
            
            # Price bounds validation
            if 'low' in df.columns and 'high' in df.columns and 'close' in df.columns:
                bounds_violations = ((df['close'] < df['low']) | (df['close'] > df['high'])).sum()
                if bounds_violations > 0:
                    validation_result['validation_passed'] = False
                    validation_result['errors'].append(f"Found {bounds_violations} price bounds violations")
            
            # Non-negative validation
            numeric_cols = ['volume', 'posts', 'comments', 'score', 'total_engagement']
            for col in numeric_cols:
                if col in df.columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        validation_result['validation_passed'] = False
                        validation_result['errors'].append(f"Found {negative_count} negative values in {col}")
            
        except Exception as e:
            validation_result['validation_passed'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def detect_outliers(self, series: pd.Series, method: str = 'z_score', threshold: float = 6) -> pd.Series:
        """Detect outliers using Z-score or MAD method."""
        if method == 'z_score':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold
        elif method == 'mad':
            median = series.median()
            mad = np.median(np.abs(series - median))
            mad_scores = np.abs(series - median) / (mad * 1.4826)  # Scale factor for normal distribution
            return mad_scores > threshold
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def compute_correlations(self, df: pd.DataFrame, asset: str) -> Dict[str, float]:
        """Compute correlations between engagement and returns."""
        correlations = {}
        
        required_cols = ['total_engagement', 'return_pct']
        if not ('total_engagement' in df.columns and 'return_pct' in df.columns):
            return correlations
        
        # Remove NaN values for correlation calculation
        clean_df = df[['total_engagement', 'return_pct']].dropna()
        
        if len(clean_df) < 10:  # Need minimum data points
            return correlations
        
        try:
            # Lag 0: same day correlation
            lag_0_corr = clean_df['total_engagement'].corr(clean_df['return_pct'])
            correlations['lag_0'] = float(lag_0_corr) if pd.notna(lag_0_corr) else 0.0
            
            # Lag +1: engagement predicting next day return
            if len(clean_df) > 1:
                engagement = clean_df['total_engagement'].iloc[:-1].reset_index(drop=True)
                next_return = clean_df['return_pct'].iloc[1:].reset_index(drop=True)
                lag_1_corr = engagement.corr(next_return)
                correlations['lag_1'] = float(lag_1_corr) if pd.notna(lag_1_corr) else 0.0
            
            # Lag +2: engagement predicting return 2 days later
            if len(clean_df) > 2:
                engagement = clean_df['total_engagement'].iloc[:-2].reset_index(drop=True)
                future_return = clean_df['return_pct'].iloc[2:].reset_index(drop=True)
                lag_2_corr = engagement.corr(future_return)
                correlations['lag_2'] = float(lag_2_corr) if pd.notna(lag_2_corr) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to compute correlations for {asset}: {e}")
        
        return correlations
    
    def write_processed_file(self, df: pd.DataFrame, asset: str, asset_type: str, 
                           suffix: str = "_aligned") -> Tuple[List[Path], Dict[str, Any]]:
        """Write processed data to CSV and/or Parquet files."""
        written_files = []
        
        # Determine output directory and filename base
        if asset_type == "stock":
            output_dir = self.output_dir / "stocks"
        else:
            output_dir = self.output_dir / "crypto"
        
        base_name = f"{asset}{suffix}"
        
        # Build metadata
        metadata = build_metadata(
            symbol=asset,
            asset_type=asset_type,
            source="alignment",
            df=df,
            notes=f"stage=processed/aligned; merge_policy={self.merge_policy}; reddit_use_keywords={self.reddit_use_keywords}"
        )
        
        # Write CSV if requested
        if self.write_csv:
            csv_path = output_dir / f"{base_name}.csv"
            final_csv_path, csv_metadata = safe_write_versioned(df, csv_path, metadata.copy())
            meta_path = write_metadata(csv_metadata, final_csv_path)
            written_files.append(final_csv_path)
            
            log_write(
                self.logger,
                final_csv_path,
                len(df),
                csv_metadata.get("checksum_sha256", "")[:8],
                csv_metadata.get("version", "")
            )
        
        # Write Parquet if requested
        if self.write_parquet:
            parquet_path = output_dir / f"{base_name}.parquet"
            
            # Handle versioning manually for Parquet (safe_write_versioned is CSV-specific)
            if parquet_path.exists():
                from common.time_utils import generate_version_timestamp
                version = generate_version_timestamp()
                from common.paths import versioned_filename
                final_parquet_path = versioned_filename(parquet_path, version)
                metadata['version'] = version
            else:
                final_parquet_path = parquet_path
            
            # Write Parquet file
            df.to_parquet(final_parquet_path, index=False)
            
            # Write metadata for Parquet
            parquet_metadata = metadata.copy()
            from common.io_utils import compute_sha256
            parquet_metadata["checksum_sha256"] = compute_sha256(final_parquet_path)
            
            parquet_meta_path = write_metadata(parquet_metadata, final_parquet_path)
            written_files.append(final_parquet_path)
            
            log_write(
                self.logger,
                final_parquet_path,
                len(df),
                parquet_metadata.get("checksum_sha256", "")[:8],
                parquet_metadata.get("version", "")
            )
        
        return written_files, metadata
    
    def create_panel_datasets(self, processed_assets: Dict[str, pd.DataFrame]):
        """Create panel datasets by stacking individual asset data."""
        
        for asset_type in ['stock', 'crypto']:
            assets_of_type = {asset: df for asset, df in processed_assets.items() 
                            if asset_type in self.report_data['assets'].get(asset, {}).get('asset_type', '')}
            
            if not assets_of_type:
                continue
            
            # Add symbol column and stack
            panel_dfs = []
            for asset, df in assets_of_type.items():
                df_with_symbol = df.copy()
                df_with_symbol.insert(0, 'symbol', asset)
                panel_dfs.append(df_with_symbol)
            
            panel_df = pd.concat(panel_dfs, ignore_index=True)
            panel_df = panel_df.sort_values(['symbol', 'date']).reset_index(drop=True)
            
            # Write panel file
            panel_dir = self.output_dir / "panel"
            base_name = f"{asset_type}s_panel"  # stocks_panel or crypto_panel
            
            metadata = build_metadata(
                symbol=f"{asset_type}_panel",
                asset_type="panel",
                source="alignment",
                df=panel_df,
                notes=f"panel_dataset; asset_type={asset_type}; assets={list(assets_of_type.keys())}"
            )
            
            if self.write_parquet:
                parquet_path = panel_dir / f"{base_name}.parquet"
                panel_df.to_parquet(parquet_path, index=False)
                write_metadata(metadata, parquet_path)
                self.logger.info(f"Created panel dataset: {parquet_path}")
            
            if self.write_csv:
                csv_path = panel_dir / f"{base_name}.csv"
                panel_df.to_csv(csv_path, index=False)
                write_metadata(metadata, csv_path)
                self.logger.info(f"Created panel dataset: {csv_path}")
    
    def generate_data_quality_report(self, start_date: str, end_date: str, 
                                   stocks: List[str], crypto: List[str]) -> str:
        """Generate markdown data quality report."""
        
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        report_path = self.report_dir / f"data_quality_{timestamp[:8]}.md"
        
        # Build report content
        report_lines = []
        
        # Header
        report_lines.extend([
            f"# Data Quality Report",
            f"",
            f"**Generated:** {now_utc_iso()}  ",
            f"**Date Range:** {start_date} to {end_date}  ",
            f"**Stocks:** {', '.join(stocks) if stocks else 'None'}  ",
            f"**Crypto:** {', '.join(crypto) if crypto else 'None'}  ",
            f"**Reddit Keywords:** {'Yes' if self.reddit_use_keywords else 'No'}  ",
            f"**Merge Policy:** {self.merge_policy}  ",
            f"",
            f"---",
            f""
        ])
        
        # Asset Summary Table
        if self.report_data['assets']:
            report_lines.extend([
                "## Asset Summary",
                "",
                "| Asset | Type | Rows | Date Range | Missing | Duplicates | Vol Outliers | Eng Outliers | Status |",
                "|-------|------|------|-------------|---------|------------|--------------|--------------|--------|"
            ])
            
            for asset, data in self.report_data['assets'].items():
                validation = data.get('validation', {})
                status = "✅ Pass" if validation.get('validation_passed', False) else "❌ Fail"
                
                report_lines.append(
                    f"| {asset} | {validation.get('asset_type', 'N/A')} | "
                    f"{validation.get('total_rows', 0)} | "
                    f"{validation.get('date_min', 'N/A')} - {validation.get('date_max', 'N/A')} | "
                    f"{validation.get('missing_dates', 0)} | "
                    f"{validation.get('duplicates', 0)} | "
                    f"{validation.get('outliers_volume', 0)} | "
                    f"{validation.get('outliers_engagement', 0)} | "
                    f"{status} |"
                )
            
            report_lines.extend(["", ""])
        
        # Top Engagement Days
        if self.report_data['assets']:
            report_lines.extend([
                "## Top 10 Days by Total Engagement",
                "",
                "| Asset | Date | Total Engagement | Return % |",
                "|-------|------|------------------|----------|"
            ])
            
            for asset, data in self.report_data['assets'].items():
                top_engagement = data.get('top_engagement', [])
                for day in top_engagement[:5]:  # Top 5 per asset to keep reasonable
                    report_lines.append(
                        f"| {asset} | {day.get('date', 'N/A')} | "
                        f"{day.get('total_engagement', 0):,} | "
                        f"{day.get('return_pct', 0):.2f}% |"
                    )
            
            report_lines.extend(["", ""])
        
        # Correlations
        if self.compute_correlations and self.report_data['assets']:
            report_lines.extend([
                "## Engagement-Return Correlations",
                "",
                "Pearson correlation between Reddit engagement and stock/crypto returns:",
                "",
                "| Asset | Same Day (Lag 0) | Next Day (Lag +1) | 2 Days Later (Lag +2) |",
                "|-------|------------------|--------------------|------------------------|"
            ])
            
            for asset, data in self.report_data['assets'].items():
                correlations = data.get('correlations', {})
                lag_0 = correlations.get('lag_0', 0)
                lag_1 = correlations.get('lag_1', 0)
                lag_2 = correlations.get('lag_2', 0)
                
                report_lines.append(
                    f"| {asset} | {lag_0:.3f} | {lag_1:.3f} | {lag_2:.3f} |"
                )
            
            report_lines.extend(["", ""])
        
        # Warnings and Errors
        if self.report_data['warnings']:
            report_lines.extend([
                "## Warnings",
                ""
            ])
            for warning in self.report_data['warnings']:
                report_lines.append(f"⚠️ {warning}")
            report_lines.extend(["", ""])
        
        if self.report_data['errors']:
            report_lines.extend([
                "## Errors", 
                ""
            ])
            for error in self.report_data['errors']:
                report_lines.append(f"❌ {error}")
            report_lines.extend(["", ""])
        
        # Write report file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Generated data quality report: {report_path}")
        return str(report_path)
    
    def process_assets(self, stocks: List[str], crypto: List[str], 
                      start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Process all assets through the validation and alignment pipeline."""
        
        processed_assets = {}
        
        log_run_start(
            self.logger,
            "validate_and_align",
            stocks=",".join(stocks) if stocks else "none",
            crypto=",".join(crypto) if crypto else "none", 
            start=start_date,
            end=end_date,
            merge_policy=self.merge_policy
        )
        
        # Process stocks
        for stock in stocks:
            try:
                self.logger.info(f"Processing stock: {stock}")
                
                # Load price data
                price_df = self.load_price_data(stock, "stock")
                if price_df is None:
                    self.report_data['warnings'].append(f"No price data found for stock {stock}")
                    continue
                
                # Load and aggregate Reddit data
                reddit_df = self.aggregate_reddit_for_asset(stock, "stocks")
                if reddit_df is None:
                    self.report_data['warnings'].append(f"No Reddit data found for stock {stock}")
                    # Create empty Reddit data frame
                    reddit_df = pd.DataFrame({
                        'date': [],
                        'posts': [],
                        'comments': [],
                        'score': [],
                        'total_engagement': [],
                        'is_weekend': []
                    })
                
                # Align and merge
                aligned_df = self.align_and_merge(price_df, reddit_df, stock, "stock")
                
                # Validate
                validation_result = self.validate_aligned_data(aligned_df, stock, "stock")
                
                # Compute correlations
                correlations = {}
                if self.compute_correlations:
                    correlations = ValidationAndAlignmentPipeline.compute_correlations(self, aligned_df, stock)
                
                # Find top engagement days
                top_engagement = []
                if 'total_engagement' in aligned_df.columns:
                    top_df = aligned_df.nlargest(10, 'total_engagement')
                    top_engagement = [
                        {
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'total_engagement': int(row['total_engagement']),
                            'return_pct': float(row.get('return_pct', 0))
                        }
                        for _, row in top_df.iterrows()
                    ]
                
                # Store results
                processed_assets[stock] = aligned_df
                self.report_data['assets'][stock] = {
                    'asset_type': 'stock',
                    'validation': validation_result,
                    'correlations': correlations,
                    'top_engagement': top_engagement
                }
                
                # Write processed files
                written_files, metadata = self.write_processed_file(aligned_df, stock, "stock")
                
                log_validation(self.logger, stock, **{
                    k: v for k, v in validation_result.items() 
                    if k in ['duplicates', 'missing_dates', 'validation_passed']
                })
                
            except Exception as e:
                error_msg = f"Failed to process stock {stock}: {str(e)}"
                self.logger.error(error_msg)
                self.report_data['errors'].append(error_msg)
        
        # Process crypto (similar to stocks)
        for crypto_asset in crypto:
            try:
                self.logger.info(f"Processing crypto: {crypto_asset}")
                
                # Load price data
                price_df = self.load_price_data(crypto_asset, "crypto")
                if price_df is None:
                    self.report_data['warnings'].append(f"No price data found for crypto {crypto_asset}")
                    continue
                
                # Load and aggregate Reddit data
                reddit_df = self.aggregate_reddit_for_asset(crypto_asset, "crypto")
                if reddit_df is None:
                    self.report_data['warnings'].append(f"No Reddit data found for crypto {crypto_asset}")
                    reddit_df = pd.DataFrame({
                        'date': [],
                        'posts': [],
                        'comments': [],
                        'score': [],
                        'total_engagement': [],
                        'is_weekend': []
                    })
                
                # Align and merge  
                aligned_df = self.align_and_merge(price_df, reddit_df, crypto_asset, "crypto")
                
                # Validate
                validation_result = self.validate_aligned_data(aligned_df, crypto_asset, "crypto")
                
                # Compute correlations
                correlations = {}
                if self.compute_correlations:
                    correlations = ValidationAndAlignmentPipeline.compute_correlations(self, aligned_df, crypto_asset)
                
                # Find top engagement days
                top_engagement = []
                if 'total_engagement' in aligned_df.columns:
                    top_df = aligned_df.nlargest(10, 'total_engagement')
                    top_engagement = [
                        {
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'total_engagement': int(row['total_engagement']),
                            'return_pct': float(row.get('return_pct', 0))
                        }
                        for _, row in top_df.iterrows()
                    ]
                
                # Store results
                processed_assets[crypto_asset] = aligned_df
                self.report_data['assets'][crypto_asset] = {
                    'asset_type': 'crypto', 
                    'validation': validation_result,
                    'correlations': correlations,
                    'top_engagement': top_engagement
                }
                
                # Write processed files
                written_files, metadata = self.write_processed_file(aligned_df, crypto_asset, "crypto")
                
                log_validation(self.logger, crypto_asset, **{
                    k: v for k, v in validation_result.items()
                    if k in ['duplicates', 'missing_dates', 'validation_passed']
                })
                
            except Exception as e:
                error_msg = f"Failed to process crypto {crypto_asset}: {str(e)}"
                self.logger.error(error_msg)
                self.report_data['errors'].append(error_msg)
        
        return processed_assets

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validation & Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_and_align.py --stocks GME AMC --crypto DOGE --start 2021-01-01 --end 2021-12-31
  
  python validate_and_align.py \
    --stocks GME AMC BB KOSS \
    --crypto DOGE SHIB PEPE \
    --subreddit-map config/asset_reddit_map.yaml \
    --reddit-use-keywords true \
    --start 2020-12-01 --end 2023-12-31 \
    --compute-correlations true
        """
    )
    
    parser.add_argument('--stocks', nargs='*', default=[], help='List of stock symbols to process')
    parser.add_argument('--crypto', nargs='*', default=[], help='List of crypto symbols to process')
    parser.add_argument('--subreddit-map', default='config/asset_reddit_map.yaml', help='Path to asset-subreddit mapping file')
    parser.add_argument('--reddit-use-keywords', type=str, choices=['true', 'false'], default='false', help='Use keyword-filtered Reddit data')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data/processed', help='Output directory for processed files')
    parser.add_argument('--report-dir', default='reports', help='Output directory for reports')
    parser.add_argument('--write-parquet', type=str, choices=['true', 'false'], default='true', help='Write Parquet files')
    parser.add_argument('--write-csv', type=str, choices=['true', 'false'], default='true', help='Write CSV files')
    parser.add_argument('--compute-correlations', type=str, choices=['true', 'false'], default='true', help='Compute correlations')
    parser.add_argument('--rolling', nargs='*', type=int, default=[7, 14, 30], help='Rolling window periods')
    parser.add_argument('--merge-policy', choices=['left_on_price', 'full_outer'], default='left_on_price', help='Data merge policy')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Parse boolean arguments
    reddit_use_keywords = args.reddit_use_keywords.lower() == 'true'
    write_parquet = args.write_parquet.lower() == 'true'
    write_csv = args.write_csv.lower() == 'true' 
    compute_correlations = args.compute_correlations.lower() == 'true'
    
    # Validate inputs
    if not args.stocks and not args.crypto:
        parser.error("Must specify at least one stock or crypto asset")
    
    try:
        # Initialize pipeline
        pipeline = ValidationAndAlignmentPipeline(
            subreddit_map_path=args.subreddit_map,
            reddit_use_keywords=reddit_use_keywords,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            write_parquet=write_parquet,
            write_csv=write_csv,
            compute_correlations=compute_correlations,
            merge_policy=args.merge_policy,
            log_level=args.log_level
        )
        
        # Process all assets
        processed_assets = pipeline.process_assets(args.stocks, args.crypto, args.start, args.end)
        
        # Create panel datasets  
        pipeline.create_panel_datasets(processed_assets)
        
        # Generate report
        report_path = pipeline.generate_data_quality_report(args.start, args.end, args.stocks, args.crypto)
        
        # Summary
        success_count = len(processed_assets)
        total_count = len(args.stocks) + len(args.crypto)
        
        if success_count < total_count:
            pipeline.logger.warning(f"Successfully processed {success_count}/{total_count} assets")
            sys.exit(1)
        else:
            pipeline.logger.info(f"✅ Successfully processed all {success_count} assets")
            pipeline.logger.info(f"📊 Data quality report: {report_path}")
            
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()