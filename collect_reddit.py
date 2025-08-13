#!/usr/bin/env python3
"""
Reddit Historical Data Collector
Executes BigQuery SQL templates to collect Reddit posts and comments data.

Usage:
    python collect_reddit.py \
      --subreddits wallstreetbets stocks BBBY GME amcstock cryptocurrency dogecoin SHIBArmy pepecoin bonk \
      --start 2020-12-01 --end 2023-12-31 \
      --use-keywords false \
      --keyword-pattern '(?i)\b(GME|AMC|BB|KOSS|BBBY|DOGE|SHIB|PEPE|BONK)\b' \
      --project <GCP_PROJECT_ID> \
      --sql-dir sql \
      --output-dir data/raw/reddit \
      --dry-run false
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import time
from datetime import datetime, timezone

# Add common package to path
sys.path.append(str(Path(__file__).parent))

from common.logging_utils import get_logger, log_run_start, log_validation, log_write, log_collection_summary
from common.paths import ensure_dirs_exist, dir_raw_reddit
from common.time_utils import now_utc_iso, parse_any_ts, is_weekend
from common.validation import validate_reddit_data, ValidationError
from common.metadata import build_metadata, write_metadata
from common.io_utils import safe_write_versioned, atomic_write_csv

class RedditCollector:
    """Reddit data collector using BigQuery SQL templates."""
    
    def __init__(self, project_id: str, sql_dir: str = "sql", output_dir: str = "data/raw/reddit", 
                 dry_run: bool = False, log_level: str = "INFO"):
        self.logger = get_logger(__name__, log_level)
        self.project_id = project_id
        self.sql_dir = Path(sql_dir)
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        
        # Ensure directories exist
        ensure_dirs_exist()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize BigQuery client
        self._init_bigquery_client()
    
    def _init_bigquery_client(self):
        """Initialize BigQuery client with proper authentication."""
        try:
            from google.cloud import bigquery
            from google.auth.exceptions import DefaultCredentialsError
            
            try:
                self.bq_client = bigquery.Client(project=self.project_id)
                self.logger.info(f"BigQuery client initialized for project: {self.project_id}")
                
                # Test connection with a simple query
                if not self.dry_run:
                    test_query = "SELECT 1 as test"
                    test_job = self.bq_client.query(test_query)
                    list(test_job.result())  # Force execution
                    self.logger.info("BigQuery connection test successful")
                
            except DefaultCredentialsError:
                raise ValueError(
                    "BigQuery authentication failed. Please set GOOGLE_APPLICATION_CREDENTIALS "
                    "environment variable or run 'gcloud auth application-default login'"
                )
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery package not found. Please install: "
                "pip install google-cloud-bigquery"
            )
    
    def load_sql_template(self, template_name: str) -> str:
        """
        Load SQL template from file.
        
        Args:
            template_name: Name of SQL template file
            
        Returns:
            SQL query string
        """
        template_path = self.sql_dir / f"{template_name}.sql"
        
        if not template_path.exists():
            raise FileNotFoundError(f"SQL template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def execute_parameterized_query(self, sql: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute parameterized BigQuery query with retry logic.
        
        Args:
            sql: SQL query string
            params: Dictionary of named parameters
            
        Returns:
            Query results as DataFrame
        """
        from google.cloud import bigquery
        
        # Convert parameters to BigQuery format
        job_config = bigquery.QueryJobConfig()
        
        for param_name, param_value in params.items():
            if isinstance(param_value, str):
                param_type = bigquery.enums.SqlTypeNames.STRING
            elif isinstance(param_value, int):
                param_type = bigquery.enums.SqlTypeNames.INT64
            elif isinstance(param_value, (datetime, pd.Timestamp)):
                param_type = bigquery.enums.SqlTypeNames.DATE
                param_value = param_value.strftime('%Y-%m-%d') if hasattr(param_value, 'strftime') else str(param_value)
            else:
                param_type = bigquery.enums.SqlTypeNames.STRING
                param_value = str(param_value)
            
            job_config.query_parameters.append(
                bigquery.ScalarQueryParameter(param_name, param_type, param_value)
            )
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would execute query with params: {params}")
            # Return empty DataFrame with expected columns for dry run
            return pd.DataFrame(columns=['date'])
        
        # Execute with retry logic
        max_retries = 5
        base_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Executing BigQuery (attempt {attempt + 1}/{max_retries})")
                
                query_job = self.bq_client.query(sql, job_config=job_config)
                df = query_job.to_dataframe()
                
                self.logger.debug(f"Query returned {len(df)} rows")
                return df
                
            except Exception as e:
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"Rate limit hit, waiting {delay}s before retry")
                        time.sleep(delay)
                        continue
                
                self.logger.error(f"BigQuery execution failed: {e}")
                raise
        
        raise RuntimeError(f"Query failed after {max_retries} attempts")
    
    def create_date_spine(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create a complete UTC date spine for the given range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with date column
        """
        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq='D',
            tz='UTC'
        )
        
        # Convert to date strings to match BigQuery output format
        dates = [d.strftime('%Y-%m-%d') for d in date_range]
        
        return pd.DataFrame({'date': dates})
    
    def compute_output_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute required output fields from joined data.
        
        Args:
            df: DataFrame with posts and comments data joined
            
        Returns:
            DataFrame with computed fields
        """
        # Fill NaN values with 0 for aggregation
        df = df.fillna(0)
        
        # Convert date to datetime for weekend calculation
        df['date_dt'] = pd.to_datetime(df['date'])
        
        # Compute required fields
        result = pd.DataFrame({
            'date': df['date'],  # Keep as string for schema compliance
            'posts': df.get('posts', 0).astype('int64'),
            'comments': df.get('comments', 0).astype('int64'),
            'score': (df.get('score_sum', 0) + df.get('comment_score_sum', 0)).astype('int64'),
            'total_engagement': (
                df.get('score_sum', 0) + 
                df.get('comments_sum', 0) + 
                df.get('comment_score_sum', 0)
            ).astype('int64'),
            'is_weekend': df['date_dt'].apply(lambda x: 1 if is_weekend(x) else 0).astype('int64')
        })
        
        return result[['date', 'posts', 'comments', 'score', 'total_engagement', 'is_weekend']]
    
    def collect_subreddit_data(self, subreddit: str, start_date: str, end_date: str, 
                             use_keywords: bool = False, keyword_pattern: str = "") -> Dict[str, Any]:
        """
        Collect data for a single subreddit.
        
        Args:
            subreddit: Subreddit name
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_keywords: Whether to use keyword filtering
            keyword_pattern: Regex pattern for keyword filtering
            
        Returns:
            Collection result dictionary
        """
        result = {
            "subreddit": subreddit,
            "status": "unknown",
            "records": 0,
            "path": "",
            "error": None
        }
        
        try:
            log_run_start(
                self.logger,
                "reddit",
                subreddit=subreddit,
                start=start_date,
                end=end_date,
                use_keywords=use_keywords
            )
            
            # Determine SQL templates to use
            posts_template = "posts_daily_keywords" if use_keywords else "posts_daily"
            comments_template = "comments_daily_keywords" if use_keywords else "comments_daily"
            
            # Load SQL templates
            posts_sql = self.load_sql_template(posts_template)
            comments_sql = self.load_sql_template(comments_template)
            
            # Prepare parameters
            params = {
                'subreddit': subreddit,
                'start': start_date,
                'end': end_date
            }
            
            if use_keywords:
                params['keyword_pattern'] = keyword_pattern
            
            # Execute queries
            self.logger.info(f"Executing posts query for r/{subreddit}")
            posts_df = self.execute_parameterized_query(posts_sql, params)
            
            self.logger.info(f"Executing comments query for r/{subreddit}")
            comments_df = self.execute_parameterized_query(comments_sql, params)
            
            self.logger.info(f"[BQ] posts_rows={len(posts_df)} comments_rows={len(comments_df)}")
            
            if self.dry_run:
                result.update({
                    "status": "dry_run",
                    "records": 0
                })
                return result
            
            # Create date spine
            date_spine = self.create_date_spine(start_date, end_date)
            
            # Join posts and comments data to spine
            merged = date_spine.merge(posts_df, on='date', how='left')
            merged = merged.merge(comments_df, on='date', how='left')
            
            # Compute output fields
            final_df = self.compute_output_fields(merged)
            
            missing_dates = final_df[final_df['posts'] + final_df['comments'] == 0].shape[0]
            self.logger.info(f"[JOIN] out_rows={len(final_df)} missing_dates={missing_dates}")
            
            # Validate data
            validation_report = validate_reddit_data(final_df, f"r/{subreddit}")
            log_validation(self.logger, f"r/{subreddit}", **validation_report["validations"])
            
            # Build metadata
            suffix = "_keywords" if use_keywords else ""
            notes_parts = [
                f"use_keywords={str(use_keywords).lower()}",
                f"templates='{posts_template}+{comments_template}'"
            ]
            if use_keywords:
                notes_parts.append(f"pattern='{keyword_pattern}'")
            
            metadata = build_metadata(
                symbol=f"r/{subreddit}",
                asset_type="reddit",
                source="bigquery",
                df=final_df,
                date_range=f"{start_date} to {end_date}",
                notes="; ".join(notes_parts)
            )
            
            # Generate output filename
            subreddit_normalized = subreddit.lower()
            filename = f"reddit_{subreddit_normalized}{suffix}.csv"
            output_path = self.output_dir / filename
            
            # Write data using atomic versioned write
            csv_path, updated_metadata = safe_write_versioned(final_df, output_path, metadata)
            
            # Write metadata
            meta_path = write_metadata(updated_metadata, csv_path)
            
            log_write(
                self.logger,
                csv_path,
                len(final_df),
                updated_metadata.get("checksum_sha256", "")[:8],
                updated_metadata.get("version", "")
            )
            
            result.update({
                "status": "success",
                "records": len(final_df),
                "path": str(csv_path),
                "meta_path": str(meta_path)
            })
            
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e)
            })
            self.logger.error(f"Failed to collect r/{subreddit}: {str(e)}")
        
        return result
    
    def collect_multiple_subreddits(self, subreddits: List[str], start_date: str, end_date: str,
                                  use_keywords: bool = False, keyword_pattern: str = "") -> Dict[str, Any]:
        """
        Collect data for multiple subreddits.
        
        Returns:
            Summary dictionary with results
        """
        results = []
        
        for subreddit in subreddits:
            # Collect base data
            base_result = self.collect_subreddit_data(
                subreddit, start_date, end_date, use_keywords=False
            )
            results.append(base_result)
            
            # Collect keyword-filtered data if requested
            if use_keywords and keyword_pattern:
                keyword_result = self.collect_subreddit_data(
                    subreddit, start_date, end_date, 
                    use_keywords=True, keyword_pattern=keyword_pattern
                )
                results.append(keyword_result)
        
        # Summary
        success_results = [r for r in results if r["status"] == "success"]
        failed_results = [r for r in results if r["status"] == "error"]
        
        log_collection_summary(
            self.logger,
            "reddit",
            len(success_results),
            len(results),
            [r["subreddit"] for r in failed_results]
        )
        
        return {
            "total": len(results),
            "success": len(success_results),
            "failed": len(failed_results),
            "results": results
        }

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reddit Historical Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_reddit.py --subreddits wallstreetbets stocks --start 2021-01-01 --end 2021-03-31 --project my-project
  
  python collect_reddit.py \
    --subreddits GME AMC dogecoin \
    --start 2021-01-01 --end 2021-12-31 \
    --use-keywords true \
    --keyword-pattern '(?i)\\b(GME|AMC|DOGE)\\b' \
    --project my-project
        """
    )
    
    parser.add_argument(
        '--subreddits',
        nargs='+',
        required=True,
        help='List of subreddit names to collect (without r/ prefix)'
    )
    parser.add_argument(
        '--start',
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end',
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--use-keywords',
        type=str,
        choices=['true', 'false'],
        default='false',
        help='Whether to use keyword filtering (default: false)'
    )
    parser.add_argument(
        '--keyword-pattern',
        default='',
        help='Regex pattern for keyword filtering (required if use-keywords=true)'
    )
    parser.add_argument(
        '--project',
        required=True,
        help='Google Cloud Project ID'
    )
    parser.add_argument(
        '--sql-dir',
        default='sql',
        help='Directory containing SQL templates (default: sql)'
    )
    parser.add_argument(
        '--output-dir', 
        default='data/raw/reddit',
        help='Output directory for CSV files (default: data/raw/reddit)'
    )
    parser.add_argument(
        '--dry-run',
        type=str,
        choices=['true', 'false'],
        default='false',
        help='Connect and compile queries but do not write files (default: false)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Parse boolean arguments
    use_keywords = args.use_keywords.lower() == 'true'
    dry_run = args.dry_run.lower() == 'true'
    
    # Validate arguments
    if use_keywords and not args.keyword_pattern:
        parser.error("--keyword-pattern is required when --use-keywords=true")
    
    try:
        # Initialize collector
        collector = RedditCollector(
            project_id=args.project,
            sql_dir=args.sql_dir,
            output_dir=args.output_dir,
            dry_run=dry_run,
            log_level=args.log_level
        )
        
        # Collect data
        summary = collector.collect_multiple_subreddits(
            subreddits=args.subreddits,
            start_date=args.start,
            end_date=args.end,
            use_keywords=use_keywords,
            keyword_pattern=args.keyword_pattern
        )
        
        # Exit code based on results
        if summary["failed"] > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()