#!/usr/bin/env python3
"""
Backfill metadata for existing CSV files.

Scans existing CSV files in data/raw/ and creates metadata files
without modifying the original data files.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add common package to path
sys.path.append(str(Path(__file__).parent))

from common.logging_utils import get_logger
from common.paths import dir_raw_stocks, dir_raw_crypto, dir_raw_reddit, ensure_dirs_exist
from common.metadata import build_metadata_from_file, write_metadata
from common.io_utils import read_index

def find_csv_files() -> List[Path]:
    """
    Find all CSV files in raw data directories that need metadata backfill.
    
    Returns:
        List of CSV file paths
    """
    csv_files = []
    
    # Scan raw data directories
    directories = [
        dir_raw_stocks(),
        dir_raw_crypto(), 
        dir_raw_reddit()
    ]
    
    for directory in directories:
        if directory.exists():
            for csv_file in directory.glob("*.csv"):
                # Skip files that already have metadata
                meta_file = csv_file.with_suffix('.meta.json')
                if not meta_file.exists():
                    csv_files.append(csv_file)
    
    return sorted(csv_files)

def backfill_single_file(csv_path: Path, logger) -> Dict[str, Any]:
    """
    Backfill metadata for a single CSV file.
    
    Args:
        csv_path: Path to CSV file
        logger: Logger instance
        
    Returns:
        Dictionary with backfill results
    """
    result = {
        "path": str(csv_path),
        "status": "unknown",
        "metadata": {},
        "error": None
    }
    
    try:
        logger.info(f"Processing {csv_path}")
        
        # Build metadata by analyzing the file
        metadata = build_metadata_from_file(csv_path)
        
        # Write metadata and update index
        meta_path = write_metadata(metadata, csv_path)
        
        result.update({
            "status": "success",
            "metadata": metadata,
            "meta_path": str(meta_path)
        })
        
        logger.info(f"✓ Created metadata for {csv_path.name}")
        
    except Exception as e:
        error_msg = str(e)
        result.update({
            "status": "error", 
            "error": error_msg
        })
        
        logger.error(f"✗ Failed to process {csv_path}: {error_msg}")
    
    return result

def print_summary_table(results: List[Dict[str, Any]], logger) -> None:
    """
    Print summary table of backfill results.
    
    Args:
        results: List of backfill result dictionaries
        logger: Logger instance
    """
    if not results:
        logger.info("No files processed.")
        return
    
    # Count results by status
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKFILL SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total files processed: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")
    
    if success_count > 0:
        logger.info(f"\nSuccessfully backfilled files:")
        logger.info(f"{'File':<40} {'Symbol':<8} {'Type':<8} {'Records':<8} {'Missing':<8}")
        logger.info(f"{'-'*76}")
        
        for result in results:
            if result["status"] == "success":
                meta = result["metadata"]
                filename = Path(result["path"]).name
                symbol = meta.get("symbol", "?")
                asset_type = meta.get("asset_type", "?")
                records = meta.get("total_records", 0)
                missing = meta.get("missing_dates", 0)
                
                logger.info(f"{filename:<40} {symbol:<8} {asset_type:<8} {records:<8} {missing:<8}")
    
    if error_count > 0:
        logger.info(f"\nFiles with errors:")
        for result in results:
            if result["status"] == "error":
                filename = Path(result["path"]).name
                error = result["error"]
                logger.info(f"  {filename}: {error}")
    
    logger.info(f"{'='*80}")

def validate_existing_index(logger) -> None:
    """
    Validate existing index entries and report any issues.
    
    Args:
        logger: Logger instance
    """
    try:
        index_entries = read_index()
        
        if not index_entries:
            logger.info("Index is empty or doesn't exist - will be created during backfill")
            return
        
        logger.info(f"Found {len(index_entries)} existing index entries")
        
        # Check for files that exist in index but not on disk
        missing_files = []
        for entry in index_entries:
            file_path = Path(entry.get("path", ""))
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.warning(f"Found {len(missing_files)} index entries for missing files:")
            for missing in missing_files[:5]:  # Show first 5
                logger.warning(f"  Missing: {missing}")
            if len(missing_files) > 5:
                logger.warning(f"  ... and {len(missing_files) - 5} more")
    
    except Exception as e:
        logger.warning(f"Could not validate existing index: {e}")

def main():
    """Main backfill execution."""
    logger = get_logger(__name__)
    
    logger.info("Starting metadata backfill for existing CSV files")
    
    # Ensure all directories exist
    ensure_dirs_exist()
    
    # Validate existing index
    validate_existing_index(logger)
    
    # Find CSV files that need metadata
    csv_files = find_csv_files()
    
    if not csv_files:
        logger.info("No CSV files found that need metadata backfill")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files without metadata")
    
    # Process each file
    results = []
    for csv_path in csv_files:
        result = backfill_single_file(csv_path, logger)
        results.append(result)
    
    # Print summary
    print_summary_table(results, logger)
    
    # Final status
    success_count = sum(1 for r in results if r["status"] == "success")
    if success_count == len(results):
        logger.info("🎉 Backfill completed successfully!")
    elif success_count > 0:
        logger.info(f"⚠️  Backfill completed with {len(results) - success_count} errors")
    else:
        logger.error("❌ Backfill failed for all files")
        sys.exit(1)

if __name__ == "__main__":
    main()