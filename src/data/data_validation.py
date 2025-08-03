"""
Data Validation Module for Meme Stock Prediction Project
Day 1: Environment Setup & Data Infrastructure

This module provides comprehensive data validation for:
- Reddit WSB dataset integrity
- Stock price data completeness
- Temporal alignment validation
- Data quality assessment
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation for meme stock prediction datasets
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataValidator with data directory path
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        
    def validate_reddit_dataset(self) -> Dict:
        """
        Validate Reddit WSB dataset integrity and quality
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating Reddit WSB dataset...")
        
        reddit_file = self.data_dir / "raw" / "reddit_wsb.csv"
        
        if not reddit_file.exists():
            return {
                "status": "ERROR",
                "message": "Reddit WSB dataset not found",
                "file_path": str(reddit_file)
            }
        
        try:
            # Load dataset
            df = pd.read_csv(reddit_file)
            
            validation_results = {
                "file_size_mb": round(os.path.getsize(reddit_file) / (1024 * 1024), 2),
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "date_range": None,
                "missing_data": {},
                "duplicates": 0,
                "data_types": df.dtypes.to_dict(),
                "sample_data": df.head(3).to_dict('records')
            }
            
            # Check for required columns
            required_columns = ['title', 'score', 'num_comments', 'created_utc']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_results["missing_columns"] = missing_columns
                validation_results["status"] = "WARNING"
            else:
                validation_results["status"] = "PASS"
            
            # Check for missing data
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    validation_results["missing_data"][col] = {
                        "count": missing_count,
                        "percentage": round(missing_count / len(df) * 100, 2)
                    }
            
            # Check for duplicates
            duplicates = df.duplicated().sum()
            validation_results["duplicates"] = duplicates
            
            # Check date range if created_utc exists
            if 'created_utc' in df.columns:
                try:
                    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
                    date_range = {
                        "start": df['created_utc'].min().strftime('%Y-%m-%d'),
                        "end": df['created_utc'].max().strftime('%Y-%m-%d'),
                        "total_days": (df['created_utc'].max() - df['created_utc'].min()).days
                    }
                    validation_results["date_range"] = date_range
                except Exception as e:
                    validation_results["date_parsing_error"] = str(e)
            
            # Check for spam/quality indicators
            if 'score' in df.columns:
                validation_results["score_stats"] = {
                    "mean": round(df['score'].mean(), 2),
                    "median": df['score'].median(),
                    "std": round(df['score'].std(), 2),
                    "min": df['score'].min(),
                    "max": df['score'].max()
                }
            
            if 'num_comments' in df.columns:
                validation_results["comment_stats"] = {
                    "mean": round(df['num_comments'].mean(), 2),
                    "median": df['num_comments'].median(),
                    "std": round(df['num_comments'].std(), 2),
                    "min": df['num_comments'].min(),
                    "max": df['num_comments'].max()
                }
            
            logger.info(f"Reddit dataset validation completed. Status: {validation_results['status']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating Reddit dataset: {str(e)}")
            return {
                "status": "ERROR",
                "message": f"Error reading Reddit dataset: {str(e)}"
            }
    
    def validate_stock_data(self) -> Dict:
        """
        Validate stock price data completeness and quality
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating stock price data...")
        
        archive_dirs = [
            self.data_dir / "raw" / "archive-2",
            self.data_dir / "raw" / "archive-3"
        ]
        
        stock_files = {}
        validation_results = {
            "status": "PASS",
            "stock_files": {},
            "total_files": 0,
            "missing_stocks": []
        }
        
        # Expected stock files
        expected_stocks = ['GME', 'AMC', 'BB']
        
        for archive_dir in archive_dirs:
            if archive_dir.exists():
                for file in archive_dir.glob("*.csv"):
                    if any(stock in file.name.upper() for stock in expected_stocks):
                        stock_name = file.stem
                        stock_files[stock_name] = {
                            "path": str(file),
                            "size_mb": round(os.path.getsize(file) / (1024 * 1024), 2)
                        }
                        
                        # Validate individual stock file
                        try:
                            df = pd.read_csv(file)
                            stock_files[stock_name].update({
                                "rows": len(df),
                                "columns": list(df.columns),
                                "date_range": None,
                                "missing_data": {}
                            })
                            
                            # Check for date column and range
                            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                            if date_cols:
                                date_col = date_cols[0]
                                try:
                                    df[date_col] = pd.to_datetime(df[date_col])
                                    stock_files[stock_name]["date_range"] = {
                                        "start": df[date_col].min().strftime('%Y-%m-%d'),
                                        "end": df[date_col].max().strftime('%Y-%m-%d'),
                                        "total_days": (df[date_col].max() - df[date_col].min()).days
                                    }
                                except Exception as e:
                                    stock_files[stock_name]["date_parsing_error"] = str(e)
                            
                            # Check for missing data
                            for col in df.columns:
                                missing_count = df[col].isnull().sum()
                                if missing_count > 0:
                                    stock_files[stock_name]["missing_data"][col] = {
                                        "count": missing_count,
                                        "percentage": round(missing_count / len(df) * 100, 2)
                                    }
                                    
                        except Exception as e:
                            stock_files[stock_name]["error"] = str(e)
                            validation_results["status"] = "WARNING"
        
        validation_results["stock_files"] = stock_files
        validation_results["total_files"] = len(stock_files)
        
        # Check for missing expected stocks
        found_stocks = [stock.upper() for stock in stock_files.keys()]
        validation_results["missing_stocks"] = [
            stock for stock in expected_stocks 
            if not any(stock in found_stock for found_stock in found_stocks)
        ]
        
        logger.info(f"Stock data validation completed. Found {len(stock_files)} stock files.")
        return validation_results
    
    def validate_temporal_alignment(self) -> Dict:
        """
        Validate temporal alignment between datasets
        
        Returns:
            Dictionary containing temporal alignment validation results
        """
        logger.info("Validating temporal alignment...")
        
        alignment_results = {
            "status": "PASS",
            "reddit_date_range": None,
            "stock_date_ranges": {},
            "overlap_analysis": {},
            "recommendations": []
        }
        
        # Get Reddit date range
        reddit_file = self.data_dir / "raw" / "reddit_wsb.csv"
        if reddit_file.exists():
            try:
                df_reddit = pd.read_csv(reddit_file)
                if 'created_utc' in df_reddit.columns:
                    df_reddit['created_utc'] = pd.to_datetime(df_reddit['created_utc'], unit='s')
                    reddit_start = df_reddit['created_utc'].min()
                    reddit_end = df_reddit['created_utc'].max()
                    
                    alignment_results["reddit_date_range"] = {
                        "start": reddit_start.strftime('%Y-%m-%d'),
                        "end": reddit_end.strftime('%Y-%m-%d'),
                        "total_days": (reddit_end - reddit_start).days
                    }
            except Exception as e:
                alignment_results["reddit_error"] = str(e)
                alignment_results["status"] = "WARNING"
        
        # Get stock date ranges
        archive_dirs = [
            self.data_dir / "raw" / "archive-2",
            self.data_dir / "raw" / "archive-3"
        ]
        
        for archive_dir in archive_dirs:
            if archive_dir.exists():
                for file in archive_dir.glob("*.csv"):
                    if any(stock in file.name.upper() for stock in ['GME', 'AMC', 'BB']):
                        try:
                            df_stock = pd.read_csv(file)
                            date_cols = [col for col in df_stock.columns if 'date' in col.lower() or 'time' in col.lower()]
                            
                            if date_cols:
                                date_col = date_cols[0]
                                df_stock[date_col] = pd.to_datetime(df_stock[date_col])
                                stock_start = df_stock[date_col].min()
                                stock_end = df_stock[date_col].max()
                                
                                stock_name = file.stem
                                alignment_results["stock_date_ranges"][stock_name] = {
                                    "start": stock_start.strftime('%Y-%m-%d'),
                                    "end": stock_end.strftime('%Y-%m-%d'),
                                    "total_days": (stock_end - stock_start).days
                                }
                                
                                # Check overlap with Reddit data
                                if alignment_results["reddit_date_range"]:
                                    reddit_start = pd.to_datetime(alignment_results["reddit_date_range"]["start"])
                                    reddit_end = pd.to_datetime(alignment_results["reddit_date_range"]["end"])
                                    
                                    overlap_start = max(reddit_start, stock_start)
                                    overlap_end = min(reddit_end, stock_end)
                                    
                                    if overlap_start <= overlap_end:
                                        overlap_days = (overlap_end - overlap_start).days
                                        alignment_results["overlap_analysis"][stock_name] = {
                                            "overlap_start": overlap_start.strftime('%Y-%m-%d'),
                                            "overlap_end": overlap_end.strftime('%Y-%m-%d'),
                                            "overlap_days": overlap_days,
                                            "reddit_coverage": round(overlap_days / (reddit_end - reddit_start).days * 100, 2),
                                            "stock_coverage": round(overlap_days / (stock_end - stock_start).days * 100, 2)
                                        }
                                    else:
                                        alignment_results["overlap_analysis"][stock_name] = {
                                            "status": "NO_OVERLAP"
                                        }
                                        alignment_results["status"] = "WARNING"
                                        
                        except Exception as e:
                            alignment_results["stock_errors"][file.stem] = str(e)
                            alignment_results["status"] = "WARNING"
        
        # Generate recommendations
        if alignment_results["overlap_analysis"]:
            min_overlap = min([
                analysis.get("overlap_days", 0) 
                for analysis in alignment_results["overlap_analysis"].values()
                if isinstance(analysis, dict) and "overlap_days" in analysis
            ])
            
            if min_overlap < 30:
                alignment_results["recommendations"].append(
                    "Consider extending data collection period for better temporal coverage"
                )
        
        logger.info("Temporal alignment validation completed.")
        return alignment_results
    
    def generate_validation_report(self) -> Dict:
        """
        Generate comprehensive validation report
        
        Returns:
            Complete validation report
        """
        logger.info("Generating comprehensive validation report...")
        
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "data_directory": str(self.data_dir),
            "reddit_validation": self.validate_reddit_dataset(),
            "stock_validation": self.validate_stock_data(),
            "temporal_alignment": self.validate_temporal_alignment(),
            "overall_status": "PASS",
            "summary": {}
        }
        
        # Determine overall status
        statuses = [
            report["reddit_validation"].get("status", "UNKNOWN"),
            report["stock_validation"].get("status", "UNKNOWN"),
            report["temporal_alignment"].get("status", "UNKNOWN")
        ]
        
        if "ERROR" in statuses:
            report["overall_status"] = "ERROR"
        elif "WARNING" in statuses:
            report["overall_status"] = "WARNING"
        
        # Generate summary
        summary = {
            "total_datasets": 0,
            "total_rows": 0,
            "total_size_mb": 0,
            "date_coverage": "Unknown",
            "data_quality_score": 0
        }
        
        # Reddit summary
        if report["reddit_validation"].get("status") != "ERROR":
            summary["total_datasets"] += 1
            summary["total_rows"] += report["reddit_validation"].get("total_rows", 0)
            summary["total_size_mb"] += report["reddit_validation"].get("file_size_mb", 0)
        
        # Stock summary
        stock_validation = report["stock_validation"]
        summary["total_datasets"] += stock_validation.get("total_files", 0)
        for stock_info in stock_validation.get("stock_files", {}).values():
            summary["total_rows"] += stock_info.get("rows", 0)
            summary["total_size_mb"] += stock_info.get("size_mb", 0)
        
        # Date coverage
        if report["temporal_alignment"].get("reddit_date_range"):
            summary["date_coverage"] = f"{report['temporal_alignment']['reddit_date_range']['total_days']} days"
        
        # Data quality score (simplified)
        quality_factors = []
        if report["reddit_validation"].get("status") == "PASS":
            quality_factors.append(1.0)
        elif report["reddit_validation"].get("status") == "WARNING":
            quality_factors.append(0.7)
        
        if report["stock_validation"].get("status") == "PASS":
            quality_factors.append(1.0)
        elif report["stock_validation"].get("status") == "WARNING":
            quality_factors.append(0.7)
        
        if report["temporal_alignment"].get("status") == "PASS":
            quality_factors.append(1.0)
        elif report["temporal_alignment"].get("status") == "WARNING":
            quality_factors.append(0.7)
        
        if quality_factors:
            summary["data_quality_score"] = round(sum(quality_factors) / len(quality_factors) * 100, 1)
        
        report["summary"] = summary
        
        logger.info(f"Validation report generated. Overall status: {report['overall_status']}")
        return report
    
    def save_validation_report(self, report: Dict, output_path: str = "results/day1_validation_report.json"):
        """
        Save validation report to file
        
        Args:
            report: Validation report dictionary
            output_path: Path to save the report
        """
        import json
        
        # Ensure results directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_file}")
        
        # Also save a human-readable summary
        summary_file = output_file.parent / "day1_validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== DAY 1 DATA VALIDATION SUMMARY ===\n\n")
            f.write(f"Validation Timestamp: {report['validation_timestamp']}\n")
            f.write(f"Overall Status: {report['overall_status']}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"  Total Datasets: {report['summary']['total_datasets']}\n")
            f.write(f"  Total Rows: {report['summary']['total_rows']:,}\n")
            f.write(f"  Total Size: {report['summary']['total_size_mb']:.1f} MB\n")
            f.write(f"  Date Coverage: {report['summary']['date_coverage']}\n")
            f.write(f"  Data Quality Score: {report['summary']['data_quality_score']}%\n\n")
            
            f.write("REDDIT DATASET:\n")
            reddit = report['reddit_validation']
            if reddit.get('status') != 'ERROR':
                f.write(f"  Status: {reddit['status']}\n")
                f.write(f"  File Size: {reddit['file_size_mb']} MB\n")
                f.write(f"  Total Rows: {reddit['total_rows']:,}\n")
                f.write(f"  Total Columns: {reddit['total_columns']}\n")
                if reddit.get('date_range'):
                    f.write(f"  Date Range: {reddit['date_range']['start']} to {reddit['date_range']['end']}\n")
                if reddit.get('missing_data'):
                    f.write(f"  Missing Data: {len(reddit['missing_data'])} columns affected\n")
            else:
                f.write(f"  Status: {reddit['status']} - {reddit.get('message', 'Unknown error')}\n")
            
            f.write("\nSTOCK DATA:\n")
            stock = report['stock_validation']
            f.write(f"  Status: {stock['status']}\n")
            f.write(f"  Total Files: {stock['total_files']}\n")
            for stock_name, stock_info in stock.get('stock_files', {}).items():
                f.write(f"  {stock_name}: {stock_info.get('rows', 0):,} rows, {stock_info.get('size_mb', 0):.1f} MB\n")
            
            f.write("\nTEMPORAL ALIGNMENT:\n")
            temporal = report['temporal_alignment']
            f.write(f"  Status: {temporal['status']}\n")
            if temporal.get('reddit_date_range'):
                f.write(f"  Reddit Range: {temporal['reddit_date_range']['start']} to {temporal['reddit_date_range']['end']}\n")
            
            if temporal.get('recommendations'):
                f.write("\nRECOMMENDATIONS:\n")
                for rec in temporal['recommendations']:
                    f.write(f"  - {rec}\n")
        
        logger.info(f"Validation summary saved to {summary_file}")


def main():
    """
    Main function to run Day 1 data validation
    """
    logger.info("Starting Day 1 Data Validation...")
    
    # Initialize validator
    validator = DataValidator()
    
    # Generate comprehensive validation report
    report = validator.generate_validation_report()
    
    # Save report
    validator.save_validation_report(report)
    
    # Print summary
    print("\n" + "="*50)
    print("DAY 1 DATA VALIDATION COMPLETE")
    print("="*50)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Total Datasets: {report['summary']['total_datasets']}")
    print(f"Total Rows: {report['summary']['total_rows']:,}")
    print(f"Total Size: {report['summary']['total_size_mb']:.1f} MB")
    print(f"Data Quality Score: {report['summary']['data_quality_score']}%")
    print("="*50)
    
    if report['overall_status'] == 'PASS':
        logger.info("✅ Day 1 validation PASSED - Ready for feature engineering")
    elif report['overall_status'] == 'WARNING':
        logger.warning("⚠️ Day 1 validation has WARNINGS - Review recommendations")
    else:
        logger.error("❌ Day 1 validation FAILED - Fix issues before proceeding")


if __name__ == "__main__":
    main() 