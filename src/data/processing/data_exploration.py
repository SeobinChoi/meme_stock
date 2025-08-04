"""
Day 2: Data Quality Assessment & Integration
Comprehensive data exploration and analysis module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExplorer:
    """
    Comprehensive data exploration and quality assessment for Day 2
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.reddit_data = None
        self.stock_data = {}
        self.exploration_results = {}
        
    def load_datasets(self) -> Dict:
        """
        Load all datasets for exploration
        """
        logger.info("Loading datasets for Day 2 exploration...")
        
        # Load Reddit data
        try:
            reddit_file = f"{self.data_dir}/raw/reddit_wsb.csv"
            self.reddit_data = pd.read_csv(reddit_file)
            logger.info(f"✅ Loaded Reddit data: {len(self.reddit_data):,} rows")
        except Exception as e:
            logger.error(f"❌ Failed to load Reddit data: {e}")
            return {"status": "ERROR", "message": f"Reddit data loading failed: {e}"}
        
        # Load stock data
        stock_symbols = ["GME", "AMC", "BB"]
        for symbol in stock_symbols:
            try:
                stock_file = f"{self.data_dir}/raw/{symbol}_stock_data.csv"
                self.stock_data[symbol] = pd.read_csv(stock_file)
                logger.info(f"✅ Loaded {symbol} data: {len(self.stock_data[symbol]):,} rows")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {symbol} data: {e}")
        
        return {"status": "SUCCESS", "datasets_loaded": len(self.stock_data) + 1}
    
    def explore_reddit_data(self) -> Dict:
        """
        Comprehensive Reddit data analysis
        """
        logger.info("🔍 Exploring Reddit data patterns...")
        
        if self.reddit_data is None:
            return {"status": "ERROR", "message": "Reddit data not loaded"}
        
        results = {
            "basic_stats": {},
            "temporal_analysis": {},
            "content_analysis": {},
            "engagement_analysis": {},
            "quality_assessment": {}
        }
        
        # Basic statistics
        results["basic_stats"] = {
            "total_posts": len(self.reddit_data),
            "total_columns": len(self.reddit_data.columns),
            "columns": list(self.reddit_data.columns),
            "memory_usage_mb": round(self.reddit_data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        # Temporal analysis
        if 'created' in self.reddit_data.columns:
            self.reddit_data['created'] = pd.to_datetime(self.reddit_data['created'], unit='s')
            results["temporal_analysis"] = {
                "date_range": {
                    "start": self.reddit_data['created'].min().strftime('%Y-%m-%d'),
                    "end": self.reddit_data['created'].max().strftime('%Y-%m-%d'),
                    "total_days": (self.reddit_data['created'].max() - self.reddit_data['created'].min()).days
                },
                "posts_per_day": len(self.reddit_data) / ((self.reddit_data['created'].max() - self.reddit_data['created'].min()).days + 1),
                "hourly_distribution": self.reddit_data['created'].dt.hour.value_counts().sort_index().to_dict()
            }
        
        # Content analysis
        if 'title' in self.reddit_data.columns:
            title_lengths = self.reddit_data['title'].str.len()
            results["content_analysis"] = {
                "avg_title_length": round(title_lengths.mean(), 2),
                "title_length_stats": {
                    "min": int(title_lengths.min()),
                    "max": int(title_lengths.max()),
                    "median": int(title_lengths.median()),
                    "std": round(title_lengths.std(), 2)
                },
                "empty_titles": int((self.reddit_data['title'].isna() | (self.reddit_data['title'] == '')).sum()),
                "unique_titles": int(self.reddit_data['title'].nunique())
            }
        
        # Engagement analysis
        if 'score' in self.reddit_data.columns:
            score_stats = self.reddit_data['score'].describe()
            results["engagement_analysis"] = {
                "score_stats": {
                    "mean": round(score_stats['mean'], 2),
                    "median": round(score_stats['50%'], 2),
                    "std": round(score_stats['std'], 2),
                    "min": int(score_stats['min']),
                    "max": int(score_stats['max'])
                },
                "high_engagement_posts": int((self.reddit_data['score'] > 1000).sum()),
                "negative_score_posts": int((self.reddit_data['score'] < 0).sum())
            }
        
        if 'comms_num' in self.reddit_data.columns:
            comment_stats = self.reddit_data['comms_num'].describe()
            results["engagement_analysis"]["comment_stats"] = {
                "mean": round(comment_stats['mean'], 2),
                "median": round(comment_stats['50%'], 2),
                "std": round(comment_stats['std'], 2),
                "min": int(comment_stats['min']),
                "max": int(comment_stats['max'])
            }
        
        # Quality assessment
        missing_data = self.reddit_data.isnull().sum()
        results["quality_assessment"] = {
            "missing_data_percentage": round((missing_data / len(self.reddit_data) * 100).to_dict(), 2),
            "duplicate_posts": int(self.reddit_data.duplicated().sum()),
            "data_completeness": round((1 - missing_data.sum() / (len(self.reddit_data) * len(self.reddit_data.columns))) * 100, 2)
        }
        
        self.exploration_results["reddit"] = results
        logger.info("✅ Reddit data exploration completed")
        return results
    
    def explore_stock_data(self) -> Dict:
        """
        Comprehensive stock data analysis
        """
        logger.info("📈 Exploring stock data patterns...")
        
        results = {}
        
        for symbol, data in self.stock_data.items():
            logger.info(f"Analyzing {symbol} data...")
            
            symbol_results = {
                "basic_stats": {},
                "price_analysis": {},
                "volume_analysis": {},
                "quality_assessment": {}
            }
            
            # Basic statistics
            symbol_results["basic_stats"] = {
                "total_records": len(data),
                "columns": list(data.columns),
                "date_range": None
            }
            
            # Date range analysis
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                symbol_results["basic_stats"]["date_range"] = {
                    "start": data['Date'].min().strftime('%Y-%m-%d'),
                    "end": data['Date'].max().strftime('%Y-%m-%d'),
                    "total_days": (data['Date'].max() - data['Date'].min()).days
                }
            
            # Price analysis
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                price_data = data[['Open', 'High', 'Low', 'Close']]
                
                # Price consistency checks
                price_issues = []
                if (data['High'] < data['Low']).any():
                    price_issues.append("High < Low violations")
                if (data['High'] < data['Open']).any():
                    price_issues.append("High < Open violations")
                if (data['High'] < data['Close']).any():
                    price_issues.append("High < Close violations")
                
                symbol_results["price_analysis"] = {
                    "price_issues": price_issues,
                    "avg_daily_range": round(((data['High'] - data['Low']) / data['Close'] * 100).mean(), 2),
                    "price_volatility": round(data['Close'].pct_change().std() * 100, 2),
                    "max_single_day_return": round(data['Close'].pct_change().max() * 100, 2),
                    "min_single_day_return": round(data['Close'].pct_change().min() * 100, 2)
                }
            
            # Volume analysis
            if 'Volume' in data.columns:
                volume_stats = data['Volume'].describe()
                symbol_results["volume_analysis"] = {
                    "avg_volume": int(volume_stats['mean']),
                    "median_volume": int(volume_stats['50%']),
                    "max_volume": int(volume_stats['max']),
                    "volume_volatility": round(data['Volume'].pct_change().std() * 100, 2)
                }
            
            # Quality assessment
            missing_data = data.isnull().sum()
            symbol_results["quality_assessment"] = {
                "missing_data_percentage": round((missing_data / len(data) * 100).to_dict(), 2),
                "data_completeness": round((1 - missing_data.sum() / (len(data) * len(data.columns))) * 100, 2)
            }
            
            results[symbol] = symbol_results
        
        self.exploration_results["stock"] = results
        logger.info("✅ Stock data exploration completed")
        return results
    
    def analyze_temporal_alignment(self) -> Dict:
        """
        Cross-dataset temporal alignment analysis
        """
        logger.info("⏰ Analyzing temporal alignment across datasets...")
        
        results = {
            "date_ranges": {},
            "overlap_analysis": {},
            "missing_data_patterns": {},
            "recommendations": []
        }
        
        # Reddit date range
        if self.reddit_data is not None and 'created' in self.reddit_data.columns:
            reddit_dates = pd.to_datetime(self.reddit_data['created'], unit='s')
            results["date_ranges"]["reddit"] = {
                "start": reddit_dates.min().strftime('%Y-%m-%d'),
                "end": reddit_dates.max().strftime('%Y-%m-%d'),
                "total_days": (reddit_dates.max() - reddit_dates.min()).days
            }
        
        # Stock date ranges
        for symbol, data in self.stock_data.items():
            if 'Date' in data.columns:
                stock_dates = pd.to_datetime(data['Date'])
                results["date_ranges"][symbol] = {
                    "start": stock_dates.min().strftime('%Y-%m-%d'),
                    "end": stock_dates.max().strftime('%Y-%m-%d'),
                    "total_days": (stock_dates.max() - stock_dates.min()).days
                }
        
        # Overlap analysis
        if len(results["date_ranges"]) > 1:
            all_starts = [range_info["start"] for range_info in results["date_ranges"].values()]
            all_ends = [range_info["end"] for range_info in results["date_ranges"].values()]
            
            optimal_start = max(all_starts)
            optimal_end = min(all_ends)
            
            results["overlap_analysis"] = {
                "optimal_start": optimal_start,
                "optimal_end": optimal_end,
                "overlap_days": (pd.to_datetime(optimal_end) - pd.to_datetime(optimal_start)).days
            }
            
            # Recommendations
            if results["overlap_analysis"]["overlap_days"] < 30:
                results["recommendations"].append("Limited temporal overlap - consider extending data collection period")
            if results["overlap_analysis"]["overlap_days"] > 0:
                results["recommendations"].append(f"Use date range {optimal_start} to {optimal_end} for unified dataset")
        
        self.exploration_results["temporal"] = results
        logger.info("✅ Temporal alignment analysis completed")
        return results
    
    def generate_exploration_report(self) -> Dict:
        """
        Generate comprehensive exploration report
        """
        logger.info("📊 Generating comprehensive exploration report...")
        
        # Run all explorations
        reddit_results = self.explore_reddit_data()
        stock_results = self.explore_stock_data()
        temporal_results = self.analyze_temporal_alignment()
        
        # Compile comprehensive report
        report = {
            "exploration_timestamp": datetime.now().isoformat(),
            "datasets_analyzed": len(self.stock_data) + 1,
            "reddit_analysis": reddit_results,
            "stock_analysis": stock_results,
            "temporal_analysis": temporal_results,
            "overall_assessment": self._generate_overall_assessment()
        }
        
        self.exploration_results = report
        return report
    
    def _generate_overall_assessment(self) -> Dict:
        """
        Generate overall data quality assessment
        """
        assessment = {
            "data_quality_score": 0,
            "recommendations": [],
            "critical_issues": [],
            "strengths": []
        }
        
        # Calculate quality score based on various factors
        quality_factors = []
        
        # Reddit data quality
        if "reddit" in self.exploration_results:
            reddit_quality = self.exploration_results["reddit"]["quality_assessment"]["data_completeness"]
            quality_factors.append(reddit_quality)
            if reddit_quality < 90:
                assessment["critical_issues"].append(f"Reddit data completeness: {reddit_quality}%")
            else:
                assessment["strengths"].append(f"Reddit data completeness: {reddit_quality}%")
        
        # Stock data quality
        stock_qualities = []
        for symbol, analysis in self.exploration_results.get("stock", {}).items():
            if "quality_assessment" in analysis:
                stock_quality = analysis["quality_assessment"]["data_completeness"]
                stock_qualities.append(stock_quality)
                if stock_quality < 90:
                    assessment["critical_issues"].append(f"{symbol} data completeness: {stock_quality}%")
                else:
                    assessment["strengths"].append(f"{symbol} data completeness: {stock_quality}%")
        
        if stock_qualities:
            quality_factors.append(np.mean(stock_qualities))
        
        # Temporal overlap quality
        temporal = self.exploration_results.get("temporal", {})
        if "overlap_analysis" in temporal:
            overlap_days = temporal["overlap_analysis"].get("overlap_days", 0)
            if overlap_days > 30:
                quality_factors.append(100)
                assessment["strengths"].append(f"Good temporal overlap: {overlap_days} days")
            elif overlap_days > 0:
                quality_factors.append(70)
                assessment["recommendations"].append(f"Limited temporal overlap: {overlap_days} days")
            else:
                quality_factors.append(30)
                assessment["critical_issues"].append("No temporal overlap between datasets")
        
        # Calculate overall score
        if quality_factors:
            assessment["data_quality_score"] = round(np.mean(quality_factors), 1)
        
        # Add recommendations
        if assessment["data_quality_score"] < 80:
            assessment["recommendations"].append("Overall data quality needs improvement")
        elif assessment["data_quality_score"] < 95:
            assessment["recommendations"].append("Data quality is good but could be optimized")
        else:
            assessment["recommendations"].append("Excellent data quality - ready for feature engineering")
        
        return assessment
    
    def save_exploration_report(self, report: Dict, output_path: str = "results/day2_exploration_report.json"):
        """
        Save exploration report to file
        """
        import json
        from pathlib import Path
        
        # Ensure results directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exploration report saved to {output_file}")
        
        # Save human-readable summary
        summary_file = output_file.parent / "day2_exploration_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== DAY 2 DATA EXPLORATION SUMMARY ===\n\n")
            f.write(f"Exploration Timestamp: {report['exploration_timestamp']}\n")
            f.write(f"Datasets Analyzed: {report['datasets_analyzed']}\n\n")
            
            # Overall assessment
            overall = report['overall_assessment']
            f.write(f"Overall Data Quality Score: {overall['data_quality_score']}%\n\n")
            
            # Strengths
            if overall['strengths']:
                f.write("STRENGTHS:\n")
                for strength in overall['strengths']:
                    f.write(f"  ✅ {strength}\n")
                f.write("\n")
            
            # Critical issues
            if overall['critical_issues']:
                f.write("CRITICAL ISSUES:\n")
                for issue in overall['critical_issues']:
                    f.write(f"  ❌ {issue}\n")
                f.write("\n")
            
            # Recommendations
            if overall['recommendations']:
                f.write("RECOMMENDATIONS:\n")
                for rec in overall['recommendations']:
                    f.write(f"  💡 {rec}\n")
                f.write("\n")
            
            # Reddit summary
            if 'reddit_analysis' in report:
                reddit = report['reddit_analysis']
                f.write("REDDIT DATA SUMMARY:\n")
                if 'basic_stats' in reddit:
                    f.write(f"  Total Posts: {reddit['basic_stats']['total_posts']:,}\n")
                if 'temporal_analysis' in reddit:
                    f.write(f"  Date Range: {reddit['temporal_analysis']['date_range']['start']} to {reddit['temporal_analysis']['date_range']['end']}\n")
                if 'quality_assessment' in reddit:
                    f.write(f"  Data Completeness: {reddit['quality_assessment']['data_completeness']}%\n")
                f.write("\n")
            
            # Stock summary
            if 'stock_analysis' in report:
                f.write("STOCK DATA SUMMARY:\n")
                for symbol, analysis in report['stock_analysis'].items():
                    f.write(f"  {symbol}: {analysis['basic_stats']['total_records']:,} records\n")
                    if 'quality_assessment' in analysis:
                        f.write(f"    Completeness: {analysis['quality_assessment']['data_completeness']}%\n")
                f.write("\n")
            
            # Temporal summary
            if 'temporal_analysis' in report:
                temporal = report['temporal_analysis']
                if 'overlap_analysis' in temporal and temporal['overlap_analysis']:
                    f.write("TEMPORAL ALIGNMENT:\n")
                    f.write(f"  Optimal Period: {temporal['overlap_analysis']['optimal_start']} to {temporal['overlap_analysis']['optimal_end']}\n")
                    f.write(f"  Overlap Days: {temporal['overlap_analysis']['overlap_days']}\n")
                else:
                    f.write("TEMPORAL ALIGNMENT:\n")
                    f.write("  No temporal overlap analysis available\n")
        
        logger.info(f"Exploration summary saved to {summary_file}")


def main():
    """
    Main function to run Day 2 data exploration
    """
    logger.info("Starting Day 2 Data Exploration...")
    
    # Initialize explorer
    explorer = DataExplorer()
    
    # Load datasets
    load_result = explorer.load_datasets()
    if load_result["status"] != "SUCCESS":
        logger.error(f"Failed to load datasets: {load_result}")
        return
    
    # Generate comprehensive exploration report
    report = explorer.generate_exploration_report()
    
    # Save report
    explorer.save_exploration_report(report)
    
    # Print summary
    print("\n" + "="*50)
    print("DAY 2 DATA EXPLORATION COMPLETE")
    print("="*50)
    print(f"Datasets Analyzed: {report['datasets_analyzed']}")
    print(f"Data Quality Score: {report['overall_assessment']['data_quality_score']}%")
    print("="*50)
    
    if report['overall_assessment']['data_quality_score'] >= 90:
        logger.info("✅ Day 2 exploration PASSED - Ready for data cleaning")
    elif report['overall_assessment']['data_quality_score'] >= 70:
        logger.warning("⚠️ Day 2 exploration has WARNINGS - Review recommendations")
    else:
        logger.error("❌ Day 2 exploration FAILED - Fix critical issues")


if __name__ == "__main__":
    main() 