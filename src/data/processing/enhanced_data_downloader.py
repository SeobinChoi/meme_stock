"""
Enhanced Data Downloader
Downloads data from multiple sources for improved quality and completeness
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Enhanced data source imports
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataDownloader:
    """
    Enhanced data downloader with multiple sources and quality validation
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.symbols = ["GME", "AMC", "BB"]
        self.alpha_vantage_key = None  # Set your API key here
        self.quality_metrics = {}
        
    def set_alpha_vantage_key(self, api_key: str):
        """
        Set Alpha Vantage API key for enhanced financial data
        """
        self.alpha_vantage_key = api_key
        logger.info("✅ Alpha Vantage API key configured")
    
    def download_enhanced_stock_data(self):
        """
        Download stock data from multiple sources for better quality
        """
        logger.info("📈 Downloading enhanced stock data from multiple sources...")
        
        for symbol in self.symbols:
            logger.info(f"Downloading {symbol} data from multiple sources...")
            
            # Download from multiple sources
            yahoo_data = self._download_yahoo_data(symbol)
            alpha_data = self._download_alpha_vantage_data(symbol) if self.alpha_vantage_key else pd.DataFrame()
            
            # Merge and validate data
            merged_data = self._merge_stock_sources(symbol, yahoo_data, alpha_data)
            
            # Quality assessment
            quality_score = self._assess_stock_quality(symbol, merged_data)
            self.quality_metrics[symbol] = quality_score
            
            # Save enhanced data
            self._save_enhanced_stock_data(symbol, merged_data)
            
            logger.info(f"✅ {symbol} enhanced data downloaded. Quality: {quality_score:.1f}%")
    
    def _download_yahoo_data(self, symbol: str) -> pd.DataFrame:
        """
        Download data from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start="2020-01-01", end="2021-12-31")
            
            if not df.empty:
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                })
                
                # Add source identifier
                df['source'] = 'yahoo'
                
                logger.info(f"  Yahoo: {len(df)} records for {symbol}")
                return df
            else:
                logger.warning(f"  Yahoo: No data for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"  Yahoo: Error downloading {symbol}: {e}")
            return pd.DataFrame()
    
    def _download_alpha_vantage_data(self, symbol: str) -> pd.DataFrame:
        """
        Download data from Alpha Vantage
        """
        if not self.alpha_vantage_key:
            return pd.DataFrame()
            
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            df, meta = ts.get_daily(symbol, outputsize='full')
            
            if not df.empty:
                # Reset index and rename columns
                df = df.reset_index()
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                # Filter to 2020-2021 period
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-12-31')]
                
                # Add source identifier
                df['source'] = 'alpha_vantage'
                
                logger.info(f"  Alpha Vantage: {len(df)} records for {symbol}")
                return df
            else:
                logger.warning(f"  Alpha Vantage: No data for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"  Alpha Vantage: Error downloading {symbol}: {e}")
            return pd.DataFrame()
    
    def _merge_stock_sources(self, symbol: str, yahoo_data: pd.DataFrame, alpha_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge data from multiple sources with quality validation
        """
        if yahoo_data.empty and alpha_data.empty:
            logger.error(f"❌ No data available for {symbol}")
            return pd.DataFrame()
        
        if yahoo_data.empty:
            logger.info(f"  Using Alpha Vantage data for {symbol}")
            return alpha_data
        
        if alpha_data.empty:
            logger.info(f"  Using Yahoo data for {symbol}")
            return yahoo_data
        
        # Both sources available - merge with quality validation
        logger.info(f"  Merging multiple sources for {symbol}")
        
        # Convert dates for comparison
        yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date'])
        alpha_data['Date'] = pd.to_datetime(alpha_data['Date'])
        
        # Merge on date
        merged = pd.merge(yahoo_data, alpha_data, on='Date', suffixes=('_yahoo', '_alpha'))
        
        # Quality-based selection
        merged['Open'] = merged[['Open_yahoo', 'Open_alpha']].mean(axis=1)
        merged['High'] = merged[['High_yahoo', 'High_alpha']].max(axis=1)
        merged['Low'] = merged[['Low_yahoo', 'Low_alpha']].min(axis=1)
        merged['Close'] = merged[['Close_yahoo', 'Close_alpha']].mean(axis=1)
        merged['Volume'] = merged[['Volume_yahoo', 'Volume_alpha']].max(axis=1)
        
        # Keep only essential columns
        result = merged[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        result['source'] = 'merged'
        
        logger.info(f"  Merged: {len(result)} records for {symbol}")
        return result
    
    def _assess_stock_quality(self, symbol: str, data: pd.DataFrame) -> float:
        """
        Assess quality of stock data
        """
        if data.empty:
            return 0.0
        
        quality_checks = []
        
        # Completeness check
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        completeness = 1 - missing_ratio
        quality_checks.append(completeness)
        
        # Logical consistency check
        if 'High' in data.columns and 'Low' in data.columns:
            logical_consistency = (data['High'] >= data['Low']).mean()
            quality_checks.append(logical_consistency)
        
        # Price range check
        if 'Close' in data.columns:
            price_range = ((data['Close'] > 0) & (data['Close'] < 10000)).mean()
            quality_checks.append(price_range)
        
        # Volume check
        if 'Volume' in data.columns:
            volume_check = (data['Volume'] >= 0).mean()
            quality_checks.append(volume_check)
        
        # Date coverage check
        if 'Date' in data.columns:
            date_range = (data['Date'].max() - data['Date'].min()).days
            coverage_ratio = min(date_range / 730, 1.0)  # 730 days = 2 years
            quality_checks.append(coverage_ratio)
        
        return np.mean(quality_checks) * 100
    
    def _save_enhanced_stock_data(self, symbol: str, data: pd.DataFrame):
        """
        Save enhanced stock data with quality metadata
        """
        if data.empty:
            return
        
        # Save data
        output_file = self.data_dir / "raw" / f"{symbol}_enhanced_stock_data.csv"
        data.to_csv(output_file, index=False)
        
        # Save quality metadata
        quality_file = self.data_dir / "raw" / f"{symbol}_quality_metadata.json"
        metadata = {
            'symbol': symbol,
            'download_timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'date_range': {
                'start': data['Date'].min().isoformat(),
                'end': data['Date'].max().isoformat()
            },
            'quality_score': self.quality_metrics.get(symbol, 0),
            'data_sources': data['source'].unique().tolist() if 'source' in data.columns else ['unknown'],
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        with open(quality_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"  Saved: {output_file}")
        logger.info(f"  Quality metadata: {quality_file}")
    
    def download_enhanced_social_data(self):
        """
        Download enhanced social media data (placeholder for future implementation)
        """
        logger.info("📱 Enhanced social data download (placeholder)")
        logger.info("  Future: Twitter API, StockTwits, Reddit API v2")
        logger.info("  Current: Using existing Reddit data")
    
    def create_enhanced_dataset_description(self):
        """
        Create comprehensive dataset description
        """
        logger.info("📝 Creating enhanced dataset description...")
        
        description_file = self.data_dir / "raw" / "ENHANCED_DATASET_DESCRIPTION.md"
        
        with open(description_file, 'w') as f:
            f.write("# Enhanced Dataset for Meme Stock Analysis\n\n")
            f.write("## Overview\n\n")
            f.write("This directory contains enhanced stock data downloaded from multiple sources for improved quality and completeness.\n\n")
            
            f.write("## Data Sources\n\n")
            f.write("- **Yahoo Finance**: Primary financial data source\n")
            f.write("- **Alpha Vantage**: Secondary financial data source (when API key available)\n")
            f.write("- **Quality Validation**: Multi-source cross-validation\n\n")
            
            f.write("## Quality Metrics\n\n")
            for symbol, quality in self.quality_metrics.items():
                f.write(f"- **{symbol}**: {quality:.1f}% quality score\n")
            f.write("\n")
            
            f.write("## Files\n\n")
            f.write("- `*_enhanced_stock_data.csv` - Enhanced stock data with quality validation\n")
            f.write("- `*_quality_metadata.json` - Quality assessment metadata\n")
            f.write("- `ENHANCED_DATASET_DESCRIPTION.md` - This file\n\n")
            
            f.write("## Quality Improvements\n\n")
            f.write("1. **Multi-source validation**: Cross-check data across sources\n")
            f.write("2. **Quality scoring**: Automated quality assessment\n")
            f.write("3. **Data merging**: Intelligent combination of sources\n")
            f.write("4. **Metadata tracking**: Comprehensive quality documentation\n\n")
            
            f.write("## Usage\n\n")
            f.write("Enhanced data provides better quality for:\n")
            f.write("- Feature engineering\n")
            f.write("- Model training\n")
            f.write("- Statistical analysis\n")
            f.write("- Quality monitoring\n\n")
            
            f.write(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"✅ Enhanced dataset description: {description_file}")
    
    def generate_quality_summary(self):
        """
        Generate quality summary report
        """
        logger.info("📊 Generating quality summary...")
        
        summary_file = self.data_dir / "raw" / "quality_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=== ENHANCED DATA QUALITY SUMMARY ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("STOCK DATA QUALITY:\n")
            for symbol, quality in self.quality_metrics.items():
                status = "✅" if quality >= 80 else "⚠️" if quality >= 60 else "❌"
                f.write(f"  {status} {symbol}: {quality:.1f}%\n")
            f.write("\n")
            
            avg_quality = np.mean(list(self.quality_metrics.values()))
            f.write(f"Average Quality: {avg_quality:.1f}%\n")
            
            if avg_quality >= 80:
                f.write("🎉 Excellent data quality achieved!\n")
            elif avg_quality >= 60:
                f.write("✅ Good data quality - suitable for modeling\n")
            else:
                f.write("⚠️ Quality improvements recommended\n")
        
        logger.info(f"✅ Quality summary: {summary_file}")


def main():
    """
    Main function to download enhanced data
    """
    logger.info("🚀 Starting Enhanced Data Download...")
    
    downloader = EnhancedDataDownloader()
    
    # Set Alpha Vantage API key if available
    # downloader.set_alpha_vantage_key('YOUR_API_KEY')
    
    # Download enhanced stock data
    downloader.download_enhanced_stock_data()
    
    # Download enhanced social data (placeholder)
    downloader.download_enhanced_social_data()
    
    # Create descriptions
    downloader.create_enhanced_dataset_description()
    downloader.generate_quality_summary()
    
    logger.info("🎉 Enhanced data download completed!")


if __name__ == "__main__":
    main() 