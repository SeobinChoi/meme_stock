#!/usr/bin/env python3
"""
Download Historical Stock Data for Meme Stock Period (2020-2021)
Downloads GME, AMC, BB data from the actual meme stock phenomenon period
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_historical_stock_data():
    """
    Download historical stock data from the meme stock period
    """
    logger.info("📈 Downloading historical stock data from meme stock period (2020-2021)...")
    
    # Meme stock symbols
    symbols = ["GME", "AMC", "BB"]
    
    # Create data/raw directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    for symbol in symbols:
        try:
            logger.info(f"Downloading {symbol} historical data...")
            
            # Download historical data from meme stock period
            ticker = yf.Ticker(symbol)
            df = ticker.history(start="2020-01-01", end="2021-12-31")
            
            if df.empty:
                logger.warning(f"No data found for {symbol} in 2020-2021 period")
                continue
            
            # Reset index to get Date as column
            df = df.reset_index()
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Select only needed columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Save to file
            output_file = f"data/raw/{symbol}_stock_data.csv"
            df.to_csv(output_file, index=False)
            
            # Create description file
            description_file = f"data/raw/{symbol}_stock_data_DESCRIPTION.txt"
            with open(description_file, 'w') as f:
                f.write(f"=== {symbol} STOCK DATA DESCRIPTION ===\n\n")
                f.write(f"Data Source: Yahoo Finance (yfinance)\n")
                f.write(f"Symbol: {symbol}\n")
                f.write(f"Period: 2020-01-01 to 2021-12-31\n")
                f.write(f"Total Records: {len(df):,}\n")
                f.write(f"Date Range: {df['Date'].min()} to {df['Date'].max()}\n")
                f.write(f"Columns: {list(df.columns)}\n\n")
                f.write(f"Description:\n")
                f.write(f"This dataset contains historical stock price data for {symbol} during the meme stock phenomenon period.\n")
                f.write(f"The data covers the period when {symbol} experienced significant price movements and high trading volumes.\n")
                f.write(f"This data will be used to analyze the relationship between Reddit sentiment and stock price movements.\n\n")
                f.write(f"Data Quality:\n")
                f.write(f"- Missing values: {df.isnull().sum().sum()}\n")
                f.write(f"- Data completeness: {round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)}%\n")
                f.write(f"- Trading days: {len(df)}\n\n")
                f.write(f"Downloaded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"✅ Downloaded {len(df):,} records for {symbol}")
            logger.info(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"   Saved to: {output_file}")
            logger.info(f"   Description: {description_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to download {symbol} data: {e}")
    
    logger.info("✅ Historical stock data download completed")

def create_dataset_description():
    """
    Create overall dataset description
    """
    logger.info("📝 Creating overall dataset description...")
    
    description_file = "data/raw/HISTORICAL_STOCK_DATA_DESCRIPTION.md"
    with open(description_file, 'w') as f:
        f.write("# Historical Stock Data for Meme Stock Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This directory contains historical stock price data for the meme stock phenomenon period (2020-2021).\n\n")
        f.write("## Files\n\n")
        f.write("- `GME_stock_data.csv` - GameStop Corp. historical data\n")
        f.write("- `AMC_stock_data.csv` - AMC Entertainment Holdings historical data\n")
        f.write("- `BB_stock_data.csv` - BlackBerry Limited historical data\n")
        f.write("- `*_DESCRIPTION.txt` - Individual file descriptions\n\n")
        f.write("## Data Period\n\n")
        f.write("- **Start Date**: January 1, 2020\n")
        f.write("- **End Date**: December 31, 2021\n")
        f.write("- **Coverage**: Full meme stock phenomenon period\n\n")
        f.write("## Data Source\n\n")
        f.write("- **Provider**: Yahoo Finance\n")
        f.write("- **API**: yfinance Python library\n")
        f.write("- **Format**: OHLCV (Open, High, Low, Close, Volume)\n\n")
        f.write("## Purpose\n\n")
        f.write("This data will be used to:\n")
        f.write("1. Analyze price movements during the meme stock period\n")
        f.write("2. Correlate with Reddit sentiment data\n")
        f.write("3. Build predictive models for meme stock behavior\n")
        f.write("4. Study the relationship between social media activity and stock prices\n\n")
        f.write("## Data Quality\n\n")
        f.write("- **Completeness**: High (trading days only)\n")
        f.write("- **Accuracy**: Market data from Yahoo Finance\n")
        f.write("- **Consistency**: Standard OHLCV format\n\n")
        f.write("## Usage\n\n")
        f.write("This data is processed by the Day 2 pipeline for:\n")
        f.write("- Data cleaning and validation\n")
        f.write("- Temporal alignment with Reddit data\n")
        f.write("- Feature engineering for machine learning models\n\n")
        f.write(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info(f"✅ Created overall description: {description_file}")

def main():
    """
    Main function to download historical stock data
    """
    logger.info("🚀 Starting historical stock data download...")
    
    # Download historical data
    download_historical_stock_data()
    
    # Create descriptions
    create_dataset_description()
    
    logger.info("🎉 Historical stock data download and documentation completed!")

if __name__ == "__main__":
    main() 