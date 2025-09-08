# Historical Stock Data for Meme Stock Analysis

## Overview

This directory contains historical stock price data for the meme stock phenomenon period (2020-2021).

## Files

- `GME_stock_data.csv` - GameStop Corp. historical data
- `AMC_stock_data.csv` - AMC Entertainment Holdings historical data
- `BB_stock_data.csv` - BlackBerry Limited historical data
- `*_DESCRIPTION.txt` - Individual file descriptions

## Data Period

- **Start Date**: January 1, 2020
- **End Date**: December 31, 2021
- **Coverage**: Full meme stock phenomenon period

## Data Source

- **Provider**: Yahoo Finance
- **API**: yfinance Python library
- **Format**: OHLCV (Open, High, Low, Close, Volume)

## Purpose

This data will be used to:
1. Analyze price movements during the meme stock period
2. Correlate with Reddit sentiment data
3. Build predictive models for meme stock behavior
4. Study the relationship between social media activity and stock prices

## Data Quality

- **Completeness**: High (trading days only)
- **Accuracy**: Market data from Yahoo Finance
- **Consistency**: Standard OHLCV format

## Usage

This data is processed by the Day 2 pipeline for:
- Data cleaning and validation
- Temporal alignment with Reddit data
- Feature engineering for machine learning models

**Generated on**: 2025-08-04 17:43:25
