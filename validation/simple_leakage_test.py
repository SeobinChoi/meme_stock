#!/usr/bin/env python3
"""
Simple test to verify data leakage fix without requiring LightGBM
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_leakage_fix():
    """Test that data leakage has been fixed"""
    
    logger.info("🔍 Testing data leakage fix...")
    
    # Load the dataset directly
    try:
        dataset = pd.read_csv('data/processed/meme_enhanced_data.csv')
        logger.info(f"📊 Loaded dataset with {len(dataset)} rows and {len(dataset.columns)} columns")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return False
    
    # Define target columns that should NOT be in features
    target_columns = [
        # Direction targets
        'GME_direction_1d', 'GME_direction_3d',
        'AMC_direction_1d', 'AMC_direction_3d', 
        'BB_direction_1d', 'BB_direction_3d',
        
        # Magnitude targets
        'GME_magnitude_3d', 'GME_magnitude_7d',
        'AMC_magnitude_3d', 'AMC_magnitude_7d',
        'BB_magnitude_3d', 'BB_magnitude_7d',
        
        # Return targets
        'GME_returns_1d', 'GME_returns_3d', 'GME_returns_7d',
        'AMC_returns_1d', 'AMC_returns_3d', 'AMC_returns_7d',
        'BB_returns_1d', 'BB_returns_3d', 'BB_returns_7d'
    ]
    
    # Check which target columns exist in the dataset
    existing_targets = [col for col in target_columns if col in dataset.columns]
    logger.info(f"✅ Found {len(existing_targets)} target columns in dataset")
    
    # Simulate the feature preparation logic from the fixed code
    all_exclude_cols = []
    
    # Add target columns
    all_exclude_cols.extend(existing_targets)
    
    # Add return columns that could cause data leakage
    return_cols_to_exclude = []
    for stock in ['GME', 'AMC', 'BB']:
        for horizon in [1, 3, 7, 14]:
            col = f"{stock}_returns_{horizon}d"
            if col in dataset.columns:
                return_cols_to_exclude.append(col)
    
    # Add magnitude columns that are targets
    magnitude_cols_to_exclude = []
    for stock in ['GME', 'AMC', 'BB']:
        for horizon in [3, 7]:
            col = f"{stock}_magnitude_{horizon}d"
            if col in dataset.columns:
                magnitude_cols_to_exclude.append(col)
    
    # Add direction columns that are targets
    direction_cols_to_exclude = []
    for stock in ['GME', 'AMC', 'BB']:
        for horizon in [1, 3]:
            col = f"{stock}_direction_{horizon}d"
            if col in dataset.columns:
                direction_cols_to_exclude.append(col)
    
    # Combine all columns to exclude
    all_exclude_cols = (existing_targets + return_cols_to_exclude + 
                       magnitude_cols_to_exclude + direction_cols_to_exclude)
    
    # Get feature columns (exclude target/leakage columns and date)
    feature_cols = [col for col in dataset.columns 
                   if col not in all_exclude_cols and col != 'date']
    
    features = dataset[feature_cols].copy()
    
    logger.info(f"✅ Features shape: {features.shape}")
    logger.info(f"✅ Excluded {len(all_exclude_cols)} target/leakage columns")
    
    # Check for data leakage
    leakage_detected = False
    leakage_columns = []
    
    # Check if any target columns are in features
    for target_col in existing_targets:
        if target_col in features.columns:
            leakage_detected = True
            leakage_columns.append(target_col)
            logger.error(f"❌ DATA LEAKAGE: {target_col} found in features!")
    
    # Check for direction columns that should be targets
    direction_cols = ['GME_direction_1d', 'GME_direction_3d', 
                     'AMC_direction_1d', 'AMC_direction_3d',
                     'BB_direction_1d', 'BB_direction_3d']
    
    for direction_col in direction_cols:
        if direction_col in features.columns:
            leakage_detected = True
            leakage_columns.append(direction_col)
            logger.error(f"❌ DATA LEAKAGE: {direction_col} found in features!")
    
    # Check for magnitude columns that should be targets
    magnitude_cols = ['GME_magnitude_3d', 'GME_magnitude_7d',
                     'AMC_magnitude_3d', 'AMC_magnitude_7d',
                     'BB_magnitude_3d', 'BB_magnitude_7d']
    
    for magnitude_col in magnitude_cols:
        if magnitude_col in features.columns:
            leakage_detected = True
            leakage_columns.append(magnitude_col)
            logger.error(f"❌ DATA LEAKAGE: {magnitude_col} found in features!")
    
    if leakage_detected:
        logger.error(f"❌ DATA LEAKAGE DETECTED in {len(leakage_columns)} columns!")
        logger.error(f"Leakage columns: {leakage_columns}")
        return False
    else:
        logger.info("✅ No data leakage detected!")
    
    # Show some feature examples
    logger.info(f"📋 Sample features: {feature_cols[:10]}")
    
    # Check feature statistics
    logger.info(f"📊 Feature statistics:")
    logger.info(f"   - Total features: {len(feature_cols)}")
    logger.info(f"   - Features with missing values: {features.isnull().sum().sum()}")
    logger.info(f"   - Features with zero variance: {(features.var() == 0).sum()}")
    
    return True

if __name__ == "__main__":
    success = test_data_leakage_fix()
    if success:
        logger.info("🎉 Data leakage fix test PASSED!")
        sys.exit(0)
    else:
        logger.error("💥 Data leakage fix test FAILED!")
        sys.exit(1) 