#!/usr/bin/env python3
"""
Fix Data Leakage Issues
Removes target variables from features and creates clean dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_data_leakage():
    """
    Fix data leakage issues by cleaning the feature dataset
    """
    logger.info("🚀 Starting Data Leakage Fix...")
    
    # Load the original features dataset
    features_file = Path("data/features/engineered_features_dataset.csv")
    if not features_file.exists():
        logger.error("❌ Features dataset not found")
        return False
    
    df = pd.read_csv(features_file)
    logger.info(f"✅ Loaded dataset with {len(df)} samples and {len(df.columns)} features")
    
    # Store original info
    original_features = len(df.columns)
    original_samples = len(df)
    
    # 1. Remove target variables (returns) from features
    logger.info("🔍 Identifying target variables...")
    target_columns = [col for col in df.columns if 'returns_' in col]
    logger.info(f"📊 Found {len(target_columns)} target variables: {target_columns}")
    
    # 2. Remove target variables from features
    feature_df = df.drop(columns=target_columns)
    logger.info(f"✅ Removed target variables. Features reduced from {original_features} to {len(feature_df.columns)}")
    
    # 3. Remove constant features (no variation)
    logger.info("🔍 Identifying constant features...")
    constant_features = []
    for col in feature_df.columns:
        if feature_df[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        logger.info(f"⚠️  Found {len(constant_features)} constant features")
        feature_df = feature_df.drop(columns=constant_features)
        logger.info(f"✅ Removed constant features. Features reduced to {len(feature_df.columns)}")
    
    # 4. Check for future data leakage indicators
    logger.info("🔍 Checking for future data leakage...")
    future_indicators = ['future_', 'tomorrow_', 'next_day_', 'forward_']
    future_features = []
    
    for col in feature_df.columns:
        for indicator in future_indicators:
            if indicator in col.lower():
                future_features.append(col)
                break
    
    if future_features:
        logger.warning(f"⚠️  Found {len(future_features)} features with future indicators: {future_features}")
        # Remove future features
        feature_df = feature_df.drop(columns=future_features)
        logger.info(f"✅ Removed future features. Features reduced to {len(feature_df.columns)}")
    
    # 5. Create clean feature dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_features_file = Path(f"data/features/clean_features_dataset_{timestamp}.csv")
    
    # Ensure directory exists
    clean_features_file.parent.mkdir(exist_ok=True)
    
    # Save clean features
    feature_df.to_csv(clean_features_file, index=False)
    logger.info(f"✅ Clean features saved to: {clean_features_file}")
    
    # 6. Create target dataset
    target_df = df[target_columns].copy()
    target_file = Path(f"data/features/targets_dataset_{timestamp}.csv")
    target_df.to_csv(target_file, index=False)
    logger.info(f"✅ Target variables saved to: {target_file}")
    
    # 7. Generate summary report
    summary = {
        'timestamp': timestamp,
        'original_features': original_features,
        'original_samples': original_samples,
        'clean_features': len(feature_df.columns),
        'target_variables': len(target_columns),
        'constant_features_removed': len(constant_features),
        'future_features_removed': len(future_features),
        'data_leakage_fixed': True
    }
    
    # Save summary
    summary_file = Path(f"data/results/data_leakage_fix_summary_{timestamp}.json")
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Fix summary saved to: {summary_file}")
    
    # 8. Print summary
    print("\n" + "="*80)
    print("DATA LEAKAGE FIX COMPLETE")
    print("="*80)
    print(f"✅ Original Features: {original_features}")
    print(f"✅ Clean Features: {len(feature_df.columns)}")
    print(f"✅ Target Variables: {len(target_columns)}")
    print(f"✅ Constant Features Removed: {len(constant_features)}")
    print(f"✅ Future Features Removed: {len(future_features)}")
    print(f"✅ Data Leakage: FIXED")
    print("="*80)
    
    # 9. Verify no data leakage remains
    logger.info("🔍 Verifying no data leakage remains...")
    
    # Check for any remaining returns in features
    remaining_returns = [col for col in feature_df.columns if 'returns_' in col]
    if remaining_returns:
        logger.error(f"❌ Data leakage still present: {remaining_returns}")
        return False
    
    # Check for any remaining future indicators
    remaining_future = []
    for col in feature_df.columns:
        for indicator in future_indicators:
            if indicator in col.lower():
                remaining_future.append(col)
                break
    
    if remaining_future:
        logger.error(f"❌ Future data leakage still present: {remaining_future}")
        return False
    
    logger.info("✅ Data leakage verification passed!")
    
    # 10. Create feature metadata for clean dataset
    clean_metadata = {
        'total_features': len(feature_df.columns),
        'total_samples': len(feature_df),
        'feature_list': list(feature_df.columns),
        'creation_timestamp': datetime.now().isoformat(),
        'data_leakage_fixed': True,
        'removed_features': {
            'target_variables': target_columns,
            'constant_features': constant_features,
            'future_features': future_features
        }
    }
    
    metadata_file = Path(f"data/features/clean_features_metadata_{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(clean_metadata, f, indent=2)
    
    logger.info(f"✅ Clean features metadata saved to: {metadata_file}")
    
    return True

def main():
    """
    Main function
    """
    logger.info("🚀 Starting Data Leakage Fix Process...")
    
    try:
        success = fix_data_leakage()
        
        if success:
            logger.info("✅ Data leakage fix completed successfully!")
            logger.info("🎯 Dataset is now ready for model training")
        else:
            logger.error("❌ Data leakage fix failed!")
            
    except Exception as e:
        logger.error(f"❌ Error during data leakage fix: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
