#!/usr/bin/env python3
"""
Test script to verify data leakage fix
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from models.baseline_models import BaselineModelTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_leakage_fix():
    """Test that data leakage has been fixed"""
    
    logger.info("🔍 Testing data leakage fix...")
    
    # Initialize trainer
    trainer = BaselineModelTrainer()
    
    # Load data
    dataset = trainer._load_engineered_dataset()
    if dataset is None:
        logger.error("❌ Failed to load dataset")
        return False
    
    logger.info(f"📊 Loaded dataset with {len(dataset)} rows and {len(dataset.columns)} columns")
    
    # Prepare targets and features
    data = trainer._prepare_targets_and_features(dataset)
    if data is None:
        logger.error("❌ Failed to prepare targets and features")
        return False
    
    features = data['features']
    targets = data['targets']
    
    logger.info(f"✅ Features shape: {features.shape}")
    logger.info(f"✅ Number of targets: {len(targets)}")
    
    # Check for data leakage
    leakage_detected = False
    leakage_columns = []
    
    # Check if any target columns are in features
    for target_col in targets.keys():
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
    
    # Test a simple model to check for overfitting
    logger.info("🧪 Testing model for overfitting...")
    
    try:
        # Train a simple LightGBM model
        from lightgbm import LGBMClassifier
        import numpy as np
        from sklearn.metrics import accuracy_score
        
        # Use first target for testing
        target_col = list(targets.keys())[0]
        logger.info(f"Testing with target: {target_col}")
        
        train_data = data['train_data']
        test_data = data['test_data']
        
        feature_cols = [col for col in train_data.columns if col not in targets.keys()]
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Train model with regularization
        model = LGBMClassifier(
            num_leaves=15,
            learning_rate=0.01,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20,
            min_data_in_leaf=10,
            verbose=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        logger.info(f"📈 Train accuracy: {train_accuracy:.4f}")
        logger.info(f"📈 Test accuracy: {test_accuracy:.4f}")
        
        # Check for overfitting (train accuracy should not be much higher than test accuracy)
        accuracy_diff = train_accuracy - test_accuracy
        
        if accuracy_diff > 0.1:  # More than 10% difference suggests overfitting
            logger.warning(f"⚠️  Potential overfitting detected: {accuracy_diff:.4f} accuracy difference")
        else:
            logger.info(f"✅ No overfitting detected: {accuracy_diff:.4f} accuracy difference")
        
        if test_accuracy > 0.95:  # Suspiciously high accuracy
            logger.warning(f"⚠️  Suspiciously high test accuracy: {test_accuracy:.4f}")
        else:
            logger.info(f"✅ Reasonable test accuracy: {test_accuracy:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    success = test_data_leakage_fix()
    if success:
        logger.info("🎉 Data leakage fix test PASSED!")
        sys.exit(0)
    else:
        logger.error("💥 Data leakage fix test FAILED!")
        sys.exit(1) 