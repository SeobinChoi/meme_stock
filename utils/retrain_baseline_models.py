#!/usr/bin/env python3
"""
Retrain Baseline Models with Clean Data (Post Data Leakage Fix)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import the baseline trainer
from src.models.baseline_models import BaselineModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanDataBaselineTrainer(BaselineModelTrainer):
    """
    Modified baseline trainer to use clean dataset
    """
    
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir)
        self.clean_features_path = None
        self.clean_targets_path = None
        self._find_clean_datasets()
    
    def _find_clean_datasets(self):
        """Find the latest clean datasets"""
        features_dir = Path(self.data_dir) / "features"
        
        # Find clean features dataset
        clean_features_files = list(features_dir.glob("clean_features_dataset_*.csv"))
        if clean_features_files:
            self.clean_features_path = sorted(clean_features_files)[-1]  # Get latest
            logger.info(f"Found clean features dataset: {self.clean_features_path}")
        
        # Find targets dataset
        target_files = list(features_dir.glob("targets_dataset_*.csv"))
        if target_files:
            self.clean_targets_path = sorted(target_files)[-1]  # Get latest
            logger.info(f"Found targets dataset: {self.clean_targets_path}")
    
    def _load_engineered_dataset(self) -> Optional[pd.DataFrame]:
        """
        Load the clean engineered dataset (no data leakage)
        """
        try:
            if not self.clean_features_path or not self.clean_targets_path:
                logger.error("Clean datasets not found")
                return None
            
            # Load clean features and targets
            features_df = pd.read_csv(self.clean_features_path)
            targets_df = pd.read_csv(self.clean_targets_path)
            
            logger.info(f"✅ Loaded clean features: {features_df.shape}")
            logger.info(f"✅ Loaded targets: {targets_df.shape}")
            
            # Combine features and targets (they should have same index)
            if 'date' in features_df.columns and 'date' in targets_df.columns:
                # Merge on date column
                combined_df = pd.merge(features_df, targets_df, on='date', how='inner')
            else:
                # Merge on index
                combined_df = pd.concat([features_df, targets_df], axis=1)
            
            logger.info(f"✅ Combined dataset shape: {combined_df.shape}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading clean dataset: {e}")
            return None
    
    def _prepare_targets_and_features(self, dataset: pd.DataFrame) -> Optional[Dict]:
        """
        Prepare targets and features from clean dataset
        """
        try:
            logger.info(f"📊 Dataset shape: {dataset.shape}")
            logger.info(f"📊 Dataset columns: {list(dataset.columns)}")
            
            # Separate features and targets based on column names
            target_cols = [col for col in dataset.columns 
                          if any(pattern in col for pattern in ['_returns_', '_direction_'])]
            
            # Get feature columns (everything except targets, dates, and indices)
            exclude_cols = target_cols + ['date', 'Unnamed: 0'] + [col for col in dataset.columns if 'index' in col.lower()]
            feature_cols = [col for col in dataset.columns if col not in exclude_cols]
            
            logger.info(f"📈 Found {len(target_cols)} target columns")
            logger.info(f"🎯 Found {len(feature_cols)} feature columns")
            
            # Create targets dataframe
            targets = dataset[target_cols]
            
            # Create features dataframe
            features = dataset[feature_cols]
            
            # Remove any remaining NaN values
            features = features.fillna(0)
            targets = targets.fillna(0)
            
            logger.info(f"✅ Features shape after cleaning: {features.shape}")
            logger.info(f"✅ Targets shape after cleaning: {targets.shape}")
            
            return {
                'features': features,
                'targets': targets,
                'target_columns': target_cols,
                'feature_columns': feature_cols
            }
            
        except Exception as e:
            logger.error(f"Error preparing targets and features: {e}")
            return None

def main():
    """
    Main function to retrain baseline models with clean data
    """
    print("🚀 Starting Baseline Model Retraining with Clean Data")
    print("="*60)
    
    # Initialize trainer
    trainer = CleanDataBaselineTrainer()
    
    # Run the baseline model development
    results = trainer.run_baseline_model_development()
    
    # Log results
    if results['status'] == 'COMPLETED':
        print("\n✅ Baseline Model Retraining COMPLETED!")
        print(f"Models trained: {results.get('summary', {}).get('models_trained', 'N/A')}")
        
        # Update our progress log
        log_path = Path("reports/autonomous_implementation_log_20250810.md")
        if log_path.exists():
            with open(log_path, 'a') as f:
                f.write(f"\n\n#### **Task 2.1: Retrain Baseline Models - COMPLETED ✅**\n")
                f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Status:** SUCCESS\n")
                f.write(f"**Models Trained:** {results.get('summary', {}).get('models_trained', 'N/A')}\n")
                f.write(f"**Performance:** Models retrained with clean dataset (no data leakage)\n")
                
    else:
        print(f"\n❌ Baseline Model Retraining FAILED: {results.get('message', 'Unknown error')}")
        
        # Log failure
        log_path = Path("reports/autonomous_implementation_log_20250810.md")
        if log_path.exists():
            with open(log_path, 'a') as f:
                f.write(f"\n\n#### **Task 2.1: Retrain Baseline Models - FAILED ❌**\n")
                f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Status:** FAILED\n")
                f.write(f"**Error:** {results.get('message', 'Unknown error')}\n")

if __name__ == "__main__":
    main()