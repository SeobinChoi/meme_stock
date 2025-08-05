"""
Day 5-6: Baseline Model Development
Establish competitive baseline models for performance benchmarking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, mean_squared_error, 
    mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModelTrainer:
    """
    Baseline model development for Day 5-6
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.scalers = {}
        
        # Model configurations
        self.classification_targets = ['GME_direction_1d', 'GME_direction_3d', 
                                     'AMC_direction_1d', 'AMC_direction_3d',
                                     'BB_direction_1d', 'BB_direction_3d']
        
        self.regression_targets = ['GME_returns_3d', 'GME_returns_7d',
                                 'AMC_returns_3d', 'AMC_returns_7d',
                                 'BB_returns_3d', 'BB_returns_7d']
        
    def run_baseline_model_development(self) -> Dict:
        """
        Run complete baseline model development pipeline
        """
        logger.info("🚀 Starting Day 5-6: Baseline Model Development")
        
        # Step 1: Load engineered dataset
        logger.info("="*50)
        logger.info("STEP 1: Loading Engineered Dataset")
        logger.info("="*50)
        
        dataset = self._load_engineered_dataset()
        if dataset is None:
            return {"status": "ERROR", "message": "Failed to load engineered dataset"}
        
        # Step 2: Prepare targets and features
        logger.info("="*50)
        logger.info("STEP 2: Preparing Targets and Features")
        logger.info("="*50)
        
        prepared_data = self._prepare_targets_and_features(dataset)
        if prepared_data is None:
            return {"status": "ERROR", "message": "Failed to prepare targets and features"}
        
        # Step 3: Train LightGBM models for classification
        logger.info("="*50)
        logger.info("STEP 3: Training LightGBM Classification Models")
        logger.info("="*50)
        
        lgb_results = self._train_lightgbm_classification(prepared_data)
        self.results['lightgbm_classification'] = lgb_results
        
        # Step 4: Train XGBoost models for regression
        logger.info("="*50)
        logger.info("STEP 4: Training XGBoost Regression Models")
        logger.info("="*50)
        
        xgb_results = self._train_xgboost_regression(prepared_data)
        self.results['xgboost_regression'] = xgb_results
        
        # Step 5: Train LSTM models (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            logger.info("="*50)
            logger.info("STEP 5: Training LSTM Models")
            logger.info("="*50)
            
            lstm_results = self._train_lstm_models(prepared_data)
            self.results['lstm'] = lstm_results
        else:
            logger.warning("TensorFlow not available. Skipping LSTM models.")
        
        # Step 6: Model evaluation and analysis
        logger.info("="*50)
        logger.info("STEP 6: Model Evaluation and Analysis")
        logger.info("="*50)
        
        evaluation_results = self._evaluate_all_models()
        
        # Step 7: Save models and results
        logger.info("="*50)
        logger.info("STEP 7: Saving Models and Results")
        logger.info("="*50)
        
        self._save_models_and_results()
        
        # Step 8: Generate completion report
        completion_report = self._generate_completion_report()
        self._save_completion_report(completion_report)
        
        logger.info("✅ Day 5-6: Baseline Model Development Completed")
        return completion_report
    
    def _load_engineered_dataset(self) -> Optional[pd.DataFrame]:
        """
        Load the engineered dataset
        """
        try:
            dataset_path = self.data_dir / "features" / "engineered_features_dataset.csv"
            if not dataset_path.exists():
                logger.error(f"Dataset not found: {dataset_path}")
                return None
            
            dataset = pd.read_csv(dataset_path)
            logger.info(f"✅ Loaded dataset with shape: {dataset.shape}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def _prepare_targets_and_features(self, dataset: pd.DataFrame) -> Optional[Dict]:
        """
        Prepare targets and features for modeling
        """
        try:
            # Create target variables
            targets = {}
            
            # Classification targets (price direction)
            for stock in ['GME', 'AMC', 'BB']:
                for horizon in [1, 3]:
                    target_col = f"{stock}_direction_{horizon}d"
                    returns_col = f"{stock}_returns_{horizon}d"
                    
                    if returns_col in dataset.columns:
                        # Create binary direction target (1 for positive, 0 for negative)
                        targets[target_col] = (dataset[returns_col] > 0).astype(int)
                        logger.info(f"Created target: {target_col}")
            
            # Regression targets (price returns)
            for stock in ['GME', 'AMC', 'BB']:
                for horizon in [3, 7]:
                    target_col = f"{stock}_returns_{horizon}d"
                    if target_col in dataset.columns:
                        targets[target_col] = dataset[target_col]
                        logger.info(f"Using target: {target_col}")
            
            # Remove target columns and their related return columns from features to prevent data leakage
            target_cols = list(targets.keys())
            
            # Also exclude return columns that could cause data leakage
            return_cols_to_exclude = []
            for stock in ['GME', 'AMC', 'BB']:
                for horizon in [1, 3, 7, 14]:
                    return_cols_to_exclude.append(f"{stock}_returns_{horizon}d")
            
            # CRITICAL FIX: Also exclude magnitude columns that are targets
            magnitude_cols_to_exclude = []
            for stock in ['GME', 'AMC', 'BB']:
                for horizon in [3, 7]:
                    magnitude_cols_to_exclude.append(f"{stock}_magnitude_{horizon}d")
            
            # CRITICAL FIX: Also exclude direction columns that are targets
            direction_cols_to_exclude = []
            for stock in ['GME', 'AMC', 'BB']:
                for horizon in [1, 3]:
                    direction_cols_to_exclude.append(f"{stock}_direction_{horizon}d")
            
            # Combine all columns to exclude
            all_exclude_cols = (target_cols + return_cols_to_exclude + 
                              magnitude_cols_to_exclude + direction_cols_to_exclude)
            
            feature_cols = [col for col in dataset.columns 
                          if col not in all_exclude_cols and col != 'date']
            
            features = dataset[feature_cols].copy()
            
            # Handle missing values
            features = features.fillna(features.mean())
            
            # Create a combined dataset for proper train/test split
            combined_data = pd.concat([features, pd.DataFrame(targets)], axis=1)
            combined_data = combined_data.dropna()
            
            if len(combined_data) == 0:
                logger.error("No valid data after removing missing values")
                return None
            
            # Create time-based train/test split to prevent data leakage
            # Use first 80% for training, last 20% for testing
            split_idx = int(len(combined_data) * 0.8)
            train_data = combined_data.iloc[:split_idx]
            test_data = combined_data.iloc[split_idx:]
            
            logger.info(f"✅ Prepared {len(features.columns)} features and {len(targets)} targets")
            logger.info(f"✅ Train set: {len(train_data)} samples, Test set: {len(test_data)} samples")
            logger.info(f"✅ Excluded {len(all_exclude_cols)} target/leakage columns")
            
            return {
                'features': features,
                'targets': targets,
                'dates': dataset['date'] if 'date' in dataset.columns else None,
                'train_data': train_data,
                'test_data': test_data
            }
            
        except Exception as e:
            logger.error(f"Error preparing targets and features: {e}")
            return None
    
    def _train_lightgbm_classification(self, data: Dict) -> Dict:
        """
        Train LightGBM models for classification tasks
        """
        results = {}
        features = data['features']
        targets = data['targets']
        
        # Filter classification targets
        classification_targets = [col for col in targets.keys() if 'direction' in col]
        
        for target in classification_targets:
            if target not in targets:
                continue
                
            logger.info(f"Training LightGBM for {target}")
            
            try:
                # Use train/test split from prepared data
                train_data = data['train_data']
                test_data = data['test_data']
                
                # Prepare features and target
                feature_cols = [col for col in train_data.columns if col not in targets.keys()]
                X_train = train_data[feature_cols]
                y_train = train_data[target]
                X_test = test_data[feature_cols]
                y_test = test_data[target]
                
                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"No valid data for {target}")
                    continue
                
                # Model parameters with regularization to prevent overfitting
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 15,  # Reduced from 31 to prevent overfitting
                    'learning_rate': 0.01,  # Reduced from 0.05 for better generalization
                    'feature_fraction': 0.7,  # Reduced from 0.9 to prevent overfitting
                    'bagging_fraction': 0.7,  # Reduced from 0.8 to prevent overfitting
                    'bagging_freq': 5,
                    'verbose': -1,
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 0.1,  # L2 regularization
                    'min_child_samples': 20,  # Minimum samples per leaf
                    'min_data_in_leaf': 10  # Minimum data in leaf
                }
                
                # Train model on training data
                train_dataset = lgb.Dataset(X_train, label=y_train)
                val_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)
                
                model = lgb.train(
                    params,
                    train_dataset,
                    valid_sets=[val_dataset],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                # Predictions on test set
                y_pred_proba = model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Feature importance
                importance = model.feature_importance(importance_type='gain')
                feature_importance = dict(zip(X_train.columns, importance))
                
                results[target] = {
                    'test_scores': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc
                    },
                    'feature_importance': feature_importance
                }
                
                logger.info(f"✅ {target}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
                
            except Exception as e:
                logger.error(f"Error training LightGBM for {target}: {e}")
                results[target] = {'error': str(e)}
        
        return results
    
    def _train_xgboost_regression(self, data: Dict) -> Dict:
        """
        Train XGBoost models for regression tasks
        """
        results = {}
        features = data['features']
        targets = data['targets']
        
        # Filter regression targets
        regression_targets = [col for col in targets.keys() if 'returns' in col]
        
        for target in regression_targets:
            if target not in targets:
                continue
                
            logger.info(f"Training XGBoost for {target}")
            
            try:
                # Use train/test split from prepared data
                train_data = data['train_data']
                test_data = data['test_data']
                
                # Prepare features and target
                feature_cols = [col for col in train_data.columns if col not in targets.keys()]
                X_train = train_data[feature_cols]
                y_train = train_data[target]
                X_test = test_data[feature_cols]
                y_test = test_data[target]
                
                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"No valid data for {target}")
                    continue
                
                # Model parameters with regularization to prevent overfitting
                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': 4,  # Reduced from 6 to prevent overfitting
                    'learning_rate': 0.05,  # Reduced from 0.1 for better generalization
                    'subsample': 0.7,  # Reduced from 0.8 to prevent overfitting
                    'colsample_bytree': 0.7,  # Reduced from 0.8 to prevent overfitting
                    'n_estimators': 100,  # Reduced from 1000 to prevent overfitting
                    'random_state': 42,
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 0.1,  # L2 regularization
                    'min_child_weight': 3,  # Minimum sum of instance weight in child
                    'gamma': 0.1  # Minimum loss reduction for split
                }
                
                # Train model on training data
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=0
                )
                
                # Predictions on test set
                y_pred = model.predict(X_test)
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Directional accuracy
                y_direction = (y_test > 0).astype(int)
                pred_direction = (y_pred > 0).astype(int)
                directional_accuracy = accuracy_score(y_direction, pred_direction)
                
                # Feature importance
                importance = model.feature_importances_
                feature_importance = dict(zip(X_train.columns, importance))
                
                results[target] = {
                    'test_scores': {
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'directional_accuracy': directional_accuracy
                    },
                    'feature_importance': feature_importance
                }
                
                logger.info(f"✅ {target}: RMSE={rmse:.4f}, R²={r2:.3f}")
                
            except Exception as e:
                logger.error(f"Error training XGBoost for {target}: {e}")
                results[target] = {'error': str(e)}
        
        return results
    
    def _train_lstm_models(self, data: Dict) -> Dict:
        """
        Train LSTM models for sequential pattern recognition
        """
        if not TENSORFLOW_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        results = {}
        features = data['features']
        targets = data['targets']
        
        # Use a subset of targets for LSTM (due to computational constraints)
        lstm_targets = ['GME_direction_1d', 'GME_returns_3d']
        
        for target in lstm_targets:
            if target not in targets:
                continue
                
            logger.info(f"Training LSTM for {target}")
            
            try:
                # Prepare data
                X = features.copy()
                y = targets[target]
                
                # Remove rows with missing targets
                valid_mask = ~y.isna()
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) < 100:  # Need sufficient data for LSTM
                    logger.warning(f"Insufficient data for LSTM: {target}")
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Create sequences (30-day lookback)
                sequence_length = 30
                X_sequences = []
                y_sequences = []
                
                for i in range(sequence_length, len(X_scaled)):
                    X_sequences.append(X_scaled[i-sequence_length:i])
                    y_sequences.append(y.iloc[i])
                
                X_sequences = np.array(X_sequences)
                y_sequences = np.array(y_sequences)
                
                # Split data
                split_idx = int(0.8 * len(X_sequences))
                X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
                y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1, activation='sigmoid' if 'direction' in target else 'linear')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy' if 'direction' in target else 'mse',
                    metrics=['accuracy'] if 'direction' in target else ['mae']
                )
                
                # Callbacks
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ]
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Predictions
                y_pred = model.predict(X_val).flatten()
                
                # Metrics
                if 'direction' in target:
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    accuracy = accuracy_score(y_val, y_pred_binary)
                    precision = precision_score(y_val, y_pred_binary, zero_division=0)
                    recall = recall_score(y_val, y_pred_binary, zero_division=0)
                    f1 = f1_score(y_val, y_pred_binary, zero_division=0)
                    auc = roc_auc_score(y_val, y_pred)
                    
                    metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc
                    }
                else:
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    metrics = {
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }
                
                results[target] = {
                    'metrics': metrics,
                    'history': history.history,
                    'model_summary': model.summary()
                }
                
                logger.info(f"✅ LSTM {target}: {list(metrics.values())[0]:.3f}")
                
            except Exception as e:
                logger.error(f"Error training LSTM for {target}: {e}")
                results[target] = {'error': str(e)}
        
        return results
    
    def _evaluate_all_models(self) -> Dict:
        """
        Comprehensive evaluation of all models
        """
        logger.info("Evaluating all models...")
        
        evaluation = {
            'summary': {},
            'best_models': {},
            'feature_importance_summary': {}
        }
        
        # Summarize results by model type
        for model_type, results in self.results.items():
            if model_type == 'lightgbm_classification':
                accuracies = []
                for target, result in results.items():
                    if 'avg_scores' in result:
                        accuracies.append(result['avg_scores']['accuracy'])
                
                if accuracies:
                    evaluation['summary']['lightgbm_classification'] = {
                        'avg_accuracy': np.mean(accuracies),
                        'best_accuracy': np.max(accuracies),
                        'num_models': len(accuracies)
                    }
            
            elif model_type == 'xgboost_regression':
                r2_scores = []
                for target, result in results.items():
                    if 'avg_scores' in result:
                        r2_scores.append(result['avg_scores']['r2'])
                
                if r2_scores:
                    evaluation['summary']['xgboost_regression'] = {
                        'avg_r2': np.mean(r2_scores),
                        'best_r2': np.max(r2_scores),
                        'num_models': len(r2_scores)
                    }
        
        logger.info("✅ Model evaluation completed")
        return evaluation
    
    def _save_models_and_results(self):
        """
        Save trained models and results
        """
        try:
            # Save results
            results_path = self.data_dir / "models" / "baseline_results.json"
            results_path.parent.mkdir(exist_ok=True)
            
            # Convert results to serializable format
            serializable_results = {}
            for model_type, results in self.results.items():
                serializable_results[model_type] = {}
                for target, result in results.items():
                    if isinstance(result, dict):
                        serializable_results[model_type][target] = result
            
            import json
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"✅ Saved results to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving models and results: {e}")
    
    def _generate_completion_report(self) -> Dict:
        """
        Generate completion report for Day 5-6
        """
        report = {
            "day": "5-6",
            "task": "Baseline Model Development",
            "status": "COMPLETED",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "models_trained": len(self.results),
                "classification_targets": len([t for t in self.classification_targets if t in self.results.get('lightgbm_classification', {})]),
                "regression_targets": len([t for t in self.regression_targets if t in self.results.get('xgboost_regression', {})]),
                "lstm_models": len(self.results.get('lstm', {}))
            },
            "performance_summary": {},
            "deliverables": self._assess_deliverables(),
            "next_steps": self._generate_next_steps()
        }
        
        # Add performance summary
        for model_type, results in self.results.items():
            if model_type == 'lightgbm_classification':
                accuracies = []
                for target, result in results.items():
                    if 'avg_scores' in result:
                        accuracies.append(result['avg_scores']['accuracy'])
                
                if accuracies:
                    report["performance_summary"]["lightgbm_classification"] = {
                        "avg_accuracy": np.mean(accuracies),
                        "best_accuracy": np.max(accuracies)
                    }
            
            elif model_type == 'xgboost_regression':
                r2_scores = []
                for target, result in results.items():
                    if 'avg_scores' in result:
                        r2_scores.append(result['avg_scores']['r2'])
                
                if r2_scores:
                    report["performance_summary"]["xgboost_regression"] = {
                        "avg_r2": np.mean(r2_scores),
                        "best_r2": np.max(r2_scores)
                    }
        
        return report
    
    def _assess_deliverables(self) -> Dict:
        """
        Assess completion of deliverables
        """
        deliverables = {
            "trained_baseline_models": len(self.results) > 0,
            "performance_evaluation": len(self.results) > 0,
            "feature_importance_analysis": any('feature_importance' in str(result) for result in self.results.values()),
            "model_comparison_framework": len(self.results) >= 2
        }
        
        return deliverables
    
    def _generate_next_steps(self) -> List[str]:
        """
        Generate next steps for Week 2
        """
        return [
            "Implement advanced meme-specific features (Day 8-9)",
            "Develop multi-modal transformer architecture (Day 10-11)",
            "Create ensemble methods and meta-learning (Day 12-13)",
            "Optimize hyperparameters and model selection (Day 14)"
        ]
    
    def _save_completion_report(self, report: Dict):
        """
        Save completion report
        """
        try:
            # Get next sequence number
            sequence_num = self._get_next_sequence_number("day")
            
            # Save report
            report_path = Path("results") / f"{sequence_num:03d}_day5_6_baseline_models_summary.txt"
            
            with open(report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write(f"DAY 5-6: BASELINE MODEL DEVELOPMENT SUMMARY\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Status: {report['status']}\n")
                f.write(f"Timestamp: {report['timestamp']}\n\n")
                
                f.write("SUMMARY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Models Trained: {report['summary']['models_trained']}\n")
                f.write(f"Classification Targets: {report['summary']['classification_targets']}\n")
                f.write(f"Regression Targets: {report['summary']['regression_targets']}\n")
                f.write(f"LSTM Models: {report['summary']['lstm_models']}\n\n")
                
                f.write("PERFORMANCE SUMMARY:\n")
                f.write("-" * 20 + "\n")
                for model_type, perf in report['performance_summary'].items():
                    f.write(f"{model_type.upper()}:\n")
                    for metric, value in perf.items():
                        f.write(f"  {metric}: {value:.3f}\n")
                    f.write("\n")
                
                f.write("DELIVERABLES:\n")
                f.write("-" * 20 + "\n")
                for deliverable, completed in report['deliverables'].items():
                    status = "✅" if completed else "❌"
                    f.write(f"{status} {deliverable}\n")
                
                f.write("\nNEXT STEPS:\n")
                f.write("-" * 20 + "\n")
                for i, step in enumerate(report['next_steps'], 1):
                    f.write(f"{i}. {step}\n")
            
            logger.info(f"✅ Saved completion report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving completion report: {e}")
    
    def _get_next_sequence_number(self, prefix: str) -> int:
        """
        Get next sequence number for file naming
        """
        results_dir = Path("results")
        if not results_dir.exists():
            return 1
        
        existing_files = [f for f in results_dir.glob(f"*_{prefix}*_summary.txt")]
        if not existing_files:
            return 1
        
        numbers = []
        for file in existing_files:
            try:
                num = int(file.name.split('_')[0])
                numbers.append(num)
            except:
                continue
        
        return max(numbers) + 1 if numbers else 1

def main():
    """
    Main function to run baseline model development
    """
    trainer = BaselineModelTrainer()
    results = trainer.run_baseline_model_development()
    
    if results['status'] == 'COMPLETED':
        print("✅ Day 5-6: Baseline Model Development completed successfully!")
        print(f"Models trained: {results['summary']['models_trained']}")
    else:
        print(f"❌ Day 5-6: Baseline Model Development failed: {results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main() 