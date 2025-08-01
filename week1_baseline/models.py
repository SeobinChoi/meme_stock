"""
Baseline Models for Meme Stock Prediction
Week 1 Implementation - Academic Competition Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    import lightgbm as lgb
    import xgboost as xgb
    LGB_AVAILABLE = True
    XGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    XGB_AVAILABLE = False
    print("⚠️ LightGBM/XGBoost not available. Using simplified models.")

# Import DL libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM model will be skipped.")

class BaselineModels:
    def __init__(self, data_path='../data/features_data.csv'):
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load feature-engineered data"""
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"✅ Loaded data: {self.data.shape}")
        except FileNotFoundError:
            print("❌ Features data not found. Please run feature_engineering.py first.")
            return False
        return True
    
    def prepare_data(self, target_col, lookback_days=60):
        """Prepare data for modeling"""
        print(f"🎯 Preparing data for target: {target_col}")
        
        # Get feature columns (exclude date and targets)
        exclude_cols = ['date'] + [col for col in self.data.columns if 'direction' in col or 'magnitude' in col]
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Remove rows with NaN in target
        clean_data = self.data.dropna(subset=[target_col])
        
        X = clean_data[feature_cols].values
        y = clean_data[target_col].values
        
        print(f"📊 Features: {X.shape[1]}, Samples: {X.shape[0]}")
        return X, y, feature_cols
    
    def train_lightgbm(self, target_col, params=None):
        """Train LightGBM model for short-term prediction"""
        if not LGB_AVAILABLE:
            print("❌ LightGBM not available")
            return None
            
        print("🌳 Training LightGBM model...")
        
        X, y, feature_cols = self.prepare_data(target_col)
        
        # Default parameters
        if params is None:
            params = {
                'objective': 'binary' if 'direction' in target_col else 'regression',
                'metric': 'binary_logloss' if 'direction' in target_col else 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        feature_importance = np.zeros(X.shape[1])
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Predictions
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            
            # Metrics
            if 'direction' in target_col:
                y_pred_binary = (y_pred > 0.5).astype(int)
                score = accuracy_score(y_val, y_pred_binary)
            else:
                score = mean_squared_error(y_val, y_pred, squared=False)
            
            cv_scores.append(score)
            feature_importance += model.feature_importance()
            
            print(f"Fold {fold+1}: Score = {score:.4f}")
        
        # Train final model on full data
        full_data = lgb.Dataset(X, label=y)
        final_model = lgb.train(params, full_data, num_boost_round=1000)
        
        # Store results
        self.models[f'lightgbm_{target_col}'] = final_model
        self.results[f'lightgbm_{target_col}'] = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'feature_importance': feature_importance / 5,
            'feature_names': feature_cols
        }
        
        print(f"✅ LightGBM training completed. Mean CV score: {np.mean(cv_scores):.4f}")
        return final_model
    
    def train_xgboost(self, target_col, params=None):
        """Train XGBoost model for long-term prediction"""
        if not XGB_AVAILABLE:
            print("❌ XGBoost not available")
            return None
            
        print("🌲 Training XGBoost model...")
        
        X, y, feature_cols = self.prepare_data(target_col)
        
        # Default parameters
        if params is None:
            params = {
                'objective': 'binary:logistic' if 'direction' in target_col else 'reg:squarederror',
                'eval_metric': 'logloss' if 'direction' in target_col else 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        feature_importance = np.zeros(X.shape[1])
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = xgb.XGBRegressor(**params) if 'magnitude' in target_col else xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            
            # Metrics
            if 'direction' in target_col:
                score = accuracy_score(y_val, y_pred)
            else:
                score = mean_squared_error(y_val, y_pred, squared=False)
            
            cv_scores.append(score)
            feature_importance += model.feature_importances_
            
            print(f"Fold {fold+1}: Score = {score:.4f}")
        
        # Train final model on full data
        final_model = xgb.XGBRegressor(**params) if 'magnitude' in target_col else xgb.XGBClassifier(**params)
        final_model.fit(X, y)
        
        # Store results
        self.models[f'xgboost_{target_col}'] = final_model
        self.results[f'xgboost_{target_col}'] = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'feature_importance': feature_importance / 5,
            'feature_names': feature_cols
        }
        
        print(f"✅ XGBoost training completed. Mean CV score: {np.mean(cv_scores):.4f}")
        return final_model
    
    def train_lstm(self, target_col, lookback_days=60):
        """Train LSTM model for sequential patterns"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
            
        print("🧠 Training LSTM model...")
        
        X, y, feature_cols = self.prepare_data(target_col)
        
        # Prepare sequential data
        X_seq, y_seq = self._prepare_sequential_data(X, y, lookback_days)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Fewer folds for LSTM due to computational cost
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_seq[train_idx], y_seq[val_idx]
            
            # Build LSTM model
            model = self._build_lstm_model(X_train.shape[1], X_train.shape[2])
            
            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predictions
            y_pred = model.predict(X_val)
            
            # Metrics
            if 'direction' in target_col:
                y_pred_binary = (y_pred > 0.5).astype(int)
                score = accuracy_score(y_val, y_pred_binary)
            else:
                score = mean_squared_error(y_val, y_pred, squared=False)
            
            cv_scores.append(score)
            print(f"Fold {fold+1}: Score = {score:.4f}")
        
        # Train final model on full data
        final_model = self._build_lstm_model(X_seq.shape[1], X_seq.shape[2])
        final_model.fit(X_seq, y_seq, epochs=100, batch_size=32, verbose=0)
        
        # Store results
        self.models[f'lstm_{target_col}'] = final_model
        self.results[f'lstm_{target_col}'] = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'feature_names': feature_cols
        }
        
        print(f"✅ LSTM training completed. Mean CV score: {np.mean(cv_scores):.4f}")
        return final_model
    
    def _prepare_sequential_data(self, X, y, lookback_days):
        """Prepare sequential data for LSTM"""
        X_seq, y_seq = [], []
        
        for i in range(lookback_days, len(X)):
            X_seq.append(X[i-lookback_days:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_lstm_model(self, lookback_days, n_features):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback_days, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid' if 'direction' in target_col else 'linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy' if 'direction' in target_col else 'mse',
            metrics=['accuracy'] if 'direction' in target_col else ['mae']
        )
        
        return model
    
    def train_all_models(self, targets=None):
        """Train all models for specified targets"""
        if targets is None:
            targets = [col for col in self.data.columns if 'direction' in col or 'magnitude' in col]
        
        print(f"🚀 Training models for {len(targets)} targets...")
        
        for target in targets:
            print(f"\n🎯 Training models for {target}")
            print("-" * 40)
            
            # Train LightGBM
            if LGB_AVAILABLE:
                self.train_lightgbm(target)
            
            # Train XGBoost
            if XGB_AVAILABLE:
                self.train_xgboost(target)
            
            # Train LSTM (only for direction targets due to computational cost)
            if TF_AVAILABLE and 'direction' in target:
                self.train_lstm(target)
    
    def save_models(self, path='../models/'):
        """Save trained models"""
        import joblib
        import os
        
        os.makedirs(path, exist_ok=True)
        
        for model_name, model in self.models.items():
            if 'lstm' in model_name:
                model.save(f"{path}{model_name}.h5")
            else:
                joblib.dump(model, f"{path}{model_name}.pkl")
        
        # Save results
        joblib.dump(self.results, f"{path}model_results.pkl")
        
        print(f"✅ Models saved to {path}")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n📊 Model Performance Report")
        print("=" * 50)
        
        report_data = []
        
        for model_name, results in self.results.items():
            report_data.append({
                'Model': model_name,
                'Mean Score': f"{results['mean_score']:.4f}",
                'Std Score': f"{results['std_score']:.4f}",
                'CV Scores': results['cv_scores']
            })
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Mean Score', ascending=False)
        
        print(report_df[['Model', 'Mean Score', 'Std Score']].to_string(index=False))
        
        # Save report
        report_df.to_csv('../data/baseline_performance.csv', index=False)
        print(f"\n✅ Performance report saved to ../data/baseline_performance.csv")
        
        return report_df

if __name__ == "__main__":
    # Train models
    models = BaselineModels()
    if models.load_data():
        models.train_all_models()
        models.save_models()
        models.generate_performance_report() 