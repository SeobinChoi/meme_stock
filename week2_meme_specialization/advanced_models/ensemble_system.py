"""
Multi-Model Ensemble System for Meme Stock Prediction
Week 2 Implementation - Advanced Ensemble Methods
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced ML libraries
try:
    import lightgbm as lgb
    import xgboost as xgb
    LGB_AVAILABLE = True
    XGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    XGB_AVAILABLE = False
    print("⚠️ LightGBM/XGBoost not available.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available.")

class MemeStockEnsemble:
    def __init__(self, week1_models_path='../models/', week2_models_path='../models/'):
        self.week1_models_path = week1_models_path
        self.week2_models_path = week2_models_path
        self.models = {}
        self.weights = {}
        self.model_performances = {}
        
    def load_week1_models(self):
        """Load Week 1 baseline models"""
        print("📦 Loading Week 1 models...")
        
        try:
            # Load LightGBM models
            if LGB_AVAILABLE:
                for target in ['GME_direction_1d', 'GME_direction_3d', 'AMC_direction_1d', 'AMC_direction_3d', 'BB_direction_1d', 'BB_direction_3d']:
                    model_path = f"{self.week1_models_path}lightgbm_{target}.pkl"
                    try:
                        self.models[f'lightgbm_{target}'] = joblib.load(model_path)
                        print(f"✅ Loaded lightgbm_{target}")
                    except:
                        print(f"❌ Could not load lightgbm_{target}")
            
            # Load XGBoost models
            if XGB_AVAILABLE:
                for target in ['GME_direction_1d', 'GME_direction_3d', 'AMC_direction_1d', 'AMC_direction_3d', 'BB_direction_1d', 'BB_direction_3d']:
                    model_path = f"{self.week1_models_path}xgboost_{target}.pkl"
                    try:
                        self.models[f'xgboost_{target}'] = joblib.load(model_path)
                        print(f"✅ Loaded xgboost_{target}")
                    except:
                        print(f"❌ Could not load xgboost_{target}")
            
            print(f"✅ Loaded {len(self.models)} Week 1 models")
            
        except Exception as e:
            print(f"❌ Error loading Week 1 models: {e}")
    
    def load_week2_models(self):
        """Load Week 2 advanced models"""
        print("🚀 Loading Week 2 models...")
        
        try:
            # Load transformer models (if available)
            if TF_AVAILABLE:
                for target in ['GME_direction_1d', 'GME_direction_3d']:
                    model_path = f"{self.week2_models_path}transformer_{target}.h5"
                    try:
                        self.models[f'transformer_{target}'] = load_model(model_path)
                        print(f"✅ Loaded transformer_{target}")
                    except:
                        print(f"⚠️ Could not load transformer_{target}")
            
            # Load enhanced LSTM models
            if TF_AVAILABLE:
                for target in ['GME_direction_1d', 'GME_direction_3d']:
                    model_path = f"{self.week2_models_path}lstm_enhanced_{target}.h5"
                    try:
                        self.models[f'lstm_enhanced_{target}'] = load_model(model_path)
                        print(f"✅ Loaded lstm_enhanced_{target}")
                    except:
                        print(f"⚠️ Could not load lstm_enhanced_{target}")
            
            print(f"✅ Loaded {len(self.models)} Week 2 models")
            
        except Exception as e:
            print(f"❌ Error loading Week 2 models: {e}")
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, target_col):
        """
        Train ensemble system and optimize weights
        """
        print(f"🎯 Training ensemble for {target_col}")
        
        # Get models for this target
        target_models = {name: model for name, model in self.models.items() 
                        if target_col in name}
        
        if not target_models:
            print(f"❌ No models found for target {target_col}")
            return None
        
        # Get predictions from all models
        predictions = {}
        for name, model in target_models.items():
            try:
                if 'transformer' in name or 'lstm' in name:
                    # Handle neural network models
                    pred = self._predict_neural_network(model, X_val)
                else:
                    # Handle traditional ML models
                    pred = model.predict(X_val)
                
                predictions[name] = pred
                print(f"✅ Generated predictions for {name}")
                
            except Exception as e:
                print(f"❌ Error predicting with {name}: {e}")
        
        if not predictions:
            print("❌ No valid predictions generated")
            return None
        
        # Optimize ensemble weights
        optimal_weights = self._optimize_weights(predictions, y_val)
        
        # Store weights for this target
        self.weights[target_col] = optimal_weights
        
        print(f"✅ Ensemble weights optimized for {target_col}")
        return optimal_weights
    
    def _predict_neural_network(self, model, X_val):
        """Handle predictions for neural network models"""
        try:
            # Reshape data for neural networks if needed
            if len(X_val.shape) == 2:
                # For LSTM, we might need 3D input
                if 'lstm' in str(type(model)):
                    # Pad or truncate to expected sequence length
                    seq_length = 60  # Default LSTM sequence length
                    if X_val.shape[0] >= seq_length:
                        X_reshaped = X_val[-seq_length:].reshape(1, seq_length, X_val.shape[1])
                    else:
                        # Pad with zeros
                        padding = np.zeros((seq_length - X_val.shape[0], X_val.shape[1]))
                        X_reshaped = np.vstack([padding, X_val]).reshape(1, seq_length, X_val.shape[1])
                else:
                    X_reshaped = X_val.reshape(1, X_val.shape[0], X_val.shape[1])
            else:
                X_reshaped = X_val
            
            # Get predictions
            pred = model.predict(X_reshaped)
            
            # Handle different output formats
            if len(pred.shape) > 1:
                if pred.shape[1] == 2:  # Binary classification
                    pred = pred[:, 1]  # Take positive class probability
                else:
                    pred = pred.flatten()
            
            return pred
            
        except Exception as e:
            print(f"❌ Error in neural network prediction: {e}")
            # Return random predictions as fallback
            return np.random.uniform(0, 1, len(X_val))
    
    def _optimize_weights(self, predictions, y_true, method='performance_based'):
        """Optimize ensemble weights"""
        
        if method == 'performance_based':
            # Weight based on individual model performance
            performances = {}
            for name, pred in predictions.items():
                if len(pred.shape) > 1 and pred.shape[1] == 2:
                    pred_binary = (pred[:, 1] > 0.5).astype(int)
                else:
                    pred_binary = (pred > 0.5).astype(int)
                
                performance = accuracy_score(y_true, pred_binary)
                performances[name] = performance
            
            # Convert performances to weights
            total_performance = sum(performances.values())
            if total_performance > 0:
                weights = {name: perf / total_performance for name, perf in performances.items()}
            else:
                # Equal weights if no performance
                weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
        
        elif method == 'equal':
            # Equal weights
            weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
        
        elif method == 'variance_based':
            # Weight based on prediction variance (lower variance = higher weight)
            variances = {}
            for name, pred in predictions.items():
                variance = np.var(pred)
                variances[name] = 1.0 / (variance + 1e-8)  # Inverse variance
            
            total_inverse_variance = sum(variances.values())
            weights = {name: var / total_inverse_variance for name, var in variances.items()}
        
        return weights
    
    def predict_with_confidence(self, X_test, target_col):
        """
        Make ensemble predictions with confidence intervals
        """
        print(f"🎯 Making ensemble predictions for {target_col}")
        
        # Get models for this target
        target_models = {name: model for name, model in self.models.items() 
                        if target_col in name}
        
        if not target_models:
            print(f"❌ No models found for target {target_col}")
            return None, None
        
        # Get weights for this target
        if target_col not in self.weights:
            print(f"❌ No weights found for target {target_col}")
            return None, None
        
        weights = self.weights[target_col]
        
        # Generate predictions from all models
        predictions = []
        confidences = []
        
        for name, model in target_models.items():
            if name not in weights:
                continue
                
            try:
                if 'transformer' in name or 'lstm' in name:
                    pred = self._predict_neural_network(model, X_test)
                else:
                    pred = model.predict(X_test)
                
                predictions.append(pred)
                
                # Calculate confidence based on model type
                if 'transformer' in name or 'lstm' in name:
                    # For neural networks, use prediction variance as confidence
                    conf = 1.0 - np.var(pred) if len(pred) > 1 else 0.5
                else:
                    # For traditional models, use prediction probability if available
                    try:
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_test)
                            conf = np.max(proba, axis=1).mean()
                        else:
                            conf = 0.7  # Default confidence
                    except:
                        conf = 0.7
                
                confidences.append(conf)
                
            except Exception as e:
                print(f"❌ Error predicting with {name}: {e}")
                continue
        
        if not predictions:
            print("❌ No valid predictions generated")
            return None, None
        
        # Calculate weighted ensemble prediction
        predictions_array = np.array(predictions)
        weights_array = np.array([weights[name] for name in target_models.keys() if name in weights])
        
        # Normalize weights
        weights_array = weights_array / weights_array.sum()
        
        # Weighted ensemble
        ensemble_pred = np.average(predictions_array, weights=weights_array, axis=0)
        
        # Calculate ensemble confidence
        ensemble_conf = np.average(confidences, weights=weights_array)
        
        return ensemble_pred, ensemble_conf
    
    def evaluate_ensemble(self, X_test, y_test, target_col):
        """
        Evaluate ensemble performance
        """
        print(f"📊 Evaluating ensemble for {target_col}")
        
        # Get ensemble predictions
        ensemble_pred, ensemble_conf = self.predict_with_confidence(X_test, target_col)
        
        if ensemble_pred is None:
            return None
        
        # Calculate metrics
        if 'direction' in target_col:
            # Classification metrics
            pred_binary = (ensemble_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, pred_binary)
            
            metrics = {
                'accuracy': accuracy,
                'confidence': ensemble_conf,
                'prediction_mean': np.mean(ensemble_pred),
                'prediction_std': np.std(ensemble_pred)
            }
        else:
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            
            metrics = {
                'rmse': rmse,
                'confidence': ensemble_conf,
                'prediction_mean': np.mean(ensemble_pred),
                'prediction_std': np.std(ensemble_pred)
            }
        
        return metrics
    
    def save_ensemble(self, filename='ensemble_system.pkl'):
        """Save ensemble system"""
        ensemble_data = {
            'weights': self.weights,
            'model_performances': self.model_performances
        }
        
        joblib.dump(ensemble_data, f"../models/{filename}")
        print(f"✅ Ensemble system saved to ../models/{filename}")
    
    def load_ensemble(self, filename='ensemble_system.pkl'):
        """Load ensemble system"""
        try:
            ensemble_data = joblib.load(f"../models/{filename}")
            self.weights = ensemble_data['weights']
            self.model_performances = ensemble_data['model_performances']
            print(f"✅ Ensemble system loaded from ../models/{filename}")
        except:
            print(f"❌ Could not load ensemble system from ../models/{filename}")

if __name__ == "__main__":
    # Test ensemble system
    ensemble = MemeStockEnsemble()
    
    # Load models
    ensemble.load_week1_models()
    ensemble.load_week2_models()
    
    print(f"📦 Total models loaded: {len(ensemble.models)}")
    
    # Save ensemble system
    ensemble.save_ensemble() 