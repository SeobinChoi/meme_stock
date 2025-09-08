#!/usr/bin/env python3
"""
Meta-ensemble framework for meme stock contrarian effect prediction.

Implements simple blending and stacking as described in the manual.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr

# Import our models
from .baseline_models import RidgeModel, LightGBMModel, XGBoostModel, evaluate_model
from .sequence_models import TCNModel, TFTModel

logger = logging.getLogger(__name__)

class SimpleBlender:
    """Simple weighted average ensemble."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize simple blender.
        
        Args:
            weights: Dictionary mapping model names to weights
        """
        self.weights = weights or {}
        self.models = {}
        self.is_fitted = False
    
    def add_model(self, name: str, model: Any, weight: Optional[float] = None):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Trained model
            weight: Optional weight for this model
        """
        self.models[name] = model
        
        if weight is not None:
            self.weights[name] = weight
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_scores: Optional[Dict[str, float]] = None):
        """
        Fit the blender by determining optimal weights.
        
        Args:
            X: Features
            y: Targets
            validation_scores: Optional validation scores for weight calculation
        """
        logger.info("Fitting simple blender...")
        
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        # Calculate weights based on validation scores if provided
        if validation_scores is not None:
            # Use inverse of validation RMSE as weights
            total_score = sum(validation_scores.values())
            for name, score in validation_scores.items():
                if name in self.models:
                    self.weights[name] = score / total_score
        
        # If no weights specified, use equal weights
        if not self.weights:
            equal_weight = 1.0 / len(self.models)
            for name in self.models.keys():
                self.weights[name] = equal_weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight
        
        self.is_fitted = True
        logger.info(f"Simple blender fitted with weights: {self.weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before making predictions")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Error getting predictions from {name}: {e}")
                continue
        
        # Weighted average
        if not predictions:
            raise ValueError("No valid predictions available")
        
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0)
            ensemble_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual model contributions to ensemble prediction."""
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before getting contributions")
        
        contributions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                weight = self.weights.get(name, 0)
                contributions[name] = weight * pred
            except Exception as e:
                logger.warning(f"Error getting contribution from {name}: {e}")
                contributions[name] = np.zeros(len(X))
        
        return contributions

class StackingEnsemble:
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, 
                 meta_learner: str = 'ridge',
                 meta_params: Optional[Dict] = None,
                 use_probas: bool = False):
        """
        Initialize stacking ensemble.
        
        Args:
            meta_learner: Type of meta-learner ('ridge', 'linear', 'rf', 'lgb')
            meta_params: Parameters for meta-learner
            use_probas: Whether to use prediction probabilities
        """
        self.meta_learner_type = meta_learner
        self.meta_params = meta_params or {}
        self.use_probas = use_probas
        
        self.base_models = {}
        self.meta_learner = None
        self.is_fitted = False
    
    def _create_meta_learner(self):
        """Create meta-learner instance."""
        if self.meta_learner_type == 'ridge':
            return Ridge(**self.meta_params)
        elif self.meta_learner_type == 'linear':
            return LinearRegression(**self.meta_params)
        elif self.meta_learner_type == 'rf':
            return RandomForestRegressor(**self.meta_params)
        elif self.meta_learner_type == 'lgb':
            return LightGBMModel(**self.meta_params)
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")
    
    def add_base_model(self, name: str, model: Any):
        """
        Add a base model to the ensemble.
        
        Args:
            name: Model name
            model: Trained model
        """
        self.base_models[name] = model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """
        Fit the stacking ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Fitting stacking ensemble...")
        
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        
        # Generate meta-features from base models
        meta_features_train = self._generate_meta_features(X_train)
        meta_features_val = self._generate_meta_features(X_val)
        
        # Create meta-learner
        self.meta_learner = self._create_meta_learner()
        
        # Fit meta-learner on validation predictions
        self.meta_learner.fit(meta_features_val, y_val)
        
        self.is_fitted = True
        logger.info("Stacking ensemble fitted successfully")
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base models."""
        meta_features = []
        
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X)
                meta_features.append(pred)
            except Exception as e:
                logger.warning(f"Error generating meta-features from {name}: {e}")
                # Use zeros if model fails
                meta_features.append(np.zeros(len(X)))
        
        return np.column_stack(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Meta-learner prediction
        return self.meta_learner.predict(meta_features)
    
    def get_base_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all base models."""
        predictions = {}
        
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Error getting prediction from {name}: {e}")
                predictions[name] = np.zeros(len(X))
        
        return predictions

class MetaEnsemble:
    """Comprehensive meta-ensemble framework."""
    
    def __init__(self, 
                 use_simple_blending: bool = True,
                 use_stacking: bool = True,
                 stacking_meta_learner: str = 'ridge'):
        """
        Initialize meta-ensemble.
        
        Args:
            use_simple_blending: Whether to use simple blending
            use_stacking: Whether to use stacking
            stacking_meta_learner: Meta-learner for stacking
        """
        self.use_simple_blending = use_simple_blending
        self.use_stacking = use_stacking
        self.stacking_meta_learner = stacking_meta_learner
        
        self.models = {}
        self.simple_blender = None
        self.stacking_ensemble = None
        self.validation_scores = {}
        self.is_fitted = False
    
    def add_model(self, name: str, model: Any):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Trained model
        """
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """
        Fit the meta-ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Fitting meta-ensemble...")
        
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        # Calculate validation scores for each model
        self._calculate_validation_scores(X_val, y_val)
        
        # Fit simple blender if requested
        if self.use_simple_blending:
            logger.info("Fitting simple blender...")
            self.simple_blender = SimpleBlender()
            
            for name, model in self.models.items():
                self.simple_blender.add_model(name, model)
            
            self.simple_blender.fit(X_val, y_val, self.validation_scores)
        
        # Fit stacking ensemble if requested
        if self.use_stacking:
            logger.info("Fitting stacking ensemble...")
            self.stacking_ensemble = StackingEnsemble(
                meta_learner=self.stacking_meta_learner
            )
            
            for name, model in self.models.items():
                self.stacking_ensemble.add_base_model(name, model)
            
            self.stacking_ensemble.fit(X_train, y_train, X_val, y_val)
        
        self.is_fitted = True
        logger.info("Meta-ensemble fitted successfully")
    
    def _calculate_validation_scores(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calculate validation scores for each model."""
        logger.info("Calculating validation scores...")
        
        for name, model in self.models.items():
            try:
                metrics = evaluate_model(model, X_val, y_val)
                # Use inverse RMSE as score (higher is better)
                self.validation_scores[name] = 1.0 / (metrics['rmse'] + 1e-8)
                logger.info(f"{name} validation score: {self.validation_scores[name]:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluating {name}: {e}")
                self.validation_scores[name] = 0.0
    
    def predict(self, X: np.ndarray, method: str = 'auto') -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features
            method: Prediction method ('auto', 'simple', 'stacking', 'average')
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = {}
        
        # Get predictions from each method
        if self.use_simple_blending and method in ['auto', 'simple']:
            predictions['simple'] = self.simple_blender.predict(X)
        
        if self.use_stacking and method in ['auto', 'stacking']:
            predictions['stacking'] = self.stacking_ensemble.predict(X)
        
        # Choose final prediction
        if method == 'auto':
            # Use stacking if available, otherwise simple blending
            if 'stacking' in predictions:
                return predictions['stacking']
            elif 'simple' in predictions:
                return predictions['simple']
            else:
                raise ValueError("No ensemble method available")
        elif method in predictions:
            return predictions[method]
        else:
            raise ValueError(f"Method {method} not available")
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights from simple blender."""
        if self.simple_blender and self.simple_blender.is_fitted:
            return self.simple_blender.weights.copy()
        else:
            return {}
    
    def get_ensemble_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ensemble performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with performance metrics for each ensemble method
        """
        logger.info("Evaluating ensemble performance...")
        
        results = {}
        
        # Evaluate individual models
        results['individual'] = {}
        for name, model in self.models.items():
            try:
                metrics = evaluate_model(model, X_test, y_test)
                results['individual'][name] = metrics
            except Exception as e:
                logger.warning(f"Error evaluating {name}: {e}")
                results['individual'][name] = {}
        
        # Evaluate ensemble methods
        if self.use_simple_blending:
            try:
                simple_pred = self.simple_blender.predict(X_test)
                results['simple_blending'] = evaluate_model(
                    type('MockModel', (), {'predict': lambda x: simple_pred})(), 
                    X_test, y_test
                )
            except Exception as e:
                logger.warning(f"Error evaluating simple blending: {e}")
                results['simple_blending'] = {}
        
        if self.use_stacking:
            try:
                stacking_pred = self.stacking_ensemble.predict(X_test)
                results['stacking'] = evaluate_model(
                    type('MockModel', (), {'predict': lambda x: stacking_pred})(), 
                    X_test, y_test
                )
            except Exception as e:
                logger.warning(f"Error evaluating stacking: {e}")
                results['stacking'] = {}
        
        return results

def create_meta_ensemble(models: Dict[str, Any],
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        use_simple_blending: bool = True,
                        use_stacking: bool = True) -> MetaEnsemble:
    """
    Create and fit a meta-ensemble.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        use_simple_blending: Whether to use simple blending
        use_stacking: Whether to use stacking
        
    Returns:
        Fitted meta-ensemble
    """
    logger.info("Creating meta-ensemble...")
    
    ensemble = MetaEnsemble(
        use_simple_blending=use_simple_blending,
        use_stacking=use_stacking
    )
    
    # Add models
    for name, model in models.items():
        ensemble.add_model(name, model)
    
    # Fit ensemble
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    return ensemble

if __name__ == "__main__":
    # Test meta-ensemble
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Split data
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Create mock models
    class MockModel:
        def __init__(self, name):
            self.name = name
        
        def predict(self, X):
            return np.random.randn(len(X)) * 0.1
    
    models = {
        'ridge': MockModel('ridge'),
        'lgb': MockModel('lgb'),
        'xgb': MockModel('xgb')
    }
    
    # Create meta-ensemble
    ensemble = create_meta_ensemble(models, X_train, y_train, X_val, y_val)
    
    # Make predictions
    pred = ensemble.predict(X_test)
    
    print(f"Ensemble prediction shape: {pred.shape}")
    print(f"Model weights: {ensemble.get_model_weights()}")
    
    # Evaluate performance
    performance = ensemble.get_ensemble_performance(X_test, y_test)
    print(f"Ensemble performance keys: {list(performance.keys())}")

