#!/usr/bin/env python3
"""
Interpretability and ablation analysis for meme stock contrarian effect prediction.

Implements SHAP analysis, ablation studies, and feature importance analysis
as described in the manual.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    print("Warning: shap not installed. Install with: pip install shap")
    shap = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
    plt = None
    sns = None

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """Analyze feature importance across different models."""
    
    def __init__(self):
        """Initialize feature importance analyzer."""
        self.importance_results = {}
    
    def extract_importance(self, model, model_name: str, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract feature importance from a model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            feature_names: Optional feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        logger.info(f"Extracting feature importance from {model_name}")
        
        try:
            # Try different methods based on model type
            if hasattr(model, 'get_feature_importance'):
                # Custom method
                importance = model.get_feature_importance()
            elif hasattr(model, 'feature_importances_'):
                # Scikit-learn style
                importance = dict(zip(feature_names or range(len(model.feature_importances_)), 
                                    model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # Linear model coefficients
                importance = dict(zip(feature_names or range(len(model.coef_)), 
                                    np.abs(model.coef_)))
            else:
                logger.warning(f"No feature importance method found for {model_name}")
                return {}
            
            self.importance_results[model_name] = importance
            return importance
            
        except Exception as e:
            logger.warning(f"Error extracting importance from {model_name}: {e}")
            return {}
    
    def compare_importance(self, models: Dict[str, Any], 
                          feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare feature importance across models.
        
        Args:
            models: Dictionary mapping model names to models
            feature_names: Optional feature names
            
        Returns:
            DataFrame with importance comparison
        """
        logger.info("Comparing feature importance across models...")
        
        all_features = set()
        importance_data = {}
        
        # Extract importance from each model
        for model_name, model in models.items():
            importance = self.extract_importance(model, model_name, feature_names)
            importance_data[model_name] = importance
            all_features.update(importance.keys())
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(index=sorted(all_features))
        
        for model_name, importance in importance_data.items():
            comparison_df[model_name] = comparison_df.index.map(importance).fillna(0)
        
        # Normalize importance scores
        for col in comparison_df.columns:
            if comparison_df[col].sum() > 0:
                comparison_df[col] = comparison_df[col] / comparison_df[col].sum()
        
        return comparison_df
    
    def get_top_features(self, n_top: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get top N features for each model."""
        top_features = {}
        
        for model_name, importance in self.importance_results.items():
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features[model_name] = sorted_features[:n_top]
        
        return top_features

class SHAPAnalyzer:
    """SHAP analysis for model interpretability."""
    
    def __init__(self):
        """Initialize SHAP analyzer."""
        if shap is None:
            raise ImportError("SHAP is required but not installed")
        
        self.explainers = {}
        self.shap_values = {}
    
    def create_explainer(self, model, X_sample: np.ndarray, model_type: str = 'auto'):
        """
        Create SHAP explainer for a model.
        
        Args:
            model: Trained model
            X_sample: Sample data for explainer
            model_type: Type of model ('tree', 'linear', 'deep', 'auto')
        """
        logger.info(f"Creating SHAP explainer for {model_type} model")
        
        try:
            if model_type == 'tree' or (hasattr(model, 'predict') and 'tree' in str(type(model)).lower()):
                explainer = shap.TreeExplainer(model)
            elif model_type == 'linear' or (hasattr(model, 'coef_')):
                explainer = shap.LinearExplainer(model, X_sample)
            elif model_type == 'deep' or 'keras' in str(type(model)).lower():
                explainer = shap.DeepExplainer(model, X_sample)
            else:
                # Use KernelExplainer as fallback
                explainer = shap.KernelExplainer(model.predict, X_sample)
            
            return explainer
            
        except Exception as e:
            logger.warning(f"Error creating SHAP explainer: {e}")
            # Fallback to KernelExplainer
            return shap.KernelExplainer(model.predict, X_sample)
    
    def analyze_model(self, model, X: np.ndarray, model_name: str, 
                     model_type: str = 'auto', max_samples: int = 1000):
        """
        Perform SHAP analysis on a model.
        
        Args:
            model: Trained model
            X: Feature data
            model_name: Name of the model
            model_type: Type of model
            max_samples: Maximum samples for analysis
        """
        logger.info(f"Performing SHAP analysis for {model_name}")
        
        # Sample data if too large
        if len(X) > max_samples:
            sample_idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        # Create explainer
        explainer = self.create_explainer(model, X_sample, model_type)
        self.explainers[model_name] = explainer
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        self.shap_values[model_name] = shap_values
        
        logger.info(f"SHAP analysis completed for {model_name}")
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get SHAP-based feature importance."""
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_name}")
        
        shap_values = self.shap_values[model_name]
        
        # Calculate mean absolute SHAP values
        if len(shap_values.shape) == 1:
            importance = np.abs(shap_values)
        else:
            importance = np.mean(np.abs(shap_values), axis=0)
        
        return dict(enumerate(importance))
    
    def plot_summary(self, model_name: str, feature_names: Optional[List[str]] = None, 
                    max_display: int = 20):
        """Plot SHAP summary."""
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_name}")
        
        if shap is None or plt is None:
            logger.warning("SHAP or matplotlib not available for plotting")
            return
        
        shap_values = self.shap_values[model_name]
        
        # Get sample data for plotting
        explainer = self.explainers[model_name]
        if hasattr(explainer, 'data'):
            X_sample = explainer.data
        else:
            X_sample = None
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         max_display=max_display, show=False)
        plt.title(f"SHAP Summary Plot - {model_name}")
        plt.tight_layout()
        plt.show()
    
    def plot_waterfall(self, model_name: str, instance_idx: int = 0, 
                      feature_names: Optional[List[str]] = None):
        """Plot SHAP waterfall for a specific instance."""
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_name}")
        
        if shap is None or plt is None:
            logger.warning("SHAP or matplotlib not available for plotting")
            return
        
        shap_values = self.shap_values[model_name]
        explainer = self.explainers[model_name]
        
        # Get base value
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
        else:
            base_value = 0
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(base_value, shap_values[instance_idx], 
                           feature_names=feature_names, show=False)
        plt.title(f"SHAP Waterfall Plot - {model_name} (Instance {instance_idx})")
        plt.tight_layout()
        plt.show()

class AblationAnalyzer:
    """Perform ablation studies to test feature contributions."""
    
    def __init__(self):
        """Initialize ablation analyzer."""
        self.ablation_results = {}
    
    def perform_ablation(self, model, X: np.ndarray, y: np.ndarray,
                        feature_groups: Dict[str, List[int]],
                        feature_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Perform ablation study by removing feature groups.
        
        Args:
            model: Model to test
            X: Feature data
            y: Target data
            feature_groups: Dictionary mapping group names to feature indices
            feature_names: Optional feature names
            
        Returns:
            Dictionary with ablation results
        """
        logger.info("Performing ablation study...")
        
        from .baseline_models import evaluate_model
        
        # Full model performance
        full_pred = model.predict(X)
        full_metrics = evaluate_model(model, X, y)
        
        ablation_results = {
            'full_model': full_metrics
        }
        
        # Test each feature group removal
        for group_name, feature_indices in feature_groups.items():
            logger.info(f"Testing ablation: removing {group_name}")
            
            # Create feature mask
            feature_mask = np.ones(X.shape[1], dtype=bool)
            feature_mask[feature_indices] = False
            
            # Remove features
            X_ablated = X[:, feature_mask]
            
            # Create new model instance
            model_class = type(model)
            if hasattr(model, '__init__'):
                # Try to copy model parameters
                try:
                    ablated_model = model_class(**model.__dict__)
                except:
                    ablated_model = model_class()
            else:
                ablated_model = model_class()
            
            # Retrain model
            try:
                ablated_model.fit(X_ablated, y)
                ablated_pred = ablated_model.predict(X_ablated)
                ablated_metrics = evaluate_model(ablated_model, X_ablated, y)
                
                ablation_results[f'without_{group_name}'] = ablated_metrics
                
            except Exception as e:
                logger.warning(f"Error in ablation for {group_name}: {e}")
                ablation_results[f'without_{group_name}'] = {}
        
        self.ablation_results = ablation_results
        return ablation_results
    
    def calculate_contribution(self, full_metrics: Dict[str, float], 
                              ablated_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate feature group contribution."""
        contribution = {}
        
        for metric in full_metrics.keys():
            if metric in ablated_metrics:
                full_val = full_metrics[metric]
                ablated_val = ablated_metrics[metric]
                
                if full_val != 0:
                    contribution[f'{metric}_contribution'] = (full_val - ablated_val) / full_val
                else:
                    contribution[f'{metric}_contribution'] = 0
        
        return contribution
    
    def get_contribution_summary(self) -> pd.DataFrame:
        """Get summary of feature contributions."""
        if not self.ablation_results:
            return pd.DataFrame()
        
        contributions = []
        
        full_metrics = self.ablation_results['full_model']
        
        for key, metrics in self.ablation_results.items():
            if key != 'full_model':
                contribution = self.calculate_contribution(full_metrics, metrics)
                contribution['feature_group'] = key.replace('without_', '')
                contributions.append(contribution)
        
        return pd.DataFrame(contributions)

class InterpretabilityReport:
    """Generate comprehensive interpretability report."""
    
    def __init__(self):
        """Initialize interpretability report generator."""
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.shap_analyzer = SHAPAnalyzer() if shap else None
        self.ablation_analyzer = AblationAnalyzer()
    
    def generate_report(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                       feature_names: Optional[List[str]] = None,
                       feature_groups: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive interpretability report.
        
        Args:
            models: Dictionary of trained models
            X: Feature data
            y: Target data
            feature_names: Optional feature names
            feature_groups: Optional feature groups for ablation
            
        Returns:
            Dictionary with interpretability results
        """
        logger.info("Generating interpretability report...")
        
        report = {}
        
        # Feature importance analysis
        logger.info("Analyzing feature importance...")
        importance_comparison = self.feature_analyzer.compare_importance(models, feature_names)
        report['feature_importance'] = {
            'comparison': importance_comparison,
            'top_features': self.feature_analyzer.get_top_features()
        }
        
        # SHAP analysis
        if self.shap_analyzer:
            logger.info("Performing SHAP analysis...")
            shap_results = {}
            
            for model_name, model in models.items():
                try:
                    self.shap_analyzer.analyze_model(model, X, model_name)
                    shap_results[model_name] = {
                        'feature_importance': self.shap_analyzer.get_feature_importance(model_name)
                    }
                except Exception as e:
                    logger.warning(f"SHAP analysis failed for {model_name}: {e}")
                    shap_results[model_name] = {}
            
            report['shap_analysis'] = shap_results
        
        # Ablation study
        if feature_groups:
            logger.info("Performing ablation study...")
            ablation_results = {}
            
            for model_name, model in models.items():
                try:
                    ablation_results[model_name] = self.ablation_analyzer.perform_ablation(
                        model, X, y, feature_groups, feature_names
                    )
                except Exception as e:
                    logger.warning(f"Ablation study failed for {model_name}: {e}")
                    ablation_results[model_name] = {}
            
            report['ablation_study'] = {
                'results': ablation_results,
                'contribution_summary': self.ablation_analyzer.get_contribution_summary()
            }
        
        logger.info("Interpretability report generated successfully")
        
        return report
    
    def validate_contrarian_hypothesis(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate contrarian hypothesis based on interpretability results.
        
        Args:
            report: Interpretability report
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating contrarian hypothesis...")
        
        validation = {
            'hypothesis_supported': False,
            'evidence': [],
            'contradictions': []
        }
        
        # Check feature importance for Reddit features
        if 'feature_importance' in report:
            importance_df = report['feature_importance']['comparison']
            
            # Look for Reddit-related features
            reddit_features = [col for col in importance_df.index 
                              if any(x in col.lower() for x in ['reddit', 'mention', 'sentiment', 'confidence'])]
            
            if reddit_features:
                validation['evidence'].append(f"Reddit features found: {reddit_features}")
                
                # Check if Reddit features have high importance
                for feature in reddit_features:
                    if feature in importance_df.index:
                        avg_importance = importance_df.loc[feature].mean()
                        if avg_importance > 0.1:  # Threshold for high importance
                            validation['evidence'].append(f"High importance Reddit feature: {feature} ({avg_importance:.3f})")
        
        # Check SHAP analysis for contrarian signals
        if 'shap_analysis' in report:
            for model_name, shap_results in report['shap_analysis'].items():
                if 'feature_importance' in shap_results:
                    # Look for negative correlations with confidence/sentiment
                    validation['evidence'].append(f"SHAP analysis completed for {model_name}")
        
        # Check ablation results
        if 'ablation_study' in report:
            ablation_results = report['ablation_study']['results']
            
            for model_name, results in ablation_results.items():
                if 'full_model' in results and any('without_reddit' in key for key in results.keys()):
                    # Compare performance with and without Reddit features
                    full_performance = results['full_model'].get('spearman_corr', 0)
                    reddit_ablated = results.get('without_reddit', {}).get('spearman_corr', 0)
                    
                    if full_performance > reddit_ablated:
                        validation['evidence'].append(f"Reddit features improve {model_name} performance")
                        validation['hypothesis_supported'] = True
        
        return validation

if __name__ == "__main__":
    # Test interpretability analysis
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create mock models
    class MockModel:
        def __init__(self, name):
            self.name = name
            self.feature_importances_ = np.random.rand(n_features)
            self.feature_importances_ = self.feature_importances_ / self.feature_importances_.sum()
        
        def predict(self, X):
            return np.random.randn(len(X))
        
        def fit(self, X, y):
            pass
    
    models = {
        'ridge': MockModel('ridge'),
        'lgb': MockModel('lgb')
    }
    
    # Test feature importance analysis
    analyzer = FeatureImportanceAnalyzer()
    importance_df = analyzer.compare_importance(models, feature_names)
    
    print("Feature Importance Comparison:")
    print(importance_df.head())
    
    # Test ablation analysis
    ablation_analyzer = AblationAnalyzer()
    feature_groups = {
        'reddit': [0, 1, 2],
        'technical': [3, 4, 5]
    }
    
    # Mock ablation results
    print(f"\nAblation groups defined: {list(feature_groups.keys())}")

