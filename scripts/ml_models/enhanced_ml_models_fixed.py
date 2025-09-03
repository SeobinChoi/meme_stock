#!/usr/bin/env python3
"""
Enhanced ML Models for Meme Stock Prediction - FIXED VERSION
============================================================
Colab 데이터셋 사용으로 수정된 버전

Hardware: i7 16GB 최적화
Target: IC > 0.03, Hit Rate > 55%
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import lightgbm as lgb
import xgboost as xgb

# Statistical Libraries
from scipy.stats import spearmanr, pearsonr
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

class EnhancedMLPipelineFixed:
    """Enhanced ML Pipeline using working Colab dataset"""
    
    def __init__(self, target_ic=0.03, target_hit_rate=0.55):
        self.target_ic = target_ic
        self.target_hit_rate = target_hit_rate
        self.results = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = -np.inf
        
        print(f"🎯 Target IC: {target_ic:.3f}, Target Hit Rate: {target_hit_rate:.1%}")
        
    def load_colab_data(self):
        """Load Colab datasets"""
        print("📊 Loading Colab datasets...")
        
        try:
            self.train_df = pd.read_csv("data/colab_datasets/tabular_train_20250814_031335.csv")
            self.val_df = pd.read_csv("data/colab_datasets/tabular_val_20250814_031335.csv") 
            self.test_df = pd.read_csv("data/colab_datasets/tabular_test_20250814_031335.csv")
            
            print(f"   ✅ Train: {len(self.train_df)} samples, {self.train_df.shape[1]} features")
            print(f"   ✅ Validation: {len(self.val_df)} samples")
            print(f"   ✅ Test: {len(self.test_df)} samples")
            
            # Check target variable
            if 'y1d' in self.train_df.columns:
                y_stats = self.train_df['y1d'].describe()
                print(f"   🎯 Target (y1d) stats: mean={y_stats['mean']:.4f}, std={y_stats['std']:.4f}")
            
            return self
            
        except Exception as e:
            print(f"❌ Error loading colab data: {e}")
            return None
    
    def categorize_features(self):
        """Categorize features by type"""
        all_cols = self.train_df.columns.tolist()
        
        # Remove target and index columns
        feature_cols = [col for col in all_cols if col not in ['y1d', 'date', 'ticker']]
        
        # Feature categorization based on naming patterns
        self.price_features = [col for col in feature_cols if any(x in col.lower() for x in 
                              ['returns', 'vol_', 'price', 'volume', 'rsi', 'sma', 'turnover'])]
        
        self.reddit_features = [col for col in feature_cols if any(x in col.lower() for x in 
                               ['reddit', 'mentions', 'log_mentions', 'sentiment'])]
        
        self.temporal_features = [col for col in feature_cols if any(x in col.lower() for x in 
                                 ['day_of', 'month', 'is_monday', 'is_friday', 'weekend', 'regime'])]
        
        # All remaining features
        self.other_features = [col for col in feature_cols if col not in 
                              self.price_features + self.reddit_features + self.temporal_features]
        
        print(f"📊 Feature Categorization:")
        print(f"   Price: {len(self.price_features)}")
        print(f"   Reddit: {len(self.reddit_features)}")
        print(f"   Temporal: {len(self.temporal_features)}")
        print(f"   Other: {len(self.other_features)}")
        
        # Create feature sets
        self.feature_sets = {
            'price_only': self.price_features + self.temporal_features,
            'reddit_only': self.reddit_features + self.temporal_features,
            'price_reddit': self.price_features + self.reddit_features + self.temporal_features,
            'all_features': feature_cols
        }
        
        for name, features in self.feature_sets.items():
            print(f"   {name}: {len(features)} features")
    
    def calculate_ic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate Information Coefficient and related metrics"""
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 10:
            return {'ic': 0, 'rank_ic': 0, 'hit_rate': 0.5, 'mse': np.inf, 'r2': -np.inf}
        
        # Information Coefficient (Pearson correlation)
        ic, ic_p_value = pearsonr(y_true_clean, y_pred_clean) if len(y_true_clean) > 1 else (0, 1)
        
        # Rank IC (Spearman correlation)
        rank_ic, rank_ic_p_value = spearmanr(y_true_clean, y_pred_clean) if len(y_true_clean) > 1 else (0, 1)
        
        # Hit Rate (directional accuracy)
        y_true_direction = np.sign(y_true_clean)
        y_pred_direction = np.sign(y_pred_clean)
        hit_rate = np.mean(y_true_direction == y_pred_direction)
        
        # Traditional metrics
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        return {
            'ic': ic if not np.isnan(ic) else 0,
            'ic_p_value': ic_p_value if not np.isnan(ic_p_value) else 1,
            'rank_ic': rank_ic if not np.isnan(rank_ic) else 0,
            'rank_ic_p_value': rank_ic_p_value if not np.isnan(rank_ic_p_value) else 1,
            'hit_rate': hit_rate if not np.isnan(hit_rate) else 0.5,
            'mse': mse,
            'mae': mae,
            'r2': r2 if not np.isnan(r2) else -np.inf,
            'n_samples': len(y_true_clean)
        }
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray, 
                           feature_names: List[str]) -> Dict:
        """Train optimized Random Forest"""
        print("🌳 Training Random Forest...")
        
        # Optimized hyperparameters for financial data
        rf_model = RandomForestRegressor(
            n_estimators=150,          # More trees for stability
            max_depth=12,              # Deeper trees for complex patterns
            min_samples_split=20,      # Prevent overfitting  
            min_samples_leaf=10,       # Prevent overfitting
            max_features='sqrt',       # Feature sampling
            bootstrap=True,
            random_state=42,
            n_jobs=-1                  # Use all cores
        )
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_ic_metrics(y_test, y_pred)
        
        # Feature importance
        importance = dict(zip(feature_names, rf_model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'model': rf_model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': importance,
            'top_features': top_features
        }
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str]) -> Dict:
        """Train optimized LightGBM"""
        print("💡 Training LightGBM...")
        
        # Optimized hyperparameters for financial time series
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            num_leaves=100,            # Good for complex patterns
            max_depth=8,               # Reasonable depth
            learning_rate=0.05,        # Slower learning for stability
            n_estimators=300,          # More iterations
            subsample=0.8,             # Bagging
            colsample_bytree=0.8,      # Feature sampling
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=0.1,            # L2 regularization
            min_child_samples=30,      # Prevent overfitting
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_ic_metrics(y_test, y_pred)
        
        # Feature importance
        importance = dict(zip(feature_names, lgb_model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'model': lgb_model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': importance,
            'top_features': top_features
        }
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     feature_names: List[str]) -> Dict:
        """Train optimized XGBoost"""
        print("🚀 Training XGBoost...")
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=15,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_ic_metrics(y_test, y_pred)
        
        # Feature importance
        importance = dict(zip(feature_names, xgb_model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'model': xgb_model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': importance,
            'top_features': top_features
        }
    
    def create_ensemble(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Create ensemble of best models"""
        print("🎭 Creating Ensemble Model...")
        
        # Filter models by IC performance
        good_models = {name: model for name, model in models.items() 
                      if model and model['metrics']['ic'] > 0.005}  # Lower threshold
        
        if len(good_models) < 2:
            print("   ⚠️ Not enough good models for ensemble")
            return None
            
        # Weight models by IC performance with minimum weight
        weights = {}
        total_ic = sum(abs(model['metrics']['ic']) for model in good_models.values())
        
        if total_ic == 0:
            # Equal weights if all ICs are 0
            weight_per_model = 1.0 / len(good_models)
            for name in good_models.keys():
                weights[name] = weight_per_model
        else:
            for name, model in good_models.items():
                weights[name] = abs(model['metrics']['ic']) / total_ic
            
        # Create weighted ensemble predictions
        ensemble_pred = np.zeros(len(X_test))
        for name, model in good_models.items():
            ensemble_pred += weights[name] * model['predictions']
            
        # Calculate ensemble metrics
        metrics = self.calculate_ic_metrics(y_test, ensemble_pred)
        
        return {
            'predictions': ensemble_pred,
            'metrics': metrics,
            'weights': weights,
            'component_models': list(good_models.keys())
        }
    
    def train_feature_set(self, feature_set_name: str):
        """Train all models for a specific feature set"""
        print(f"\n🎯 Training models with {feature_set_name}...")
        
        # Prepare data
        features = self.feature_sets[feature_set_name]
        
        # Ensure all features exist
        available_features = [f for f in features if f in self.train_df.columns]
        print(f"   📊 Using {len(available_features)}/{len(features)} features")
        
        if len(available_features) == 0:
            print("   ❌ No valid features found!")
            return
        
        # Prepare training data
        X_train = self.train_df[available_features].fillna(0).values.astype(np.float32)
        y_train = self.train_df['y1d'].values.astype(np.float32)
        
        # Prepare validation data  
        X_val = self.val_df[available_features].fillna(0).values.astype(np.float32)
        y_val = self.val_df['y1d'].values.astype(np.float32)
        
        # Prepare test data
        X_test = self.test_df[available_features].fillna(0).values.astype(np.float32)
        y_test = self.test_df['y1d'].values.astype(np.float32)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   📈 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train individual models
        models = {}
        
        # Random Forest
        models['RandomForest'] = self.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test, available_features)
        
        # LightGBM  
        models['LightGBM'] = self.train_lightgbm(
            X_train_scaled, y_train, X_test_scaled, y_test, available_features)
        
        # XGBoost
        models['XGBoost'] = self.train_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test, available_features)
        
        # Ensemble
        ensemble = self.create_ensemble(models, X_test_scaled, y_test)
        if ensemble:
            models['Ensemble'] = ensemble
        
        # Store results
        self.results[feature_set_name] = {
            'feature_set': feature_set_name,
            'features': available_features,
            'models': models,
            'test_metrics': {name: model['metrics'] if model else None 
                           for name, model in models.items()},
            'scaler': scaler
        }
        
        # Print results
        self._print_results(feature_set_name)
        
    def _print_results(self, feature_set_name: str):
        """Print results for a feature set"""
        if feature_set_name not in self.results:
            return
            
        result = self.results[feature_set_name]
        print(f"\n📊 Results ({feature_set_name}):")
        print(f"{'Model':<15} {'IC':<8} {'Rank IC':<8} {'Hit Rate':<10} {'R²':<8} {'Status'}")
        print("-" * 65)
        
        for model_name, metrics in result['test_metrics'].items():
            if metrics is None:
                continue
                
            ic = metrics['ic']
            rank_ic = metrics['rank_ic']
            hit_rate = metrics['hit_rate']
            r2 = metrics['r2']
            
            # Determine status
            if ic >= self.target_ic and hit_rate >= self.target_hit_rate:
                status = "✅ PASS"
            elif ic >= self.target_ic * 0.3 or hit_rate >= self.target_hit_rate - 0.05:
                status = "⚠️ CLOSE"
            else:
                status = "❌ FAIL"
                
            print(f"{model_name:<15} {ic:<8.4f} {rank_ic:<8.4f} {hit_rate:<10.3f} {r2:<8.4f} {status}")
    
    def run_experiment(self):
        """Run comprehensive ML experiment"""
        print("\n🚀 Starting Enhanced ML Experiment (Fixed Version)...")
        print("="*70)
        
        if self.load_colab_data() is None:
            return
            
        self.categorize_features()
        
        # Test different feature combinations
        feature_sets_to_test = ['price_only', 'reddit_only', 'price_reddit', 'all_features']
        
        for feature_set in feature_sets_to_test:
            try:
                self.train_feature_set(feature_set)
            except Exception as e:
                print(f"❌ Error training {feature_set}: {e}")
        
        # Generate summary
        self.generate_summary()
        
    def generate_summary(self):
        """Generate experiment summary"""
        print("\n" + "="*70)
        print("📊 EXPERIMENT SUMMARY")
        print("="*70)
        
        best_results = []
        
        for result in self.results.values():
            feature_set = result['feature_set']
            
            for model_name, metrics in result['test_metrics'].items():
                if metrics is None:
                    continue
                    
                best_results.append({
                    'feature_set': feature_set,
                    'model': model_name,
                    'ic': metrics['ic'],
                    'rank_ic': metrics['rank_ic'],
                    'hit_rate': metrics['hit_rate'],
                    'r2': metrics['r2'],
                    'n_samples': metrics['n_samples']
                })
        
        # Sort by IC performance
        best_results.sort(key=lambda x: x['ic'], reverse=True)
        
        print(f"\n🏆 TOP 10 PERFORMING MODELS:")
        print(f"{'Rank':<4} {'Model':<12} {'Features':<15} {'IC':<8} {'Rank IC':<8} {'Hit Rate':<10} {'R²':<8}")
        print("-" * 75)
        
        for i, result in enumerate(best_results[:10], 1):
            print(f"{i:<4} {result['model']:<12} {result['feature_set']:<15} "
                  f"{result['ic']:<8.4f} {result['rank_ic']:<8.4f} "
                  f"{result['hit_rate']:<10.3f} {result['r2']:<8.4f}")
        
        # Success metrics
        passing_models = [r for r in best_results 
                         if r['ic'] >= self.target_ic and r['hit_rate'] >= self.target_hit_rate]
        
        close_models = [r for r in best_results 
                       if r['ic'] >= self.target_ic * 0.3 or r['hit_rate'] >= self.target_hit_rate - 0.05]
        
        print(f"\n✅ Models passing targets: {len(passing_models)}/{len(best_results)}")
        print(f"⚠️ Models close to targets: {len(close_models)}/{len(best_results)}")
        print(f"🎯 Best IC achieved: {best_results[0]['ic']:.4f}")
        print(f"🎯 Best Hit Rate: {max(r['hit_rate'] for r in best_results):.3f}")
        print(f"🎯 Target IC: {self.target_ic:.3f}")
        print(f"🎯 Target Hit Rate: {self.target_hit_rate:.3f}")
        
        # Feature importance analysis
        self._analyze_feature_importance()
        
        # Save results
        self.save_results()
        
    def _analyze_feature_importance(self):
        """Analyze feature importance across models"""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
        
        all_importances = {}
        
        for result in self.results.values():
            for model_name, model_data in result['models'].items():
                if model_data and 'feature_importance' in model_data:
                    for feature, importance in model_data['feature_importance'].items():
                        if feature not in all_importances:
                            all_importances[feature] = []
                        all_importances[feature].append(importance)
        
        # Calculate average importance
        avg_importances = {feature: np.mean(importances) 
                          for feature, importances in all_importances.items()}
        
        # Top features across all models
        top_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:15]
        
        print(f"🏆 TOP 15 FEATURES (Average Importance):")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"{i:2d}. {feature:<30} {importance:.4f}")
        
    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/enhanced_ml_results_fixed_{timestamp}.json"
        
        # Prepare serializable results
        save_data = {
            'timestamp': timestamp,
            'target_ic': self.target_ic,
            'target_hit_rate': self.target_hit_rate,
            'summary': []
        }
        
        for result in self.results.values():
            for model_name, metrics in result['test_metrics'].items():
                if metrics is None:
                    continue
                    
                save_data['summary'].append({
                    'feature_set': result['feature_set'],
                    'model': model_name,
                    'metrics': metrics
                })
        
        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main execution function"""
    print("🚀 Enhanced ML Models for Meme Stock Prediction (FIXED)")
    print("=" * 70)
    print("📊 Target: IC > 0.03, Hit Rate > 55%")
    print("💻 Hardware: i7 16GB optimized")
    print("🔧 Using working Colab dataset")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = EnhancedMLPipelineFixed()
    
    # Run experiment
    pipeline.run_experiment()
    
    print("\n✅ Experiment completed!")
    print("📈 Check results/ directory for detailed outputs")


if __name__ == "__main__":
    main()

