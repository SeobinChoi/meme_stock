#!/usr/bin/env python3
"""
Enhanced ML Models for Meme Stock Prediction
============================================
RandomForest, LightGBM, XGBoost, CatBoost, Ensemble 모델 구현

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

# Optional advanced models
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, skipping...")

warnings.filterwarnings('ignore')
np.random.seed(42)

class EnhancedMLPipeline:
    """Enhanced ML Pipeline for Meme Stock Prediction"""
    
    def __init__(self, target_ic=0.03, target_hit_rate=0.55):
        self.target_ic = target_ic
        self.target_hit_rate = target_hit_rate
        self.results = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = -np.inf
        
        print(f"🎯 Target IC: {target_ic:.3f}, Target Hit Rate: {target_hit_rate:.1%}")
        
    def load_data(self, data_path: str = "data/features/advanced_meme_features_dataset.csv"):
        """Load and prepare dataset"""
        print(f"📊 Loading data from {data_path}...")
        
        try:
            self.df = pd.read_csv(data_path)
            print(f"   ✅ Loaded {len(self.df)} samples with {self.df.shape[1]} features")
            
            # Convert date column
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
            
            # Feature categorization
            self._categorize_features()
            
            return self
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _categorize_features(self):
        """Categorize features by type"""
        all_cols = self.df.columns.tolist()
        
        # Target features
        self.target_features = [col for col in all_cols if 'returns_1d' in col or col.endswith('_y1d')]
        
        # Reddit/Text features
        self.reddit_features = [col for col in all_cols if 'reddit' in col.lower() or 'sentiment' in col.lower()]
        
        # Price features  
        self.price_features = [col for col in all_cols if any(stock in col for stock in ['GME', 'AMC', 'BB']) 
                              and 'reddit' not in col.lower() and 'sentiment' not in col.lower()]
        
        # Temporal features
        self.temporal_features = [col for col in all_cols if 'temporal' in col.lower() or 'day_of' in col]
        
        # Cross-modal features
        self.cross_modal_features = [col for col in all_cols if 'cross_modal' in col.lower()]
        
        # Viral/Social features
        self.viral_features = [col for col in all_cols if 'viral' in col.lower() or 'social' in col.lower()]
        
        print(f"📊 Feature Categorization:")
        print(f"   Reddit/Text: {len(self.reddit_features)}")
        print(f"   Price: {len(self.price_features)}")
        print(f"   Temporal: {len(self.temporal_features)}")
        print(f"   Cross-modal: {len(self.cross_modal_features)}")
        print(f"   Viral/Social: {len(self.viral_features)}")
        
    def prepare_targets(self):
        """Prepare prediction targets for each stock"""
        self.targets = {}
        
        for stock in ['GME', 'AMC', 'BB']:
            target_col = f"{stock}_returns_1d"
            if target_col in self.df.columns:
                self.targets[stock] = self.df[target_col].fillna(0).values
                print(f"   {stock}: {len(self.targets[stock])} target values")
        
        if not self.targets:
            print("❌ No target columns found!")
            return False
            
        return True
    
    def create_feature_sets(self):
        """Create different feature combinations"""
        self.feature_sets = {
            'price_only': self.price_features + self.temporal_features,
            'reddit_only': self.reddit_features + self.temporal_features,
            'cross_modal': self.cross_modal_features + self.temporal_features,
            'viral_social': self.viral_features + self.temporal_features,
            'full_features': (self.price_features + self.reddit_features + 
                            self.temporal_features + self.cross_modal_features + 
                            self.viral_features),
            'selected_features': []  # Will be filled by feature selection
        }
        
        # Remove duplicates and ensure columns exist
        for name, features in self.feature_sets.items():
            existing_features = [f for f in features if f in self.df.columns and f != 'date']
            self.feature_sets[name] = list(set(existing_features))
            print(f"   {name}: {len(self.feature_sets[name])} features")
            
    def feature_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                         top_k: int = 50) -> Tuple[np.ndarray, List[str]]:
        """Advanced feature selection"""
        print(f"🔍 Feature selection: {len(feature_names)} -> {top_k}")
        
        # Method 1: Statistical selection
        selector = SelectKBest(score_func=f_regression, k=min(top_k, len(feature_names)))
        X_selected = selector.fit_transform(X, y)
        selected_mask = selector.get_support()
        
        # Method 2: Random Forest importance
        rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_selector.fit(X, y)
        feature_importance = rf_selector.feature_importances_
        
        # Combine selections
        statistical_scores = selector.scores_
        combined_scores = (stats.rankdata(-statistical_scores) + stats.rankdata(-feature_importance)) / 2
        
        # Select top features
        top_indices = np.argsort(combined_scores)[:top_k]
        selected_features = [feature_names[i] for i in top_indices]
        
        print(f"   ✅ Selected {len(selected_features)} features")
        return X[:, top_indices], selected_features
    
    def create_time_series_splits(self, n_splits: int = 5):
        """Create time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)  # 1-day gap to prevent lookahead
        return tscv
    
    def calculate_ic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate Information Coefficient and related metrics"""
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 10:
            return {'ic': 0, 'rank_ic': 0, 'hit_rate': 0, 'mse': np.inf, 'r2': -np.inf}
        
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
            'ic': ic,
            'ic_p_value': ic_p_value,
            'rank_ic': rank_ic,
            'rank_ic_p_value': rank_ic_p_value,
            'hit_rate': hit_rate,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'n_samples': len(y_true_clean)
        }
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray, 
                           feature_names: List[str]) -> Dict:
        """Train optimized Random Forest"""
        print("🌳 Training Random Forest...")
        
        # Optimized hyperparameters for financial data
        rf_model = RandomForestRegressor(
            n_estimators=200,          # More trees for stability
            max_depth=15,              # Deeper trees for complex patterns
            min_samples_split=10,      # Prevent overfitting
            min_samples_leaf=5,        # Prevent overfitting
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
            num_leaves=127,            # Complex model for rich features
            max_depth=10,              # Deep model
            learning_rate=0.05,        # Slower learning for stability
            n_estimators=500,          # More iterations
            subsample=0.8,             # Bagging
            colsample_bytree=0.8,      # Feature sampling
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=0.1,            # L2 regularization
            min_child_samples=20,      # Prevent overfitting
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
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=10,
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
    
    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str]) -> Optional[Dict]:
        """Train CatBoost if available"""
        if not CATBOOST_AVAILABLE:
            return None
            
        print("🐱 Training CatBoost...")
        
        cb_model = cb.CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        
        cb_model.fit(X_train, y_train)
        y_pred = cb_model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_ic_metrics(y_test, y_pred)
        
        # Feature importance
        importance = dict(zip(feature_names, cb_model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'model': cb_model,
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
                      if model and model['metrics']['ic'] > 0.001}
        
        if len(good_models) < 2:
            print("   ⚠️ Not enough good models for ensemble")
            return None
            
        # Weight models by IC performance
        weights = {}
        total_ic = sum(model['metrics']['ic'] for model in good_models.values())
        
        for name, model in good_models.items():
            weights[name] = model['metrics']['ic'] / total_ic
            
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
    
    def train_stock_models(self, stock: str, feature_set_name: str = 'full_features'):
        """Train all models for a specific stock"""
        print(f"\n🎯 Training {stock} models with {feature_set_name}...")
        
        if stock not in self.targets:
            print(f"❌ No target data for {stock}")
            return
            
        # Prepare data
        features = self.feature_sets[feature_set_name]
        X = self.df[features].fillna(0).values.astype(np.float32)
        y = self.targets[stock]
        
        # Remove samples with missing targets
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        print(f"   📊 Using {X.shape[1]} features, {len(y)} samples")
        
        # Feature selection for full features
        if feature_set_name == 'full_features' and len(features) > 50:
            X, features = self.feature_selection(X, y, features, top_k=50)
            self.feature_sets['selected_features'] = features
        
        # Train/test split (time series aware)
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   📈 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train individual models
        models = {}
        
        # Random Forest
        models['RandomForest'] = self.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test, features)
        
        # LightGBM  
        models['LightGBM'] = self.train_lightgbm(
            X_train_scaled, y_train, X_test_scaled, y_test, features)
        
        # XGBoost
        models['XGBoost'] = self.train_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test, features)
        
        # CatBoost (if available)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = self.train_catboost(
                X_train_scaled, y_train, X_test_scaled, y_test, features)
        
        # Ensemble
        ensemble = self.create_ensemble(models, X_test_scaled, y_test)
        if ensemble:
            models['Ensemble'] = ensemble
        
        # Store results
        self.results[f"{stock}_{feature_set_name}"] = {
            'stock': stock,
            'feature_set': feature_set_name,
            'features': features,
            'models': models,
            'test_metrics': {name: model['metrics'] if model else None 
                           for name, model in models.items()},
            'scaler': scaler
        }
        
        # Print results
        self._print_stock_results(stock, feature_set_name)
        
    def _print_stock_results(self, stock: str, feature_set_name: str):
        """Print results for a stock"""
        key = f"{stock}_{feature_set_name}"
        if key not in self.results:
            return
            
        result = self.results[key]
        print(f"\n📊 {stock} Results ({feature_set_name}):")
        print(f"{'Model':<15} {'IC':<8} {'Hit Rate':<10} {'R²':<8} {'Status'}")
        print("-" * 50)
        
        for model_name, metrics in result['test_metrics'].items():
            if metrics is None:
                continue
                
            ic = metrics['ic']
            hit_rate = metrics['hit_rate']
            r2 = metrics['r2']
            
            # Determine status
            status = "✅ PASS" if ic >= self.target_ic and hit_rate >= self.target_hit_rate else "❌ FAIL"
            if ic >= self.target_ic * 0.5:  # Half target
                status = "⚠️ CLOSE"
                
            print(f"{model_name:<15} {ic:<8.4f} {hit_rate:<10.3f} {r2:<8.4f} {status}")
    
    def run_comprehensive_experiment(self):
        """Run comprehensive ML experiment"""
        print("\n🚀 Starting Comprehensive ML Experiment...")
        print("="*60)
        
        if not self.prepare_targets():
            return
            
        self.create_feature_sets()
        
        # Test different feature combinations
        feature_sets_to_test = ['price_only', 'reddit_only', 'full_features']
        stocks_to_test = ['GME', 'AMC', 'BB']
        
        for stock in stocks_to_test:
            for feature_set in feature_sets_to_test:
                try:
                    self.train_stock_models(stock, feature_set)
                except Exception as e:
                    print(f"❌ Error training {stock} with {feature_set}: {e}")
        
        # Summary
        self.generate_summary()
        
    def generate_summary(self):
        """Generate experiment summary"""
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY")
        print("="*60)
        
        best_results = []
        
        for key, result in self.results.items():
            stock = result['stock']
            feature_set = result['feature_set']
            
            for model_name, metrics in result['test_metrics'].items():
                if metrics is None:
                    continue
                    
                best_results.append({
                    'stock': stock,
                    'feature_set': feature_set,
                    'model': model_name,
                    'ic': metrics['ic'],
                    'hit_rate': metrics['hit_rate'],
                    'r2': metrics['r2'],
                    'n_samples': metrics['n_samples']
                })
        
        # Sort by IC performance
        best_results.sort(key=lambda x: x['ic'], reverse=True)
        
        print(f"\n🏆 TOP 10 PERFORMING MODELS:")
        print(f"{'Rank':<4} {'Stock':<5} {'Model':<12} {'Features':<12} {'IC':<8} {'Hit Rate':<10} {'R²':<8}")
        print("-" * 70)
        
        for i, result in enumerate(best_results[:10], 1):
            print(f"{i:<4} {result['stock']:<5} {result['model']:<12} "
                  f"{result['feature_set']:<12} {result['ic']:<8.4f} "
                  f"{result['hit_rate']:<10.3f} {result['r2']:<8.4f}")
        
        # Success rate
        passing_models = [r for r in best_results 
                         if r['ic'] >= self.target_ic and r['hit_rate'] >= self.target_hit_rate]
        
        print(f"\n✅ Models passing targets: {len(passing_models)}/{len(best_results)}")
        print(f"🎯 Best IC achieved: {best_results[0]['ic']:.4f}")
        print(f"🎯 Target IC: {self.target_ic:.3f}")
        
        # Save results
        self.save_results()
        
    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/enhanced_ml_results_{timestamp}.json"
        
        # Prepare serializable results
        save_data = {
            'timestamp': timestamp,
            'target_ic': self.target_ic,
            'target_hit_rate': self.target_hit_rate,
            'summary': []
        }
        
        for key, result in self.results.items():
            for model_name, metrics in result['test_metrics'].items():
                if metrics is None:
                    continue
                    
                save_data['summary'].append({
                    'stock': result['stock'],
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
    print("🚀 Enhanced ML Models for Meme Stock Prediction")
    print("=" * 60)
    print("📊 Target: IC > 0.03, Hit Rate > 55%")
    print("💻 Hardware: i7 16GB optimized")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EnhancedMLPipeline()
    
    # Load data
    if pipeline.load_data() is None:
        print("❌ Failed to load data")
        return
    
    # Run experiment
    pipeline.run_comprehensive_experiment()
    
    print("\n✅ Experiment completed!")
    print("📈 Check results/ directory for detailed outputs")


if __name__ == "__main__":
    main()

