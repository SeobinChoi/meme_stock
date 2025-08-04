"""
Advanced Feature Engineering and Validation Module
Day 4: Feature Selection, Importance Analysis, and Stability Assessment
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, RFE, 
    SelectFromModel, mutual_info_classif, mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering with selection, importance analysis, and stability assessment
    """
    
    def __init__(self, data_path='data/features/engineered_features_dataset.csv'):
        self.data_path = data_path
        self.features_df = None
        self.feature_importance_scores = {}
        self.feature_stability_scores = {}
        self.selected_features = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the engineered features dataset"""
        print("📊 Loading engineered features dataset...")
        self.features_df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded {len(self.features_df)} samples with {len(self.features_df.columns)} features")
        return self.features_df
    
    def prepare_targets(self):
        """Prepare target variables for different prediction tasks"""
        print("🎯 Preparing target variables...")
        
        # Calculate returns for different stocks
        for stock in ['GME', 'AMC', 'BB']:
            price_col = f'{stock}_close'
            if price_col in self.features_df.columns:
                # 1-day returns
                self.features_df[f'{stock}_return_1d'] = self.features_df[price_col].pct_change(1)
                
                # 3-day returns
                self.features_df[f'{stock}_return_3d'] = self.features_df[price_col].pct_change(3)
                
                # 7-day returns
                self.features_df[f'{stock}_return_7d'] = self.features_df[price_col].pct_change(7)
                
                # Direction targets (1 for positive, 0 for negative)
                for period in [1, 3, 7]:
                    return_col = f'{stock}_return_{period}d'
                    direction_col = f'{stock}_direction_{period}d'
                    self.features_df[direction_col] = (self.features_df[return_col] > 0).astype(int)
        
        print("✅ Target variables prepared")
        return self.features_df
    
    def implement_feature_selection(self):
        """Implement multiple feature selection algorithms"""
        print("🔍 Implementing feature selection algorithms...")
        
        # Get feature columns (exclude targets and price columns)
        exclude_cols = [col for col in self.features_df.columns 
                       if any(x in col for x in ['return', 'direction', 'close', 'open', 'high', 'low', 'volume'])]
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        clean_df = self.features_df[feature_cols].dropna()
        
        # Prepare sample targets for selection (using GME direction as example)
        target_col = 'GME_direction_1d'
        if target_col in self.features_df.columns:
            target = self.features_df[target_col].dropna()
            common_idx = clean_df.index.intersection(target.index)
            X = clean_df.loc[common_idx]
            y = target.loc[common_idx]
            
            # 1. Univariate Selection (F-test)
            print("  - Running univariate selection (F-test)...")
            f_selector = SelectKBest(score_func=f_classif, k=50)
            f_selector.fit(X, y)
            f_scores = pd.Series(f_selector.scores_, index=feature_cols)
            self.feature_importance_scores['f_test'] = f_scores.sort_values(ascending=False)
            
            # 2. Mutual Information
            print("  - Running mutual information selection...")
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_scores_series = pd.Series(mi_scores, index=feature_cols)
            self.feature_importance_scores['mutual_info'] = mi_scores_series.sort_values(ascending=False)
            
            # 3. Random Forest Importance
            print("  - Running random forest importance...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = pd.Series(rf.feature_importances_, index=feature_cols)
            self.feature_importance_scores['random_forest'] = rf_importance.sort_values(ascending=False)
            
            # 4. Lasso-based selection
            print("  - Running Lasso-based selection...")
            X_scaled = self.scaler.fit_transform(X)
            lasso = Lasso(alpha=0.01, random_state=42)
            lasso.fit(X_scaled, y)
            lasso_coef = pd.Series(np.abs(lasso.coef_), index=feature_cols)
            self.feature_importance_scores['lasso'] = lasso_coef.sort_values(ascending=False)
            
            # 5. Recursive Feature Elimination
            print("  - Running recursive feature elimination...")
            rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=30)
            rfe.fit(X, y)
            rfe_support = pd.Series(rfe.support_, index=feature_cols)
            self.selected_features['rfe'] = feature_cols[rfe.support_]
            
            print("✅ Feature selection algorithms completed")
        else:
            print("⚠️  Target column not found, skipping feature selection")
    
    def analyze_feature_importance(self):
        """Analyze feature importance across different methods"""
        print("📈 Analyzing feature importance...")
        
        importance_analysis = {}
        
        for method, scores in self.feature_importance_scores.items():
            # Top 20 features for each method
            top_features = scores.head(20)
            importance_analysis[method] = {
                'top_features': top_features.to_dict(),
                'mean_importance': scores.mean(),
                'std_importance': scores.std(),
                'feature_count': len(scores)
            }
        
        # Consensus analysis - features that appear in top 20 across multiple methods
        all_top_features = []
        for method, scores in self.feature_importance_scores.items():
            all_top_features.extend(scores.head(20).index.tolist())
        
        feature_counts = pd.Series(all_top_features).value_counts()
        consensus_features = feature_counts[feature_counts >= 2].index.tolist()
        
        importance_analysis['consensus'] = {
            'consensus_features': consensus_features,
            'consensus_count': len(consensus_features),
            'feature_agreement': feature_counts.to_dict()
        }
        
        self.feature_importance_scores['consensus_analysis'] = importance_analysis
        print(f"✅ Feature importance analysis completed. Found {len(consensus_features)} consensus features")
        
        return importance_analysis
    
    def assess_feature_stability(self):
        """Assess feature stability across different time periods"""
        print("🔄 Assessing feature stability...")
        
        # Get feature columns
        exclude_cols = [col for col in self.features_df.columns 
                       if any(x in col for x in ['return', 'direction', 'close', 'open', 'high', 'low', 'volume'])]
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Prepare data
        clean_df = self.features_df[feature_cols].dropna()
        
        # Split data into time periods
        n_periods = 4
        period_length = len(clean_df) // n_periods
        
        stability_scores = {}
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else len(clean_df)
            
            period_data = clean_df.iloc[start_idx:end_idx]
            period_name = f"period_{i+1}"
            
            # Calculate feature importance for this period
            if 'GME_direction_1d' in self.features_df.columns:
                target = self.features_df['GME_direction_1d'].iloc[start_idx:end_idx]
                common_idx = period_data.index.intersection(target.index)
                
                if len(common_idx) > 10:  # Minimum sample size
                    X_period = period_data.loc[common_idx]
                    y_period = target.loc[common_idx]
                    
                    # Use random forest importance for stability assessment
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf.fit(X_period, y_period)
                    
                    period_importance = pd.Series(rf.feature_importances_, index=feature_cols)
                    stability_scores[period_name] = period_importance
        
        # Calculate stability metrics
        if len(stability_scores) > 1:
            # Calculate correlation between importance rankings across periods
            importance_df = pd.DataFrame(stability_scores)
            
            # Calculate rank correlation
            rank_corr = importance_df.rank().corr()
            stability_metric = rank_corr.mean().mean()
            
            # Calculate coefficient of variation for each feature
            feature_cv = importance_df.std() / importance_df.mean()
            feature_cv = feature_cv.replace([np.inf, -np.inf], np.nan).dropna()
            
            self.feature_stability_scores = {
                'period_importance': stability_scores,
                'rank_correlation': rank_corr.to_dict(),
                'overall_stability': stability_metric,
                'feature_cv': feature_cv.to_dict(),
                'stable_features': feature_cv[feature_cv < 0.5].index.tolist()  # Low CV = stable
            }
            
            print(f"✅ Feature stability assessment completed. Overall stability: {stability_metric:.3f}")
        else:
            print("⚠️  Insufficient data for stability assessment")
    
    def create_feature_subsets(self):
        """Create different feature subsets based on selection results"""
        print("📦 Creating feature subsets...")
        
        feature_subsets = {}
        
        # Get all feature columns
        exclude_cols = [col for col in self.features_df.columns 
                       if any(x in col for x in ['return', 'direction', 'close', 'open', 'high', 'low', 'volume'])]
        all_features = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # 1. Top features from each method
        for method, scores in self.feature_importance_scores.items():
            if isinstance(scores, pd.Series):
                top_features = scores.head(30).index.tolist()
                feature_subsets[f'top_{method}'] = top_features
        
        # 2. Consensus features
        if 'consensus_analysis' in self.feature_importance_scores:
            consensus_features = self.feature_importance_scores['consensus_analysis']['consensus_features']
            feature_subsets['consensus'] = consensus_features
        
        # 3. Stable features
        if 'stable_features' in self.feature_stability_scores:
            stable_features = self.feature_stability_scores['stable_features']
            feature_subsets['stable'] = stable_features
        
        # 4. RFE selected features
        if 'rfe' in self.selected_features:
            feature_subsets['rfe'] = self.selected_features['rfe']
        
        # 5. Combined subset (intersection of top methods)
        top_methods = ['random_forest', 'mutual_info', 'f_test']
        combined_features = set(all_features)
        for method in top_methods:
            if method in self.feature_importance_scores:
                top_features = set(self.feature_importance_scores[method].head(20).index)
                combined_features = combined_features.intersection(top_features)
        
        if combined_features:
            feature_subsets['combined'] = list(combined_features)
        
        self.feature_subsets = feature_subsets
        print(f"✅ Created {len(feature_subsets)} feature subsets")
        
        return feature_subsets
    
    def evaluate_feature_subsets(self):
        """Evaluate performance of different feature subsets"""
        print("🎯 Evaluating feature subsets...")
        
        if not hasattr(self, 'feature_subsets'):
            print("⚠️  No feature subsets available")
            return
        
        evaluation_results = {}
        
        # Prepare target
        target_col = 'GME_direction_1d'
        if target_col not in self.features_df.columns:
            print("⚠️  Target column not found")
            return
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for subset_name, features in self.feature_subsets.items():
            print(f"  - Evaluating {subset_name} ({len(features)} features)...")
            
            # Prepare data
            X = self.features_df[features].dropna()
            y = self.features_df[target_col].dropna()
            
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) < 50:  # Minimum sample size
                print(f"    ⚠️  Insufficient data for {subset_name}")
                continue
            
            # Evaluate with random forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            try:
                scores = cross_val_score(rf, X, y, cv=tscv, scoring='accuracy')
                evaluation_results[subset_name] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std(),
                    'feature_count': len(features),
                    'sample_count': len(X)
                }
            except Exception as e:
                print(f"    ⚠️  Error evaluating {subset_name}: {e}")
        
        self.evaluation_results = evaluation_results
        print("✅ Feature subset evaluation completed")
        
        return evaluation_results
    
    def generate_advanced_report(self):
        """Generate comprehensive advanced feature engineering report"""
        print("📋 Generating advanced feature engineering report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.features_df),
                'total_features': len([col for col in self.features_df.columns 
                                     if not any(x in col for x in ['return', 'direction', 'close', 'open', 'high', 'low', 'volume'])]),
                'target_variables': [col for col in self.features_df.columns if 'direction' in col]
            },
            'feature_selection_results': {
                'methods_used': list(self.feature_importance_scores.keys()),
                'top_features_by_method': {}
            },
            'feature_importance_analysis': {},
            'feature_stability_analysis': {},
            'feature_subsets': {},
            'evaluation_results': {}
        }
        
        # Add top features by method
        for method, scores in self.feature_importance_scores.items():
            if isinstance(scores, pd.Series):
                report['feature_selection_results']['top_features_by_method'][method] = {
                    'top_10_features': scores.head(10).to_dict(),
                    'mean_importance': float(scores.mean()),
                    'std_importance': float(scores.std())
                }
        
        # Add consensus analysis
        if 'consensus_analysis' in self.feature_importance_scores:
            report['feature_importance_analysis'] = self.feature_importance_scores['consensus_analysis']
        
        # Add stability analysis
        if hasattr(self, 'feature_stability_scores'):
            report['feature_stability_analysis'] = {
                'overall_stability': self.feature_stability_scores.get('overall_stability', 0),
                'stable_features_count': len(self.feature_stability_scores.get('stable_features', [])),
                'stable_features': self.feature_stability_scores.get('stable_features', [])
            }
        
        # Add feature subsets
        if hasattr(self, 'feature_subsets'):
            report['feature_subsets'] = {
                name: {'feature_count': len(features), 'features': features[:10]}  # Show first 10
                for name, features in self.feature_subsets.items()
            }
        
        # Add evaluation results
        if hasattr(self, 'evaluation_results'):
            report['evaluation_results'] = self.evaluation_results
        
        # Save report
        report_path = 'results/002_day4_advanced_features_internal.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary
        self._generate_summary_report(report)
        
        print("✅ Advanced feature engineering report generated")
        return report
    
    def _generate_summary_report(self, report):
        """Generate human-readable summary report"""
        summary_lines = [
            "=" * 60,
            "DAY 4: ADVANCED FEATURE ENGINEERING & VALIDATION",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DATASET OVERVIEW:",
            f"- Total samples: {report['dataset_info']['total_samples']:,}",
            f"- Total features: {report['dataset_info']['total_features']}",
            f"- Target variables: {len(report['dataset_info']['target_variables'])}",
            "",
            "FEATURE SELECTION RESULTS:",
        ]
        
        for method, data in report['feature_selection_results']['top_features_by_method'].items():
            summary_lines.extend([
                f"- {method.upper()}:",
                f"  * Top feature: {list(data['top_10_features'].keys())[0]}",
                f"  * Mean importance: {data['mean_importance']:.4f}",
                f"  * Std importance: {data['std_importance']:.4f}"
            ])
        
        if report['feature_importance_analysis']:
            consensus_count = report['feature_importance_analysis'].get('consensus_count', 0)
            summary_lines.extend([
                "",
                "CONSENSUS ANALYSIS:",
                f"- Consensus features: {consensus_count}",
            ])
        
        if report['feature_stability_analysis']:
            stability = report['feature_stability_analysis']['overall_stability']
            stable_count = report['feature_stability_analysis']['stable_features_count']
            summary_lines.extend([
                "",
                "STABILITY ANALYSIS:",
                f"- Overall stability: {stability:.3f}",
                f"- Stable features: {stable_count}",
            ])
        
        if report['evaluation_results']:
            summary_lines.extend([
                "",
                "FEATURE SUBSET EVALUATION:",
            ])
            
            # Sort by mean accuracy
            sorted_results = sorted(
                report['evaluation_results'].items(),
                key=lambda x: x[1]['mean_accuracy'],
                reverse=True
            )
            
            for subset_name, results in sorted_results:
                summary_lines.extend([
                    f"- {subset_name}:",
                    f"  * Accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}",
                    f"  * Features: {results['feature_count']}",
                    f"  * Samples: {results['sample_count']:,}"
                ])
        
        summary_lines.extend([
            "",
            "NEXT STEPS:",
            "- Proceed to Day 5: Baseline Model Development",
            "- Use selected feature subsets for model training",
            "- Focus on stable and consensus features",
            "",
            "=" * 60
        ])
        
        # Save summary
        summary_path = 'results/002_day4_advanced_features_summary.txt'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"📄 Summary saved to {summary_path}")
    
    def run_advanced_pipeline(self):
        """Run the complete advanced feature engineering pipeline"""
        print("🚀 Starting Advanced Feature Engineering Pipeline...")
        print("=" * 60)
        
        # Load and prepare data
        self.load_data()
        self.prepare_targets()
        
        # Implement feature selection
        self.implement_feature_selection()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Assess feature stability
        self.assess_feature_stability()
        
        # Create feature subsets
        self.create_feature_subsets()
        
        # Evaluate feature subsets
        self.evaluate_feature_subsets()
        
        # Generate comprehensive report
        self.generate_advanced_report()
        
        print("=" * 60)
        print("✅ Advanced Feature Engineering Pipeline Complete!")
        print("📊 Reports saved to results/ directory")
        
        return True

if __name__ == "__main__":
    # Run the advanced feature engineering pipeline
    advanced_engineer = AdvancedFeatureEngineer()
    advanced_engineer.run_advanced_pipeline() 