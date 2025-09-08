#!/usr/bin/env python3
"""
🚀 금융 시계열 예측을 위한 ML 모델 비교 실험
Reddit 데이터 포함/제외에 따른 성능 차이 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Try to import CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, skipping CatBoost models")

class FinancialMLExperiment:
    """금융 ML 모델 비교 실험 클래스"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.scalers = {}
        
        # Define feature groups
        self.price_features = [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
            'vol_5d', 'vol_10d', 'vol_20d', 'price_ratio_sma10', 
            'price_ratio_sma20', 'rsi_14', 'volume_ratio', 'turnover',
            'day_of_week', 'month', 'is_monday', 'is_friday', 
            'is_weekend_effect', 'market_vol_regime'
        ]
        
        self.reddit_features = [
            'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
            'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95',
            'reddit_momentum_3', 'reddit_momentum_7', 'reddit_momentum_14',
            'reddit_momentum_21', 'reddit_vol_5', 'reddit_vol_10', 'reddit_vol_20',
            'reddit_percentile', 'reddit_high_regime', 'reddit_low_regime',
            'market_sentiment', 'price_reddit_momentum', 'vol_reddit_attention'
        ]
    
    def load_data(self):
        """데이터 로드 및 준비"""
        print("📊 Loading data...")
        
        # Load datasets
        self.train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
        self.val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
        self.test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        # Convert dates
        for df in [self.train_df, self.val_df, self.test_df]:
            df['date'] = pd.to_datetime(df['date'])
        
        print(f"✅ Data loaded:")
        print(f"   Train: {len(self.train_df)} samples")
        print(f"   Val: {len(self.val_df)} samples") 
        print(f"   Test: {len(self.test_df)} samples")
        
        # Verify feature availability
        all_features = set(self.train_df.columns)
        
        available_price_features = [f for f in self.price_features if f in all_features]
        available_reddit_features = [f for f in self.reddit_features if f in all_features]
        
        print(f"   Available price features: {len(available_price_features)}/{len(self.price_features)}")
        print(f"   Available reddit features: {len(available_reddit_features)}/{len(self.reddit_features)}")
        
        # Update feature lists
        self.price_features = available_price_features
        self.reddit_features = available_reddit_features
        
        return self
    
    def prepare_data(self, include_reddit=False):
        """데이터 준비 (특성 선택 및 스케일링)"""
        
        if include_reddit:
            features = self.price_features + self.reddit_features
            experiment_type = "Enhanced (Price + Reddit)"
        else:
            features = self.price_features
            experiment_type = "Baseline (Price Only)"
        
        print(f"🔧 Preparing data for: {experiment_type}")
        print(f"   Using {len(features)} features")
        
        # Prepare features
        X_train = self.train_df[features].fillna(0).values.astype(np.float32)
        X_val = self.val_df[features].fillna(0).values.astype(np.float32)
        X_test = self.test_df[features].fillna(0).values.astype(np.float32)
        
        # Prepare targets
        y_train = self.train_df['y1d'].values.astype(np.float32)
        y_val = self.val_df['y1d'].values.astype(np.float32)
        y_test = self.test_df['y1d'].values.astype(np.float32)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'features': features,
            'scaler': scaler,
            'experiment_type': experiment_type
        }
    
    def calculate_metrics(self, y_true, y_pred, model_name=""):
        """성능 지표 계산"""
        
        # Remove NaN values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            return {
                'ic': 0, 'rank_ic': 0, 'hit_rate': 0.5,
                'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf
            }
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # IC metrics
        ic, _ = pearsonr(y_pred_clean, y_true_clean)
        rank_ic, _ = spearmanr(y_pred_clean, y_true_clean)
        
        # Hit rate (directional accuracy)
        hit_rate = np.mean(np.sign(y_pred_clean) == np.sign(y_true_clean))
        
        # Error metrics
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return {
            'ic': ic if not np.isnan(ic) else 0,
            'rank_ic': rank_ic if not np.isnan(rank_ic) else 0,
            'hit_rate': hit_rate,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_samples': len(y_true_clean)
        }
    
    def train_traditional_ml(self, data):
        """전통적인 ML 모델 훈련"""
        
        print(f"🤖 Training Traditional ML models...")
        
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        
        models = {
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.1, random_state=self.random_state, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=2000),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                learning_rate=0.1, max_depth=6, n_estimators=50,
                random_state=self.random_state, verbosity=0
            ),
            'LightGBM': lgb.LGBMRegressor(
                learning_rate=0.1, num_leaves=31, n_estimators=50,
                random_state=self.random_state, verbosity=-1
            )
        }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = cb.CatBoostRegressor(
                learning_rate=0.1, depth=6, iterations=50,
                random_state=self.random_state, verbose=False
            )
        
        results = {}
        
        for name, model in models.items():
            try:
                print(f"   Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_val = model.predict(X_val)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                val_metrics = self.calculate_metrics(y_val, y_pred_val, name)
                test_metrics = self.calculate_metrics(y_test, y_pred_test, name)
                
                results[name] = {
                    'model': model,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'predictions': {
                        'val': y_pred_val,
                        'test': y_pred_test
                    }
                }
                
                print(f"     Val IC: {val_metrics['rank_ic']:.4f}, Test IC: {test_metrics['rank_ic']:.4f}")
                
            except Exception as e:
                print(f"     ❌ {name} failed: {e}")
                continue
        
        return results
    
    def train_deep_learning(self, data):
        """딥러닝 모델 훈련 (간단한 MLP만)"""
        
        print(f"🧠 Training Deep Learning models...")
        
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        
        results = {}
        
        # MLP Model
        try:
            print(f"   Training MLP...")
            mlp_results = self._train_mlp(X_train, X_val, X_test, y_train, y_val, y_test, device)
            results['MLP'] = mlp_results
        except Exception as e:
            print(f"     ❌ MLP failed: {e}")
        
        return results
    
    def _train_mlp(self, X_train, X_val, X_test, y_train, y_val, y_test, device):
        """MLP 모델 훈련"""
        
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
                super(SimpleMLP, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # Create model
        model = SimpleMLP(X_train.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Training
        model.train()
        for epoch in range(20):  # Faster training
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Predictions
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            y_pred_val = model(X_val_tensor).cpu().numpy().flatten()
            y_pred_test = model(X_test_tensor).cpu().numpy().flatten()
        
        # Calculate metrics
        val_metrics = self.calculate_metrics(y_val, y_pred_val, 'MLP')
        test_metrics = self.calculate_metrics(y_test, y_pred_test, 'MLP')
        
        print(f"     Val IC: {val_metrics['rank_ic']:.4f}, Test IC: {test_metrics['rank_ic']:.4f}")
        
        return {
            'model': model,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'predictions': {
                'val': y_pred_val,
                'test': y_pred_test
            }
        }
    
    def run_experiment(self):
        """전체 실험 실행"""
        
        print("🚀 Starting Financial ML Comparison Experiment")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        experiments = [
            {'include_reddit': False, 'name': 'Baseline'},
            {'include_reddit': True, 'name': 'Enhanced'}
        ]
        
        all_results = {}
        
        for exp in experiments:
            print(f"\n📊 Running {exp['name']} experiment...")
            
            # Prepare data
            data = self.prepare_data(include_reddit=exp['include_reddit'])
            
            # Train models
            traditional_results = self.train_traditional_ml(data)
            dl_results = self.train_deep_learning(data)
            
            # Combine results
            exp_results = {**traditional_results, **dl_results}
            
            all_results[exp['name']] = {
                'data': data,
                'results': exp_results
            }
        
        self.results = all_results
        return self
    
    def analyze_results(self):
        """결과 분석 및 시각화"""
        
        print("\n📈 Analyzing Results...")
        
        # Collect metrics for comparison
        comparison_data = []
        
        for exp_name, exp_data in self.results.items():
            for model_name, model_data in exp_data['results'].items():
                
                val_metrics = model_data['val_metrics']
                test_metrics = model_data['test_metrics']
                
                comparison_data.append({
                    'Experiment': exp_name,
                    'Model': model_name,
                    'Val_IC': val_metrics['rank_ic'],
                    'Test_IC': test_metrics['rank_ic'],
                    'Val_Hit_Rate': val_metrics['hit_rate'],
                    'Test_Hit_Rate': test_metrics['hit_rate'],
                    'Val_RMSE': val_metrics['rmse'],
                    'Test_RMSE': test_metrics['rmse'],
                    'Test_R2': test_metrics['r2']
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Print summary table
        print("\n📊 Performance Summary (Test Set):")
        print("=" * 80)
        
        summary_cols = ['Experiment', 'Model', 'Test_IC', 'Test_Hit_Rate', 'Test_RMSE']
        print(self.comparison_df[summary_cols].round(4).to_string(index=False))
        
        # Calculate improvements
        self._calculate_improvements()
        
        # Create visualizations
        self._create_visualizations()
        
        return self
    
    def _calculate_improvements(self):
        """Reddit 데이터 추가로 인한 개선도 계산"""
        
        print("\n🚀 Reddit Data Impact Analysis:")
        print("=" * 50)
        
        improvements = []
        
        # Get models that exist in both experiments
        baseline_models = set(self.results['Baseline']['results'].keys())
        enhanced_models = set(self.results['Enhanced']['results'].keys())
        common_models = baseline_models & enhanced_models
        
        for model in common_models:
            baseline_ic = self.results['Baseline']['results'][model]['test_metrics']['rank_ic']
            enhanced_ic = self.results['Enhanced']['results'][model]['test_metrics']['rank_ic']
            
            baseline_hit = self.results['Baseline']['results'][model]['test_metrics']['hit_rate']
            enhanced_hit = self.results['Enhanced']['results'][model]['test_metrics']['hit_rate']
            
            ic_improvement = enhanced_ic - baseline_ic
            hit_improvement = enhanced_hit - baseline_hit
            
            improvements.append({
                'Model': model,
                'IC_Improvement': ic_improvement,
                'Hit_Rate_Improvement': hit_improvement,
                'IC_Improvement_Pct': (ic_improvement / abs(baseline_ic)) * 100 if baseline_ic != 0 else 0
            })
            
            print(f"{model:12s}: IC +{ic_improvement:+.4f} ({ic_improvement/abs(baseline_ic)*100:+.1f}%), "
                  f"Hit Rate +{hit_improvement:+.4f}")
        
        self.improvements_df = pd.DataFrame(improvements)
        
        # Overall statistics
        if len(self.improvements_df) > 0:
            avg_ic_improvement = self.improvements_df['IC_Improvement'].mean()
            avg_hit_improvement = self.improvements_df['Hit_Rate_Improvement'].mean()
            
            print(f"\n📊 Average Improvements:")
            print(f"   IC: +{avg_ic_improvement:.4f}")
            print(f"   Hit Rate: +{avg_hit_improvement:.4f}")
        
        # Best performing models
        best_baseline = self.comparison_df[self.comparison_df['Experiment'] == 'Baseline'].nlargest(1, 'Test_IC')
        best_enhanced = self.comparison_df[self.comparison_df['Experiment'] == 'Enhanced'].nlargest(1, 'Test_IC')
        
        print(f"\n🏆 Best Models:")
        if not best_baseline.empty:
            print(f"   Baseline: {best_baseline.iloc[0]['Model']} (IC: {best_baseline.iloc[0]['Test_IC']:.4f})")
        if not best_enhanced.empty:
            print(f"   Enhanced: {best_enhanced.iloc[0]['Model']} (IC: {best_enhanced.iloc[0]['Test_IC']:.4f})")
    
    def _create_visualizations(self):
        """결과 시각화"""
        
        print("\n📊 Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Financial ML Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. IC Comparison
        ax1 = axes[0, 0]
        ic_pivot = self.comparison_df.pivot(index='Model', columns='Experiment', values='Test_IC')
        ic_pivot.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Information Coefficient (IC) Comparison')
        ax1.set_ylabel('Rank IC')
        ax1.legend(title='Experiment')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.03, color='red', linestyle='--', alpha=0.7, label='Target (0.03)')
        
        # 2. Hit Rate Comparison
        ax2 = axes[0, 1]
        hit_pivot = self.comparison_df.pivot(index='Model', columns='Experiment', values='Test_Hit_Rate')
        hit_pivot.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Hit Rate Comparison')
        ax2.set_ylabel('Hit Rate')
        ax2.legend(title='Experiment')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
        
        # 3. RMSE Comparison
        ax3 = axes[1, 0]
        rmse_pivot = self.comparison_df.pivot(index='Model', columns='Experiment', values='Test_RMSE')
        rmse_pivot.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('RMSE Comparison (Lower is Better)')
        ax3.set_ylabel('RMSE')
        ax3.legend(title='Experiment')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Improvement Analysis
        ax4 = axes[1, 1]
        if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
            improvements_plot = self.improvements_df.set_index('Model')['IC_Improvement']
            improvements_plot.plot(kind='bar', ax=ax4, color='green', alpha=0.7)
            ax4.set_title('IC Improvement (Enhanced - Baseline)')
            ax4.set_ylabel('IC Improvement')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        else:
            ax4.text(0.5, 0.5, 'No improvement data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('IC Improvement Analysis')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'financial_ml_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Saved visualization: financial_ml_comparison_{timestamp}.png")
        plt.show()
    
    def generate_report(self):
        """종합 보고서 생성"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'financial_ml_experiment_report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🚀 FINANCIAL ML MODEL COMPARISON EXPERIMENT REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Experiment overview
            f.write("📊 EXPERIMENT OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Target Variable: y1d (1-day return)\n")
            f.write(f"Training Samples: {len(self.train_df)}\n")
            f.write(f"Validation Samples: {len(self.val_df)}\n")
            f.write(f"Test Samples: {len(self.test_df)}\n")
            f.write(f"Price Features: {len(self.price_features)}\n")
            f.write(f"Reddit Features: {len(self.reddit_features)}\n\n")
            
            # Performance summary
            f.write("📈 PERFORMANCE SUMMARY (TEST SET)\n")
            f.write("-" * 30 + "\n")
            summary_cols = ['Experiment', 'Model', 'Test_IC', 'Test_Hit_Rate', 'Test_RMSE']
            f.write(self.comparison_df[summary_cols].round(4).to_string(index=False))
            f.write("\n\n")
            
            # Improvement analysis
            if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
                f.write("🚀 REDDIT DATA IMPACT ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(self.improvements_df.round(4).to_string(index=False))
                f.write("\n\n")
                
                avg_ic_improvement = self.improvements_df['IC_Improvement'].mean()
                avg_hit_improvement = self.improvements_df['Hit_Rate_Improvement'].mean()
                
                f.write(f"📊 Average IC Improvement: {avg_ic_improvement:+.4f}\n")
                f.write(f"📊 Average Hit Rate Improvement: {avg_hit_improvement:+.4f}\n\n")
            
            # Best models
            best_baseline = self.comparison_df[self.comparison_df['Experiment'] == 'Baseline'].nlargest(1, 'Test_IC')
            best_enhanced = self.comparison_df[self.comparison_df['Experiment'] == 'Enhanced'].nlargest(1, 'Test_IC')
            
            f.write("🏆 BEST PERFORMING MODELS\n")
            f.write("-" * 30 + "\n")
            if not best_baseline.empty:
                f.write(f"Best Baseline: {best_baseline.iloc[0]['Model']} (IC: {best_baseline.iloc[0]['Test_IC']:.4f})\n")
            if not best_enhanced.empty:
                f.write(f"Best Enhanced: {best_enhanced.iloc[0]['Model']} (IC: {best_enhanced.iloc[0]['Test_IC']:.4f})\n\n")
            
            # Conclusions
            f.write("💡 KEY INSIGHTS\n")
            f.write("-" * 30 + "\n")
            
            if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
                positive_improvements = (self.improvements_df['IC_Improvement'] > 0).sum()
                total_models = len(self.improvements_df)
                avg_ic_improvement = self.improvements_df['IC_Improvement'].mean()
                
                f.write(f"• {positive_improvements}/{total_models} models improved with Reddit data\n")
                
                if avg_ic_improvement > 0.01:
                    f.write(f"• Strong evidence for Reddit data value (avg improvement: {avg_ic_improvement:.4f})\n")
                elif avg_ic_improvement > 0.005:
                    f.write(f"• Moderate evidence for Reddit data value (avg improvement: {avg_ic_improvement:.4f})\n")
                else:
                    f.write(f"• Limited evidence for Reddit data value (avg improvement: {avg_ic_improvement:.4f})\n")
            
            # Check if any model achieves target performance
            best_ic = self.comparison_df['Test_IC'].max()
            if best_ic >= 0.03:
                f.write(f"• Target IC ≥ 0.03 achieved! Best IC: {best_ic:.4f}\n")
            else:
                f.write(f"• Target IC ≥ 0.03 not achieved. Best IC: {best_ic:.4f}\n")
            
            f.write("\n📋 RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            if hasattr(self, 'improvements_df') and len(self.improvements_df) > 0:
                avg_ic_improvement = self.improvements_df['IC_Improvement'].mean()
                if avg_ic_improvement > 0.01:
                    f.write("• Reddit data provides significant value - recommend inclusion\n")
                    f.write("• Consider developing more sophisticated Reddit features\n")
                elif avg_ic_improvement > 0.005:
                    f.write("• Reddit data provides moderate value - consider cost-benefit\n")
                    f.write("• Explore advanced Reddit sentiment analysis\n")
                else:
                    f.write("• Limited evidence for Reddit data value in current form\n")
                    f.write("• Focus on improving price-based features\n")
            
            if best_ic < 0.03:
                f.write("• Consider ensemble methods or feature engineering\n")
                f.write("• Explore alternative targets or prediction horizons\n")
        
        print(f"\n📋 Report saved: {report_file}")
        return report_file


def main():
    """메인 실행 함수"""
    
    print("🚀 Starting Financial ML Comparison Experiment")
    print("=" * 60)
    
    # Run experiment
    experiment = FinancialMLExperiment(random_state=42)
    experiment.run_experiment()
    experiment.analyze_results()
    report_file = experiment.generate_report()
    
    print("\n" + "=" * 60)
    print("✅ Experiment completed successfully!")
    print(f"📊 Results visualization and report generated")
    print(f"📋 Report file: {report_file}")
    
    return experiment


if __name__ == "__main__":
    experiment = main()