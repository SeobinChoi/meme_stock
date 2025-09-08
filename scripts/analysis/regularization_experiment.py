#!/usr/bin/env python3
"""
Regularization Experiment for Reddit Features
정규화 기법을 활용한 Reddit 피처 가중치 자동 조절 실험

실험 설계:
1. Ridge Regression: L2 정규화로 가중치 축소
2. Lasso Regression: L1 정규화로 피처 선택
3. Elastic Net: L1 + L2 정규화 조합
4. 하이퍼파라미터 튜닝 및 성능 비교
5. 피처 가중치 분석 및 해석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class RegularizationExperiment:
    """정규화 실험 클래스"""
    
    def __init__(self):
        self.df = None
        self.reddit_features = []
        self.price_features = []
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_weights = {}
        
    def load_data(self):
        """데이터 로드"""
        print("📊 Loading dataset for regularization experiment...")
        
        # 통합 데이터셋 로드
        train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
        val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
        test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        # 데이터 통합
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # 다음날 수익률 타겟 생성
        df['target_1d'] = df.groupby('ticker')['returns_1d'].shift(-1)
        
        # 미래 데이터 마스킹
        df['mask'] = False
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_indices = df[ticker_mask].index
            if len(ticker_indices) > 5:
                df.loc[ticker_indices[-5:], 'mask'] = True
        
        # 마스킹된 데이터 제외
        df = df[~df['mask']].copy()
        
        print(f"   ✅ Total data: {len(df)} records")
        print(f"   ✅ Date range: {df['date'].min()} ~ {df['date'].max()}")
        print(f"   ✅ Tickers: {df['ticker'].unique()}")
        
        self.df = df
        return df
    
    def prepare_features(self):
        """특성 준비"""
        print("🔧 Preparing features...")
        
        # 주가 관련 특성
        self.price_features = [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
            'vol_5d', 'vol_10d', 'vol_20d',
            'price_ratio_sma10', 'price_ratio_sma20',
            'rsi_14', 'volume_ratio', 'turnover',
            'day_of_week', 'month', 'is_monday', 'is_friday', 'is_weekend_effect'
        ]
        
        # Reddit 관련 특성
        self.reddit_features = [
            'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
            'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95',
            'reddit_momentum_3', 'reddit_momentum_7', 'reddit_momentum_14', 'reddit_momentum_21',
            'reddit_vol_5', 'reddit_vol_10', 'reddit_vol_20',
            'reddit_percentile', 'reddit_high_regime', 'reddit_low_regime',
            'market_sentiment', 'price_reddit_momentum', 'vol_reddit_attention'
        ]
        
        # 존재하는 특성만 선택
        available_price_features = [col for col in self.price_features if col in self.df.columns]
        available_reddit_features = [col for col in self.reddit_features if col in self.df.columns]
        
        self.price_features = available_price_features
        self.reddit_features = available_reddit_features
        
        print(f"   ✅ Price features: {len(self.price_features)}")
        print(f"   ✅ Reddit features: {len(self.reddit_features)}")
        
        return self.price_features, self.reddit_features
    
    def strict_time_series_split(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """엄격한 시계열 분할"""
        print("📊 Performing strict time series split...")
        
        # 각 종목별로 시간 순서대로 분할
        train_data = []
        val_data = []
        test_data = []
        
        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            
            # 시간 순서대로 분할
            n = len(ticker_data)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_data.append(ticker_data.iloc[:train_end])
            val_data.append(ticker_data.iloc[train_end:val_end])
            test_data.append(ticker_data.iloc[val_end:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        print(f"   ✅ Train: {len(train_df)} records")
        print(f"   ✅ Validation: {len(val_df)} records")
        print(f"   ✅ Test: {len(test_df)} records")
        
        return train_df, val_df, test_df
    
    def prepare_model_data(self, train_df, val_df, test_df):
        """모델 데이터 준비"""
        print("🔧 Preparing model data...")
        
        # 특성과 타겟 준비
        all_features = self.price_features + self.reddit_features
        
        # 종목별 더미 변수 추가
        ticker_dummies_train = pd.get_dummies(train_df['ticker'], prefix='ticker')
        ticker_dummies_val = pd.get_dummies(val_df['ticker'], prefix='ticker')
        ticker_dummies_test = pd.get_dummies(test_df['ticker'], prefix='ticker')
        
        # 모든 더미 변수 컬럼 통일
        all_ticker_cols = set(ticker_dummies_train.columns) | set(ticker_dummies_val.columns) | set(ticker_dummies_test.columns)
        
        for col in all_ticker_cols:
            if col not in ticker_dummies_train.columns:
                ticker_dummies_train[col] = 0
            if col not in ticker_dummies_val.columns:
                ticker_dummies_val[col] = 0
            if col not in ticker_dummies_test.columns:
                ticker_dummies_test[col] = 0
        
        # 데이터 타입을 float로 변환
        ticker_dummies_train = ticker_dummies_train.astype(float)
        ticker_dummies_val = ticker_dummies_val.astype(float)
        ticker_dummies_test = ticker_dummies_test.astype(float)
        
        # 최종 특성 세트
        final_features = all_features + list(all_ticker_cols)
        
        # 데이터 준비
        X_train = train_df[all_features].fillna(0)
        X_val = val_df[all_features].fillna(0)
        X_test = test_df[all_features].fillna(0)
        
        # 더미 변수 추가
        X_train = pd.concat([X_train, ticker_dummies_train], axis=1)
        X_val = pd.concat([X_val, ticker_dummies_val], axis=1)
        X_test = pd.concat([X_test, ticker_dummies_test], axis=1)
        
        y_train = train_df['target_1d'].fillna(0)
        y_val = val_df['target_1d'].fillna(0)
        y_test = test_df['target_1d'].fillna(0)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        print(f"   ✅ Final features: {len(final_features)}")
        print(f"   ✅ Data shapes: Train {X_train_scaled.shape}, Val {X_val_scaled.shape}, Test {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, final_features
    
    def calculate_spearman_ic(self, y_true, y_pred):
        """Spearman rank correlation 기반 IC 계산"""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return np.nan, np.nan
        
        corr, p_value = spearmanr(y_true_clean, y_pred_clean)
        return corr, p_value
    
    def calculate_hit_rate(self, y_true, y_pred, threshold=0.0):
        """Hit Rate 계산"""
        true_direction = (y_true > threshold).astype(int)
        pred_direction = (y_pred > threshold).astype(int)
        
        hit_rate = (true_direction == pred_direction).mean()
        return hit_rate
    
    def train_ridge_models(self, X_train, y_train, X_val, y_val):
        """Ridge Regression 모델 훈련"""
        print("📈 Training Ridge Regression models...")
        
        # 하이퍼파라미터 그리드
        alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        ridge_results = {}
        
        for alpha in alpha_values:
            print(f"   🔍 Training Ridge with alpha={alpha}...")
            
            ridge = Ridge(alpha=alpha, random_state=42)
            ridge.fit(X_train, y_train)
            
            # 검증 데이터로 예측
            y_pred_val = ridge.predict(X_val)
            
            # 성능 평가
            mse = mean_squared_error(y_val, y_pred_val)
            mae = mean_absolute_error(y_val, y_pred_val)
            r2 = r2_score(y_val, y_pred_val)
            ic_spearman, ic_p_spearman = self.calculate_spearman_ic(y_val, y_pred_val)
            hit_rate = self.calculate_hit_rate(y_val, y_pred_val)
            
            ridge_results[alpha] = {
                'model': ridge,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'ic_spearman': ic_spearman,
                'ic_spearman_p': ic_p_spearman,
                'hit_rate': hit_rate
            }
            
            print(f"      IC: {ic_spearman:.4f}, Hit Rate: {hit_rate:.4f}, R²: {r2:.4f}")
        
        # 최적 모델 선택 (IC 기준)
        best_alpha = max(ridge_results.keys(), key=lambda x: ridge_results[x]['ic_spearman'])
        best_model = ridge_results[best_alpha]['model']
        
        print(f"   ✅ Best Ridge model: alpha={best_alpha} (IC={ridge_results[best_alpha]['ic_spearman']:.4f})")
        
        self.models['ridge'] = best_model
        self.results['ridge'] = ridge_results
        
        return ridge_results, best_alpha
    
    def train_lasso_models(self, X_train, y_train, X_val, y_val):
        """Lasso Regression 모델 훈련"""
        print("📈 Training Lasso Regression models...")
        
        # 하이퍼파라미터 그리드
        alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        lasso_results = {}
        
        for alpha in alpha_values:
            print(f"   🔍 Training Lasso with alpha={alpha}...")
            
            lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
            lasso.fit(X_train, y_train)
            
            # 검증 데이터로 예측
            y_pred_val = lasso.predict(X_val)
            
            # 성능 평가
            mse = mean_squared_error(y_val, y_pred_val)
            mae = mean_absolute_error(y_val, y_pred_val)
            r2 = r2_score(y_val, y_pred_val)
            ic_spearman, ic_p_spearman = self.calculate_spearman_ic(y_val, y_pred_val)
            hit_rate = self.calculate_hit_rate(y_val, y_pred_val)
            
            # 선택된 피처 수 계산
            n_selected_features = np.sum(np.abs(lasso.coef_) > 1e-6)
            
            lasso_results[alpha] = {
                'model': lasso,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'ic_spearman': ic_spearman,
                'ic_spearman_p': ic_p_spearman,
                'hit_rate': hit_rate,
                'n_selected_features': n_selected_features
            }
            
            print(f"      IC: {ic_spearman:.4f}, Hit Rate: {hit_rate:.4f}, Features: {n_selected_features}")
        
        # 최적 모델 선택 (IC 기준)
        best_alpha = max(lasso_results.keys(), key=lambda x: lasso_results[x]['ic_spearman'])
        best_model = lasso_results[best_alpha]['model']
        
        print(f"   ✅ Best Lasso model: alpha={best_alpha} (IC={lasso_results[best_alpha]['ic_spearman']:.4f})")
        
        self.models['lasso'] = best_model
        self.results['lasso'] = lasso_results
        
        return lasso_results, best_alpha
    
    def train_elastic_net_models(self, X_train, y_train, X_val, y_val):
        """Elastic Net 모델 훈련"""
        print("📈 Training Elastic Net models...")
        
        # 하이퍼파라미터 그리드
        alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0]
        l1_ratio_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        elastic_net_results = {}
        
        for alpha in alpha_values:
            for l1_ratio in l1_ratio_values:
                print(f"   🔍 Training Elastic Net with alpha={alpha}, l1_ratio={l1_ratio}...")
                
                elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
                elastic_net.fit(X_train, y_train)
                
                # 검증 데이터로 예측
                y_pred_val = elastic_net.predict(X_val)
                
                # 성능 평가
                mse = mean_squared_error(y_val, y_pred_val)
                mae = mean_absolute_error(y_val, y_pred_val)
                r2 = r2_score(y_val, y_pred_val)
                ic_spearman, ic_p_spearman = self.calculate_spearman_ic(y_val, y_pred_val)
                hit_rate = self.calculate_hit_rate(y_val, y_pred_val)
                
                # 선택된 피처 수 계산
                n_selected_features = np.sum(np.abs(elastic_net.coef_) > 1e-6)
                
                param_key = f"alpha_{alpha}_l1_{l1_ratio}"
                elastic_net_results[param_key] = {
                    'model': elastic_net,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'ic_spearman': ic_spearman,
                    'ic_spearman_p': ic_p_spearman,
                    'hit_rate': hit_rate,
                    'n_selected_features': n_selected_features
                }
                
                print(f"      IC: {ic_spearman:.4f}, Hit Rate: {hit_rate:.4f}, Features: {n_selected_features}")
        
        # 최적 모델 선택 (IC 기준)
        best_params = max(elastic_net_results.keys(), key=lambda x: elastic_net_results[x]['ic_spearman'])
        best_model = elastic_net_results[best_params]['model']
        
        print(f"   ✅ Best Elastic Net model: {best_params} (IC={elastic_net_results[best_params]['ic_spearman']:.4f})")
        
        self.models['elastic_net'] = best_model
        self.results['elastic_net'] = elastic_net_results
        
        return elastic_net_results, best_params
    
    def evaluate_models(self, X_test, y_test, feature_names):
        """모델 평가"""
        print("📊 Evaluating models on test data...")
        
        test_results = {}
        
        for model_name, model in self.models.items():
            print(f"   🔍 Evaluating {model_name}...")
            
            # 예측
            y_pred = model.predict(X_test)
            
            # 기본 메트릭
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Spearman IC 계산
            ic_spearman, ic_p_spearman = self.calculate_spearman_ic(y_test, y_pred)
            
            # Hit Rate 계산
            hit_rate = self.calculate_hit_rate(y_test, y_pred)
            
            # 피처 가중치 분석
            feature_weights = model.coef_
            feature_importance = np.abs(feature_weights)
            
            # Reddit 피처 가중치만 추출
            reddit_indices = [i for i, name in enumerate(feature_names) if name in self.reddit_features]
            reddit_weights = feature_weights[reddit_indices]
            reddit_importance = feature_importance[reddit_indices]
            reddit_feature_names = [feature_names[i] for i in reddit_indices]
            
            test_results[model_name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'ic_spearman': ic_spearman,
                'ic_spearman_p': ic_p_spearman,
                'hit_rate': hit_rate,
                'feature_weights': feature_weights,
                'feature_importance': feature_importance,
                'reddit_weights': reddit_weights,
                'reddit_importance': reddit_importance,
                'reddit_feature_names': reddit_feature_names,
                'predictions': y_pred
            }
            
            print(f"      IC (Spearman): {ic_spearman:.4f} (p={ic_p_spearman:.4f})")
            print(f"      Hit Rate: {hit_rate:.4f}")
            print(f"      R²: {r2:.4f}")
        
        self.feature_weights = test_results
        return test_results
    
    def analyze_feature_weights(self):
        """피처 가중치 분석"""
        print("🔍 Analyzing feature weights...")
        
        weight_analysis = {}
        
        for model_name, results in self.feature_weights.items():
            print(f"   📊 Analyzing {model_name} weights...")
            
            reddit_weights = results['reddit_weights']
            reddit_importance = results['reddit_importance']
            reddit_feature_names = results['reddit_feature_names']
            
            # 가중치 통계
            weight_stats = {
                'mean_weight': np.mean(reddit_weights),
                'std_weight': np.std(reddit_weights),
                'max_weight': np.max(reddit_weights),
                'min_weight': np.min(reddit_weights),
                'mean_importance': np.mean(reddit_importance),
                'std_importance': np.std(reddit_importance),
                'max_importance': np.max(reddit_importance),
                'min_importance': np.min(reddit_importance)
            }
            
            # 상위 중요도 피처
            importance_df = pd.DataFrame({
                'feature': reddit_feature_names,
                'weight': reddit_weights,
                'importance': reddit_importance
            }).sort_values('importance', ascending=False)
            
            weight_analysis[model_name] = {
                'stats': weight_stats,
                'importance_df': importance_df,
                'top_features': importance_df.head(10)
            }
            
            print(f"      Mean weight: {weight_stats['mean_weight']:.4f}")
            print(f"      Mean importance: {weight_stats['mean_importance']:.4f}")
            print(f"      Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.4f})")
        
        return weight_analysis
    
    def create_performance_comparison(self):
        """성능 비교 표 생성"""
        print("📋 Creating performance comparison table...")
        
        comparison_data = []
        
        for model_name, results in self.feature_weights.items():
            comparison_data.append({
                'Model': model_name.title(),
                'IC_Spearman': f"{results['ic_spearman']:.4f}",
                'Hit_Rate': f"{results['hit_rate']:.4f}",
                'R2': f"{results['r2']:.4f}",
                'MSE': f"{results['mse']:.4f}",
                'MAE': f"{results['mae']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*100)
        print("REGULARIZATION MODELS PERFORMANCE COMPARISON")
        print("="*100)
        print(comparison_df.to_string(index=False))
        print("="*100)
        
        return comparison_df
    
    def create_weight_visualization(self, weight_analysis):
        """피처 가중치 시각화"""
        print("📈 Creating feature weight visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Regularization Models Feature Weight Analysis', fontsize=16, fontweight='bold')
        
        # 1. 모델별 IC 비교
        ax1 = axes[0, 0]
        model_names = list(self.feature_weights.keys())
        ic_values = [self.feature_weights[model]['ic_spearman'] for model in model_names]
        
        bars = ax1.bar(model_names, ic_values, alpha=0.8, color=['skyblue', 'lightgreen', 'orange'])
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Information Coefficient Comparison')
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, ic_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. 모델별 Hit Rate 비교
        ax2 = axes[0, 1]
        hit_rates = [self.feature_weights[model]['hit_rate'] for model in model_names]
        
        bars = ax2.bar(model_names, hit_rates, alpha=0.8, color=['skyblue', 'lightgreen', 'orange'])
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, hit_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. 모델별 R² 비교
        ax3 = axes[0, 2]
        r2_values = [self.feature_weights[model]['r2'] for model in model_names]
        
        bars = ax3.bar(model_names, r2_values, alpha=0.8, color=['skyblue', 'lightgreen', 'orange'])
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score Comparison')
        ax3.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, r2_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 4-6. 각 모델별 상위 피처 중요도
        for i, (model_name, analysis) in enumerate(weight_analysis.items()):
            ax = axes[1, i]
            
            top_features = analysis['top_features'].head(10)
            
            bars = ax.barh(range(len(top_features)), top_features['importance'], alpha=0.8)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=8)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_name.title()} Top Features')
            ax.grid(True, alpha=0.3)
            
            # 값 표시
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/regularization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Regularization visualization saved to results/regularization_analysis.png")
    
    def create_hyperparameter_analysis(self):
        """하이퍼파라미터 분석 시각화"""
        print("📈 Creating hyperparameter analysis...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Hyperparameter Analysis', fontsize=16, fontweight='bold')
        
        # 1. Ridge Alpha vs IC
        ax1 = axes[0]
        ridge_results = self.results['ridge']
        alphas = list(ridge_results.keys())
        ic_values = [ridge_results[alpha]['ic_spearman'] for alpha in alphas]
        
        ax1.semilogx(alphas, ic_values, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Alpha (log scale)')
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Ridge: Alpha vs IC')
        ax1.grid(True, alpha=0.3)
        
        # 최적점 표시
        best_alpha = max(alphas, key=lambda x: ridge_results[x]['ic_spearman'])
        best_ic = ridge_results[best_alpha]['ic_spearman']
        ax1.axvline(best_alpha, color='red', linestyle='--', alpha=0.7)
        ax1.text(best_alpha, best_ic, f'Best: {best_alpha}', ha='center', va='bottom')
        
        # 2. Lasso Alpha vs IC
        ax2 = axes[1]
        lasso_results = self.results['lasso']
        alphas = list(lasso_results.keys())
        ic_values = [lasso_results[alpha]['ic_spearman'] for alpha in alphas]
        n_features = [lasso_results[alpha]['n_selected_features'] for alpha in alphas]
        
        ax2.semilogx(alphas, ic_values, marker='o', linewidth=2, markersize=8, label='IC')
        ax2.set_xlabel('Alpha (log scale)')
        ax2.set_ylabel('IC (Spearman)')
        ax2.set_title('Lasso: Alpha vs IC')
        ax2.grid(True, alpha=0.3)
        
        # 최적점 표시
        best_alpha = max(alphas, key=lambda x: lasso_results[x]['ic_spearman'])
        best_ic = lasso_results[best_alpha]['ic_spearman']
        ax2.axvline(best_alpha, color='red', linestyle='--', alpha=0.7)
        ax2.text(best_alpha, best_ic, f'Best: {best_alpha}', ha='center', va='bottom')
        
        # 3. Elastic Net Heatmap
        ax3 = axes[2]
        elastic_net_results = self.results['elastic_net']
        
        # 파라미터별 IC 매트릭스 생성
        alpha_values = sorted(set([result['alpha'] for result in elastic_net_results.values()]))
        l1_ratio_values = sorted(set([result['l1_ratio'] for result in elastic_net_results.values()]))
        
        ic_matrix = np.zeros((len(alpha_values), len(l1_ratio_values)))
        
        for i, alpha in enumerate(alpha_values):
            for j, l1_ratio in enumerate(l1_ratio_values):
                param_key = f"alpha_{alpha}_l1_{l1_ratio}"
                if param_key in elastic_net_results:
                    ic_matrix[i, j] = elastic_net_results[param_key]['ic_spearman']
        
        im = ax3.imshow(ic_matrix, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(l1_ratio_values)))
        ax3.set_xticklabels([f'{l1:.1f}' for l1 in l1_ratio_values])
        ax3.set_yticks(range(len(alpha_values)))
        ax3.set_yticklabels([f'{a:.3f}' for a in alpha_values])
        ax3.set_xlabel('L1 Ratio')
        ax3.set_ylabel('Alpha')
        ax3.set_title('Elastic Net: IC Heatmap')
        
        # 컬러바 추가
        plt.colorbar(im, ax=ax3, label='IC (Spearman)')
        
        plt.tight_layout()
        plt.savefig('results/hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Hyperparameter analysis saved to results/hyperparameter_analysis.png")
    
    def generate_final_report(self, comparison_df, weight_analysis):
        """최종 리포트 생성"""
        print("📝 Generating final regularization report...")
        
        report = []
        report.append("=" * 120)
        report.append("REGULARIZATION EXPERIMENT REPORT")
        report.append("=" * 120)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Ridge Regression: L2 regularization")
        report.append("- Lasso Regression: L1 regularization (feature selection)")
        report.append("- Elastic Net: L1 + L2 regularization")
        report.append("- Hyperparameter tuning: Alpha and L1 ratio")
        report.append("- Evaluation: IC (Spearman), Hit Rate, R²")
        report.append("")
        
        # 성능 비교
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 50)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # 최고 성능 모델
        best_model = max(self.feature_weights.keys(), key=lambda x: self.feature_weights[x]['ic_spearman'])
        best_ic = self.feature_weights[best_model]['ic_spearman']
        
        report.append("BEST MODEL")
        report.append("-" * 50)
        report.append(f"Best Model: {best_model.title()}")
        report.append(f"Best IC: {best_ic:.4f}")
        report.append("")
        
        # 피처 가중치 분석
        report.append("FEATURE WEIGHT ANALYSIS")
        report.append("-" * 50)
        
        for model_name, analysis in weight_analysis.items():
            report.append(f"{model_name.title()}:")
            report.append(f"  Mean weight: {analysis['stats']['mean_weight']:.4f}")
            report.append(f"  Mean importance: {analysis['stats']['mean_importance']:.4f}")
            report.append(f"  Weight std: {analysis['stats']['std_weight']:.4f}")
            report.append("")
            
            report.append(f"  Top 5 Reddit Features:")
            for _, row in analysis['top_features'].head(5).iterrows():
                report.append(f"    {row['feature']}: {row['importance']:.4f}")
            report.append("")
        
        # 정규화 효과 분석
        report.append("REGULARIZATION EFFECTIVENESS")
        report.append("-" * 50)
        
        # 가중치 분산 분석
        weight_vars = {}
        for model_name, results in self.feature_weights.items():
            weight_vars[model_name] = np.var(results['reddit_weights'])
        
        report.append("Weight Variance (Lower = More Regularized):")
        for model_name, var in weight_vars.items():
            report.append(f"  {model_name.title()}: {var:.6f}")
        report.append("")
        
        # 결론
        report.append("CONCLUSIONS")
        report.append("-" * 50)
        
        if best_ic > 0:
            report.append("✅ Regularization shows positive contribution to prediction performance")
        else:
            report.append("❌ Regularization shows negative impact on prediction performance")
        
        report.append(f"Best regularization approach: {best_model.title()}")
        report.append("")
        
        # 정규화 효과
        if weight_vars['ridge'] < weight_vars['lasso']:
            report.append("Ridge provides better regularization than Lasso")
        else:
            report.append("Lasso provides better regularization than Ridge")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/regularization_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Report saved to results/regularization_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Regularization Experiment")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = RegularizationExperiment()
    
    # 1. 데이터 로드
    df = experiment.load_data()
    
    # 2. 특성 준비
    price_features, reddit_features = experiment.prepare_features()
    
    # 3. 시계열 분할
    train_df, val_df, test_df = experiment.strict_time_series_split()
    
    # 4. 모델 데이터 준비
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = experiment.prepare_model_data(train_df, val_df, test_df)
    
    # 5. Ridge Regression 실험
    print("\n" + "="*50)
    print("RIDGE REGRESSION EXPERIMENT")
    print("="*50)
    ridge_results, best_ridge_alpha = experiment.train_ridge_models(X_train, y_train, X_val, y_val)
    
    # 6. Lasso Regression 실험
    print("\n" + "="*50)
    print("LASSO REGRESSION EXPERIMENT")
    print("="*50)
    lasso_results, best_lasso_alpha = experiment.train_lasso_models(X_train, y_train, X_val, y_val)
    
    # 7. Elastic Net 실험
    print("\n" + "="*50)
    print("ELASTIC NET EXPERIMENT")
    print("="*50)
    elastic_net_results, best_elastic_params = experiment.train_elastic_net_models(X_train, y_train, X_val, y_val)
    
    # 8. 테스트 데이터 평가
    print("\n" + "="*50)
    print("TEST DATA EVALUATION")
    print("="*50)
    test_results = experiment.evaluate_models(X_test, y_test, feature_names)
    
    # 9. 피처 가중치 분석
    print("\n" + "="*50)
    print("FEATURE WEIGHT ANALYSIS")
    print("="*50)
    weight_analysis = experiment.analyze_feature_weights()
    
    # 10. 성능 비교
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    comparison_df = experiment.create_performance_comparison()
    
    # 11. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    experiment.create_weight_visualization(weight_analysis)
    experiment.create_hyperparameter_analysis()
    
    # 12. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_final_report(comparison_df, weight_analysis)
    
    print("\n🎉 Regularization experiment completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df, weight_analysis

if __name__ == "__main__":
    experiment, comparison_df, weight_analysis = main()
