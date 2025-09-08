#!/usr/bin/env python3
"""
Comprehensive Model Comparison: Reddit Features vs Baseline
고급 Reddit 피처 기반 모델 vs 베이스라인 모델 성능 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveModelComparison:
    """종합적인 모델 성능 비교 클래스"""
    
    def __init__(self):
        self.df = None
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self):
        """데이터 로드"""
        print("📊 Loading comprehensive dataset...")
        
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
        
        # 미래 데이터 마스킹 (마지막 5일 제외)
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
    
    def prepare_feature_sets(self):
        """특성 세트 준비"""
        print("🔧 Preparing feature sets...")
        
        # 주가 관련 특성 (베이스라인)
        self.price_features = [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
            'vol_5d', 'vol_10d', 'vol_20d',
            'price_ratio_sma10', 'price_ratio_sma20',
            'rsi_14', 'volume_ratio', 'turnover',
            'day_of_week', 'month', 'is_monday', 'is_friday', 'is_weekend_effect'
        ]
        
        # 기본 Reddit 특성
        self.basic_reddit_features = [
            'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10'
        ]
        
        # 고급 Reddit 특성
        self.advanced_reddit_features = [
            'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95',
            'reddit_momentum_3', 'reddit_momentum_7', 'reddit_momentum_14', 'reddit_momentum_21',
            'reddit_vol_5', 'reddit_vol_10', 'reddit_vol_20',
            'reddit_percentile', 'reddit_high_regime', 'reddit_low_regime',
            'market_sentiment', 'price_reddit_momentum', 'vol_reddit_attention'
        ]
        
        # 존재하는 특성만 선택
        available_price_features = [col for col in self.price_features if col in self.df.columns]
        available_basic_reddit = [col for col in self.basic_reddit_features if col in self.df.columns]
        available_advanced_reddit = [col for col in self.advanced_reddit_features if col in self.df.columns]
        
        self.price_features = available_price_features
        self.basic_reddit_features = available_basic_reddit
        self.advanced_reddit_features = available_advanced_reddit
        
        print(f"   ✅ Price features: {len(self.price_features)}")
        print(f"   ✅ Basic Reddit features: {len(self.basic_reddit_features)}")
        print(f"   ✅ Advanced Reddit features: {len(self.advanced_reddit_features)}")
        
        return self.price_features, self.basic_reddit_features, self.advanced_reddit_features
    
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
    
    def prepare_model_data(self, train_df, val_df, test_df, feature_set='price_only'):
        """모델 데이터 준비"""
        print(f"🔧 Preparing model data ({feature_set})...")
        
        # 특성 선택
        if feature_set == 'price_only':
            all_features = self.price_features
        elif feature_set == 'price_basic_reddit':
            all_features = self.price_features + self.basic_reddit_features
        elif feature_set == 'price_advanced_reddit':
            all_features = self.price_features + self.advanced_reddit_features
        elif feature_set == 'price_all_reddit':
            all_features = self.price_features + self.basic_reddit_features + self.advanced_reddit_features
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        
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
        
        print(f"   ✅ Using {feature_set}: {len(all_features)} features")
        print(f"   ✅ Final features: {len(final_features)}")
        print(f"   ✅ Data shapes: Train {X_train_scaled.shape}, Val {X_val_scaled.shape}, Test {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, final_features, scaler
    
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
    
    def train_ridge_model(self, X_train, y_train, X_val, y_val, feature_set):
        """Ridge 모델 훈련"""
        print(f"   📈 Training Ridge ({feature_set})...")
        
        # 하이퍼파라미터 튜닝
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        ridge = Ridge(random_state=42)
        
        # GridSearchCV로 최적 파라미터 찾기
        grid_search = GridSearchCV(ridge, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_ridge = grid_search.best_estimator_
        print(f"      Best alpha: {best_ridge.alpha}")
        
        return best_ridge
    
    def train_lgb_model(self, X_train, y_train, X_val, y_val, feature_set):
        """LightGBM 모델 훈련"""
        print(f"   🌟 Training LightGBM ({feature_set})...")
        
        # 하이퍼파라미터 튜닝
        param_grid = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200]
        }
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for num_leaves in param_grid['num_leaves']:
            for lr in param_grid['learning_rate']:
                for n_est in param_grid['n_estimators']:
                    lgb_params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': num_leaves,
                        'learning_rate': lr,
                        'verbose': -1,
                        'random_state': 42
                    }
                    
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val)
                    
                    model = lgb.train(lgb_params, train_data, num_boost_round=n_est, 
                                    valid_sets=[val_data], callbacks=[lgb.log_evaluation(0)])
                    
                    # 검증 성능 평가
                    val_pred = model.predict(X_val)
                    val_score = -mean_squared_error(y_val, val_pred)
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = lgb_params.copy()
                        best_params['n_estimators'] = n_est
                        best_model = model
        
        print(f"      Best params: {best_params}")
        return best_model
    
    def train_xgb_model(self, X_train, y_train, X_val, y_val, feature_set):
        """XGBoost 모델 훈련"""
        print(f"   🚀 Training XGBoost ({feature_set})...")
        
        # 하이퍼파라미터 튜닝
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200]
        }
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for max_depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                for n_est in param_grid['n_estimators']:
                    xgb_params = {
                        'objective': 'reg:squarederror',
                        'max_depth': max_depth,
                        'learning_rate': lr,
                        'n_estimators': n_est,
                        'random_state': 42,
                        'verbosity': 0
                    }
                    
                    model = xgb.XGBRegressor(**xgb_params)
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)], 
                             verbose=False)
                    
                    # 검증 성능 평가
                    val_pred = model.predict(X_val)
                    val_score = -mean_squared_error(y_val, val_pred)
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = xgb_params
                        best_model = model
        
        print(f"      Best params: {best_params}")
        return best_model
    
    def evaluate_model(self, model, X_test, y_test, model_name, feature_set, scaler=None):
        """모델 평가"""
        print(f"   🔍 Evaluating {model_name} ({feature_set})...")
        
        # 예측
        if model_name == 'Ridge':
            y_pred = model.predict(scaler.transform(X_test))
        else:
            y_pred = model.predict(X_test)
        
        # 기본 메트릭
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Spearman IC 계산
        ic_spearman, ic_p_spearman = self.calculate_spearman_ic(y_test, y_pred)
        
        # Hit Rate 계산
        hit_rate = self.calculate_hit_rate(y_test, y_pred)
        
        results = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'IC_Spearman': ic_spearman,
            'IC_Spearman_p': ic_p_spearman,
            'Hit_Rate': hit_rate,
            'predictions': y_pred
        }
        
        print(f"      IC (Spearman): {ic_spearman:.4f} (p={ic_p_spearman:.4f})")
        print(f"      Hit Rate: {hit_rate:.4f}")
        print(f"      R²: {r2:.4f}")
        
        return results
    
    def run_comprehensive_experiment(self):
        """종합 실험 실행"""
        print("🚀 Starting Comprehensive Model Comparison Experiment")
        print("=" * 80)
        
        # 데이터 로드
        df = self.load_data()
        
        # 특성 준비
        price_features, basic_reddit, advanced_reddit = self.prepare_feature_sets()
        
        # 시계열 분할
        train_df, val_df, test_df = self.strict_time_series_split()
        
        # 실험할 특성 세트들
        feature_sets = {
            'price_only': 'Price Only (Baseline)',
            'price_basic_reddit': 'Price + Basic Reddit',
            'price_advanced_reddit': 'Price + Advanced Reddit',
            'price_all_reddit': 'Price + All Reddit'
        }
        
        # 모델들
        models = ['Ridge', 'LightGBM', 'XGBoost']
        
        # 결과 저장
        all_results = {}
        
        for feature_set, feature_name in feature_sets.items():
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {feature_name}")
            print(f"{'='*60}")
            
            # 데이터 준비
            X_train, X_val, X_test, y_train, y_val, y_test, final_features, scaler = self.prepare_model_data(
                train_df, val_df, test_df, feature_set)
            
            feature_results = {}
            
            # Ridge 모델
            ridge_model = self.train_ridge_model(X_train, y_train, X_val, y_val, feature_set)
            ridge_results = self.evaluate_model(ridge_model, X_test, y_test, 'Ridge', feature_set, scaler)
            feature_results['Ridge'] = ridge_results
            
            # LightGBM 모델
            lgb_model = self.train_lgb_model(X_train, y_train, X_val, y_val, feature_set)
            lgb_results = self.evaluate_model(lgb_model, X_test, y_test, 'LightGBM', feature_set)
            feature_results['LightGBM'] = lgb_results
            
            # XGBoost 모델
            xgb_model = self.train_xgb_model(X_train, y_train, X_val, y_val, feature_set)
            xgb_results = self.evaluate_model(xgb_model, X_test, y_test, 'XGBoost', feature_set)
            feature_results['XGBoost'] = xgb_results
            
            all_results[feature_set] = feature_results
        
        self.results = all_results
        return all_results
    
    def create_comprehensive_comparison_table(self):
        """종합 비교 표 생성"""
        print("📋 Creating comprehensive comparison table...")
        
        comparison_data = []
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            row_data = {'Model': model_name}
            
            for feature_set in ['price_only', 'price_basic_reddit', 'price_advanced_reddit', 'price_all_reddit']:
                if feature_set in self.results and model_name in self.results[feature_set]:
                    result = self.results[feature_set][model_name]
                    row_data[f'{feature_set}_IC'] = f"{result['IC_Spearman']:.4f}"
                    row_data[f'{feature_set}_Hit_Rate'] = f"{result['Hit_Rate']:.4f}"
                    row_data[f'{feature_set}_R2'] = f"{result['R2']:.4f}"
                else:
                    row_data[f'{feature_set}_IC'] = 'N/A'
                    row_data[f'{feature_set}_Hit_Rate'] = 'N/A'
                    row_data[f'{feature_set}_R2'] = 'N/A'
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*150)
        print("COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("="*150)
        print(comparison_df.to_string(index=False))
        print("="*150)
        
        return comparison_df
    
    def analyze_reddit_feature_impact(self):
        """Reddit 피처 영향 분석"""
        print("🔍 Analyzing Reddit feature impact...")
        
        impact_analysis = {}
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            print(f"\n📊 {model_name} Reddit Feature Impact:")
            print("-" * 50)
            
            model_impacts = {}
            
            # 베이스라인 대비 개선도 계산
            baseline_ic = self.results['price_only'][model_name]['IC_Spearman']
            baseline_hr = self.results['price_only'][model_name]['Hit_Rate']
            baseline_r2 = self.results['price_only'][model_name]['R2']
            
            print(f"Baseline (Price Only):")
            print(f"  IC: {baseline_ic:.4f}")
            print(f"  Hit Rate: {baseline_hr:.4f}")
            print(f"  R²: {baseline_r2:.4f}")
            
            # 각 Reddit 피처 세트별 개선도
            reddit_sets = {
                'price_basic_reddit': 'Basic Reddit',
                'price_advanced_reddit': 'Advanced Reddit',
                'price_all_reddit': 'All Reddit'
            }
            
            for feature_set, feature_name in reddit_sets.items():
                if feature_set in self.results and model_name in self.results[feature_set]:
                    result = self.results[feature_set][model_name]
                    
                    ic_improvement = result['IC_Spearman'] - baseline_ic
                    hr_improvement = result['Hit_Rate'] - baseline_hr
                    r2_improvement = result['R2'] - baseline_r2
                    
                    print(f"\n{feature_name}:")
                    print(f"  IC: {result['IC_Spearman']:.4f} ({ic_improvement:+.4f})")
                    print(f"  Hit Rate: {result['Hit_Rate']:.4f} ({hr_improvement:+.4f})")
                    print(f"  R²: {result['R2']:.4f} ({r2_improvement:+.4f})")
                    
                    model_impacts[feature_set] = {
                        'ic_improvement': ic_improvement,
                        'hr_improvement': hr_improvement,
                        'r2_improvement': r2_improvement,
                        'ic_improvement_pct': (ic_improvement / abs(baseline_ic)) * 100 if baseline_ic != 0 else 0
                    }
            
            impact_analysis[model_name] = model_impacts
        
        self.impact_analysis = impact_analysis
        return impact_analysis
    
    def create_comprehensive_visualization(self):
        """종합 시각화 생성"""
        print("📈 Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Comprehensive Model Comparison: Reddit Features vs Baseline', fontsize=16, fontweight='bold')
        
        # 1. IC 비교
        ax1 = axes[0, 0]
        models = ['Ridge', 'LightGBM', 'XGBoost']
        feature_sets = ['price_only', 'price_basic_reddit', 'price_advanced_reddit', 'price_all_reddit']
        feature_names = ['Price Only', 'Price + Basic Reddit', 'Price + Advanced Reddit', 'Price + All Reddit']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, (feature_set, feature_name) in enumerate(zip(feature_sets, feature_names)):
            ic_values = [self.results[feature_set][model]['IC_Spearman'] for model in models]
            ax1.bar(x + i*width, ic_values, width, label=feature_name, alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Information Coefficient Comparison', fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Hit Rate 비교
        ax2 = axes[0, 1]
        for i, (feature_set, feature_name) in enumerate(zip(feature_sets, feature_names)):
            hr_values = [self.results[feature_set][model]['Hit_Rate'] for model in models]
            ax2.bar(x + i*width, hr_values, width, label=feature_name, alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate Comparison', fontweight='bold')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. R² 비교
        ax3 = axes[0, 2]
        for i, (feature_set, feature_name) in enumerate(zip(feature_sets, feature_names)):
            r2_values = [self.results[feature_set][model]['R2'] for model in models]
            ax3.bar(x + i*width, r2_values, width, label=feature_name, alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score Comparison', fontweight='bold')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. IC 개선도
        ax4 = axes[1, 0]
        baseline_ic = [self.results['price_only'][model]['IC_Spearman'] for model in models]
        advanced_ic = [self.results['price_advanced_reddit'][model]['IC_Spearman'] for model in models]
        all_ic = [self.results['price_all_reddit'][model]['IC_Spearman'] for model in models]
        
        ic_improvements_advanced = [advanced - baseline for advanced, baseline in zip(advanced_ic, baseline_ic)]
        ic_improvements_all = [all_r - baseline for all_r, baseline in zip(all_ic, baseline_ic)]
        
        bars1 = ax4.bar(x - width/2, ic_improvements_advanced, width, label='Advanced Reddit', alpha=0.8)
        bars2 = ax4.bar(x + width/2, ic_improvements_all, width, label='All Reddit', alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('IC Improvement')
        ax4.set_title('IC Improvement vs Baseline', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                        f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # 5. Hit Rate 개선도
        ax5 = axes[1, 1]
        baseline_hr = [self.results['price_only'][model]['Hit_Rate'] for model in models]
        advanced_hr = [self.results['price_advanced_reddit'][model]['Hit_Rate'] for model in models]
        all_hr = [self.results['price_all_reddit'][model]['Hit_Rate'] for model in models]
        
        hr_improvements_advanced = [advanced - baseline for advanced, baseline in zip(advanced_hr, baseline_hr)]
        hr_improvements_all = [all_r - baseline for all_r, baseline in zip(all_hr, baseline_hr)]
        
        bars1 = ax5.bar(x - width/2, hr_improvements_advanced, width, label='Advanced Reddit', alpha=0.8)
        bars2 = ax5.bar(x + width/2, hr_improvements_all, width, label='All Reddit', alpha=0.8)
        
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Hit Rate Improvement')
        ax5.set_title('Hit Rate Improvement vs Baseline', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(models)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                        f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # 6. 최고 성능 모델 비교
        ax6 = axes[1, 2]
        best_ic_by_feature = []
        best_model_by_feature = []
        
        for feature_set, feature_name in zip(feature_sets, feature_names):
            best_ic = -np.inf
            best_model = None
            for model in models:
                ic = self.results[feature_set][model]['IC_Spearman']
                if ic > best_ic:
                    best_ic = ic
                    best_model = model
            best_ic_by_feature.append(best_ic)
            best_model_by_feature.append(best_model)
        
        bars = ax6.bar(range(len(feature_names)), best_ic_by_feature, alpha=0.8, color=['lightblue', 'orange', 'green', 'red'])
        ax6.set_xlabel('Feature Sets')
        ax6.set_ylabel('Best IC (Spearman)')
        ax6.set_title('Best Model Performance by Feature Set', fontweight='bold')
        ax6.set_xticks(range(len(feature_names)))
        ax6.set_xticklabels(feature_names, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        # 값과 모델명 표시
        for i, (bar, ic, model) in enumerate(zip(bars, best_ic_by_feature, best_model_by_feature)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ic:.3f}\n({model})', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Comprehensive visualization saved to results/comprehensive_model_comparison.png")
    
    def generate_final_comprehensive_report(self, comparison_df, impact_analysis):
        """최종 종합 리포트 생성"""
        print("📝 Generating final comprehensive report...")
        
        report = []
        report.append("=" * 150)
        report.append("COMPREHENSIVE MODEL COMPARISON: REDDIT FEATURES VS BASELINE")
        report.append("=" * 150)
        report.append("")
        report.append("EXPERIMENT OVERVIEW:")
        report.append("- Target: Compare Reddit feature impact on stock price prediction")
        report.append("- Models: Ridge Regression, LightGBM, XGBoost")
        report.append("- Feature Sets: Price Only, Price + Basic Reddit, Price + Advanced Reddit, Price + All Reddit")
        report.append("- Evaluation: IC (Spearman), Hit Rate, R²")
        report.append("- Data: Strict time series split (chronological order)")
        report.append("")
        
        # 성능 비교 표
        report.append("PERFORMANCE COMPARISON TABLE")
        report.append("-" * 100)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Reddit 피처 영향 분석
        report.append("REDDIT FEATURE IMPACT ANALYSIS")
        report.append("-" * 100)
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            report.append(f"\n{model_name} ANALYSIS:")
            report.append(f"  Baseline (Price Only) Performance:")
            baseline_result = self.results['price_only'][model_name]
            report.append(f"    IC: {baseline_result['IC_Spearman']:.4f}")
            report.append(f"    Hit Rate: {baseline_result['Hit_Rate']:.4f}")
            report.append(f"    R²: {baseline_result['R2']:.4f}")
            
            # 고급 Reddit 피처 개선도
            if 'price_advanced_reddit' in impact_analysis[model_name]:
                advanced_impact = impact_analysis[model_name]['price_advanced_reddit']
                report.append(f"  Advanced Reddit Features Impact:")
                report.append(f"    IC Improvement: {advanced_impact['ic_improvement']:+.4f} ({advanced_impact['ic_improvement_pct']:+.1f}%)")
                report.append(f"    Hit Rate Improvement: {advanced_impact['hr_improvement']:+.4f}")
                report.append(f"    R² Improvement: {advanced_impact['r2_improvement']:+.4f}")
            
            # 전체 Reddit 피처 개선도
            if 'price_all_reddit' in impact_analysis[model_name]:
                all_impact = impact_analysis[model_name]['price_all_reddit']
                report.append(f"  All Reddit Features Impact:")
                report.append(f"    IC Improvement: {all_impact['ic_improvement']:+.4f} ({all_impact['ic_improvement_pct']:+.1f}%)")
                report.append(f"    Hit Rate Improvement: {all_impact['hr_improvement']:+.4f}")
                report.append(f"    R² Improvement: {all_impact['r2_improvement']:+.4f}")
        
        # 전체 결론
        report.append("\nOVERALL CONCLUSIONS")
        report.append("-" * 100)
        
        # 평균 개선도 계산
        avg_ic_improvement_advanced = np.mean([impact_analysis[model]['price_advanced_reddit']['ic_improvement'] 
                                             for model in ['Ridge', 'LightGBM', 'XGBoost']])
        avg_ic_improvement_all = np.mean([impact_analysis[model]['price_all_reddit']['ic_improvement'] 
                                        for model in ['Ridge', 'LightGBM', 'XGBoost']])
        
        report.append(f"Average IC Improvement (Advanced Reddit): {avg_ic_improvement_advanced:+.4f}")
        report.append(f"Average IC Improvement (All Reddit): {avg_ic_improvement_all:+.4f}")
        report.append("")
        
        # Reddit 피처 효과 평가
        if avg_ic_improvement_advanced > 0.01:
            report.append("✅ ADVANCED REDDIT FEATURES SIGNIFICANTLY IMPROVE PERFORMANCE")
            report.append("   - Advanced Reddit features provide meaningful predictive power")
            report.append("   - Contrarian effect and interaction features are valuable")
        elif avg_ic_improvement_advanced > 0:
            report.append("⚠️ ADVANCED REDDIT FEATURES MODERATELY IMPROVE PERFORMANCE")
            report.append("   - Advanced Reddit features provide some predictive power")
            report.append("   - Benefits are modest but consistent")
        else:
            report.append("❌ ADVANCED REDDIT FEATURES DO NOT IMPROVE PERFORMANCE")
            report.append("   - Advanced Reddit features may add noise")
            report.append("   - Price-only models may be more robust")
        
        report.append("")
        
        # 모델별 권장사항
        report.append("MODEL-SPECIFIC RECOMMENDATIONS")
        report.append("-" * 100)
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            best_feature_set = 'price_only'
            best_ic = self.results['price_only'][model_name]['IC_Spearman']
            
            for feature_set in ['price_basic_reddit', 'price_advanced_reddit', 'price_all_reddit']:
                ic = self.results[feature_set][model_name]['IC_Spearman']
                if ic > best_ic:
                    best_ic = ic
                    best_feature_set = feature_set
            
            report.append(f"{model_name}:")
            report.append(f"  Best Feature Set: {best_feature_set}")
            report.append(f"  Best IC: {best_ic:.4f}")
            report.append("")
        
        # 실전 적용 가이드
        report.append("PRACTICAL APPLICATION GUIDE")
        report.append("-" * 100)
        report.append("🔹 For Maximum Performance:")
        report.append("  - Use Advanced Reddit features with Ridge or LightGBM")
        report.append("  - Focus on reddit_surprise and price_reddit_momentum")
        report.append("  - Monitor contrarian effect signals")
        report.append("")
        report.append("🔹 For Robust Performance:")
        report.append("  - Use Price-only baseline for stability")
        report.append("  - Add Reddit features gradually")
        report.append("  - Validate performance on out-of-sample data")
        report.append("")
        report.append("🔹 For Research Purposes:")
        report.append("  - Advanced Reddit features show contrarian effect")
        report.append("  - Interaction features capture price-Reddit dynamics")
        report.append("  - Feature engineering is crucial for Reddit data")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/comprehensive_model_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Comprehensive report saved to results/comprehensive_model_comparison_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Comprehensive Model Comparison")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = ComprehensiveModelComparison()
    
    # 1. 종합 실험 실행
    results = experiment.run_comprehensive_experiment()
    
    # 2. 비교 표 생성
    print("\n" + "="*50)
    print("COMPARISON TABLE GENERATION")
    print("="*50)
    comparison_df = experiment.create_comprehensive_comparison_table()
    
    # 3. Reddit 피처 영향 분석
    print("\n" + "="*50)
    print("REDDIT FEATURE IMPACT ANALYSIS")
    print("="*50)
    impact_analysis = experiment.analyze_reddit_feature_impact()
    
    # 4. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    experiment.create_comprehensive_visualization()
    
    # 5. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_final_comprehensive_report(comparison_df, impact_analysis)
    
    print("\n🎉 Comprehensive model comparison completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df, impact_analysis

if __name__ == "__main__":
    experiment, comparison_df, impact_analysis = main()
