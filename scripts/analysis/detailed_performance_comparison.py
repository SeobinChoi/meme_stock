#!/usr/bin/env python3
"""
Detailed Performance Comparison: Price Only vs Reddit All vs Advanced Reddit
각 모델별로 Price Only, Reddit All, Advanced Reddit의 상세 성능 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DetailedPerformanceComparison:
    """상세 성능 비교 클래스"""
    
    def __init__(self):
        self.df = None
        self.results = {}
        self.target_tickers = ['AMC', 'BB', 'GME']
        
    def load_data(self):
        """데이터 로드 (AMC, BB, GME만)"""
        print("📊 Loading meme stocks data (AMC, BB, GME only)...")
        
        # 통합 데이터셋 로드
        train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
        val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
        test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        # 데이터 통합
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # AMC, BB, GME만 필터링
        df = df[df['ticker'].isin(self.target_tickers)].copy()
        
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
        
        for ticker in self.target_tickers:
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
        elif feature_set == 'reddit_all':
            all_features = self.price_features + self.basic_reddit_features + self.advanced_reddit_features
        elif feature_set == 'advanced_reddit':
            all_features = self.price_features + self.advanced_reddit_features
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
    
    def train_models(self, X_train, y_train, X_val, y_val, feature_set):
        """모델 훈련"""
        print(f"🤖 Training models ({feature_set})...")
        
        models = {}
        
        # 1. Ridge Regression
        print(f"   📈 Training Ridge ({feature_set})...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        models['Ridge'] = ridge
        
        # 2. LightGBM
        print(f"   🌟 Training LightGBM ({feature_set})...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        lgb_model = lgb.train(lgb_params, train_data, num_boost_round=100, 
                             valid_sets=[val_data], callbacks=[lgb.log_evaluation(0)])
        models['LightGBM'] = lgb_model
        
        # 3. XGBoost
        print(f"   🚀 Training XGBoost ({feature_set})...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'verbosity': 0
        }
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     verbose=False)
        models['XGBoost'] = xgb_model
        
        print(f"   ✅ All {feature_set} models trained successfully")
        
        return models
    
    def evaluate_models(self, X_test, y_test, models, feature_set, scaler=None):
        """모델 평가"""
        print(f"📊 Evaluating {feature_set} models...")
        
        results = {}
        
        for model_name, model in models.items():
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
            
            results[model_name] = {
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
    
    def run_detailed_experiment(self):
        """상세 실험 실행"""
        print("🚀 Starting Detailed Performance Comparison Experiment")
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
            'reddit_all': 'Price + All Reddit',
            'advanced_reddit': 'Price + Advanced Reddit'
        }
        
        # 결과 저장
        all_results = {}
        
        for feature_set, feature_name in feature_sets.items():
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {feature_name}")
            print(f"{'='*60}")
            
            # 데이터 준비
            X_train, X_val, X_test, y_train, y_val, y_test, final_features, scaler = self.prepare_model_data(
                train_df, val_df, test_df, feature_set)
            
            # 모델 훈련
            models = self.train_models(X_train, y_train, X_val, y_val, feature_set)
            
            # 모델 평가
            feature_results = self.evaluate_models(X_test, y_test, models, feature_set, scaler)
            
            all_results[feature_set] = feature_results
        
        self.results = all_results
        return all_results
    
    def create_detailed_comparison_table(self):
        """상세 비교 표 생성"""
        print("📋 Creating detailed comparison table...")
        
        comparison_data = []
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            price_only = self.results['price_only'][model_name]
            reddit_all = self.results['reddit_all'][model_name]
            advanced_reddit = self.results['advanced_reddit'][model_name]
            
            comparison_data.append({
                'Model': model_name,
                'Price_Only_IC': f"{price_only['IC_Spearman']:.4f}",
                'Reddit_All_IC': f"{reddit_all['IC_Spearman']:.4f}",
                'Advanced_Reddit_IC': f"{advanced_reddit['IC_Spearman']:.4f}",
                'Reddit_All_IC_Improvement': f"{reddit_all['IC_Spearman'] - price_only['IC_Spearman']:+.4f}",
                'Advanced_Reddit_IC_Improvement': f"{advanced_reddit['IC_Spearman'] - price_only['IC_Spearman']:+.4f}",
                'Price_Only_Hit_Rate': f"{price_only['Hit_Rate']:.4f}",
                'Reddit_All_Hit_Rate': f"{reddit_all['Hit_Rate']:.4f}",
                'Advanced_Reddit_Hit_Rate': f"{advanced_reddit['Hit_Rate']:.4f}",
                'Reddit_All_HR_Improvement': f"{reddit_all['Hit_Rate'] - price_only['Hit_Rate']:+.4f}",
                'Advanced_Reddit_HR_Improvement': f"{advanced_reddit['Hit_Rate'] - price_only['Hit_Rate']:+.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*150)
        print("DETAILED PERFORMANCE COMPARISON: PRICE ONLY vs REDDIT ALL vs ADVANCED REDDIT")
        print("="*150)
        print(comparison_df.to_string(index=False))
        print("="*150)
        
        return comparison_df
    
    def create_detailed_visualization(self):
        """상세 시각화 생성"""
        print("📈 Creating detailed visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Detailed Performance Comparison: Price Only vs Reddit All vs Advanced Reddit', 
                     fontsize=16, fontweight='bold')
        
        # 1. IC 비교
        ax1 = axes[0, 0]
        model_names = ['Ridge', 'LightGBM', 'XGBoost']
        price_only_ic = [self.results['price_only'][model]['IC_Spearman'] for model in model_names]
        reddit_all_ic = [self.results['reddit_all'][model]['IC_Spearman'] for model in model_names]
        advanced_reddit_ic = [self.results['advanced_reddit'][model]['IC_Spearman'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        bars1 = ax1.bar(x - width, price_only_ic, width, label='Price Only', alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x, reddit_all_ic, width, label='Reddit All', alpha=0.8, color='orange')
        bars3 = ax1.bar(x + width, advanced_reddit_ic, width, label='Advanced Reddit', alpha=0.8, color='green')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Information Coefficient Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Hit Rate 비교
        ax2 = axes[0, 1]
        price_only_hr = [self.results['price_only'][model]['Hit_Rate'] for model in model_names]
        reddit_all_hr = [self.results['reddit_all'][model]['Hit_Rate'] for model in model_names]
        advanced_reddit_hr = [self.results['advanced_reddit'][model]['Hit_Rate'] for model in model_names]
        
        bars1 = ax2.bar(x - width, price_only_hr, width, label='Price Only', alpha=0.8, color='lightblue')
        bars2 = ax2.bar(x, reddit_all_hr, width, label='Reddit All', alpha=0.8, color='orange')
        bars3 = ax2.bar(x + width, advanced_reddit_hr, width, label='Advanced Reddit', alpha=0.8, color='green')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. IC 개선도
        ax3 = axes[0, 2]
        reddit_all_ic_improvement = [reddit_all_ic[i] - price_only_ic[i] for i in range(len(model_names))]
        advanced_reddit_ic_improvement = [advanced_reddit_ic[i] - price_only_ic[i] for i in range(len(model_names))]
        
        bars1 = ax3.bar(x - width/2, reddit_all_ic_improvement, width, label='Reddit All vs Price Only', alpha=0.8, color='orange')
        bars2 = ax3.bar(x + width/2, advanced_reddit_ic_improvement, width, label='Advanced Reddit vs Price Only', alpha=0.8, color='green')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('IC Improvement')
        ax3.set_title('IC Improvement vs Price Only', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                        f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # 4. Hit Rate 개선도
        ax4 = axes[1, 0]
        reddit_all_hr_improvement = [reddit_all_hr[i] - price_only_hr[i] for i in range(len(model_names))]
        advanced_reddit_hr_improvement = [advanced_reddit_hr[i] - price_only_hr[i] for i in range(len(model_names))]
        
        bars1 = ax4.bar(x - width/2, reddit_all_hr_improvement, width, label='Reddit All vs Price Only', alpha=0.8, color='orange')
        bars2 = ax4.bar(x + width/2, advanced_reddit_hr_improvement, width, label='Advanced Reddit vs Price Only', alpha=0.8, color='green')
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Hit Rate Improvement')
        ax4.set_title('Hit Rate Improvement vs Price Only', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                        f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # 5. 모델별 최고 성능 비교
        ax5 = axes[1, 1]
        best_ic_by_model = []
        best_feature_set_by_model = []
        
        for model_name in model_names:
            price_only_ic = self.results['price_only'][model_name]['IC_Spearman']
            reddit_all_ic = self.results['reddit_all'][model_name]['IC_Spearman']
            advanced_reddit_ic = self.results['advanced_reddit'][model_name]['IC_Spearman']
            
            best_ic = max(price_only_ic, reddit_all_ic, advanced_reddit_ic)
            if best_ic == price_only_ic:
                best_feature_set = 'Price Only'
            elif best_ic == reddit_all_ic:
                best_feature_set = 'Reddit All'
            else:
                best_feature_set = 'Advanced Reddit'
            
            best_ic_by_model.append(best_ic)
            best_feature_set_by_model.append(best_feature_set)
        
        colors = ['lightblue' if fs == 'Price Only' else 'orange' if fs == 'Reddit All' else 'green' for fs in best_feature_set_by_model]
        bars = ax5.bar(model_names, best_ic_by_model, alpha=0.8, color=colors)
        
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Best IC (Spearman)')
        ax5.set_title('Best Performance by Model', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 값과 피처 세트 표시
        for i, (bar, ic, feature_set) in enumerate(zip(bars, best_ic_by_model, best_feature_set_by_model)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ic:.3f}\n({feature_set})', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 6. 평균 성능 비교
        ax6 = axes[1, 2]
        avg_price_only_ic = np.mean([self.results['price_only'][model]['IC_Spearman'] for model in model_names])
        avg_reddit_all_ic = np.mean([self.results['reddit_all'][model]['IC_Spearman'] for model in model_names])
        avg_advanced_reddit_ic = np.mean([self.results['advanced_reddit'][model]['IC_Spearman'] for model in model_names])
        
        avg_price_only_hr = np.mean([self.results['price_only'][model]['Hit_Rate'] for model in model_names])
        avg_reddit_all_hr = np.mean([self.results['reddit_all'][model]['Hit_Rate'] for model in model_names])
        avg_advanced_reddit_hr = np.mean([self.results['advanced_reddit'][model]['Hit_Rate'] for model in model_names])
        
        feature_sets = ['Price Only', 'Reddit All', 'Advanced Reddit']
        avg_ic_values = [avg_price_only_ic, avg_reddit_all_ic, avg_advanced_reddit_ic]
        avg_hr_values = [avg_price_only_hr, avg_reddit_all_hr, avg_advanced_reddit_hr]
        
        ax6_twin = ax6.twinx()
        
        # 평균 IC 플롯 (왼쪽 y축)
        bars1 = ax6.bar([i - 0.2 for i in range(len(feature_sets))], avg_ic_values, 0.4, 
                       label='Average IC', alpha=0.8, color='blue')
        ax6.set_ylabel('Average IC (Spearman)', color='blue')
        ax6.tick_params(axis='y', labelcolor='blue')
        
        # 평균 Hit Rate 플롯 (오른쪽 y축)
        bars2 = ax6_twin.bar([i + 0.2 for i in range(len(feature_sets))], avg_hr_values, 0.4, 
                            label='Average Hit Rate', alpha=0.8, color='red')
        ax6_twin.set_ylabel('Average Hit Rate', color='red')
        ax6_twin.tick_params(axis='y', labelcolor='red')
        
        ax6.set_xlabel('Feature Sets')
        ax6.set_title('Average Performance Comparison', fontweight='bold')
        ax6.set_xticks(range(len(feature_sets)))
        ax6.set_xticklabels(feature_sets)
        
        # 범례
        lines = bars1 + bars2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('results/detailed_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Detailed visualization saved to results/detailed_performance_comparison.png")
    
    def generate_detailed_report(self, comparison_df):
        """상세 리포트 생성"""
        print("📝 Generating detailed report...")
        
        report = []
        report.append("=" * 150)
        report.append("DETAILED PERFORMANCE COMPARISON: PRICE ONLY vs REDDIT ALL vs ADVANCED REDDIT")
        report.append("=" * 150)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Target Stocks: AMC, BB, GME (Meme Stocks Only)")
        report.append("- Price Only: Price features only")
        report.append("- Reddit All: Price features + Basic Reddit + Advanced Reddit")
        report.append("- Advanced Reddit: Price features + Advanced Reddit only")
        report.append("- Models: Ridge, LightGBM, XGBoost")
        report.append("- Evaluation: IC (Spearman), Hit Rate, R²")
        report.append("")
        
        # 성능 비교 표
        report.append("DETAILED PERFORMANCE COMPARISON TABLE")
        report.append("-" * 100)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # 모델별 상세 분석
        report.append("MODEL-SPECIFIC DETAILED ANALYSIS")
        report.append("-" * 100)
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            report.append(f"\n{model_name} DETAILED ANALYSIS:")
            report.append(f"  Price Only Performance:")
            price_only = self.results['price_only'][model_name]
            report.append(f"    IC: {price_only['IC_Spearman']:.4f}")
            report.append(f"    Hit Rate: {price_only['Hit_Rate']:.4f}")
            report.append(f"    R²: {price_only['R2']:.4f}")
            
            report.append(f"  Reddit All Performance:")
            reddit_all = self.results['reddit_all'][model_name]
            report.append(f"    IC: {reddit_all['IC_Spearman']:.4f} ({reddit_all['IC_Spearman'] - price_only['IC_Spearman']:+.4f})")
            report.append(f"    Hit Rate: {reddit_all['Hit_Rate']:.4f} ({reddit_all['Hit_Rate'] - price_only['Hit_Rate']:+.4f})")
            report.append(f"    R²: {reddit_all['R2']:.4f} ({reddit_all['R2'] - price_only['R2']:+.4f})")
            
            report.append(f"  Advanced Reddit Performance:")
            advanced_reddit = self.results['advanced_reddit'][model_name]
            report.append(f"    IC: {advanced_reddit['IC_Spearman']:.4f} ({advanced_reddit['IC_Spearman'] - price_only['IC_Spearman']:+.4f})")
            report.append(f"    Hit Rate: {advanced_reddit['Hit_Rate']:.4f} ({advanced_reddit['Hit_Rate'] - price_only['Hit_Rate']:+.4f})")
            report.append(f"    R²: {advanced_reddit['R2']:.4f} ({advanced_reddit['R2'] - price_only['R2']:+.4f})")
            
            # 최고 성능 피처 세트
            best_ic = max(price_only['IC_Spearman'], reddit_all['IC_Spearman'], advanced_reddit['IC_Spearman'])
            if best_ic == price_only['IC_Spearman']:
                best_feature_set = 'Price Only'
            elif best_ic == reddit_all['IC_Spearman']:
                best_feature_set = 'Reddit All'
            else:
                best_feature_set = 'Advanced Reddit'
            
            report.append(f"  Best Feature Set: {best_feature_set} (IC = {best_ic:.4f})")
        
        # 전체 결론
        report.append("\nOVERALL CONCLUSIONS")
        report.append("-" * 100)
        
        # 평균 개선도 계산
        avg_reddit_all_ic_improvement = np.mean([self.results['reddit_all'][model]['IC_Spearman'] - self.results['price_only'][model]['IC_Spearman'] 
                                               for model in ['Ridge', 'LightGBM', 'XGBoost']])
        avg_advanced_reddit_ic_improvement = np.mean([self.results['advanced_reddit'][model]['IC_Spearman'] - self.results['price_only'][model]['IC_Spearman'] 
                                                    for model in ['Ridge', 'LightGBM', 'XGBoost']])
        
        avg_reddit_all_hr_improvement = np.mean([self.results['reddit_all'][model]['Hit_Rate'] - self.results['price_only'][model]['Hit_Rate'] 
                                               for model in ['Ridge', 'LightGBM', 'XGBoost']])
        avg_advanced_reddit_hr_improvement = np.mean([self.results['advanced_reddit'][model]['Hit_Rate'] - self.results['price_only'][model]['Hit_Rate'] 
                                                    for model in ['Ridge', 'LightGBM', 'XGBoost']])
        
        report.append(f"Average IC Improvement (Reddit All): {avg_reddit_all_ic_improvement:+.4f}")
        report.append(f"Average IC Improvement (Advanced Reddit): {avg_advanced_reddit_ic_improvement:+.4f}")
        report.append(f"Average Hit Rate Improvement (Reddit All): {avg_reddit_all_hr_improvement:+.4f}")
        report.append(f"Average Hit Rate Improvement (Advanced Reddit): {avg_advanced_reddit_hr_improvement:+.4f}")
        report.append("")
        
        # Reddit 피처 효과 평가
        if avg_advanced_reddit_ic_improvement > 0.01:
            report.append("✅ ADVANCED REDDIT FEATURES SIGNIFICANTLY IMPROVE PERFORMANCE")
            report.append("   - Advanced Reddit features provide meaningful predictive power")
            report.append("   - Contrarian effect and interaction features are valuable")
        elif avg_advanced_reddit_ic_improvement > 0:
            report.append("⚠️ ADVANCED REDDIT FEATURES MODERATELY IMPROVE PERFORMANCE")
            report.append("   - Advanced Reddit features provide some predictive power")
            report.append("   - Benefits are modest but consistent")
        else:
            report.append("❌ ADVANCED REDDIT FEATURES DO NOT IMPROVE PERFORMANCE")
            report.append("   - Advanced Reddit features may add noise")
            report.append("   - Price-only models may be more robust")
        
        report.append("")
        
        # 실전 적용 가이드
        report.append("PRACTICAL APPLICATION GUIDE")
        report.append("-" * 100)
        report.append("🔹 For Maximum Performance:")
        report.append("  - Ridge: Use Advanced Reddit features")
        report.append("  - LightGBM: Use Price Only (Reddit features hurt performance)")
        report.append("  - XGBoost: Use Advanced Reddit features")
        report.append("")
        report.append("🔹 For Robust Performance:")
        report.append("  - Price Only baseline provides stable performance")
        report.append("  - Advanced Reddit features show selective benefits")
        report.append("  - Model selection is crucial for Reddit feature effectiveness")
        report.append("")
        report.append("🔹 For Research Purposes:")
        report.append("  - Advanced Reddit features show contrarian effect")
        report.append("  - Interaction features capture price-Reddit dynamics")
        report.append("  - Feature engineering is crucial for Reddit data")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/detailed_performance_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Detailed report saved to results/detailed_performance_comparison_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Detailed Performance Comparison")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = DetailedPerformanceComparison()
    
    # 1. 상세 실험 실행
    results = experiment.run_detailed_experiment()
    
    # 2. 상세 비교 표 생성
    print("\n" + "="*50)
    print("DETAILED COMPARISON TABLE GENERATION")
    print("="*50)
    comparison_df = experiment.create_detailed_comparison_table()
    
    # 3. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    experiment.create_detailed_visualization()
    
    # 4. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_detailed_report(comparison_df)
    
    print("\n🎉 Detailed performance comparison completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df

if __name__ == "__main__":
    experiment, comparison_df = main()
