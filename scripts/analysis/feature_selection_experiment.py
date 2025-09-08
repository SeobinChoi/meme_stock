#!/usr/bin/env python3
"""
Feature Selection Experiment for Reddit Features
Reddit 피처 셀렉션을 통한 최적화 실험

실험 설계:
1. Forward IC 검사: 월별 Spearman 상관계수 분석
2. Feature Importance 검사: LightGBM/XGBoost 기반 중요도 분석
3. 피처 필터링: ICIR > 0.1, pos_ratio > 0.6 기준
4. 최적화된 모델 실험 및 성능 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class FeatureSelectionExperiment:
    """피처 셀렉션 실험 클래스"""
    
    def __init__(self):
        self.df = None
        self.reddit_features = []
        self.price_features = []
        self.selected_features = []
        self.ic_results = {}
        self.importance_results = {}
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """데이터 로드"""
        print("📊 Loading dataset for feature selection...")
        
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
    
    def identify_features(self):
        """특성 분류"""
        print("🔧 Identifying feature categories...")
        
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
    
    def monthly_ic(self, df, feature, target='target_1d'):
        """월별 IC 계산"""
        try:
            # 월별로 그룹화하여 IC 계산
            monthly_ic = df.groupby(df['date'].dt.to_period('M')).apply(
                lambda g: g[feature].corr(g[target], method='spearman')
            )
            return monthly_ic.dropna()
        except:
            return pd.Series(dtype=float)
    
    def forward_ic_analysis(self):
        """Forward IC 검사"""
        print("🔍 Performing Forward IC Analysis...")
        
        results = []
        
        for col in self.reddit_features:
            print(f"   📊 Analyzing {col}...")
            
            # 월별 IC 계산
            ic_series = self.monthly_ic(self.df, col)
            
            if len(ic_series) > 3:  # 최소 3개월 데이터 필요
                mean_ic = ic_series.mean()
                std_ic = ic_series.std()
                icir = mean_ic / (std_ic + 1e-9)  # IC Information Ratio
                pos_ratio = (ic_series > 0).mean()  # 양수 비율
                
                results.append({
                    'feature': col,
                    'mean_ic': mean_ic,
                    'std_ic': std_ic,
                    'icir': icir,
                    'pos_ratio': pos_ratio,
                    'n_months': len(ic_series)
                })
                
                print(f"      IC: {mean_ic:.4f}, ICIR: {icir:.4f}, Pos Ratio: {pos_ratio:.4f}")
        
        ic_df = pd.DataFrame(results)
        ic_df = ic_df.sort_values('icir', ascending=False)
        
        print(f"\n📋 Forward IC Analysis Results:")
        print(ic_df.to_string(index=False))
        
        self.ic_results = ic_df
        return ic_df
    
    def feature_importance_analysis(self):
        """Feature Importance 분석"""
        print("🔍 Performing Feature Importance Analysis...")
        
        # 데이터 준비
        all_features = self.price_features + self.reddit_features
        X = self.df[all_features].fillna(0)
        y = self.df['target_1d'].fillna(0)
        
        # 종목별 더미 변수 추가
        ticker_dummies = pd.get_dummies(self.df['ticker'], prefix='ticker')
        X = pd.concat([X, ticker_dummies], axis=1)
        
        # LightGBM 모델 훈련
        print("   🌟 Training LightGBM for importance...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X, label=y)
        lgb_model = lgb.train(lgb_params, train_data, num_boost_round=100, 
                             callbacks=[lgb.log_evaluation(0)])
        
        # LightGBM Feature Importance
        lgb_importance = pd.DataFrame({
            'feature': X.columns,
            'lgb_gain': lgb_model.feature_importance(importance_type='gain'),
            'lgb_split': lgb_model.feature_importance(importance_type='split')
        })
        
        # XGBoost 모델 훈련
        print("   🚀 Training XGBoost for importance...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'verbosity': 0
        }
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X, y)
        
        # XGBoost Feature Importance
        xgb_importance = pd.DataFrame({
            'feature': X.columns,
            'xgb_gain': xgb_model.feature_importances_
        })
        
        # 중요도 통합
        importance_df = lgb_importance.merge(xgb_importance, on='feature')
        
        # Reddit 피처만 필터링
        reddit_importance = importance_df[
            importance_df['feature'].isin(self.reddit_features)
        ].copy()
        
        # 정규화된 중요도 계산
        reddit_importance['lgb_gain_norm'] = reddit_importance['lgb_gain'] / reddit_importance['lgb_gain'].sum()
        reddit_importance['xgb_gain_norm'] = reddit_importance['xgb_gain'] / reddit_importance['xgb_gain'].sum()
        reddit_importance['avg_importance'] = (reddit_importance['lgb_gain_norm'] + reddit_importance['xgb_gain_norm']) / 2
        
        reddit_importance = reddit_importance.sort_values('avg_importance', ascending=False)
        
        print(f"\n📋 Feature Importance Analysis Results:")
        print(reddit_importance[['feature', 'lgb_gain_norm', 'xgb_gain_norm', 'avg_importance']].to_string(index=False))
        
        self.importance_results = reddit_importance
        return reddit_importance
    
    def select_features(self, icir_threshold=0.1, pos_ratio_threshold=0.6, importance_threshold=0.01):
        """피처 선택"""
        print("🎯 Selecting optimal features...")
        
        # IC 기준 필터링
        ic_selected = self.ic_results[
            (self.ic_results['icir'] > icir_threshold) & 
            (self.ic_results['pos_ratio'] > pos_ratio_threshold)
        ].copy()
        
        print(f"   📊 IC-based selection: {len(ic_selected)} features")
        print(f"      ICIR > {icir_threshold}, Pos Ratio > {pos_ratio_threshold}")
        
        # Importance 기준 필터링
        importance_selected = self.importance_results[
            self.importance_results['avg_importance'] > importance_threshold
        ].copy()
        
        print(f"   🔍 Importance-based selection: {len(importance_selected)} features")
        print(f"      Avg Importance > {importance_threshold}")
        
        # 두 기준 모두 만족하는 피처
        ic_features = set(ic_selected['feature'].tolist())
        importance_features = set(importance_selected['feature'].tolist())
        
        # 교집합 (두 기준 모두 만족)
        intersection_features = ic_features & importance_features
        
        # 합집합 (둘 중 하나라도 만족)
        union_features = ic_features | importance_features
        
        print(f"   🎯 Intersection (both criteria): {len(intersection_features)} features")
        print(f"   🎯 Union (either criteria): {len(union_features)} features")
        
        # 최종 선택 (교집합 우선, 부족하면 합집합에서 보충)
        if len(intersection_features) >= 5:
            final_features = list(intersection_features)
            selection_method = "intersection"
        else:
            final_features = list(union_features)
            selection_method = "union"
        
        self.selected_features = final_features
        
        print(f"\n✅ Final selected features ({selection_method}):")
        for i, feature in enumerate(final_features, 1):
            print(f"   {i:2d}. {feature}")
        
        return final_features, selection_method
    
    def create_feature_groups(self):
        """피처 그룹화 (PCA/Factor Analysis)"""
        print("🔧 Creating feature groups...")
        
        if len(self.selected_features) < 3:
            print("   ⚠️  Not enough features for grouping")
            return None
        
        # 선택된 Reddit 피처만으로 PCA 수행
        reddit_data = self.df[self.selected_features].fillna(0)
        
        # PCA 적용
        pca = PCA(n_components=min(3, len(self.selected_features)))
        pca_result = pca.fit_transform(reddit_data)
        
        # PCA 결과를 DataFrame으로 변환
        pca_df = pd.DataFrame(pca_result, columns=[f'reddit_factor_{i+1}' for i in range(pca_result.shape[1])])
        pca_df.index = self.df.index
        
        # 설명 분산 비율
        explained_variance = pca.explained_variance_ratio_
        
        print(f"   📊 PCA Results:")
        for i, ratio in enumerate(explained_variance):
            print(f"      Factor {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        
        print(f"   📊 Total explained variance: {explained_variance.sum():.4f} ({explained_variance.sum()*100:.2f}%)")
        
        # 피처별 기여도 분석
        feature_contributions = pd.DataFrame(
            pca.components_.T,
            columns=[f'factor_{i+1}' for i in range(pca_result.shape[1])],
            index=self.selected_features
        )
        
        print(f"\n📋 Feature Contributions to Factors:")
        print(feature_contributions.round(4).to_string())
        
        # 해석 가능한 그룹명 제안
        factor_names = []
        for i in range(pca_result.shape[1]):
            # 각 팩터에 가장 기여도가 높은 피처들 확인
            top_features = feature_contributions[f'factor_{i+1}'].abs().nlargest(3).index.tolist()
            
            # 그룹명 제안
            if any('momentum' in f for f in top_features):
                factor_names.append('Reddit_Momentum')
            elif any('vol' in f for f in top_features):
                factor_names.append('Reddit_Volatility')
            elif any('surprise' in f or 'spike' in f for f in top_features):
                factor_names.append('Reddit_Attention')
            else:
                factor_names.append(f'Reddit_Factor_{i+1}')
        
        print(f"\n🎯 Suggested Factor Names:")
        for i, name in enumerate(factor_names):
            print(f"   Factor {i+1}: {name}")
        
        return pca_df, feature_contributions, factor_names
    
    def train_optimized_models(self, pca_df=None):
        """최적화된 모델 훈련"""
        print("🤖 Training optimized models...")
        
        # 시계열 분할
        train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
        
        train_data = []
        val_data = []
        test_data = []
        
        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            
            n = len(ticker_data)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_data.append(ticker_data.iloc[:train_end])
            val_data.append(ticker_data.iloc[train_end:val_end])
            test_data.append(ticker_data.iloc[val_end:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        # 특성 준비
        base_features = self.price_features.copy()
        
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
        
        # 데이터 타입을 float로 변환 (LightGBM 호환성)
        ticker_dummies_train = ticker_dummies_train.astype(float)
        ticker_dummies_val = ticker_dummies_val.astype(float)
        ticker_dummies_test = ticker_dummies_test.astype(float)
        
        # 특성 선택
        if pca_df is not None:
            # PCA 기반 모델
            pca_features = [f'reddit_factor_{i+1}' for i in range(pca_df.shape[1])]
            selected_reddit_features = pca_features
            model_type = 'pca'
        else:
            # 개별 피처 기반 모델
            selected_reddit_features = self.selected_features
            model_type = 'individual'
        
        # 최종 특성 세트
        final_features = base_features + list(all_ticker_cols) + selected_reddit_features
        
        print(f"   📊 Using {model_type} features: {len(selected_reddit_features)} Reddit features")
        
        # 데이터 준비
        X_train = train_df[base_features].fillna(0)
        X_val = val_df[base_features].fillna(0)
        X_test = test_df[base_features].fillna(0)
        
        # 더미 변수 추가
        X_train = pd.concat([X_train, ticker_dummies_train], axis=1)
        X_val = pd.concat([X_val, ticker_dummies_val], axis=1)
        X_test = pd.concat([X_test, ticker_dummies_test], axis=1)
        
        # Reddit 피처 추가
        if pca_df is not None:
            # PCA 결과 추가 - 인덱스 재설정 후 매칭
            train_df_reset = train_df.reset_index(drop=True)
            val_df_reset = val_df.reset_index(drop=True)
            test_df_reset = test_df.reset_index(drop=True)
            
            # PCA 결과도 인덱스 재설정
            pca_df_reset = pca_df.reset_index(drop=True)
            
            # 각 분할의 길이에 맞게 PCA 결과 슬라이싱
            train_len = len(X_train)
            val_len = len(X_val)
            test_len = len(X_test)
            
            X_train = pd.concat([X_train, pca_df_reset.iloc[:train_len]], axis=1)
            X_val = pd.concat([X_val, pca_df_reset.iloc[train_len:train_len+val_len]], axis=1)
            X_test = pd.concat([X_test, pca_df_reset.iloc[train_len+val_len:train_len+val_len+test_len]], axis=1)
        else:
            # 개별 Reddit 피처 추가
            X_train = pd.concat([X_train, train_df[selected_reddit_features].fillna(0)], axis=1)
            X_val = pd.concat([X_val, val_df[selected_reddit_features].fillna(0)], axis=1)
            X_test = pd.concat([X_test, test_df[selected_reddit_features].fillna(0)], axis=1)
        
        y_train = train_df['target_1d'].fillna(0)
        y_val = val_df['target_1d'].fillna(0)
        y_test = test_df['target_1d'].fillna(0)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[f'optimized_{model_type}'] = scaler
        
        # 모델 훈련
        models = {}
        
        # 1. Ridge Regression
        print(f"   📈 Training Ridge Regression ({model_type})...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        models['Ridge'] = ridge
        
        # 2. LightGBM
        print(f"   🌟 Training LightGBM ({model_type})...")
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
        print(f"   🚀 Training XGBoost ({model_type})...")
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
        
        self.models[f'optimized_{model_type}'] = models
        
        return X_train, y_train, X_val, y_val, X_test, y_test, models
    
    def evaluate_optimized_models(self, X_test, y_test, models, model_type):
        """최적화된 모델 평가"""
        print(f"📊 Evaluating optimized {model_type} models...")
        
        results = {}
        
        for model_name, model in models.items():
            print(f"   🔍 Evaluating {model_name} ({model_type})...")
            
            # 예측
            if model_name == 'Ridge':
                y_pred = model.predict(self.scalers[f'optimized_{model_type}'].transform(X_test))
            else:
                y_pred = model.predict(X_test)
            
            # 기본 메트릭
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Spearman IC 계산
            mask = ~(np.isnan(y_test) | np.isnan(y_pred))
            y_true_clean = y_test[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) > 0:
                ic_spearman, ic_p_spearman = spearmanr(y_true_clean, y_pred_clean)
            else:
                ic_spearman, ic_p_spearman = np.nan, np.nan
            
            # Hit Rate 계산
            true_direction = (y_test > 0).astype(int)
            pred_direction = (y_pred > 0).astype(int)
            hit_rate = (true_direction == pred_direction).mean()
            
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
    
    def create_comparison_table(self, baseline_results, optimized_results):
        """비교 표 생성"""
        print("📋 Creating performance comparison table...")
        
        comparison_data = []
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            baseline = baseline_results[model_name]
            optimized = optimized_results[model_name]
            
            comparison_data.append({
                'Model': model_name,
                'Baseline_IC': f"{baseline['IC_Spearman']:.4f}",
                'Optimized_IC': f"{optimized['IC_Spearman']:.4f}",
                'IC_Improvement': f"{optimized['IC_Spearman'] - baseline['IC_Spearman']:+.4f}",
                'Baseline_Hit_Rate': f"{baseline['Hit_Rate']:.4f}",
                'Optimized_Hit_Rate': f"{optimized['Hit_Rate']:.4f}",
                'Hit_Rate_Improvement': f"{optimized['Hit_Rate'] - baseline['Hit_Rate']:+.4f}",
                'Baseline_R2': f"{baseline['R2']:.4f}",
                'Optimized_R2': f"{optimized['R2']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*120)
        print("FEATURE SELECTION OPTIMIZATION RESULTS")
        print("="*120)
        print(comparison_df.to_string(index=False))
        print("="*120)
        
        return comparison_df
    
    def create_feature_analysis_visualization(self):
        """피처 분석 시각화"""
        print("📈 Creating feature analysis visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Selection Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. IC 분석 결과
        ax1 = axes[0, 0]
        ic_data = self.ic_results.sort_values('icir', ascending=True)
        bars = ax1.barh(range(len(ic_data)), ic_data['icir'], alpha=0.8)
        ax1.set_yticks(range(len(ic_data)))
        ax1.set_yticklabels(ic_data['feature'], fontsize=8)
        ax1.set_xlabel('ICIR (Information Coefficient Information Ratio)')
        ax1.set_title('Reddit Features ICIR Ranking')
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # 2. Feature Importance 결과
        ax2 = axes[0, 1]
        importance_data = self.importance_results.sort_values('avg_importance', ascending=True)
        bars = ax2.barh(range(len(importance_data)), importance_data['avg_importance'], alpha=0.8)
        ax2.set_yticks(range(len(importance_data)))
        ax2.set_yticklabels(importance_data['feature'], fontsize=8)
        ax2.set_xlabel('Average Importance')
        ax2.set_title('Reddit Features Importance Ranking')
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # 3. IC vs Importance 산점도
        ax3 = axes[0, 2]
        merged_data = self.ic_results.merge(self.importance_results, on='feature', how='inner')
        scatter = ax3.scatter(merged_data['icir'], merged_data['avg_importance'], 
                             alpha=0.7, s=100)
        
        # 선택된 피처 하이라이트
        selected_data = merged_data[merged_data['feature'].isin(self.selected_features)]
        ax3.scatter(selected_data['icir'], selected_data['avg_importance'], 
                   color='red', s=150, alpha=0.8, label='Selected')
        
        ax3.set_xlabel('ICIR')
        ax3.set_ylabel('Average Importance')
        ax3.set_title('ICIR vs Importance (Red = Selected)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 피처명 표시
        for _, row in merged_data.iterrows():
            ax3.annotate(row['feature'], (row['icir'], row['avg_importance']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. 월별 IC 시계열 (상위 3개 피처)
        ax4 = axes[1, 0]
        top_features = self.ic_results.head(3)['feature'].tolist()
        
        for feature in top_features:
            ic_series = self.monthly_ic(self.df, feature)
            if len(ic_series) > 0:
                # Period 인덱스를 문자열로 변환
                x_values = [str(period) for period in ic_series.index]
                ax4.plot(x_values, ic_series.values, marker='o', 
                        label=feature, alpha=0.7)
        
        ax4.set_xlabel('Month')
        ax4.set_ylabel('IC (Spearman)')
        ax4.set_title('Monthly IC Time Series (Top 3 Features)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 선택 기준 시각화
        ax5 = axes[1, 1]
        
        # IC 기준
        ic_pass = self.ic_results[
            (self.ic_results['icir'] > 0.1) & 
            (self.ic_results['pos_ratio'] > 0.6)
        ]
        
        # Importance 기준
        importance_pass = self.importance_results[
            self.importance_results['avg_importance'] > 0.01
        ]
        
        # 교집합과 합집합
        ic_features = set(ic_pass['feature'].tolist())
        importance_features = set(importance_pass['feature'].tolist())
        intersection = ic_features & importance_features
        union = ic_features | importance_features
        
        categories = ['IC Pass', 'Importance Pass', 'Intersection', 'Union']
        counts = [len(ic_features), len(importance_features), len(intersection), len(union)]
        
        bars = ax5.bar(categories, counts, alpha=0.8, color=['skyblue', 'lightgreen', 'orange', 'purple'])
        ax5.set_ylabel('Number of Features')
        ax5.set_title('Feature Selection Criteria Results')
        ax5.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, count in zip(bars, counts):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        # 6. 선택된 피처 요약
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = "SELECTED FEATURES SUMMARY\n\n"
        summary_text += f"Total Reddit Features: {len(self.reddit_features)}\n"
        summary_text += f"Selected Features: {len(self.selected_features)}\n"
        summary_text += f"Selection Rate: {len(self.selected_features)/len(self.reddit_features)*100:.1f}%\n\n"
        
        summary_text += "Selected Features:\n"
        for i, feature in enumerate(self.selected_features, 1):
            summary_text += f"{i:2d}. {feature}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('results/feature_selection_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Feature analysis visualization saved to results/feature_selection_analysis.png")
    
    def generate_final_report(self, comparison_df, selection_method):
        """최종 리포트 생성"""
        print("📝 Generating final feature selection report...")
        
        report = []
        report.append("=" * 120)
        report.append("FEATURE SELECTION EXPERIMENT REPORT")
        report.append("=" * 120)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Forward IC Analysis: Monthly Spearman correlation")
        report.append("- Feature Importance Analysis: LightGBM + XGBoost")
        report.append("- Selection Criteria: ICIR > 0.1, Pos Ratio > 0.6, Importance > 0.01")
        report.append(f"- Selection Method: {selection_method}")
        report.append("")
        
        # 피처 선택 결과
        report.append("FEATURE SELECTION RESULTS")
        report.append("-" * 50)
        report.append(f"Total Reddit Features: {len(self.reddit_features)}")
        report.append(f"Selected Features: {len(self.selected_features)}")
        report.append(f"Selection Rate: {len(self.selected_features)/len(self.reddit_features)*100:.1f}%")
        report.append("")
        
        report.append("Selected Features:")
        for i, feature in enumerate(self.selected_features, 1):
            report.append(f"  {i:2d}. {feature}")
        report.append("")
        
        # 성능 비교
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 50)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # 주요 발견
        report.append("KEY FINDINGS")
        report.append("-" * 50)
        
        # 평균 성능 개선
        ic_improvements = []
        hr_improvements = []
        
        for _, row in comparison_df.iterrows():
            ic_imp = float(row['IC_Improvement'])
            hr_imp = float(row['Hit_Rate_Improvement'])
            ic_improvements.append(ic_imp)
            hr_improvements.append(hr_imp)
        
        avg_ic_improvement = np.mean(ic_improvements)
        avg_hr_improvement = np.mean(hr_improvements)
        
        report.append(f"Average IC Improvement: {avg_ic_improvement:+.4f}")
        report.append(f"Average Hit Rate Improvement: {avg_hr_improvement:+.4f}")
        report.append("")
        
        # 최고 성능 모델
        best_model = comparison_df.loc[comparison_df['IC_Improvement'].astype(float).idxmax()]
        report.append(f"Best Improved Model: {best_model['Model']} (IC Improvement: {best_model['IC_Improvement']})")
        report.append("")
        
        # 결론
        report.append("CONCLUSIONS")
        report.append("-" * 50)
        
        if avg_ic_improvement > 0:
            report.append("✅ Feature selection shows positive contribution to prediction performance")
        else:
            report.append("❌ Feature selection shows negative impact on prediction performance")
        
        report.append(f"Selection effectiveness: {'Positive' if avg_ic_improvement > 0 else 'Negative'}")
        report.append("")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/feature_selection_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Report saved to results/feature_selection_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Feature Selection Experiment")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = FeatureSelectionExperiment()
    
    # 1. 데이터 로드
    df = experiment.load_data()
    
    # 2. 특성 분류
    price_features, reddit_features = experiment.identify_features()
    
    # 3. Forward IC 분석
    print("\n" + "="*50)
    print("FORWARD IC ANALYSIS")
    print("="*50)
    ic_results = experiment.forward_ic_analysis()
    
    # 4. Feature Importance 분석
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    importance_results = experiment.feature_importance_analysis()
    
    # 5. 피처 선택
    print("\n" + "="*50)
    print("FEATURE SELECTION")
    print("="*50)
    selected_features, selection_method = experiment.select_features()
    
    # 6. 피처 그룹화 (PCA)
    print("\n" + "="*50)
    print("FEATURE GROUPING (PCA)")
    print("="*50)
    pca_df, feature_contributions, factor_names = experiment.create_feature_groups()
    
    # 7. 최적화된 모델 훈련 (개별 피처)
    print("\n" + "="*50)
    print("OPTIMIZED MODEL TRAINING (Individual Features)")
    print("="*50)
    X_train_i, y_train_i, X_val_i, y_val_i, X_test_i, y_test_i, models_i = experiment.train_optimized_models()
    results_i = experiment.evaluate_optimized_models(X_test_i, y_test_i, models_i, 'individual')
    
    # 8. 최적화된 모델 훈련 (PCA 피처) - 건너뛰기
    print("\n" + "="*50)
    print("OPTIMIZED MODEL TRAINING (PCA Features) - SKIPPED")
    print("="*50)
    print("   ⚠️  PCA feature training skipped due to data alignment issues")
    results_p = None
    
    # 9. 성능 비교 (Baseline vs Optimized)
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    # Baseline 결과 로드 (이전 실험에서)
    baseline_results = {
        'Ridge': {'IC_Spearman': -0.0086, 'Hit_Rate': 0.4977, 'R2': -0.0825},
        'LightGBM': {'IC_Spearman': 0.0202, 'Hit_Rate': 0.4735, 'R2': -0.1872},
        'XGBoost': {'IC_Spearman': 0.0585, 'Hit_Rate': 0.4800, 'R2': -0.5023}
    }
    
    # 개별 피처 결과와 비교
    comparison_i = experiment.create_comparison_table(baseline_results, results_i)
    
    # PCA 피처 결과와 비교 (건너뛰기)
    if results_p is not None:
        comparison_p = experiment.create_comparison_table(baseline_results, results_p)
    else:
        comparison_p = None
    
    # 10. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    experiment.create_feature_analysis_visualization()
    
    # 11. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_final_report(comparison_i, selection_method)
    
    print("\n🎉 Feature selection experiment completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_i, comparison_p

if __name__ == "__main__":
    experiment, comparison_i, comparison_p = main()
