#!/usr/bin/env python3
"""
Contrarian Effect Experiment
Reddit 데이터를 -1로 곱해서 반대로 적용하는 실험

실험 설계:
1. Reddit 피처에 -1을 곱해서 반전
2. 반전된 피처로 모델 훈련
3. 원본 vs 반전 성능 비교
4. Contrarian Effect 검증
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

class ContrarianEffectExperiment:
    """Contrarian Effect 실험 클래스"""
    
    def __init__(self):
        self.df = None
        self.reddit_features = []
        self.price_features = []
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_data(self):
        """데이터 로드"""
        print("📊 Loading dataset for contrarian effect experiment...")
        
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
    
    def create_contrarian_features(self, df):
        """Contrarian 피처 생성 (Reddit 피처에 -1 곱하기)"""
        print("🔄 Creating contrarian features (Reddit features × -1)...")
        
        df_contrarian = df.copy()
        
        # Reddit 피처에 -1 곱하기
        for feature in self.reddit_features:
            if feature in df_contrarian.columns:
                df_contrarian[f'{feature}_contrarian'] = df_contrarian[feature] * -1
                print(f"   📊 Created {feature}_contrarian")
        
        # Contrarian 피처 목록 업데이트
        contrarian_features = [f'{feature}_contrarian' for feature in self.reddit_features 
                             if feature in df_contrarian.columns]
        
        print(f"   ✅ Created {len(contrarian_features)} contrarian features")
        
        return df_contrarian, contrarian_features
    
    def strict_time_series_split(self, df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """엄격한 시계열 분할"""
        print("📊 Performing strict time series split...")
        
        # 각 종목별로 시간 순서대로 분할
        train_data = []
        val_data = []
        test_data = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
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
    
    def prepare_model_data(self, train_df, val_df, test_df, use_contrarian=False):
        """모델 데이터 준비"""
        print(f"🔧 Preparing model data (Contrarian: {use_contrarian})...")
        
        # 특성 선택
        if use_contrarian:
            reddit_features_to_use = [f'{feature}_contrarian' for feature in self.reddit_features 
                                     if f'{feature}_contrarian' in train_df.columns]
            feature_type = "contrarian"
        else:
            reddit_features_to_use = self.reddit_features
            feature_type = "original"
        
        all_features = self.price_features + reddit_features_to_use
        
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
        
        self.scalers[feature_type] = scaler
        
        print(f"   ✅ Using {feature_type} Reddit features: {len(reddit_features_to_use)}")
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
    
    def train_models(self, X_train, y_train, X_val, y_val, model_type='original'):
        """모델 훈련"""
        print(f"🤖 Training {model_type} models...")
        
        models = {}
        
        # 1. Ridge Regression
        print(f"   📈 Training Ridge Regression ({model_type})...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
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
        
        self.models[model_type] = models
        
        print(f"   ✅ All {model_type} models trained successfully")
        
        return models
    
    def evaluate_models(self, X_test, y_test, models, model_type='original'):
        """모델 평가"""
        print(f"📊 Evaluating {model_type} models...")
        
        results = {}
        
        for model_name, model in models.items():
            print(f"   🔍 Evaluating {model_name} ({model_type})...")
            
            # 예측
            if model_name == 'Ridge':
                y_pred = model.predict(self.scalers[model_type].transform(X_test))
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
        
        self.results[model_type] = results
        
        return results
    
    def create_comparison_table(self):
        """성능 비교 표 생성"""
        print("📋 Creating performance comparison table...")
        
        comparison_data = []
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            original = self.results['original'][model_name]
            contrarian = self.results['contrarian'][model_name]
            
            comparison_data.append({
                'Model': model_name,
                'Original_IC': f"{original['IC_Spearman']:.4f}",
                'Contrarian_IC': f"{contrarian['IC_Spearman']:.4f}",
                'IC_Improvement': f"{contrarian['IC_Spearman'] - original['IC_Spearman']:+.4f}",
                'Original_Hit_Rate': f"{original['Hit_Rate']:.4f}",
                'Contrarian_Hit_Rate': f"{contrarian['Hit_Rate']:.4f}",
                'Hit_Rate_Improvement': f"{contrarian['Hit_Rate'] - original['Hit_Rate']:+.4f}",
                'Original_R2': f"{original['R2']:.4f}",
                'Contrarian_R2': f"{contrarian['R2']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*120)
        print("CONTRARIAN EFFECT EXPERIMENT RESULTS")
        print("="*120)
        print(comparison_df.to_string(index=False))
        print("="*120)
        
        return comparison_df
    
    def analyze_contrarian_effect(self):
        """Contrarian Effect 분석"""
        print("🔍 Analyzing contrarian effect...")
        
        # IC 개선 정도 계산
        ic_improvements = []
        hr_improvements = []
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            original_ic = self.results['original'][model_name]['IC_Spearman']
            contrarian_ic = self.results['contrarian'][model_name]['IC_Spearman']
            ic_improvement = contrarian_ic - original_ic
            ic_improvements.append(ic_improvement)
            
            original_hr = self.results['original'][model_name]['Hit_Rate']
            contrarian_hr = self.results['contrarian'][model_name]['Hit_Rate']
            hr_improvement = contrarian_hr - original_hr
            hr_improvements.append(hr_improvement)
        
        avg_ic_improvement = np.mean(ic_improvements)
        avg_hr_improvement = np.mean(hr_improvements)
        
        print(f"   📊 Average IC Improvement: {avg_ic_improvement:+.4f}")
        print(f"   📊 Average Hit Rate Improvement: {avg_hr_improvement:+.4f}")
        
        # Contrarian Effect 검증
        if avg_ic_improvement > 0:
            print("   ✅ Contrarian Effect CONFIRMED: Reddit features × (-1) improves performance")
        else:
            print("   ❌ Contrarian Effect NOT CONFIRMED: Reddit features × (-1) does not improve performance")
        
        return avg_ic_improvement, avg_hr_improvement
    
    def create_contrarian_visualization(self):
        """Contrarian Effect 시각화"""
        print("📈 Creating contrarian effect visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Contrarian Effect Analysis (Reddit Features × -1)', fontsize=16, fontweight='bold')
        
        # 1. IC 비교
        ax1 = axes[0, 0]
        model_names = ['Ridge', 'LightGBM', 'XGBoost']
        original_ic = [self.results['original'][model]['IC_Spearman'] for model in model_names]
        contrarian_ic = [self.results['contrarian'][model]['IC_Spearman'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, original_ic, width, label='Original', alpha=0.8)
        bars2 = ax1.bar(x + width/2, contrarian_ic, width, label='Contrarian', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Information Coefficient Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 2. Hit Rate 비교
        ax2 = axes[0, 1]
        original_hr = [self.results['original'][model]['Hit_Rate'] for model in model_names]
        contrarian_hr = [self.results['contrarian'][model]['Hit_Rate'] for model in model_names]
        
        bars1 = ax2.bar(x - width/2, original_hr, width, label='Original', alpha=0.8)
        bars2 = ax2.bar(x + width/2, contrarian_hr, width, label='Contrarian', alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 3. 성능 개선 정도
        ax3 = axes[0, 2]
        ic_improvement = [contrarian_ic[i] - original_ic[i] for i in range(len(model_names))]
        hr_improvement = [contrarian_hr[i] - original_hr[i] for i in range(len(model_names))]
        
        bars1 = ax3.bar(x - width/2, ic_improvement, width, label='IC Improvement', alpha=0.8)
        bars2 = ax3.bar(x + width/2, hr_improvement, width, label='Hit Rate Improvement', alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Improvement')
        ax3.set_title('Performance Improvement (Contrarian - Original)')
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
                        f'{height:+.3f}', ha='center', va='bottom')
        
        # 4. R² 비교
        ax4 = axes[1, 0]
        original_r2 = [self.results['original'][model]['R2'] for model in model_names]
        contrarian_r2 = [self.results['contrarian'][model]['R2'] for model in model_names]
        
        bars1 = ax4.bar(x - width/2, original_r2, width, label='Original', alpha=0.8)
        bars2 = ax4.bar(x + width/2, contrarian_r2, width, label='Contrarian', alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('R² Score')
        ax4.set_title('R² Score Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 5. MSE 비교
        ax5 = axes[1, 1]
        original_mse = [self.results['original'][model]['MSE'] for model in model_names]
        contrarian_mse = [self.results['contrarian'][model]['MSE'] for model in model_names]
        
        bars1 = ax5.bar(x - width/2, original_mse, width, label='Original', alpha=0.8)
        bars2 = ax5.bar(x + width/2, contrarian_mse, width, label='Contrarian', alpha=0.8)
        
        ax5.set_xlabel('Models')
        ax5.set_ylabel('MSE')
        ax5.set_title('Mean Squared Error Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(model_names)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2, height + 0.0001, 
                        f'{height:.4f}', ha='center', va='bottom')
        
        # 6. Contrarian Effect 요약
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 평균 개선 정도 계산
        avg_ic_improvement = np.mean(ic_improvement)
        avg_hr_improvement = np.mean(hr_improvement)
        
        summary_text = "CONTRARIAN EFFECT SUMMARY\n\n"
        summary_text += f"Average IC Improvement: {avg_ic_improvement:+.4f}\n"
        summary_text += f"Average Hit Rate Improvement: {avg_hr_improvement:+.4f}\n\n"
        
        if avg_ic_improvement > 0:
            summary_text += "✅ CONTRARIAN EFFECT CONFIRMED\n"
            summary_text += "Reddit features × (-1) improves performance\n\n"
        else:
            summary_text += "❌ CONTRARIAN EFFECT NOT CONFIRMED\n"
            summary_text += "Reddit features × (-1) does not improve performance\n\n"
        
        summary_text += "Model Performance:\n"
        for i, model in enumerate(model_names):
            summary_text += f"  {model}: IC {ic_improvement[i]:+.4f}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('results/contrarian_effect_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Contrarian effect visualization saved to results/contrarian_effect_analysis.png")
    
    def generate_final_report(self, comparison_df, avg_ic_improvement, avg_hr_improvement):
        """최종 리포트 생성"""
        print("📝 Generating final contrarian effect report...")
        
        report = []
        report.append("=" * 120)
        report.append("CONTRARIAN EFFECT EXPERIMENT REPORT")
        report.append("=" * 120)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Original: Reddit features as-is")
        report.append("- Contrarian: Reddit features × (-1)")
        report.append("- Models: Ridge, LightGBM, XGBoost")
        report.append("- Evaluation: IC (Spearman), Hit Rate, R²")
        report.append("")
        
        # 성능 비교
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 50)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Contrarian Effect 분석
        report.append("CONTRARIAN EFFECT ANALYSIS")
        report.append("-" * 50)
        report.append(f"Average IC Improvement: {avg_ic_improvement:+.4f}")
        report.append(f"Average Hit Rate Improvement: {avg_hr_improvement:+.4f}")
        report.append("")
        
        # 모델별 개선 정도
        report.append("Model-wise Improvements:")
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            original_ic = self.results['original'][model_name]['IC_Spearman']
            contrarian_ic = self.results['contrarian'][model_name]['IC_Spearman']
            improvement = contrarian_ic - original_ic
            
            report.append(f"  {model_name}: IC {improvement:+.4f}")
        report.append("")
        
        # 결론
        report.append("CONCLUSIONS")
        report.append("-" * 50)
        
        if avg_ic_improvement > 0:
            report.append("✅ CONTRARIAN EFFECT CONFIRMED")
            report.append("Reddit features × (-1) improves prediction performance")
            report.append("This supports the hypothesis that Reddit attention has a contrarian effect")
        else:
            report.append("❌ CONTRARIAN EFFECT NOT CONFIRMED")
            report.append("Reddit features × (-1) does not improve prediction performance")
            report.append("This suggests Reddit attention may not have a strong contrarian effect")
        
        report.append("")
        
        # 최고 성능 모델
        best_original_model = max(self.results['original'].keys(), 
                                key=lambda x: self.results['original'][x]['IC_Spearman'])
        best_contrarian_model = max(self.results['contrarian'].keys(), 
                                  key=lambda x: self.results['contrarian'][x]['IC_Spearman'])
        
        report.append(f"Best Original Model: {best_original_model} (IC = {self.results['original'][best_original_model]['IC_Spearman']:.4f})")
        report.append(f"Best Contrarian Model: {best_contrarian_model} (IC = {self.results['contrarian'][best_contrarian_model]['IC_Spearman']:.4f})")
        report.append("")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/contrarian_effect_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Report saved to results/contrarian_effect_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Contrarian Effect Experiment")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = ContrarianEffectExperiment()
    
    # 1. 데이터 로드
    df = experiment.load_data()
    
    # 2. 특성 준비
    price_features, reddit_features = experiment.prepare_features()
    
    # 3. Contrarian 피처 생성
    print("\n" + "="*50)
    print("CREATING CONTRARIAN FEATURES")
    print("="*50)
    df_contrarian, contrarian_features = experiment.create_contrarian_features(df)
    
    # 4. 시계열 분할
    train_df, val_df, test_df = experiment.strict_time_series_split(df_contrarian)
    
    # 5. 원본 모델 실험
    print("\n" + "="*50)
    print("ORIGINAL MODEL EXPERIMENT")
    print("="*50)
    X_train_o, X_val_o, X_test_o, y_train_o, y_val_o, y_test_o, feature_names_o = experiment.prepare_model_data(
        train_df, val_df, test_df, use_contrarian=False)
    models_o = experiment.train_models(X_train_o, y_train_o, X_val_o, y_val_o, 'original')
    results_o = experiment.evaluate_models(X_test_o, y_test_o, models_o, 'original')
    
    # 6. Contrarian 모델 실험
    print("\n" + "="*50)
    print("CONTRARIAN MODEL EXPERIMENT")
    print("="*50)
    X_train_c, X_val_c, X_test_c, y_train_c, y_val_c, y_test_c, feature_names_c = experiment.prepare_model_data(
        train_df, val_df, test_df, use_contrarian=True)
    models_c = experiment.train_models(X_train_c, y_train_c, X_val_c, y_val_c, 'contrarian')
    results_c = experiment.evaluate_models(X_test_c, y_test_c, models_c, 'contrarian')
    
    # 7. 성능 비교
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    comparison_df = experiment.create_comparison_table()
    
    # 8. Contrarian Effect 분석
    print("\n" + "="*50)
    print("CONTRARIAN EFFECT ANALYSIS")
    print("="*50)
    avg_ic_improvement, avg_hr_improvement = experiment.analyze_contrarian_effect()
    
    # 9. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    experiment.create_contrarian_visualization()
    
    # 10. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_final_report(comparison_df, avg_ic_improvement, avg_hr_improvement)
    
    print("\n🎉 Contrarian effect experiment completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df, avg_ic_improvement, avg_hr_improvement

if __name__ == "__main__":
    experiment, comparison_df, avg_ic_improvement, avg_hr_improvement = main()
