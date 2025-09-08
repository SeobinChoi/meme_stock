#!/usr/bin/env python3
"""
Meme Stocks Reddit Feature Test (AMC, BB, GME)
AMC, BB, GME 종목에 대한 Reddit 피처 테스트
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

class MemeStocksRedditTest:
    """밈스톡 Reddit 피처 테스트 클래스"""
    
    def __init__(self):
        self.df = None
        self.reddit_features = []
        self.price_features = []
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_data(self):
        """데이터 로드 (AMC, BB, GME만)"""
        print("📊 Loading meme stocks data (AMC, BB, GME)...")
        
        # 통합 데이터셋 로드
        train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
        val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
        test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        # 데이터 통합
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # AMC, BB, GME만 필터링
        meme_stocks = ['AMC', 'BB', 'GME']
        df = df[df['ticker'].isin(meme_stocks)].copy()
        
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
        
        # 종목별 데이터 수
        for ticker in meme_stocks:
            count = len(df[df['ticker'] == ticker])
            print(f"   📊 {ticker}: {count} records")
        
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
    
    def prepare_model_data(self, train_df, val_df, test_df, use_reddit=True):
        """모델 데이터 준비"""
        feature_type = "with_reddit" if use_reddit else "price_only"
        print(f"🔧 Preparing model data ({feature_type})...")
        
        # 특성 선택
        if use_reddit:
            all_features = self.price_features + self.reddit_features
        else:
            all_features = self.price_features
        
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
        
        print(f"   ✅ Using {feature_type}: {len(all_features)} features")
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
    
    def train_models(self, X_train, y_train, X_val, y_val, model_type='price_only'):
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
    
    def evaluate_models(self, X_test, y_test, models, model_type='price_only'):
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
            price_only = self.results['price_only'][model_name]
            with_reddit = self.results['with_reddit'][model_name]
            
            comparison_data.append({
                'Model': model_name,
                'Price_Only_IC': f"{price_only['IC_Spearman']:.4f}",
                'With_Reddit_IC': f"{with_reddit['IC_Spearman']:.4f}",
                'IC_Improvement': f"{with_reddit['IC_Spearman'] - price_only['IC_Spearman']:+.4f}",
                'Price_Only_Hit_Rate': f"{price_only['Hit_Rate']:.4f}",
                'With_Reddit_Hit_Rate': f"{with_reddit['Hit_Rate']:.4f}",
                'Hit_Rate_Improvement': f"{with_reddit['Hit_Rate'] - price_only['Hit_Rate']:+.4f}",
                'Price_Only_R2': f"{price_only['R2']:.4f}",
                'With_Reddit_R2': f"{with_reddit['R2']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*120)
        print("MEME STOCKS (AMC, BB, GME) REDDIT FEATURE TEST RESULTS")
        print("="*120)
        print(comparison_df.to_string(index=False))
        print("="*120)
        
        return comparison_df
    
    def analyze_reddit_effect(self):
        """Reddit 피처 효과 분석"""
        print("🔍 Analyzing Reddit feature effect...")
        
        # IC 개선 정도 계산
        ic_improvements = []
        hr_improvements = []
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            price_only_ic = self.results['price_only'][model_name]['IC_Spearman']
            with_reddit_ic = self.results['with_reddit'][model_name]['IC_Spearman']
            ic_improvement = with_reddit_ic - price_only_ic
            ic_improvements.append(ic_improvement)
            
            price_only_hr = self.results['price_only'][model_name]['Hit_Rate']
            with_reddit_hr = self.results['with_reddit'][model_name]['Hit_Rate']
            hr_improvement = with_reddit_hr - price_only_hr
            hr_improvements.append(hr_improvement)
        
        avg_ic_improvement = np.mean(ic_improvements)
        avg_hr_improvement = np.mean(hr_improvements)
        
        print(f"   📊 Average IC Improvement: {avg_ic_improvement:+.4f}")
        print(f"   📊 Average Hit Rate Improvement: {avg_hr_improvement:+.4f}")
        
        # Reddit 피처 효과 검증
        if avg_ic_improvement > 0:
            print("   ✅ Reddit features IMPROVE performance")
        else:
            print("   ❌ Reddit features HURT performance")
        
        return avg_ic_improvement, avg_hr_improvement
    
    def create_meme_stocks_visualization(self):
        """밈스톡 시각화"""
        print("📈 Creating meme stocks visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Meme Stocks (AMC, BB, GME) Reddit Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. IC 비교
        ax1 = axes[0, 0]
        model_names = ['Ridge', 'LightGBM', 'XGBoost']
        price_only_ic = [self.results['price_only'][model]['IC_Spearman'] for model in model_names]
        with_reddit_ic = [self.results['with_reddit'][model]['IC_Spearman'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, price_only_ic, width, label='Price Only', alpha=0.8)
        bars2 = ax1.bar(x + width/2, with_reddit_ic, width, label='With Reddit', alpha=0.8)
        
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
        price_only_hr = [self.results['price_only'][model]['Hit_Rate'] for model in model_names]
        with_reddit_hr = [self.results['with_reddit'][model]['Hit_Rate'] for model in model_names]
        
        bars1 = ax2.bar(x - width/2, price_only_hr, width, label='Price Only', alpha=0.8)
        bars2 = ax2.bar(x + width/2, with_reddit_hr, width, label='With Reddit', alpha=0.8)
        
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
        ic_improvement = [with_reddit_ic[i] - price_only_ic[i] for i in range(len(model_names))]
        hr_improvement = [with_reddit_hr[i] - price_only_hr[i] for i in range(len(model_names))]
        
        bars1 = ax3.bar(x - width/2, ic_improvement, width, label='IC Improvement', alpha=0.8)
        bars2 = ax3.bar(x + width/2, hr_improvement, width, label='Hit Rate Improvement', alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Improvement')
        ax3.set_title('Performance Improvement (With Reddit - Price Only)')
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
        
        # 4-6. 종목별 Reddit 관심도 시계열
        tickers = ['AMC', 'BB', 'GME']
        for i, ticker in enumerate(tickers):
            ax = axes[1, i]
            
            # 해당 종목 데이터 필터링
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # Reddit 관심도와 수익률 플롯
            ax2 = ax.twinx()
            
            # 수익률 플롯 (왼쪽 y축)
            line1 = ax.plot(ticker_data['date'], ticker_data['returns_1d'], 
                           color='blue', linewidth=1, label='Daily Returns', alpha=0.7)
            ax.set_ylabel('Daily Returns', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            
            # Reddit 관심도 플롯 (오른쪽 y축)
            line2 = ax2.plot(ticker_data['date'], ticker_data['log_mentions'], 
                            color='red', linewidth=1, label='Reddit Interest', alpha=0.7)
            ax2.set_ylabel('Log Mentions', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title(f'{ticker} - Returns vs Reddit Interest', fontweight='bold')
            ax.set_xlabel('Date')
            
            # 범례
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            # 격자
            ax.grid(True, alpha=0.3)
            
            # x축 날짜 회전
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/meme_stocks_reddit_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Meme stocks visualization saved to results/meme_stocks_reddit_analysis.png")
    
    def generate_final_report(self, comparison_df, avg_ic_improvement, avg_hr_improvement):
        """최종 리포트 생성"""
        print("📝 Generating final meme stocks report...")
        
        report = []
        report.append("=" * 120)
        report.append("MEME STOCKS (AMC, BB, GME) REDDIT FEATURE TEST REPORT")
        report.append("=" * 120)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Target Stocks: AMC, BB, GME (Meme Stocks)")
        report.append("- Price Only: Price features only")
        report.append("- With Reddit: Price features + Reddit features")
        report.append("- Models: Ridge, LightGBM, XGBoost")
        report.append("- Evaluation: IC (Spearman), Hit Rate, R²")
        report.append("")
        
        # 성능 비교
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 50)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Reddit 피처 효과 분석
        report.append("REDDIT FEATURE EFFECT ANALYSIS")
        report.append("-" * 50)
        report.append(f"Average IC Improvement: {avg_ic_improvement:+.4f}")
        report.append(f"Average Hit Rate Improvement: {avg_hr_improvement:+.4f}")
        report.append("")
        
        # 모델별 개선 정도
        report.append("Model-wise Improvements:")
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            price_only_ic = self.results['price_only'][model_name]['IC_Spearman']
            with_reddit_ic = self.results['with_reddit'][model_name]['IC_Spearman']
            improvement = with_reddit_ic - price_only_ic
            
            report.append(f"  {model_name}: IC {improvement:+.4f}")
        report.append("")
        
        # 결론
        report.append("CONCLUSIONS")
        report.append("-" * 50)
        
        if avg_ic_improvement > 0:
            report.append("✅ REDDIT FEATURES IMPROVE PERFORMANCE")
            report.append("Reddit features enhance prediction performance for meme stocks")
            report.append("Social media sentiment is valuable for meme stock prediction")
        else:
            report.append("❌ REDDIT FEATURES HURT PERFORMANCE")
            report.append("Reddit features degrade prediction performance for meme stocks")
            report.append("Social media sentiment may be noise for meme stock prediction")
        
        report.append("")
        
        # 최고 성능 모델
        best_price_model = max(self.results['price_only'].keys(), 
                             key=lambda x: self.results['price_only'][x]['IC_Spearman'])
        best_reddit_model = max(self.results['with_reddit'].keys(), 
                              key=lambda x: self.results['with_reddit'][x]['IC_Spearman'])
        
        report.append(f"Best Price Only Model: {best_price_model} (IC = {self.results['price_only'][best_price_model]['IC_Spearman']:.4f})")
        report.append(f"Best With Reddit Model: {best_reddit_model} (IC = {self.results['with_reddit'][best_reddit_model]['IC_Spearman']:.4f})")
        report.append("")
        
        # 종목별 특성
        report.append("STOCK-SPECIFIC INSIGHTS")
        report.append("-" * 50)
        
        for ticker in ['AMC', 'BB', 'GME']:
            ticker_data = self.df[self.df['ticker'] == ticker]
            avg_mentions = ticker_data['log_mentions'].mean()
            avg_returns = ticker_data['returns_1d'].mean()
            correlation = ticker_data['log_mentions'].corr(ticker_data['returns_1d'])
            
            report.append(f"{ticker}:")
            report.append(f"  Average Log Mentions: {avg_mentions:.4f}")
            report.append(f"  Average Daily Returns: {avg_returns:.4f}")
            report.append(f"  Reddit-Return Correlation: {correlation:.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/meme_stocks_reddit_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Report saved to results/meme_stocks_reddit_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Meme Stocks Reddit Feature Test")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = MemeStocksRedditTest()
    
    # 1. 데이터 로드
    df = experiment.load_data()
    
    # 2. 특성 준비
    price_features, reddit_features = experiment.prepare_features()
    
    # 3. 시계열 분할
    train_df, val_df, test_df = experiment.strict_time_series_split()
    
    # 4. 주가만 사용한 모델 실험
    print("\n" + "="*50)
    print("PRICE ONLY MODEL EXPERIMENT")
    print("="*50)
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, feature_names_p = experiment.prepare_model_data(
        train_df, val_df, test_df, use_reddit=False)
    models_p = experiment.train_models(X_train_p, y_train_p, X_val_p, y_val_p, 'price_only')
    results_p = experiment.evaluate_models(X_test_p, y_test_p, models_p, 'price_only')
    
    # 5. 주가 + Reddit 피처 모델 실험
    print("\n" + "="*50)
    print("PRICE + REDDIT MODEL EXPERIMENT")
    print("="*50)
    X_train_r, X_val_r, X_test_r, y_train_r, y_val_r, y_test_r, feature_names_r = experiment.prepare_model_data(
        train_df, val_df, test_df, use_reddit=True)
    models_r = experiment.train_models(X_train_r, y_train_r, X_val_r, y_val_r, 'with_reddit')
    results_r = experiment.evaluate_models(X_test_r, y_test_r, models_r, 'with_reddit')
    
    # 6. 성능 비교
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    comparison_df = experiment.create_comparison_table()
    
    # 7. Reddit 피처 효과 분석
    print("\n" + "="*50)
    print("REDDIT FEATURE EFFECT ANALYSIS")
    print("="*50)
    avg_ic_improvement, avg_hr_improvement = experiment.analyze_reddit_effect()
    
    # 8. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    experiment.create_meme_stocks_visualization()
    
    # 9. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_final_report(comparison_df, avg_ic_improvement, avg_hr_improvement)
    
    print("\n🎉 Meme stocks Reddit feature test completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df, avg_ic_improvement, avg_hr_improvement

if __name__ == "__main__":
    experiment, comparison_df, avg_ic_improvement, avg_hr_improvement = main()
