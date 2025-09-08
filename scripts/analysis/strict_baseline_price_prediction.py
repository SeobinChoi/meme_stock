#!/usr/bin/env python3
"""
Strict Baseline Price Prediction Models for Meme Stock Analysis
엄격한 시계열 데이터 처리 조건을 적용한 베이스라인 모델

조건:
1. 시간 순서를 유지한 train/validation/test split
2. 미래 데이터 누수 방지를 위한 마스킹
3. 과거 주가 관련 피처만 사용
4. 기본 하이퍼파라미터 사용
5. Spearman rank correlation 기반 IC 계산
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

class StrictBaselinePricePredictor:
    """엄격한 시계열 조건을 적용한 베이스라인 예측 모델"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        
    def load_stock_data(self):
        """주가 데이터 로드"""
        print("📈 Loading stock price data...")
        
        # GME, AMC, BB 데이터 로드
        stocks = ['GME', 'AMC', 'BB']
        stock_data = {}
        
        for stock in stocks:
            try:
                # 원본 데이터 로드
                df = pd.read_csv(f'data/raw/stocks/{stock}_stock_data.csv')
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
                df = df.sort_values('Date').reset_index(drop=True)
                
                # 종목명 추가
                df['ticker'] = stock
                
                stock_data[stock] = df
                print(f"   ✅ {stock}: {len(df)} records ({df['Date'].min()} ~ {df['Date'].max()})")
                
            except FileNotFoundError:
                print(f"   ⚠️  {stock} data not found, using sample data")
                # 샘플 데이터 사용
                df = pd.read_csv(f'sample_data/{stock}_sample_price_data.csv')
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df['ticker'] = stock
                
                # 컬럼명 통일
                df = df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 
                                      'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                
                stock_data[stock] = df
                print(f"   ✅ {stock} (sample): {len(df)} records")
        
        # 데이터 통합
        all_data = pd.concat(list(stock_data.values()), ignore_index=True)
        all_data = all_data.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        print(f"📊 Total data: {len(all_data)} records")
        return all_data
    
    def create_past_only_features(self, df):
        """과거 주가 관련 피처만 생성 (미래 데이터 누수 방지)"""
        print("🔧 Creating past-only technical features...")
        
        features_df = df.copy()
        
        # 기본 수익률 계산 (과거 기준)
        features_df['returns_1d'] = features_df.groupby('ticker')['Close'].pct_change()
        features_df['returns_3d'] = features_df.groupby('ticker')['Close'].pct_change(3)
        features_df['returns_5d'] = features_df.groupby('ticker')['Close'].pct_change(5)
        features_df['returns_10d'] = features_df.groupby('ticker')['Close'].pct_change(10)
        
        # 이동평균 (과거 데이터만 사용)
        for window in [5, 10, 20, 50]:
            features_df[f'sma_{window}'] = features_df.groupby('ticker')['Close'].rolling(window).mean().reset_index(0, drop=True)
            features_df[f'price_ratio_sma{window}'] = features_df['Close'] / features_df[f'sma_{window}']
        
        # 볼린저 밴드 (과거 데이터만 사용)
        features_df['bb_middle'] = features_df.groupby('ticker')['Close'].rolling(20).mean().reset_index(0, drop=True)
        features_df['bb_std'] = features_df.groupby('ticker')['Close'].rolling(20).std().reset_index(0, drop=True)
        features_df['bb_upper'] = features_df['bb_middle'] + (features_df['bb_std'] * 2)
        features_df['bb_lower'] = features_df['bb_middle'] - (features_df['bb_std'] * 2)
        features_df['bb_position'] = (features_df['Close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # RSI 계산 (과거 데이터만 사용)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        features_df['rsi_14'] = features_df.groupby('ticker')['Close'].apply(calculate_rsi).reset_index(0, drop=True)
        
        # MACD (과거 데이터만 사용)
        features_df['ema_12'] = features_df.groupby('ticker')['Close'].ewm(span=12).mean().reset_index(0, drop=True)
        features_df['ema_26'] = features_df.groupby('ticker')['Close'].ewm(span=26).mean().reset_index(0, drop=True)
        features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
        features_df['macd_signal'] = features_df.groupby('ticker')['macd'].ewm(span=9).mean().reset_index(0, drop=True)
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # 변동성 (과거 데이터만 사용)
        for window in [5, 10, 20]:
            features_df[f'volatility_{window}d'] = features_df.groupby('ticker')['returns_1d'].rolling(window).std().reset_index(0, drop=True)
        
        # 거래량 지표 (과거 데이터만 사용)
        features_df['volume_sma_10'] = features_df.groupby('ticker')['Volume'].rolling(10).mean().reset_index(0, drop=True)
        features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_sma_10']
        
        # 가격 모멘텀 (과거 데이터만 사용)
        for window in [5, 10, 20]:
            features_df[f'momentum_{window}d'] = features_df['Close'] / features_df.groupby('ticker')['Close'].shift(window) - 1
        
        # 고점/저점 대비 위치 (과거 데이터만 사용)
        for window in [10, 20]:
            features_df[f'high_{window}d'] = features_df.groupby('ticker')['High'].rolling(window).max().reset_index(0, drop=True)
            features_df[f'low_{window}d'] = features_df.groupby('ticker')['Low'].rolling(window).min().reset_index(0, drop=True)
            features_df[f'position_{window}d'] = (features_df['Close'] - features_df[f'low_{window}d']) / (features_df[f'high_{window}d'] - features_df[f'low_{window}d'])
        
        # 시간적 특성 (과거 정보만 사용)
        features_df['day_of_week'] = features_df['Date'].dt.dayofweek
        features_df['month'] = features_df['Date'].dt.month
        features_df['quarter'] = features_df['Date'].dt.quarter
        features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
        features_df['is_friday'] = (features_df['day_of_week'] == 4).astype(int)
        
        print(f"   ✅ Created {len(features_df.columns)} past-only features")
        return features_df
    
    def prepare_targets_with_masking(self, df):
        """예측 대상 생성 및 미래 데이터 마스킹"""
        print("🎯 Preparing prediction targets with future data masking...")
        
        # 다음날 수익률 (1일 후)
        df['target_1d'] = df.groupby('ticker')['returns_1d'].shift(-1)
        
        # 다음날 방향 (상승/하락)
        df['target_direction_1d'] = (df['target_1d'] > 0).astype(int)
        
        # 미래 데이터 마스킹: 마지막 N일은 예측 불가능하므로 제외
        # 각 종목별로 마지막 5일은 마스킹 (충분한 과거 데이터 확보를 위해)
        df['mask'] = False
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_indices = df[ticker_mask].index
            if len(ticker_indices) > 5:
                # 마지막 5일 마스킹
                df.loc[ticker_indices[-5:], 'mask'] = True
        
        print("   ✅ Created prediction targets with future data masking")
        return df
    
    def select_past_features(self, df):
        """과거 주가 관련 특성만 선택"""
        print("🔍 Selecting past-only features...")
        
        # 과거 주가 관련 특성들만 선택
        feature_cols = [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
            'price_ratio_sma5', 'price_ratio_sma10', 'price_ratio_sma20', 'price_ratio_sma50',
            'bb_position', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'volume_ratio', 'momentum_5d', 'momentum_10d', 'momentum_20d',
            'position_10d', 'position_20d',
            'day_of_week', 'month', 'quarter', 'is_monday', 'is_friday'
        ]
        
        # 존재하는 특성만 선택
        available_features = [col for col in feature_cols if col in df.columns]
        
        # 종목별 더미 변수
        ticker_dummies = pd.get_dummies(df['ticker'], prefix='ticker')
        df = pd.concat([df, ticker_dummies], axis=1)
        
        # 더미 변수 컬럼 추가
        ticker_cols = [col for col in ticker_dummies.columns]
        available_features.extend(ticker_cols)
        
        self.feature_names = available_features
        print(f"   ✅ Selected {len(available_features)} past-only features")
        
        return df, available_features
    
    def strict_time_series_split(self, df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """엄격한 시계열 분할 (시간 순서 유지)"""
        print("📊 Performing strict time series split...")
        
        # 마스킹된 데이터 제외
        df_clean = df[~df['mask']].copy()
        df_clean = df_clean.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        # 각 종목별로 시간 순서대로 분할
        train_data = []
        val_data = []
        test_data = []
        
        for ticker in df_clean['ticker'].unique():
            ticker_data = df_clean[df_clean['ticker'] == ticker].copy()
            
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
    
    def train_models_with_default_params(self, train_df, val_df, test_df, features):
        """기본 하이퍼파라미터로 모델 훈련"""
        print("🤖 Training models with default hyperparameters...")
        
        # 특성과 타겟 준비
        X_train = train_df[features].fillna(0)
        y_train = train_df['target_1d'].fillna(0)
        X_val = val_df[features].fillna(0)
        y_val = val_df['target_1d'].fillna(0)
        X_test = test_df[features].fillna(0)
        y_test = test_df['target_1d'].fillna(0)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # 1. Ridge Regression (기본 파라미터)
        print("   📈 Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        self.models['Ridge'] = ridge
        
        # 2. LightGBM (기본 파라미터)
        print("   🌟 Training LightGBM...")
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
        self.models['LightGBM'] = lgb_model
        
        # 3. XGBoost (기본 파라미터)
        print("   🚀 Training XGBoost...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'verbosity': 0
        }
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     verbose=False)
        self.models['XGBoost'] = xgb_model
        
        print("   ✅ All models trained successfully")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def calculate_spearman_ic(self, y_true, y_pred):
        """Spearman rank correlation 기반 IC 계산"""
        # NaN 값 제거
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return np.nan, np.nan
        
        corr, p_value = spearmanr(y_true_clean, y_pred_clean)
        return corr, p_value
    
    def calculate_hit_rate(self, y_true, y_pred, threshold=0.0):
        """Hit Rate 계산"""
        # 방향 예측 정확도
        true_direction = (y_true > threshold).astype(int)
        pred_direction = (y_pred > threshold).astype(int)
        
        hit_rate = (true_direction == pred_direction).mean()
        return hit_rate
    
    def evaluate_models(self, X_test, y_test):
        """모델 평가"""
        print("📊 Evaluating models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"   🔍 Evaluating {model_name}...")
            
            # 예측
            if model_name == 'Ridge':
                y_pred = model.predict(self.scalers['standard'].transform(X_test))
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
        
        self.results = results
        return results
    
    def create_performance_table(self):
        """성능 비교 표 생성"""
        print("📋 Creating performance comparison table...")
        
        # 결과 표 생성
        performance_data = []
        for model_name, results in self.results.items():
            performance_data.append({
                'Model': model_name,
                'IC (Spearman)': f"{results['IC_Spearman']:.4f}",
                'IC p-value': f"{results['IC_Spearman_p']:.4f}",
                'Hit Rate': f"{results['Hit_Rate']:.4f}",
                'R²': f"{results['R2']:.4f}",
                'MSE': f"{results['MSE']:.6f}",
                'MAE': f"{results['MAE']:.6f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        print("\n" + "="*80)
        print("BASELINE MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(performance_df.to_string(index=False))
        print("="*80)
        
        return performance_df
    
    def analyze_model_suitability(self):
        """모델 적합성 분석"""
        print("\n🔍 Analyzing model suitability for baseline...")
        
        # IC 기준으로 모델 순위
        ic_ranking = sorted(self.results.items(), key=lambda x: abs(x[1]['IC_Spearman']), reverse=True)
        
        # Hit Rate 기준으로 모델 순위
        hit_rate_ranking = sorted(self.results.items(), key=lambda x: x[1]['Hit_Rate'], reverse=True)
        
        print("\n📊 MODEL RANKING ANALYSIS")
        print("-" * 50)
        
        print("\n🏆 IC (Spearman) Ranking:")
        for i, (model, results) in enumerate(ic_ranking, 1):
            significance = "✅ Significant" if results['IC_Spearman_p'] < 0.05 else "❌ Not Significant"
            print(f"   {i}. {model}: IC = {results['IC_Spearman']:.4f} (p={results['IC_Spearman_p']:.4f}) {significance}")
        
        print("\n🎯 Hit Rate Ranking:")
        for i, (model, results) in enumerate(hit_rate_ranking, 1):
            print(f"   {i}. {model}: Hit Rate = {results['Hit_Rate']:.4f}")
        
        # 베이스라인 적합성 평가
        print("\n💡 BASELINE SUITABILITY ANALYSIS")
        print("-" * 50)
        
        best_ic_model = ic_ranking[0][0]
        best_hit_model = hit_rate_ranking[0][0]
        
        print(f"• Best IC Model: {best_ic_model}")
        print(f"• Best Hit Rate Model: {best_hit_model}")
        
        # 통계적 유의성 확인
        significant_models = [model for model, results in self.results.items() 
                             if results['IC_Spearman_p'] < 0.05]
        
        if significant_models:
            print(f"• Statistically Significant Models: {', '.join(significant_models)}")
        else:
            print("• No models show statistical significance (p < 0.05)")
        
        # 베이스라인 추천
        if best_ic_model == best_hit_model:
            recommended_model = best_ic_model
            reason = "best performance in both IC and Hit Rate"
        else:
            # IC가 더 중요한 지표로 간주
            recommended_model = best_ic_model
            reason = "best IC performance (primary metric for stock prediction)"
        
        print(f"\n🎯 RECOMMENDED BASELINE MODEL: {recommended_model}")
        print(f"   Reason: {reason}")
        
        return {
            'best_ic_model': best_ic_model,
            'best_hit_model': best_hit_model,
            'recommended_model': recommended_model,
            'significant_models': significant_models
        }
    
    def create_visualizations(self, X_test, y_test):
        """결과 시각화"""
        print("📈 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strict Baseline Price Prediction Models Performance', fontsize=16, fontweight='bold')
        
        # IC 비교
        ax1 = axes[0, 0]
        model_names = list(self.results.keys())
        ic_spearman = [self.results[model]['IC_Spearman'] for model in model_names]
        
        bars = ax1.bar(model_names, ic_spearman, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_xlabel('Models')
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Information Coefficient Comparison')
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, ic in zip(bars, ic_spearman):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ic:.3f}', ha='center', va='bottom')
        
        # Hit Rate 비교
        ax2 = axes[0, 1]
        hit_rates = [self.results[model]['Hit_Rate'] for model in model_names]
        bars = ax2.bar(model_names, hit_rates, alpha=0.8, color=['#d62728', '#9467bd', '#8c564b'])
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate Comparison')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, rate in zip(bars, hit_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # 예측 vs 실제 산점도 (LightGBM)
        ax3 = axes[1, 0]
        lgb_model = self.models['LightGBM']
        y_pred_lgb = lgb_model.predict(X_test)
        
        ax3.scatter(y_test, y_pred_lgb, alpha=0.6, s=20, color='#ff7f0e')
        min_val = min(y_test.min(), y_pred_lgb.min())
        max_val = max(y_test.max(), y_pred_lgb.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax3.set_xlabel('Actual Returns')
        ax3.set_ylabel('Predicted Returns')
        ax3.set_title('LightGBM Predictions vs Actual')
        ax3.grid(True, alpha=0.3)
        
        # IC 표시
        ic = self.results['LightGBM']['IC_Spearman']
        ax3.text(0.05, 0.95, f'IC: {ic:.3f}', transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 모델별 성능 요약
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 성능 요약 텍스트
        summary_text = "PERFORMANCE SUMMARY\n\n"
        for model_name, results in self.results.items():
            significance = "✅" if results['IC_Spearman_p'] < 0.05 else "❌"
            summary_text += f"{model_name}:\n"
            summary_text += f"  IC: {results['IC_Spearman']:.4f} {significance}\n"
            summary_text += f"  Hit Rate: {results['Hit_Rate']:.4f}\n\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('results/strict_baseline_price_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Visualizations saved to results/strict_baseline_price_prediction_results.png")

def main():
    """메인 실행 함수"""
    print("🚀 Starting Strict Baseline Price Prediction Analysis")
    print("=" * 70)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 엄격한 베이스라인 예측기 초기화
    predictor = StrictBaselinePricePredictor()
    
    # 1. 데이터 로드
    stock_data = predictor.load_stock_data()
    
    # 2. 과거 전용 특성 생성
    features_df = predictor.create_past_only_features(stock_data)
    
    # 3. 타겟 생성 및 미래 데이터 마스킹
    features_df = predictor.prepare_targets_with_masking(features_df)
    
    # 4. 과거 특성 선택
    features_df, selected_features = predictor.select_past_features(features_df)
    
    # 5. 엄격한 시계열 분할
    train_df, val_df, test_df = predictor.strict_time_series_split(features_df)
    
    # 6. 기본 파라미터로 모델 훈련
    X_train, y_train, X_val, y_val, X_test, y_test = predictor.train_models_with_default_params(
        train_df, val_df, test_df, selected_features)
    
    # 7. 모델 평가
    results = predictor.evaluate_models(X_test, y_test)
    
    # 8. 성능 비교 표 생성
    performance_df = predictor.create_performance_table()
    
    # 9. 모델 적합성 분석
    analysis = predictor.analyze_model_suitability()
    
    # 10. 시각화
    predictor.create_visualizations(X_test, y_test)
    
    print("\n🎉 Strict baseline price prediction analysis completed!")
    print("📁 Results saved in 'results/' directory")
    
    return predictor, results, analysis

if __name__ == "__main__":
    predictor, results, analysis = main()
