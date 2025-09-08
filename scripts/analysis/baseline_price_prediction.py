#!/usr/bin/env python3
"""
Baseline Price Prediction Models for Meme Stock Analysis
주가 데이터만 사용한 베이스라인 모델 (Ridge, LightGBM, XGBoost)
논문용 IC 및 Hit Rate 분석 포함
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
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class BaselinePricePredictor:
    """주가 데이터만 사용한 베이스라인 예측 모델"""
    
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
                df['Date'] = pd.to_datetime(df['Date'])
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
    
    def create_technical_features(self, df):
        """기술적 지표 생성"""
        print("🔧 Creating technical features...")
        
        features_df = df.copy()
        
        # Date 컬럼을 datetime으로 변환 (UTC로 변환)
        features_df['Date'] = pd.to_datetime(features_df['Date'], utc=True).dt.tz_localize(None)
        
        # 기본 수익률 계산
        features_df['returns_1d'] = features_df.groupby('ticker')['Close'].pct_change()
        features_df['returns_3d'] = features_df.groupby('ticker')['Close'].pct_change(3)
        features_df['returns_5d'] = features_df.groupby('ticker')['Close'].pct_change(5)
        features_df['returns_10d'] = features_df.groupby('ticker')['Close'].pct_change(10)
        
        # 이동평균
        for window in [5, 10, 20, 50]:
            features_df[f'sma_{window}'] = features_df.groupby('ticker')['Close'].rolling(window).mean().reset_index(0, drop=True)
            features_df[f'price_ratio_sma{window}'] = features_df['Close'] / features_df[f'sma_{window}']
        
        # 볼린저 밴드
        features_df['bb_middle'] = features_df.groupby('ticker')['Close'].rolling(20).mean().reset_index(0, drop=True)
        features_df['bb_std'] = features_df.groupby('ticker')['Close'].rolling(20).std().reset_index(0, drop=True)
        features_df['bb_upper'] = features_df['bb_middle'] + (features_df['bb_std'] * 2)
        features_df['bb_lower'] = features_df['bb_middle'] - (features_df['bb_std'] * 2)
        features_df['bb_position'] = (features_df['Close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # RSI 계산
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        features_df['rsi_14'] = features_df.groupby('ticker')['Close'].apply(calculate_rsi).reset_index(0, drop=True)
        
        # MACD
        features_df['ema_12'] = features_df.groupby('ticker')['Close'].ewm(span=12).mean().reset_index(0, drop=True)
        features_df['ema_26'] = features_df.groupby('ticker')['Close'].ewm(span=26).mean().reset_index(0, drop=True)
        features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
        features_df['macd_signal'] = features_df.groupby('ticker')['macd'].ewm(span=9).mean().reset_index(0, drop=True)
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # 변동성
        for window in [5, 10, 20]:
            features_df[f'volatility_{window}d'] = features_df.groupby('ticker')['returns_1d'].rolling(window).std().reset_index(0, drop=True)
        
        # 거래량 지표
        features_df['volume_sma_10'] = features_df.groupby('ticker')['Volume'].rolling(10).mean().reset_index(0, drop=True)
        features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_sma_10']
        
        # 가격 모멘텀
        for window in [5, 10, 20]:
            features_df[f'momentum_{window}d'] = features_df['Close'] / features_df.groupby('ticker')['Close'].shift(window) - 1
        
        # 고점/저점 대비 위치
        for window in [10, 20]:
            features_df[f'high_{window}d'] = features_df.groupby('ticker')['High'].rolling(window).max().reset_index(0, drop=True)
            features_df[f'low_{window}d'] = features_df.groupby('ticker')['Low'].rolling(window).min().reset_index(0, drop=True)
            features_df[f'position_{window}d'] = (features_df['Close'] - features_df[f'low_{window}d']) / (features_df[f'high_{window}d'] - features_df[f'low_{window}d'])
        
        # 시간적 특성
        features_df['day_of_week'] = features_df['Date'].dt.dayofweek
        features_df['month'] = features_df['Date'].dt.month
        features_df['quarter'] = features_df['Date'].dt.quarter
        features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
        features_df['is_friday'] = (features_df['day_of_week'] == 4).astype(int)
        
        print(f"   ✅ Created {len(features_df.columns)} features")
        return features_df
    
    def prepare_targets(self, df):
        """예측 대상 생성"""
        print("🎯 Preparing prediction targets...")
        
        # 다음날 수익률 (1일 후)
        df['target_1d'] = df.groupby('ticker')['returns_1d'].shift(-1)
        
        # 다음날 방향 (상승/하락)
        df['target_direction_1d'] = (df['target_1d'] > 0).astype(int)
        
        # 3일 후 수익률
        df['target_3d'] = df.groupby('ticker')['returns_1d'].shift(-3)
        
        # 5일 후 수익률  
        df['target_5d'] = df.groupby('ticker')['returns_1d'].shift(-5)
        
        print("   ✅ Created prediction targets")
        return df
    
    def select_features(self, df):
        """특성 선택"""
        print("🔍 Selecting features...")
        
        # 기술적 지표 특성들
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
        print(f"   ✅ Selected {len(available_features)} features")
        
        return df, available_features
    
    def train_test_split(self, df, test_size=0.2):
        """훈련/테스트 분할"""
        print("📊 Splitting train/test data...")
        
        # 날짜 기준으로 분할 (시간 순서 유지)
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        # 각 종목별로 분할
        train_data = []
        test_data = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # 날짜 기준으로 분할
            split_idx = int(len(ticker_data) * (1 - test_size))
            
            train_data.append(ticker_data.iloc[:split_idx])
            test_data.append(ticker_data.iloc[split_idx:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        print(f"   ✅ Train: {len(train_df)} records, Test: {len(test_df)} records")
        
        return train_df, test_df
    
    def train_models(self, train_df, test_df, features):
        """모델 훈련"""
        print("🤖 Training baseline models...")
        
        # 특성과 타겟 준비
        X_train = train_df[features].fillna(0)
        y_train = train_df['target_1d'].fillna(0)
        X_test = test_df[features].fillna(0)
        y_test = test_df['target_1d'].fillna(0)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # 1. Ridge Regression
        print("   📈 Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        self.models['Ridge'] = ridge
        
        # 2. LightGBM
        print("   🌟 Training LightGBM...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        lgb_model = lgb.train(lgb_params, train_data, num_boost_round=1000, 
                             valid_sets=[train_data], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        self.models['LightGBM'] = lgb_model
        
        # 3. XGBoost
        print("   🚀 Training XGBoost...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, 
                     eval_set=[(X_test, y_test)], 
                     verbose=False)
        self.models['XGBoost'] = xgb_model
        
        print("   ✅ All models trained successfully")
        
        return X_train, y_train, X_test, y_test
    
    def calculate_ic(self, y_true, y_pred, method='pearson'):
        """Information Coefficient 계산"""
        # NaN 값 제거
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return np.nan, np.nan
        
        if method == 'pearson':
            corr, p_value = pearsonr(y_true_clean, y_pred_clean)
        elif method == 'spearman':
            corr, p_value = spearmanr(y_true_clean, y_pred_clean)
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        
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
            
            # IC 계산
            ic_pearson, ic_p_pearson = self.calculate_ic(y_test, y_pred, 'pearson')
            ic_spearman, ic_p_spearman = self.calculate_ic(y_test, y_pred, 'spearman')
            
            # Hit Rate 계산
            hit_rate = self.calculate_hit_rate(y_test, y_pred)
            
            results[model_name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'IC_Pearson': ic_pearson,
                'IC_Pearson_p': ic_p_pearson,
                'IC_Spearman': ic_spearman,
                'IC_Spearman_p': ic_p_spearman,
                'Hit_Rate': hit_rate,
                'predictions': y_pred
            }
            
            print(f"      IC (Pearson): {ic_pearson:.4f} (p={ic_p_pearson:.4f})")
            print(f"      IC (Spearman): {ic_spearman:.4f} (p={ic_p_spearman:.4f})")
            print(f"      Hit Rate: {hit_rate:.4f}")
            print(f"      R²: {r2:.4f}")
        
        self.results = results
        return results
    
    def create_visualizations(self, X_test, y_test):
        """결과 시각화"""
        print("📈 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Baseline Price Prediction Models Performance', fontsize=16, fontweight='bold')
        
        # IC 비교
        ax1 = axes[0, 0]
        model_names = list(self.results.keys())
        ic_pearson = [self.results[model]['IC_Pearson'] for model in model_names]
        ic_spearman = [self.results[model]['IC_Spearman'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, ic_pearson, width, label='Pearson IC', alpha=0.8)
        ax1.bar(x + width/2, ic_spearman, width, label='Spearman IC', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Information Coefficient')
        ax1.set_title('IC Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Hit Rate 비교
        ax2 = axes[0, 1]
        hit_rates = [self.results[model]['Hit_Rate'] for model in model_names]
        bars = ax2.bar(model_names, hit_rates, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate Comparison')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, rate in zip(bars, hit_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # R² 비교
        ax3 = axes[0, 2]
        r2_scores = [self.results[model]['R2'] for model in model_names]
        bars = ax3.bar(model_names, r2_scores, alpha=0.8, color=['#d62728', '#9467bd', '#8c564b'])
        ax3.set_xlabel('Models')
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score Comparison')
        ax3.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, score in zip(bars, r2_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 예측 vs 실제 산점도 (각 모델별)
        for i, (model_name, model) in enumerate(self.models.items()):
            ax = axes[1, i]
            
            if model_name == 'Ridge':
                y_pred = model.predict(self.scalers['standard'].transform(X_test))
            else:
                y_pred = model.predict(X_test)
            
            # 산점도
            ax.scatter(y_test, y_pred, alpha=0.6, s=20)
            
            # 대각선 (완벽한 예측)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('Actual Returns')
            ax.set_ylabel('Predicted Returns')
            ax.set_title(f'{model_name} Predictions')
            ax.grid(True, alpha=0.3)
            
            # IC 표시
            ic = self.results[model_name]['IC_Pearson']
            ax.text(0.05, 0.95, f'IC: {ic:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('results/baseline_price_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Visualizations saved to results/baseline_price_prediction_results.png")
    
    def generate_report(self):
        """결과 리포트 생성"""
        print("📝 Generating baseline model report...")
        
        report = []
        report.append("=" * 80)
        report.append("BASELINE PRICE PREDICTION MODELS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("Models: Ridge Regression, LightGBM, XGBoost")
        report.append("Features: Technical indicators only (price-based)")
        report.append("Target: Next-day returns")
        report.append("")
        
        # 모델별 성능 요약
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 50)
        report.append(f"{'Model':<12} {'IC(Pearson)':<12} {'IC(Spearman)':<12} {'Hit Rate':<10} {'R²':<8}")
        report.append("-" * 50)
        
        for model_name, results in self.results.items():
            report.append(f"{model_name:<12} {results['IC_Pearson']:<12.4f} {results['IC_Spearman']:<12.4f} "
                        f"{results['Hit_Rate']:<10.4f} {results['R2']:<8.4f}")
        
        report.append("")
        
        # 상세 결과
        report.append("DETAILED RESULTS")
        report.append("-" * 50)
        
        for model_name, results in self.results.items():
            report.append(f"\n{model_name}:")
            report.append(f"  Information Coefficient (Pearson): {results['IC_Pearson']:.4f} (p={results['IC_Pearson_p']:.4f})")
            report.append(f"  Information Coefficient (Spearman): {results['IC_Spearman']:.4f} (p={results['IC_Spearman_p']:.4f})")
            report.append(f"  Hit Rate: {results['Hit_Rate']:.4f}")
            report.append(f"  R² Score: {results['R2']:.4f}")
            report.append(f"  MSE: {results['MSE']:.6f}")
            report.append(f"  MAE: {results['MAE']:.6f}")
        
        report.append("")
        report.append("INTERPRETATION")
        report.append("-" * 50)
        report.append("• IC (Information Coefficient): Correlation between predictions and actual returns")
        report.append("  - Higher IC indicates better predictive power")
        report.append("  - IC > 0.05 is considered good for stock prediction")
        report.append("• Hit Rate: Percentage of correct direction predictions")
        report.append("  - Hit Rate > 0.5 indicates positive directional accuracy")
        report.append("• R²: Proportion of variance explained by the model")
        report.append("")
        
        # 최고 성능 모델
        best_ic_model = max(self.results.keys(), key=lambda x: abs(self.results[x]['IC_Pearson']))
        best_hit_model = max(self.results.keys(), key=lambda x: self.results[x]['Hit_Rate'])
        
        report.append("BEST PERFORMING MODELS")
        report.append("-" * 50)
        report.append(f"Best IC (Pearson): {best_ic_model} (IC = {self.results[best_ic_model]['IC_Pearson']:.4f})")
        report.append(f"Best Hit Rate: {best_hit_model} (Hit Rate = {self.results[best_hit_model]['Hit_Rate']:.4f})")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/baseline_price_prediction_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Report saved to results/baseline_price_prediction_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Baseline Price Prediction Analysis")
    print("=" * 60)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 베이스라인 예측기 초기화
    predictor = BaselinePricePredictor()
    
    # 1. 데이터 로드
    stock_data = predictor.load_stock_data()
    
    # 2. 특성 생성
    features_df = predictor.create_technical_features(stock_data)
    
    # 3. 타겟 생성
    features_df = predictor.prepare_targets(features_df)
    
    # 4. 특성 선택
    features_df, selected_features = predictor.select_features(features_df)
    
    # 5. 훈련/테스트 분할
    train_df, test_df = predictor.train_test_split(features_df, test_size=0.2)
    
    # 6. 모델 훈련
    X_train, y_train, X_test, y_test = predictor.train_models(train_df, test_df, selected_features)
    
    # 7. 모델 평가
    results = predictor.evaluate_models(X_test, y_test)
    
    # 8. 시각화
    predictor.create_visualizations(X_test, y_test)
    
    # 9. 리포트 생성
    predictor.generate_report()
    
    print("\n🎉 Baseline price prediction analysis completed!")
    print("📁 Results saved in 'results/' directory")

if __name__ == "__main__":
    main()
