#!/usr/bin/env python3
"""
Comprehensive Performance Metrics Analysis
6가지 핵심 지표로 상세 성능 비교: IC, Hit Rate, ICIR, Sharpe Ratio, MDD, Quintile Spread Return
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ComprehensivePerformanceMetrics:
    """6가지 핵심 지표로 종합 성능 분석 클래스"""
    
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
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, predictions_df=None):
        """6가지 핵심 지표 계산"""
        metrics = {}
        
        # 1. IC (Information Coefficient) - Spearman 상관계수
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) > 0:
            ic_spearman, ic_p = spearmanr(y_true_clean, y_pred_clean)
            metrics['IC'] = ic_spearman
            metrics['IC_p_value'] = ic_p
        else:
            metrics['IC'] = np.nan
            metrics['IC_p_value'] = np.nan
        
        # 2. Hit Rate (방향성 예측 정확도)
        true_direction = (y_true > 0).astype(int)
        pred_direction = (y_pred > 0).astype(int)
        hit_rate = (true_direction == pred_direction).mean()
        metrics['Hit_Rate'] = hit_rate
        
        # 3. ICIR (Information Coefficient Information Ratio) - 안정성
        if predictions_df is not None and 'date' in predictions_df.columns:
            # 월별 IC 계산
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            monthly_ic = []
            
            for year_month in predictions_df['date'].dt.to_period('M').unique():
                month_data = predictions_df[predictions_df['date'].dt.to_period('M') == year_month]
                if len(month_data) > 10:  # 최소 10개 데이터 포인트
                    month_ic, _ = spearmanr(month_data['y_true'], month_data['y_pred'])
                    if not np.isnan(month_ic):
                        monthly_ic.append(month_ic)
            
            if len(monthly_ic) > 1:
                ic_mean = np.mean(monthly_ic)
                ic_std = np.std(monthly_ic)
                icir = ic_mean / (ic_std + 1e-9)
                metrics['ICIR'] = icir
            else:
                metrics['ICIR'] = np.nan
        else:
            metrics['ICIR'] = np.nan
        
        # 4. Sharpe Ratio (투자 관점)
        if predictions_df is not None and 'date' in predictions_df.columns:
            # 예측 기반 포트폴리오 수익률 계산
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            predictions_df = predictions_df.sort_values('date')
            
            # 예측값을 기반으로 포지션 결정 (상위 20% 매수, 하위 20% 매도)
            predictions_df['quintile'] = predictions_df.groupby('date')['y_pred'].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
            )
            
            # 포트폴리오 수익률 계산
            portfolio_returns = []
            for date in predictions_df['date'].unique():
                date_data = predictions_df[predictions_df['date'] == date]
                if len(date_data) >= 5:  # 최소 5개 종목
                    # 상위 quintile 매수, 하위 quintile 매도
                    long_positions = date_data[date_data['quintile'] == 4]['y_true'].mean()
                    short_positions = date_data[date_data['quintile'] == 0]['y_true'].mean()
                    portfolio_return = (long_positions - short_positions) / 2  # 롱숏 포트폴리오
                    portfolio_returns.append(portfolio_return)
            
            if len(portfolio_returns) > 1:
                portfolio_returns = np.array(portfolio_returns)
                sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-9) * np.sqrt(252)
                metrics['Sharpe_Ratio'] = sharpe_ratio
            else:
                metrics['Sharpe_Ratio'] = np.nan
        else:
            metrics['Sharpe_Ratio'] = np.nan
        
        # 5. MDD (Maximum Drawdown) - 투자 관점
        if predictions_df is not None and 'date' in predictions_df.columns:
            # 누적 수익률 계산
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            predictions_df = predictions_df.sort_values('date')
            
            # 예측값을 기반으로 포지션 결정
            predictions_df['quintile'] = predictions_df.groupby('date')['y_pred'].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
            )
            
            # 포트폴리오 수익률 계산
            portfolio_returns = []
            for date in predictions_df['date'].unique():
                date_data = predictions_df[predictions_df['date'] == date]
                if len(date_data) >= 5:
                    long_positions = date_data[date_data['quintile'] == 4]['y_true'].mean()
                    short_positions = date_data[date_data['quintile'] == 0]['y_true'].mean()
                    portfolio_return = (long_positions - short_positions) / 2
                    portfolio_returns.append(portfolio_return)
            
            if len(portfolio_returns) > 1:
                portfolio_returns = np.array(portfolio_returns)
                cumulative_returns = np.cumprod(1 + portfolio_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                mdd = np.min(drawdown)
                metrics['MDD'] = mdd
            else:
                metrics['MDD'] = np.nan
        else:
            metrics['MDD'] = np.nan
        
        # 6. Quintile Spread Return (팩터 검증 관점)
        if predictions_df is not None and 'date' in predictions_df.columns:
            # Quintile별 수익률 계산
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            predictions_df = predictions_df.sort_values('date')
            
            # 예측값을 기반으로 quintile 분류
            predictions_df['quintile'] = predictions_df.groupby('date')['y_pred'].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
            )
            
            # Quintile별 평균 수익률 계산
            quintile_returns = []
            for quintile in range(5):
                quintile_data = predictions_df[predictions_df['quintile'] == quintile]
                if len(quintile_data) > 0:
                    quintile_return = quintile_data['y_true'].mean()
                    quintile_returns.append(quintile_return)
                else:
                    quintile_returns.append(0)
            
            if len(quintile_returns) == 5:
                # Q5 (상위) - Q1 (하위) 스프레드
                quintile_spread = quintile_returns[4] - quintile_returns[0]
                metrics['Quintile_Spread_Return'] = quintile_spread
            else:
                metrics['Quintile_Spread_Return'] = np.nan
        else:
            metrics['Quintile_Spread_Return'] = np.nan
        
        return metrics
    
    def evaluate_models(self, X_test, y_test, models, feature_set, scaler=None, test_df=None):
        """모델 평가 (6가지 지표)"""
        print(f"📊 Evaluating {feature_set} models with comprehensive metrics...")
        
        results = {}
        
        for model_name, model in models.items():
            print(f"   🔍 Evaluating {model_name} ({feature_set})...")
            
            # 예측
            if model_name == 'Ridge':
                y_pred = model.predict(scaler.transform(X_test))
            else:
                y_pred = model.predict(X_test)
            
            # 예측 결과를 DataFrame으로 변환 (지표 계산용)
            predictions_df = None
            if test_df is not None:
                predictions_df = test_df.copy()
                predictions_df['y_true'] = y_test
                predictions_df['y_pred'] = y_pred
            
            # 6가지 지표 계산
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, predictions_df)
            
            results[model_name] = metrics
            
            print(f"      IC: {metrics['IC']:.4f}")
            print(f"      Hit Rate: {metrics['Hit_Rate']:.4f}")
            print(f"      ICIR: {metrics['ICIR']:.4f}")
            print(f"      Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")
            print(f"      MDD: {metrics['MDD']:.4f}")
            print(f"      Quintile Spread: {metrics['Quintile_Spread_Return']:.4f}")
        
        return results
    
    def run_comprehensive_experiment(self):
        """종합 실험 실행"""
        print("🚀 Starting Comprehensive Performance Metrics Experiment")
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
            
            # 모델 평가 (6가지 지표)
            feature_results = self.evaluate_models(X_test, y_test, models, feature_set, scaler, test_df)
            
            all_results[feature_set] = feature_results
        
        self.results = all_results
        return all_results
    
    def create_comprehensive_comparison_table(self):
        """종합 비교 표 생성"""
        print("📋 Creating comprehensive comparison table...")
        
        comparison_data = []
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            price_only = self.results['price_only'][model_name]
            reddit_all = self.results['reddit_all'][model_name]
            advanced_reddit = self.results['advanced_reddit'][model_name]
            
            comparison_data.append({
                'Model': model_name,
                'Price_Only_IC': f"{price_only['IC']:.4f}",
                'Reddit_All_IC': f"{reddit_all['IC']:.4f}",
                'Advanced_Reddit_IC': f"{advanced_reddit['IC']:.4f}",
                'Price_Only_Hit_Rate': f"{price_only['Hit_Rate']:.4f}",
                'Reddit_All_Hit_Rate': f"{reddit_all['Hit_Rate']:.4f}",
                'Advanced_Reddit_Hit_Rate': f"{advanced_reddit['Hit_Rate']:.4f}",
                'Price_Only_ICIR': f"{price_only['ICIR']:.4f}",
                'Reddit_All_ICIR': f"{reddit_all['ICIR']:.4f}",
                'Advanced_Reddit_ICIR': f"{advanced_reddit['ICIR']:.4f}",
                'Price_Only_Sharpe': f"{price_only['Sharpe_Ratio']:.4f}",
                'Reddit_All_Sharpe': f"{reddit_all['Sharpe_Ratio']:.4f}",
                'Advanced_Reddit_Sharpe': f"{advanced_reddit['Sharpe_Ratio']:.4f}",
                'Price_Only_MDD': f"{price_only['MDD']:.4f}",
                'Reddit_All_MDD': f"{reddit_all['MDD']:.4f}",
                'Advanced_Reddit_MDD': f"{advanced_reddit['MDD']:.4f}",
                'Price_Only_Quintile_Spread': f"{price_only['Quintile_Spread_Return']:.4f}",
                'Reddit_All_Quintile_Spread': f"{reddit_all['Quintile_Spread_Return']:.4f}",
                'Advanced_Reddit_Quintile_Spread': f"{advanced_reddit['Quintile_Spread_Return']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*200)
        print("COMPREHENSIVE PERFORMANCE METRICS: 6 KEY INDICATORS")
        print("="*200)
        print(comparison_df.to_string(index=False))
        print("="*200)
        
        return comparison_df
    
    def create_comprehensive_visualization(self):
        """종합 시각화 생성"""
        print("📈 Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Comprehensive Performance Metrics: 6 Key Indicators', 
                     fontsize=16, fontweight='bold')
        
        model_names = ['Ridge', 'LightGBM', 'XGBoost']
        
        # 1. IC 비교
        ax1 = axes[0, 0]
        price_only_ic = [self.results['price_only'][model]['IC'] for model in model_names]
        reddit_all_ic = [self.results['reddit_all'][model]['IC'] for model in model_names]
        advanced_reddit_ic = [self.results['advanced_reddit'][model]['IC'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        bars1 = ax1.bar(x - width, price_only_ic, width, label='Price Only', alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x, reddit_all_ic, width, label='Reddit All', alpha=0.8, color='orange')
        bars3 = ax1.bar(x + width, advanced_reddit_ic, width, label='Advanced Reddit', alpha=0.8, color='green')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Information Coefficient', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
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
        ax2.set_title('Hit Rate (Directional Accuracy)', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ICIR 비교 (안정성)
        ax3 = axes[1, 0]
        price_only_icir = [self.results['price_only'][model]['ICIR'] for model in model_names]
        reddit_all_icir = [self.results['reddit_all'][model]['ICIR'] for model in model_names]
        advanced_reddit_icir = [self.results['advanced_reddit'][model]['ICIR'] for model in model_names]
        
        bars1 = ax3.bar(x - width, price_only_icir, width, label='Price Only', alpha=0.8, color='lightblue')
        bars2 = ax3.bar(x, reddit_all_icir, width, label='Reddit All', alpha=0.8, color='orange')
        bars3 = ax3.bar(x + width, advanced_reddit_icir, width, label='Advanced Reddit', alpha=0.8, color='green')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('ICIR')
        ax3.set_title('ICIR (Stability)', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sharpe Ratio 비교 (투자 관점)
        ax4 = axes[1, 1]
        price_only_sharpe = [self.results['price_only'][model]['Sharpe_Ratio'] for model in model_names]
        reddit_all_sharpe = [self.results['reddit_all'][model]['Sharpe_Ratio'] for model in model_names]
        advanced_reddit_sharpe = [self.results['advanced_reddit'][model]['Sharpe_Ratio'] for model in model_names]
        
        bars1 = ax4.bar(x - width, price_only_sharpe, width, label='Price Only', alpha=0.8, color='lightblue')
        bars2 = ax4.bar(x, reddit_all_sharpe, width, label='Reddit All', alpha=0.8, color='orange')
        bars3 = ax4.bar(x + width, advanced_reddit_sharpe, width, label='Advanced Reddit', alpha=0.8, color='green')
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Sharpe Ratio (Investment Perspective)', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. MDD 비교 (투자 관점)
        ax5 = axes[2, 0]
        price_only_mdd = [self.results['price_only'][model]['MDD'] for model in model_names]
        reddit_all_mdd = [self.results['reddit_all'][model]['MDD'] for model in model_names]
        advanced_reddit_mdd = [self.results['advanced_reddit'][model]['MDD'] for model in model_names]
        
        bars1 = ax5.bar(x - width, price_only_mdd, width, label='Price Only', alpha=0.8, color='lightblue')
        bars2 = ax5.bar(x, reddit_all_mdd, width, label='Reddit All', alpha=0.8, color='orange')
        bars3 = ax5.bar(x + width, advanced_reddit_mdd, width, label='Advanced Reddit', alpha=0.8, color='green')
        
        ax5.set_xlabel('Models')
        ax5.set_ylabel('MDD (Maximum Drawdown)')
        ax5.set_title('MDD (Investment Perspective)', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(model_names)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Quintile Spread Return 비교 (팩터 검증 관점)
        ax6 = axes[2, 1]
        price_only_quintile = [self.results['price_only'][model]['Quintile_Spread_Return'] for model in model_names]
        reddit_all_quintile = [self.results['reddit_all'][model]['Quintile_Spread_Return'] for model in model_names]
        advanced_reddit_quintile = [self.results['advanced_reddit'][model]['Quintile_Spread_Return'] for model in model_names]
        
        bars1 = ax6.bar(x - width, price_only_quintile, width, label='Price Only', alpha=0.8, color='lightblue')
        bars2 = ax6.bar(x, reddit_all_quintile, width, label='Reddit All', alpha=0.8, color='orange')
        bars3 = ax6.bar(x + width, advanced_reddit_quintile, width, label='Advanced Reddit', alpha=0.8, color='green')
        
        ax6.set_xlabel('Models')
        ax6.set_ylabel('Quintile Spread Return')
        ax6.set_title('Quintile Spread Return (Factor Validation)', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(model_names)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Comprehensive visualization saved to results/comprehensive_performance_metrics.png")
    
    def generate_comprehensive_report(self, comparison_df):
        """종합 리포트 생성"""
        print("📝 Generating comprehensive report...")
        
        report = []
        report.append("=" * 200)
        report.append("COMPREHENSIVE PERFORMANCE METRICS: 6 KEY INDICATORS")
        report.append("=" * 200)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Target Stocks: AMC, BB, GME (Meme Stocks Only)")
        report.append("- Price Only: Price features only")
        report.append("- Reddit All: Price features + Basic Reddit + Advanced Reddit")
        report.append("- Advanced Reddit: Price features + Advanced Reddit only")
        report.append("- Models: Ridge, LightGBM, XGBoost")
        report.append("- Evaluation: 6 Key Indicators (IC, Hit Rate, ICIR, Sharpe Ratio, MDD, Quintile Spread Return)")
        report.append("")
        
        # 성능 비교 표
        report.append("COMPREHENSIVE PERFORMANCE METRICS TABLE")
        report.append("-" * 150)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # 지표별 상세 분석
        report.append("METRIC-SPECIFIC DETAILED ANALYSIS")
        report.append("-" * 150)
        
        # 1. IC 분석
        report.append("\n1. INFORMATION COEFFICIENT (IC) ANALYSIS:")
        report.append("   - Measures predictive power of models")
        report.append("   - Higher IC indicates better prediction accuracy")
        report.append("   - Spearman rank correlation between predictions and actual returns")
        
        # 2. Hit Rate 분석
        report.append("\n2. HIT RATE ANALYSIS:")
        report.append("   - Measures directional prediction accuracy")
        report.append("   - Percentage of correct directional predictions")
        report.append("   - Higher hit rate indicates better directional forecasting")
        
        # 3. ICIR 분석
        report.append("\n3. ICIR (INFORMATION COEFFICIENT INFORMATION RATIO) ANALYSIS:")
        report.append("   - Measures stability of predictive power")
        report.append("   - ICIR = Mean IC / Std IC")
        report.append("   - Higher ICIR indicates more consistent performance")
        
        # 4. Sharpe Ratio 분석
        report.append("\n4. SHARPE RATIO ANALYSIS:")
        report.append("   - Measures risk-adjusted returns")
        report.append("   - Higher Sharpe Ratio indicates better risk-adjusted performance")
        report.append("   - Investment perspective metric")
        
        # 5. MDD 분석
        report.append("\n5. MAXIMUM DRAWDOWN (MDD) ANALYSIS:")
        report.append("   - Measures maximum loss from peak")
        report.append("   - Lower MDD indicates better risk management")
        report.append("   - Investment perspective metric")
        
        # 6. Quintile Spread Return 분석
        report.append("\n6. QUINTILE SPREAD RETURN ANALYSIS:")
        report.append("   - Measures factor effectiveness")
        report.append("   - Q5 (top) - Q1 (bottom) return spread")
        report.append("   - Factor validation perspective metric")
        
        # 모델별 상세 분석
        report.append("\nMODEL-SPECIFIC DETAILED ANALYSIS")
        report.append("-" * 150)
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost']:
            report.append(f"\n{model_name} DETAILED ANALYSIS:")
            
            # 각 지표별 최고 성능 피처 세트 찾기
            best_ic = max(self.results['price_only'][model_name]['IC'], 
                         self.results['reddit_all'][model_name]['IC'], 
                         self.results['advanced_reddit'][model_name]['IC'])
            
            best_hit_rate = max(self.results['price_only'][model_name]['Hit_Rate'], 
                              self.results['reddit_all'][model_name]['Hit_Rate'], 
                              self.results['advanced_reddit'][model_name]['Hit_Rate'])
            
            best_icir = max(self.results['price_only'][model_name]['ICIR'], 
                           self.results['reddit_all'][model_name]['ICIR'], 
                           self.results['advanced_reddit'][model_name]['ICIR'])
            
            best_sharpe = max(self.results['price_only'][model_name]['Sharpe_Ratio'], 
                             self.results['reddit_all'][model_name]['Sharpe_Ratio'], 
                             self.results['advanced_reddit'][model_name]['Sharpe_Ratio'])
            
            best_mdd = min(self.results['price_only'][model_name]['MDD'], 
                         self.results['reddit_all'][model_name]['MDD'], 
                         self.results['advanced_reddit'][model_name]['MDD'])
            
            best_quintile = max(self.results['price_only'][model_name]['Quintile_Spread_Return'], 
                               self.results['reddit_all'][model_name]['Quintile_Spread_Return'], 
                               self.results['advanced_reddit'][model_name]['Quintile_Spread_Return'])
            
            report.append(f"  Best IC: {best_ic:.4f}")
            report.append(f"  Best Hit Rate: {best_hit_rate:.4f}")
            report.append(f"  Best ICIR: {best_icir:.4f}")
            report.append(f"  Best Sharpe Ratio: {best_sharpe:.4f}")
            report.append(f"  Best MDD: {best_mdd:.4f}")
            report.append(f"  Best Quintile Spread: {best_quintile:.4f}")
        
        # 전체 결론
        report.append("\nOVERALL CONCLUSIONS")
        report.append("-" * 150)
        
        # 평균 성능 계산
        avg_metrics = {}
        for metric in ['IC', 'Hit_Rate', 'ICIR', 'Sharpe_Ratio', 'MDD', 'Quintile_Spread_Return']:
            avg_metrics[metric] = {}
            for feature_set in ['price_only', 'reddit_all', 'advanced_reddit']:
                avg_metrics[metric][feature_set] = np.mean([
                    self.results[feature_set][model][metric] for model in ['Ridge', 'LightGBM', 'XGBoost']
                ])
        
        report.append("Average Performance Across All Models:")
        for metric, feature_sets in avg_metrics.items():
            report.append(f"\n{metric}:")
            for feature_set, avg_value in feature_sets.items():
                report.append(f"  {feature_set}: {avg_value:.4f}")
        
        # 실전 적용 가이드
        report.append("\nPRACTICAL APPLICATION GUIDE")
        report.append("-" * 150)
        report.append("🔹 For Maximum IC (Predictive Power):")
        report.append("  - Focus on models with highest IC values")
        report.append("  - Consider ICIR for stability")
        report.append("")
        report.append("🔹 For Investment Strategy:")
        report.append("  - Use Sharpe Ratio and MDD for risk assessment")
        report.append("  - Higher Sharpe Ratio, Lower MDD preferred")
        report.append("")
        report.append("🔹 For Factor Validation:")
        report.append("  - Use Quintile Spread Return")
        report.append("  - Higher spread indicates better factor effectiveness")
        report.append("")
        report.append("🔹 For Directional Trading:")
        report.append("  - Use Hit Rate")
        report.append("  - Higher hit rate indicates better directional accuracy")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/comprehensive_performance_metrics_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Comprehensive report saved to results/comprehensive_performance_metrics_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Comprehensive Performance Metrics Analysis")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = ComprehensivePerformanceMetrics()
    
    # 1. 종합 실험 실행
    results = experiment.run_comprehensive_experiment()
    
    # 2. 종합 비교 표 생성
    print("\n" + "="*50)
    print("COMPREHENSIVE COMPARISON TABLE GENERATION")
    print("="*50)
    comparison_df = experiment.create_comprehensive_comparison_table()
    
    # 3. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    experiment.create_comprehensive_visualization()
    
    # 4. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_comprehensive_report(comparison_df)
    
    print("\n🎉 Comprehensive performance metrics analysis completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df

if __name__ == "__main__":
    experiment, comparison_df = main()
