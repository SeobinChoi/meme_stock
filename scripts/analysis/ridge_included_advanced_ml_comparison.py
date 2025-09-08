#!/usr/bin/env python3
"""
Ridge Included Advanced ML Models Comparison
Ridge 모델을 포함한 고급 ML 모델 비교 (딥러닝 + 전통 ML)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 딥러닝 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 기존 ML 라이브러리
import lightgbm as lgb
import xgboost as xgb

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class RidgeIncludedAdvancedMLComparison:
    """Ridge 포함 고급 ML 모델 비교 클래스"""
    
    def __init__(self):
        self.df = None
        self.results = {}
        self.target_tickers = ['AMC', 'BB', 'GME']
        self.sequence_length = 20  # 시퀀스 길이
        
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
    
    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Baseline 모델 훈련 (Ridge, XGBoost, LightGBM)"""
        print("🤖 Training baseline models...")
        
        models = {}
        
        # 1. Ridge Regression
        print("   📈 Training Ridge...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        models['Ridge'] = ridge
        
        # 2. LightGBM
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
        models['LightGBM'] = lgb_model
        
        # 3. XGBoost
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
        models['XGBoost'] = xgb_model
        
        print("   ✅ Baseline models trained successfully")
        
        return models
    
    def prepare_sequence_data(self, df, feature_cols, target_col='target_1d'):
        """시퀀스 데이터 준비 (딥러닝용)"""
        print(f"🔧 Preparing sequence data (length: {self.sequence_length})...")
        
        sequences = []
        targets = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # 특성 데이터 준비
            feature_data = ticker_data[feature_cols].fillna(0).values
            
            # 시퀀스 생성
            for i in range(self.sequence_length, len(ticker_data)):
                seq = feature_data[i-self.sequence_length:i]
                target = ticker_data.iloc[i][target_col]
                
                if not np.isnan(target):
                    sequences.append(seq)
                    targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"   ✅ Sequences: {sequences.shape}")
        print(f"   ✅ Targets: {targets.shape}")
        
        return sequences, targets
    
    def create_lstm_model(self, input_shape):
        """LSTM 모델 생성"""
        print("🧠 Creating LSTM model...")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_gru_model(self, input_shape):
        """GRU 모델 생성"""
        print("🧠 Creating GRU model...")
        
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_cnn_lstm_model(self, input_shape):
        """CNN-LSTM 모델 생성"""
        print("🧠 Creating CNN-LSTM model...")
        
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_transformer_model(self, input_shape):
        """Transformer 모델 생성 (간단한 버전)"""
        print("🧠 Creating Transformer model...")
        
        # 입력 레이어
        inputs = Input(shape=input_shape)
        
        # Multi-Head Attention
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention = LayerNormalization()(attention + inputs)
        
        # Feed Forward
        ffn = Dense(128, activation='relu')(attention)
        ffn = Dense(input_shape[-1])(ffn)
        ffn = LayerNormalization()(ffn + attention)
        
        # Global Average Pooling
        pooled = GlobalAveragePooling1D()(ffn)
        
        # 출력 레이어
        outputs = Dense(16, activation='relu')(pooled)
        outputs = Dropout(0.2)(outputs)
        outputs = Dense(1, activation='linear')(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train_deep_learning_models(self, X_train, y_train, X_val, y_val):
        """딥러닝 모델 훈련"""
        print("🤖 Training deep learning models...")
        
        models = {}
        
        # 데이터 스케일링
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # 1. LSTM
        print("   🧠 Training LSTM...")
        lstm_model = self.create_lstm_model((X_train.shape[1], X_train.shape[2]))
        lstm_model.fit(X_train_scaled, y_train, 
                      validation_data=(X_val_scaled, y_val),
                      epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
        models['LSTM'] = {'model': lstm_model, 'scaler': scaler}
        
        # 2. GRU
        print("   🧠 Training GRU...")
        gru_model = self.create_gru_model((X_train.shape[1], X_train.shape[2]))
        gru_model.fit(X_train_scaled, y_train, 
                     validation_data=(X_val_scaled, y_val),
                     epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
        models['GRU'] = {'model': gru_model, 'scaler': scaler}
        
        # 3. CNN-LSTM
        print("   🧠 Training CNN-LSTM...")
        cnn_lstm_model = self.create_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
        cnn_lstm_model.fit(X_train_scaled, y_train, 
                          validation_data=(X_val_scaled, y_val),
                          epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
        models['CNN-LSTM'] = {'model': cnn_lstm_model, 'scaler': scaler}
        
        # 4. Transformer
        print("   🧠 Training Transformer...")
        transformer_model = self.create_transformer_model((X_train.shape[1], X_train.shape[2]))
        transformer_model.fit(X_train_scaled, y_train, 
                             validation_data=(X_val_scaled, y_val),
                             epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
        models['Transformer'] = {'model': transformer_model, 'scaler': scaler}
        
        print("   ✅ Deep learning models trained successfully")
        
        return models
    
    def calculate_key_metrics(self, y_true, y_pred, predictions_df=None):
        """4가지 핵심 지표 계산"""
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
        
        # 4. QSR (Quintile Spread Return) - 팩터 검증
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
                metrics['QSR'] = quintile_spread
            else:
                metrics['QSR'] = np.nan
        else:
            metrics['QSR'] = np.nan
        
        return metrics
    
    def evaluate_models(self, X_test, y_test, models, test_df=None, is_sequence=False):
        """모델 평가 (4가지 핵심 지표)"""
        print("📊 Evaluating models with key metrics...")
        
        results = {}
        
        for model_name, model in models.items():
            print(f"   🔍 Evaluating {model_name}...")
            
            # 예측
            if model_name in ['Ridge', 'LightGBM', 'XGBoost']:
                y_pred = model.predict(X_test)
            else:
                # 딥러닝 모델
                X_test_scaled = model['scaler'].transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
                y_pred = model['model'].predict(X_test_scaled, verbose=0).flatten()
            
            # 예측 결과를 DataFrame으로 변환 (지표 계산용)
            predictions_df = None
            if test_df is not None and not is_sequence:
                predictions_df = test_df.copy()
                predictions_df['y_true'] = y_test
                predictions_df['y_pred'] = y_pred
            elif test_df is not None and is_sequence:
                # 시퀀스 데이터의 경우 길이에 맞게 조정
                min_len = min(len(y_test), len(y_pred), len(test_df))
                predictions_df = test_df.iloc[:min_len].copy()
                predictions_df['y_true'] = y_test[:min_len]
                predictions_df['y_pred'] = y_pred[:min_len]
            
            # 4가지 지표 계산
            metrics = self.calculate_key_metrics(y_test, y_pred, predictions_df)
            
            results[model_name] = metrics
            
            print(f"      IC: {metrics['IC']:.4f}")
            print(f"      Hit Rate: {metrics['Hit_Rate']:.4f}")
            print(f"      ICIR: {metrics['ICIR']:.4f}")
            print(f"      QSR: {metrics['QSR']:.4f}")
        
        return results
    
    def run_comprehensive_experiment(self):
        """종합 실험 실행"""
        print("🚀 Starting Ridge Included Advanced ML Models Comparison Experiment")
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
            
            # Baseline 모델 훈련
            baseline_models = self.train_baseline_models(X_train, y_train, X_val, y_val)
            
            # Baseline 모델 평가
            baseline_results = self.evaluate_models(X_test, y_test, baseline_models, test_df)
            
            # 딥러닝 모델용 데이터 준비 (Price Only일 때만)
            if feature_set == 'price_only':
                print(f"\n{'='*40}")
                print("DEEP LEARNING MODELS")
                print(f"{'='*40}")
                
                # 시퀀스 데이터 준비
                X_train_seq, y_train_seq = self.prepare_sequence_data(train_df, price_features)
                X_val_seq, y_val_seq = self.prepare_sequence_data(val_df, price_features)
                X_test_seq, y_test_seq = self.prepare_sequence_data(test_df, price_features)
                
                # 딥러닝 모델 훈련
                deep_models = self.train_deep_learning_models(X_train_seq, y_train_seq, 
                                                            X_val_seq, y_val_seq)
                
                # 딥러닝 모델 평가
                deep_results = self.evaluate_models(X_test_seq, y_test_seq, deep_models, test_df, is_sequence=True)
                
                # 결과 통합
                feature_results = {**baseline_results, **deep_results}
            else:
                feature_results = baseline_results
            
            all_results[feature_set] = feature_results
        
        self.results = all_results
        return all_results
    
    def create_comprehensive_comparison_table(self):
        """종합 비교 표 생성"""
        print("📋 Creating comprehensive comparison table...")
        
        comparison_data = []
        
        # 모든 모델 리스트
        all_models = set()
        for feature_set_results in self.results.values():
            all_models.update(feature_set_results.keys())
        
        for model_name in sorted(all_models):
            row = {'Model': model_name}
            
            for feature_set in ['price_only', 'reddit_all', 'advanced_reddit']:
                if model_name in self.results[feature_set]:
                    metrics = self.results[feature_set][model_name]
                    row[f'{feature_set}_IC'] = f"{metrics['IC']:.4f}"
                    row[f'{feature_set}_Hit_Rate'] = f"{metrics['Hit_Rate']:.4f}"
                    row[f'{feature_set}_ICIR'] = f"{metrics['ICIR']:.4f}"
                    row[f'{feature_set}_QSR'] = f"{metrics['QSR']:.4f}"
                else:
                    row[f'{feature_set}_IC'] = "N/A"
                    row[f'{feature_set}_Hit_Rate'] = "N/A"
                    row[f'{feature_set}_ICIR'] = "N/A"
                    row[f'{feature_set}_QSR'] = "N/A"
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*200)
        print("RIDGE INCLUDED ADVANCED ML MODELS COMPARISON: 4 KEY INDICATORS")
        print("="*200)
        print(comparison_df.to_string(index=False))
        print("="*200)
        
        return comparison_df
    
    def create_comprehensive_visualization(self):
        """종합 시각화 생성"""
        print("📈 Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('Ridge Included Advanced ML Models Comparison: 4 Key Indicators', 
                     fontsize=16, fontweight='bold')
        
        # 모든 모델 리스트
        all_models = set()
        for feature_set_results in self.results.values():
            all_models.update(feature_set_results.keys())
        
        model_names = sorted(all_models)
        
        # 각 특성 세트별로 플롯
        feature_sets = ['price_only', 'reddit_all', 'advanced_reddit']
        feature_names = ['Price Only', 'Reddit All', 'Advanced Reddit']
        
        for i, (feature_set, feature_name) in enumerate(zip(feature_sets, feature_names)):
            # IC 비교
            ax1 = axes[0, i]
            ic_values = [self.results[feature_set][model]['IC'] if model in self.results[feature_set] else 0 
                        for model in model_names]
            bars1 = ax1.bar(model_names, ic_values, alpha=0.8, color='skyblue')
            ax1.set_xlabel('Models')
            ax1.set_ylabel('IC (Spearman)')
            ax1.set_title(f'{feature_name} - IC', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Hit Rate 비교
            ax2 = axes[1, i]
            hit_rate_values = [self.results[feature_set][model]['Hit_Rate'] if model in self.results[feature_set] else 0 
                              for model in model_names]
            bars2 = ax2.bar(model_names, hit_rate_values, alpha=0.8, color='lightgreen')
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Hit Rate')
            ax2.set_title(f'{feature_name} - Hit Rate', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # ICIR 비교
            ax3 = axes[2, i]
            icir_values = [self.results[feature_set][model]['ICIR'] if model in self.results[feature_set] else 0 
                          for model in model_names]
            bars3 = ax3.bar(model_names, icir_values, alpha=0.8, color='orange')
            ax3.set_xlabel('Models')
            ax3.set_ylabel('ICIR')
            ax3.set_title(f'{feature_name} - ICIR', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # QSR 비교 (전체 특성 세트)
        ax4 = axes[0, 3]
        qsr_values = [self.results['price_only'][model]['QSR'] if model in self.results['price_only'] else 0 
                     for model in model_names]
        bars4 = ax4.bar(model_names, qsr_values, alpha=0.8, color='lightcoral')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('QSR')
        ax4.set_title('Price Only - QSR', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 빈 공간 제거
        axes[1, 3].axis('off')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/ridge_included_advanced_ml_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Comprehensive visualization saved to results/ridge_included_advanced_ml_comparison.png")
    
    def generate_comprehensive_report(self, comparison_df):
        """종합 리포트 생성"""
        print("📝 Generating comprehensive report...")
        
        report = []
        report.append("=" * 200)
        report.append("RIDGE INCLUDED ADVANCED ML MODELS COMPARISON: 4 KEY INDICATORS")
        report.append("=" * 200)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Target Stocks: AMC, BB, GME (Meme Stocks Only)")
        report.append("- Models: Ridge, LightGBM, XGBoost, LSTM, GRU, CNN-LSTM, Transformer")
        report.append("- Features: Price Only, Price + All Reddit, Price + Advanced Reddit")
        report.append("- Evaluation: 4 Key Indicators (IC, Hit Rate, ICIR, QSR)")
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
        
        # 4. QSR 분석
        report.append("\n4. QUINTILE SPREAD RETURN (QSR) ANALYSIS:")
        report.append("   - Measures factor effectiveness")
        report.append("   - Q5 (top) - Q1 (bottom) return spread")
        report.append("   - Factor validation perspective metric")
        
        # 모델별 상세 분석
        report.append("\nMODEL-SPECIFIC DETAILED ANALYSIS")
        report.append("-" * 150)
        
        for model_name in ['Ridge', 'LightGBM', 'XGBoost', 'LSTM', 'GRU', 'CNN-LSTM', 'Transformer']:
            if model_name in self.results['price_only']:
                report.append(f"\n{model_name} DETAILED ANALYSIS:")
                metrics = self.results['price_only'][model_name]
                report.append(f"  IC: {metrics['IC']:.4f}")
                report.append(f"  Hit Rate: {metrics['Hit_Rate']:.4f}")
                report.append(f"  ICIR: {metrics['ICIR']:.4f}")
                report.append(f"  QSR: {metrics['QSR']:.4f}")
        
        # 전체 결론
        report.append("\nOVERALL CONCLUSIONS")
        report.append("-" * 150)
        
        # 평균 성능 계산
        avg_metrics = {}
        for metric in ['IC', 'Hit_Rate', 'ICIR', 'QSR']:
            avg_metrics[metric] = {}
            for feature_set in ['price_only', 'reddit_all', 'advanced_reddit']:
                values = [self.results[feature_set][model][metric] 
                         for model in self.results[feature_set].keys()]
                avg_metrics[metric][feature_set] = np.mean(values)
        
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
        report.append("🔹 For Directional Trading:")
        report.append("  - Use Hit Rate")
        report.append("  - Higher hit rate indicates better directional accuracy")
        report.append("")
        report.append("🔹 For Factor Validation:")
        report.append("  - Use QSR")
        report.append("  - Higher spread indicates better factor effectiveness")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/ridge_included_advanced_ml_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Comprehensive report saved to results/ridge_included_advanced_ml_comparison_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Ridge Included Advanced ML Models Comparison Analysis")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = RidgeIncludedAdvancedMLComparison()
    
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
    
    print("\n🎉 Ridge included advanced ML models comparison analysis completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df

if __name__ == "__main__":
    experiment, comparison_df = main()
