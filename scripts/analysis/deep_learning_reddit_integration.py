#!/usr/bin/env python3
"""
Deep Learning Reddit Integration
딥러닝 모델에 Reddit 데이터를 효과적으로 통합
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
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Concatenate, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 기존 ML 라이브러리
import lightgbm as lgb
import xgboost as xgb

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DeepLearningRedditIntegration:
    """딥러닝 Reddit 통합 클래스"""
    
    def __init__(self):
        self.df = None
        self.results = {}
        self.target_tickers = ['AMC', 'BB', 'GME']
        self.sequence_length = 20
        
    def load_data(self):
        """데이터 로드"""
        print("📊 Loading meme stocks data...")
        
        train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
        val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
        test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        df = df[df['ticker'].isin(self.target_tickers)].copy()
        df['target_1d'] = df.groupby('ticker')['returns_1d'].shift(-1)
        
        # 미래 데이터 마스킹
        df['mask'] = False
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_indices = df[ticker_mask].index
            if len(ticker_indices) > 5:
                df.loc[ticker_indices[-5:], 'mask'] = True
        
        df = df[~df['mask']].copy()
        
        print(f"   ✅ Total data: {len(df)} records")
        self.df = df
        return df
    
    def prepare_feature_sets(self):
        """특성 세트 준비"""
        print("🔧 Preparing feature sets...")
        
        self.price_features = [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
            'vol_5d', 'vol_10d', 'vol_20d',
            'price_ratio_sma10', 'price_ratio_sma20',
            'rsi_14', 'volume_ratio', 'turnover',
            'day_of_week', 'month', 'is_monday', 'is_friday', 'is_weekend_effect'
        ]
        
        self.basic_reddit_features = [
            'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10'
        ]
        
        self.advanced_reddit_features = [
            'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95',
            'reddit_momentum_3', 'reddit_momentum_7', 'reddit_momentum_14', 'reddit_momentum_21',
            'reddit_vol_5', 'reddit_vol_10', 'reddit_vol_20',
            'reddit_percentile', 'reddit_high_regime', 'reddit_low_regime',
            'market_sentiment', 'price_reddit_momentum', 'vol_reddit_attention'
        ]
        
        # 존재하는 특성만 선택
        self.price_features = [col for col in self.price_features if col in self.df.columns]
        self.basic_reddit_features = [col for col in self.basic_reddit_features if col in self.df.columns]
        self.advanced_reddit_features = [col for col in self.advanced_reddit_features if col in self.df.columns]
        
        print(f"   ✅ Price features: {len(self.price_features)}")
        print(f"   ✅ Basic Reddit features: {len(self.basic_reddit_features)}")
        print(f"   ✅ Advanced Reddit features: {len(self.advanced_reddit_features)}")
        
        return self.price_features, self.basic_reddit_features, self.advanced_reddit_features
    
    def strict_time_series_split(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """엄격한 시계열 분할"""
        print("📊 Performing strict time series split...")
        
        train_data, val_data, test_data = [], [], []
        
        for ticker in self.target_tickers:
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
        
        print(f"   ✅ Train: {len(train_df)} records")
        print(f"   ✅ Validation: {len(val_df)} records")
        print(f"   ✅ Test: {len(test_df)} records")
        
        return train_df, val_df, test_df
    
    def prepare_multimodal_sequence_data(self, df, price_features, reddit_features):
        """멀티모달 시퀀스 데이터 준비"""
        print(f"🔧 Preparing multimodal sequence data (length: {self.sequence_length})...")
        
        price_sequences, reddit_sequences, targets = [], [], []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # 특성 데이터 준비
            price_data = ticker_data[price_features].fillna(0).values
            reddit_data = ticker_data[reddit_features].fillna(0).values
            
            # 시퀀스 생성
            for i in range(self.sequence_length, len(ticker_data)):
                price_seq = price_data[i-self.sequence_length:i]
                reddit_seq = reddit_data[i-self.sequence_length:i]
                target = ticker_data.iloc[i]['target_1d']
                
                if not np.isnan(target):
                    price_sequences.append(price_seq)
                    reddit_sequences.append(reddit_seq)
                    targets.append(target)
        
        price_sequences = np.array(price_sequences)
        reddit_sequences = np.array(reddit_sequences)
        targets = np.array(targets)
        
        print(f"   ✅ Price sequences: {price_sequences.shape}")
        print(f"   ✅ Reddit sequences: {reddit_sequences.shape}")
        print(f"   ✅ Targets: {targets.shape}")
        
        return price_sequences, reddit_sequences, targets
    
    def create_multimodal_lstm_model(self, price_input_shape, reddit_input_shape):
        """멀티모달 LSTM 모델 생성"""
        print("🧠 Creating Multimodal LSTM model...")
        
        # Price Branch
        price_input = Input(shape=price_input_shape, name='price_input')
        price_lstm = LSTM(64, return_sequences=True)(price_input)
        price_lstm = Dropout(0.2)(price_lstm)
        price_lstm = LSTM(32, return_sequences=False)(price_lstm)
        price_features = Dense(16, activation='relu')(price_lstm)
        
        # Reddit Branch
        reddit_input = Input(shape=reddit_input_shape, name='reddit_input')
        reddit_lstm = LSTM(64, return_sequences=True)(reddit_input)
        reddit_lstm = Dropout(0.2)(reddit_lstm)
        reddit_lstm = LSTM(32, return_sequences=False)(reddit_lstm)
        reddit_features = Dense(16, activation='relu')(reddit_lstm)
        
        # Cross-Attention (간단한 버전으로 수정)
        # attention = MultiHeadAttention(num_heads=4, key_dim=16)(price_features, reddit_features)
        attention = Dense(16, activation='relu')(Concatenate()([price_features, reddit_features]))
        
        # Fusion
        fused = Concatenate()([price_features, reddit_features, attention])
        fused = Dense(32, activation='relu')(fused)
        fused = Dropout(0.2)(fused)
        output = Dense(1, activation='linear')(fused)
        
        model = Model(inputs=[price_input, reddit_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_multimodal_gru_model(self, price_input_shape, reddit_input_shape):
        """멀티모달 GRU 모델 생성"""
        print("🧠 Creating Multimodal GRU model...")
        
        # Price Branch
        price_input = Input(shape=price_input_shape, name='price_input')
        price_gru = GRU(64, return_sequences=True)(price_input)
        price_gru = Dropout(0.2)(price_gru)
        price_gru = GRU(32, return_sequences=False)(price_gru)
        price_features = Dense(16, activation='relu')(price_gru)
        
        # Reddit Branch
        reddit_input = Input(shape=reddit_input_shape, name='reddit_input')
        reddit_gru = GRU(64, return_sequences=True)(reddit_input)
        reddit_gru = Dropout(0.2)(reddit_gru)
        reddit_gru = GRU(32, return_sequences=False)(reddit_gru)
        reddit_features = Dense(16, activation='relu')(reddit_gru)
        
        # Feature Interaction
        interaction = Multiply()([price_features, reddit_features])
        
        # Fusion
        fused = Concatenate()([price_features, reddit_features, interaction])
        fused = Dense(32, activation='relu')(fused)
        fused = Dropout(0.2)(fused)
        output = Dense(1, activation='linear')(fused)
        
        model = Model(inputs=[price_input, reddit_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_multimodal_cnn_lstm_model(self, price_input_shape, reddit_input_shape):
        """멀티모달 CNN-LSTM 모델 생성"""
        print("🧠 Creating Multimodal CNN-LSTM model...")
        
        # Price Branch
        price_input = Input(shape=price_input_shape, name='price_input')
        price_conv = Conv1D(64, 3, activation='relu')(price_input)
        price_conv = MaxPooling1D(2)(price_conv)
        price_conv = Conv1D(32, 3, activation='relu')(price_conv)
        price_conv = MaxPooling1D(2)(price_conv)
        price_lstm = LSTM(32, return_sequences=False)(price_conv)
        price_features = Dense(16, activation='relu')(price_lstm)
        
        # Reddit Branch
        reddit_input = Input(shape=reddit_input_shape, name='reddit_input')
        reddit_conv = Conv1D(64, 3, activation='relu')(reddit_input)
        reddit_conv = MaxPooling1D(2)(reddit_conv)
        reddit_conv = Conv1D(32, 3, activation='relu')(reddit_conv)
        reddit_conv = MaxPooling1D(2)(reddit_conv)
        reddit_lstm = LSTM(32, return_sequences=False)(reddit_conv)
        reddit_features = Dense(16, activation='relu')(reddit_lstm)
        
        # Fusion
        fused = Concatenate()([price_features, reddit_features])
        fused = Dense(32, activation='relu')(fused)
        fused = Dropout(0.2)(fused)
        output = Dense(1, activation='linear')(fused)
        
        model = Model(inputs=[price_input, reddit_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_multimodal_transformer_model(self, price_input_shape, reddit_input_shape):
        """멀티모달 Transformer 모델 생성"""
        print("🧠 Creating Multimodal Transformer model...")
        
        # Price Branch
        price_input = Input(shape=price_input_shape, name='price_input')
        price_attention = MultiHeadAttention(num_heads=8, key_dim=64)(price_input, price_input)
        price_attention = LayerNormalization()(price_attention + price_input)
        price_ffn = Dense(128, activation='relu')(price_attention)
        price_ffn = Dense(price_input_shape[-1])(price_ffn)
        price_ffn = LayerNormalization()(price_ffn + price_attention)
        price_pooled = GlobalAveragePooling1D()(price_ffn)
        price_features = Dense(16, activation='relu')(price_pooled)
        
        # Reddit Branch
        reddit_input = Input(shape=reddit_input_shape, name='reddit_input')
        reddit_attention = MultiHeadAttention(num_heads=8, key_dim=64)(reddit_input, reddit_input)
        reddit_attention = LayerNormalization()(reddit_attention + reddit_input)
        reddit_ffn = Dense(128, activation='relu')(reddit_attention)
        reddit_ffn = Dense(reddit_input_shape[-1])(reddit_ffn)
        reddit_ffn = LayerNormalization()(reddit_ffn + reddit_attention)
        reddit_pooled = GlobalAveragePooling1D()(reddit_ffn)
        reddit_features = Dense(16, activation='relu')(reddit_pooled)
        
        # Cross-Attention (간단한 버전으로 수정)
        # cross_attention = MultiHeadAttention(num_heads=4, key_dim=16)(price_features, reddit_features)
        cross_attention = Dense(16, activation='relu')(Concatenate()([price_features, reddit_features]))
        
        # Fusion
        fused = Concatenate()([price_features, reddit_features, cross_attention])
        fused = Dense(32, activation='relu')(fused)
        fused = Dropout(0.2)(fused)
        output = Dense(1, activation='linear')(fused)
        
        model = Model(inputs=[price_input, reddit_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train_multimodal_models(self, price_train, reddit_train, y_train, price_val, reddit_val, y_val):
        """멀티모달 모델 훈련"""
        print("🤖 Training multimodal models...")
        
        models = {}
        
        # 데이터 스케일링
        price_scaler = MinMaxScaler()
        reddit_scaler = MinMaxScaler()
        
        price_train_scaled = price_scaler.fit_transform(price_train.reshape(-1, price_train.shape[-1])).reshape(price_train.shape)
        price_val_scaled = price_scaler.transform(price_val.reshape(-1, price_val.shape[-1])).reshape(price_val.shape)
        
        reddit_train_scaled = reddit_scaler.fit_transform(reddit_train.reshape(-1, reddit_train.shape[-1])).reshape(reddit_train.shape)
        reddit_val_scaled = reddit_scaler.transform(reddit_val.reshape(-1, reddit_val.shape[-1])).reshape(reddit_val.shape)
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # 1. Multimodal LSTM
        print("   🧠 Training Multimodal LSTM...")
        multimodal_lstm = self.create_multimodal_lstm_model(
            (price_train.shape[1], price_train.shape[2]),
            (reddit_train.shape[1], reddit_train.shape[2])
        )
        multimodal_lstm.fit(
            [price_train_scaled, reddit_train_scaled], y_train,
            validation_data=([price_val_scaled, reddit_val_scaled], y_val),
            epochs=50, batch_size=32, callbacks=callbacks, verbose=0
        )
        models['Multimodal_LSTM'] = {
            'model': multimodal_lstm, 
            'price_scaler': price_scaler, 
            'reddit_scaler': reddit_scaler
        }
        
        # 2. Multimodal GRU
        print("   🧠 Training Multimodal GRU...")
        multimodal_gru = self.create_multimodal_gru_model(
            (price_train.shape[1], price_train.shape[2]),
            (reddit_train.shape[1], reddit_train.shape[2])
        )
        multimodal_gru.fit(
            [price_train_scaled, reddit_train_scaled], y_train,
            validation_data=([price_val_scaled, reddit_val_scaled], y_val),
            epochs=50, batch_size=32, callbacks=callbacks, verbose=0
        )
        models['Multimodal_GRU'] = {
            'model': multimodal_gru, 
            'price_scaler': price_scaler, 
            'reddit_scaler': reddit_scaler
        }
        
        # 3. Multimodal CNN-LSTM
        print("   🧠 Training Multimodal CNN-LSTM...")
        multimodal_cnn_lstm = self.create_multimodal_cnn_lstm_model(
            (price_train.shape[1], price_train.shape[2]),
            (reddit_train.shape[1], reddit_train.shape[2])
        )
        multimodal_cnn_lstm.fit(
            [price_train_scaled, reddit_train_scaled], y_train,
            validation_data=([price_val_scaled, reddit_val_scaled], y_val),
            epochs=50, batch_size=32, callbacks=callbacks, verbose=0
        )
        models['Multimodal_CNN-LSTM'] = {
            'model': multimodal_cnn_lstm, 
            'price_scaler': price_scaler, 
            'reddit_scaler': reddit_scaler
        }
        
        # 4. Multimodal Transformer
        print("   🧠 Training Multimodal Transformer...")
        multimodal_transformer = self.create_multimodal_transformer_model(
            (price_train.shape[1], price_train.shape[2]),
            (reddit_train.shape[1], reddit_train.shape[2])
        )
        multimodal_transformer.fit(
            [price_train_scaled, reddit_train_scaled], y_train,
            validation_data=([price_val_scaled, reddit_val_scaled], y_val),
            epochs=50, batch_size=32, callbacks=callbacks, verbose=0
        )
        models['Multimodal_Transformer'] = {
            'model': multimodal_transformer, 
            'price_scaler': price_scaler, 
            'reddit_scaler': reddit_scaler
        }
        
        print("   ✅ Multimodal models trained successfully")
        
        return models
    
    def calculate_key_metrics(self, y_true, y_pred, predictions_df=None):
        """4가지 핵심 지표 계산"""
        metrics = {}
        
        # 1. IC (Information Coefficient)
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
        
        # 2. Hit Rate
        true_direction = (y_true > 0).astype(int)
        pred_direction = (y_pred > 0).astype(int)
        hit_rate = (true_direction == pred_direction).mean()
        metrics['Hit_Rate'] = hit_rate
        
        # 3. ICIR
        if predictions_df is not None and 'date' in predictions_df.columns:
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            monthly_ic = []
            
            for year_month in predictions_df['date'].dt.to_period('M').unique():
                month_data = predictions_df[predictions_df['date'].dt.to_period('M') == year_month]
                if len(month_data) > 10:
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
        
        # 4. QSR
        if predictions_df is not None and 'date' in predictions_df.columns:
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            predictions_df = predictions_df.sort_values('date')
            
            predictions_df['quintile'] = predictions_df.groupby('date')['y_pred'].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
            )
            
            quintile_returns = []
            for quintile in range(5):
                quintile_data = predictions_df[predictions_df['quintile'] == quintile]
                if len(quintile_data) > 0:
                    quintile_return = quintile_data['y_true'].mean()
                    quintile_returns.append(quintile_return)
                else:
                    quintile_returns.append(0)
            
            if len(quintile_returns) == 5:
                quintile_spread = quintile_returns[4] - quintile_returns[0]
                metrics['QSR'] = quintile_spread
            else:
                metrics['QSR'] = np.nan
        else:
            metrics['QSR'] = np.nan
        
        return metrics
    
    def evaluate_multimodal_models(self, price_test, reddit_test, y_test, models, test_df=None):
        """멀티모달 모델 평가"""
        print("📊 Evaluating multimodal models...")
        
        results = {}
        
        for model_name, model_info in models.items():
            print(f"   🔍 Evaluating {model_name}...")
            
            # 예측
            price_test_scaled = model_info['price_scaler'].transform(
                price_test.reshape(-1, price_test.shape[-1])).reshape(price_test.shape)
            reddit_test_scaled = model_info['reddit_scaler'].transform(
                reddit_test.reshape(-1, reddit_test.shape[-1])).reshape(reddit_test.shape)
            
            y_pred = model_info['model'].predict([price_test_scaled, reddit_test_scaled], verbose=0).flatten()
            
            # 예측 결과를 DataFrame으로 변환
            predictions_df = None
            if test_df is not None:
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
    
    def run_multimodal_experiment(self):
        """멀티모달 실험 실행"""
        print("🚀 Starting Deep Learning Reddit Integration Experiment")
        print("=" * 80)
        
        # 데이터 로드
        df = self.load_data()
        
        # 특성 준비
        price_features, basic_reddit, advanced_reddit = self.prepare_feature_sets()
        
        # 시계열 분할
        train_df, val_df, test_df = self.strict_time_series_split()
        
        # 실험할 Reddit 특성 세트들
        reddit_sets = {
            'basic_reddit': basic_reddit,
            'advanced_reddit': advanced_reddit,
            'all_reddit': basic_reddit + advanced_reddit
        }
        
        # 결과 저장
        all_results = {}
        
        for reddit_set_name, reddit_features in reddit_sets.items():
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: Price + {reddit_set_name.upper()}")
            print(f"{'='*60}")
            
            # 멀티모달 시퀀스 데이터 준비
            price_train, reddit_train, y_train = self.prepare_multimodal_sequence_data(
                train_df, price_features, reddit_features)
            price_val, reddit_val, y_val = self.prepare_multimodal_sequence_data(
                val_df, price_features, reddit_features)
            price_test, reddit_test, y_test = self.prepare_multimodal_sequence_data(
                test_df, price_features, reddit_features)
            
            # 멀티모달 모델 훈련
            multimodal_models = self.train_multimodal_models(
                price_train, reddit_train, y_train,
                price_val, reddit_val, y_val
            )
            
            # 멀티모달 모델 평가
            multimodal_results = self.evaluate_multimodal_models(
                price_test, reddit_test, y_test, multimodal_models, test_df
            )
            
            all_results[reddit_set_name] = multimodal_results
        
        self.results = all_results
        return all_results
    
    def create_comparison_table(self):
        """비교 표 생성"""
        print("📋 Creating comparison table...")
        
        comparison_data = []
        
        for reddit_set_name, results in self.results.items():
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Reddit_Set': reddit_set_name,
                    'Model': model_name,
                    'IC': f"{metrics['IC']:.4f}",
                    'Hit_Rate': f"{metrics['Hit_Rate']:.4f}",
                    'ICIR': f"{metrics['ICIR']:.4f}",
                    'QSR': f"{metrics['QSR']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*100)
        print("DEEP LEARNING REDDIT INTEGRATION: 4 KEY INDICATORS")
        print("="*100)
        print(comparison_df.to_string(index=False))
        print("="*100)
        
        return comparison_df
    
    def create_visualization(self):
        """시각화 생성"""
        print("📈 Creating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deep Learning Reddit Integration: 4 Key Indicators', 
                     fontsize=16, fontweight='bold')
        
        # 데이터 준비
        reddit_sets = list(self.results.keys())
        model_names = list(self.results[reddit_sets[0]].keys())
        
        # IC 비교
        ax1 = axes[0, 0]
        ic_data = []
        for reddit_set in reddit_sets:
            ic_values = [self.results[reddit_set][model]['IC'] for model in model_names]
            ic_data.append(ic_values)
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, (reddit_set, ic_values) in enumerate(zip(reddit_sets, ic_data)):
            ax1.bar(x + i*width, ic_values, width, label=reddit_set, alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Information Coefficient', fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Hit Rate 비교
        ax2 = axes[0, 1]
        hit_rate_data = []
        for reddit_set in reddit_sets:
            hit_rate_values = [self.results[reddit_set][model]['Hit_Rate'] for model in model_names]
            hit_rate_data.append(hit_rate_values)
        
        for i, (reddit_set, hit_rate_values) in enumerate(zip(reddit_sets, hit_rate_data)):
            ax2.bar(x + i*width, hit_rate_values, width, label=reddit_set, alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate (Directional Accuracy)', fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ICIR 비교
        ax3 = axes[1, 0]
        icir_data = []
        for reddit_set in reddit_sets:
            icir_values = [self.results[reddit_set][model]['ICIR'] for model in model_names]
            icir_data.append(icir_values)
        
        for i, (reddit_set, icir_values) in enumerate(zip(reddit_sets, icir_data)):
            ax3.bar(x + i*width, icir_values, width, label=reddit_set, alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('ICIR')
        ax3.set_title('ICIR (Stability)', fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # QSR 비교
        ax4 = axes[1, 1]
        qsr_data = []
        for reddit_set in reddit_sets:
            qsr_values = [self.results[reddit_set][model]['QSR'] for model in model_names]
            qsr_data.append(qsr_values)
        
        for i, (reddit_set, qsr_values) in enumerate(zip(reddit_sets, qsr_data)):
            ax4.bar(x + i*width, qsr_values, width, label=reddit_set, alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('QSR')
        ax4.set_title('Quintile Spread Return', fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/deep_learning_reddit_integration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Visualization saved to results/deep_learning_reddit_integration.png")
    
    def generate_report(self, comparison_df):
        """리포트 생성"""
        print("📝 Generating report...")
        
        report = []
        report.append("=" * 150)
        report.append("DEEP LEARNING REDDIT INTEGRATION: 4 KEY INDICATORS")
        report.append("=" * 150)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Target Stocks: AMC, BB, GME (Meme Stocks Only)")
        report.append("- Models: Multimodal LSTM, GRU, CNN-LSTM, Transformer")
        report.append("- Features: Price + Basic Reddit, Price + Advanced Reddit, Price + All Reddit")
        report.append("- Architecture: Multimodal with Cross-Attention")
        report.append("- Evaluation: 4 Key Indicators (IC, Hit Rate, ICIR, QSR)")
        report.append("")
        
        # 성능 비교 표
        report.append("COMPREHENSIVE PERFORMANCE METRICS TABLE")
        report.append("-" * 100)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # 지표별 상세 분석
        report.append("METRIC-SPECIFIC DETAILED ANALYSIS")
        report.append("-" * 100)
        
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
        report.append("-" * 100)
        
        for reddit_set_name, results in self.results.items():
            report.append(f"\n{reddit_set_name.upper()} RESULTS:")
            for model_name, metrics in results.items():
                report.append(f"  {model_name}:")
                report.append(f"    IC: {metrics['IC']:.4f}")
                report.append(f"    Hit Rate: {metrics['Hit_Rate']:.4f}")
                report.append(f"    ICIR: {metrics['ICIR']:.4f}")
                report.append(f"    QSR: {metrics['QSR']:.4f}")
        
        # 전체 결론
        report.append("\nOVERALL CONCLUSIONS")
        report.append("-" * 100)
        
        # 평균 성능 계산
        avg_metrics = {}
        for metric in ['IC', 'Hit_Rate', 'ICIR', 'QSR']:
            avg_metrics[metric] = {}
            for reddit_set_name, results in self.results.items():
                values = [results[model][metric] for model in results.keys()]
                avg_metrics[metric][reddit_set_name] = np.mean(values)
        
        report.append("Average Performance Across All Models:")
        for metric, reddit_sets in avg_metrics.items():
            report.append(f"\n{metric}:")
            for reddit_set, avg_value in reddit_sets.items():
                report.append(f"  {reddit_set}: {avg_value:.4f}")
        
        # 실전 적용 가이드
        report.append("\nPRACTICAL APPLICATION GUIDE")
        report.append("-" * 100)
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
        with open('results/deep_learning_reddit_integration_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Report saved to results/deep_learning_reddit_integration_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Deep Learning Reddit Integration Analysis")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = DeepLearningRedditIntegration()
    
    # 1. 멀티모달 실험 실행
    results = experiment.run_multimodal_experiment()
    
    # 2. 비교 표 생성
    print("\n" + "="*50)
    print("COMPARISON TABLE GENERATION")
    print("="*50)
    comparison_df = experiment.create_comparison_table()
    
    # 3. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    experiment.create_visualization()
    
    # 4. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_report(comparison_df)
    
    print("\n🎉 Deep learning Reddit integration analysis completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df

if __name__ == "__main__":
    experiment, comparison_df = main()
