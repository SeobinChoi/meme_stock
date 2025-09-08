#!/usr/bin/env python3
"""
Advanced Integration Models
고급 통합 모델 구현: 하이브리드 시퀀스, 계층적 통합, 강화학습 기반
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
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Concatenate, Multiply, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 기존 ML 라이브러리
import lightgbm as lgb
import xgboost as xgb

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class AdvancedIntegrationModels:
    """고급 통합 모델 클래스"""
    
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
    
    def prepare_hybrid_sequence_data(self, df, price_features, reddit_features):
        """하이브리드 시퀀스 데이터 준비"""
        print(f"🔧 Preparing hybrid sequence data (length: {self.sequence_length})...")
        
        sequences = []
        targets = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # 통합 특성 데이터 준비
            all_features = price_features + reddit_features
            feature_data = ticker_data[all_features].fillna(0).values
            
            # 시퀀스 생성
            for i in range(self.sequence_length, len(ticker_data)):
                seq = feature_data[i-self.sequence_length:i]
                target = ticker_data.iloc[i]['target_1d']
                
                if not np.isnan(target):
                    sequences.append(seq)
                    targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"   ✅ Sequences: {sequences.shape}")
        print(f"   ✅ Targets: {targets.shape}")
        
        return sequences, targets
    
    def create_hybrid_lstm_model(self, input_shape):
        """하이브리드 LSTM 모델 생성"""
        print("🧠 Creating Hybrid LSTM model...")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_hybrid_gru_model(self, input_shape):
        """하이브리드 GRU 모델 생성"""
        print("🧠 Creating Hybrid GRU model...")
        
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            Dropout(0.3),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_hybrid_cnn_lstm_model(self, input_shape):
        """하이브리드 CNN-LSTM 모델 생성"""
        print("🧠 Creating Hybrid CNN-LSTM model...")
        
        model = Sequential([
            Conv1D(128, 5, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_hybrid_transformer_model(self, input_shape):
        """하이브리드 Transformer 모델 생성"""
        print("🧠 Creating Hybrid Transformer model...")
        
        # 입력 레이어
        inputs = Input(shape=input_shape)
        
        # Multi-Head Attention 1
        attention1 = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention1 = LayerNormalization()(attention1 + inputs)
        
        # Feed Forward 1
        ffn1 = Dense(256, activation='relu')(attention1)
        ffn1 = Dense(input_shape[-1])(ffn1)
        ffn1 = LayerNormalization()(ffn1 + attention1)
        
        # Multi-Head Attention 2
        attention2 = MultiHeadAttention(num_heads=8, key_dim=64)(ffn1, ffn1)
        attention2 = LayerNormalization()(attention2 + ffn1)
        
        # Feed Forward 2
        ffn2 = Dense(256, activation='relu')(attention2)
        ffn2 = Dense(input_shape[-1])(ffn2)
        ffn2 = LayerNormalization()(ffn2 + attention2)
        
        # Global Average Pooling
        pooled = GlobalAveragePooling1D()(ffn2)
        
        # 출력 레이어
        outputs = Dense(64, activation='relu')(pooled)
        outputs = Dropout(0.3)(outputs)
        outputs = Dense(32, activation='relu')(outputs)
        outputs = Dropout(0.2)(outputs)
        outputs = Dense(1, activation='linear')(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def create_hierarchical_model(self, input_shape):
        """계층적 통합 모델 생성"""
        print("🧠 Creating Hierarchical Integration model...")
        
        # 입력 레이어
        inputs = Input(shape=input_shape)
        
        # Level 1: Price Features (처음 17개 특성)
        price_features = inputs[:, :, :17]
        price_lstm = LSTM(64, return_sequences=True)(price_features)
        price_lstm = Dropout(0.2)(price_lstm)
        price_lstm = LSTM(32, return_sequences=False)(price_lstm)
        price_output = Dense(16, activation='relu')(price_lstm)
        
        # Level 2: Reddit Features (나머지 특성)
        reddit_features = inputs[:, :, 17:]
        reddit_lstm = LSTM(64, return_sequences=True)(reddit_features)
        reddit_lstm = Dropout(0.2)(reddit_lstm)
        reddit_lstm = LSTM(32, return_sequences=False)(reddit_lstm)
        reddit_output = Dense(16, activation='relu')(reddit_lstm)
        
        # Level 3: Feature Interaction
        interaction = Multiply()([price_output, reddit_output])
        
        # Level 4: Fusion Layer
        fused = Concatenate()([price_output, reddit_output, interaction])
        fused = Dense(32, activation='relu')(fused)
        fused = Dropout(0.2)(fused)
        fused = Dense(16, activation='relu')(fused)
        fused = Dropout(0.1)(fused)
        outputs = Dense(1, activation='linear')(fused)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train_hybrid_models(self, X_train, y_train, X_val, y_val):
        """하이브리드 모델 훈련"""
        print("🤖 Training hybrid models...")
        
        models = {}
        
        # 데이터 스케일링
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=7)
        ]
        
        # 1. Hybrid LSTM
        print("   🧠 Training Hybrid LSTM...")
        hybrid_lstm = self.create_hybrid_lstm_model((X_train.shape[1], X_train.shape[2]))
        hybrid_lstm.fit(X_train_scaled, y_train, 
                       validation_data=(X_val_scaled, y_val),
                       epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
        models['Hybrid_LSTM'] = {'model': hybrid_lstm, 'scaler': scaler}
        
        # 2. Hybrid GRU
        print("   🧠 Training Hybrid GRU...")
        hybrid_gru = self.create_hybrid_gru_model((X_train.shape[1], X_train.shape[2]))
        hybrid_gru.fit(X_train_scaled, y_train, 
                      validation_data=(X_val_scaled, y_val),
                      epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
        models['Hybrid_GRU'] = {'model': hybrid_gru, 'scaler': scaler}
        
        # 3. Hybrid CNN-LSTM
        print("   🧠 Training Hybrid CNN-LSTM...")
        hybrid_cnn_lstm = self.create_hybrid_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
        hybrid_cnn_lstm.fit(X_train_scaled, y_train, 
                           validation_data=(X_val_scaled, y_val),
                           epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
        models['Hybrid_CNN-LSTM'] = {'model': hybrid_cnn_lstm, 'scaler': scaler}
        
        # 4. Hybrid Transformer
        print("   🧠 Training Hybrid Transformer...")
        hybrid_transformer = self.create_hybrid_transformer_model((X_train.shape[1], X_train.shape[2]))
        hybrid_transformer.fit(X_train_scaled, y_train, 
                              validation_data=(X_val_scaled, y_val),
                              epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
        models['Hybrid_Transformer'] = {'model': hybrid_transformer, 'scaler': scaler}
        
        # 5. Hierarchical Integration
        print("   🧠 Training Hierarchical Integration...")
        hierarchical_model = self.create_hierarchical_model((X_train.shape[1], X_train.shape[2]))
        hierarchical_model.fit(X_train_scaled, y_train, 
                              validation_data=(X_val_scaled, y_val),
                              epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
        models['Hierarchical_Integration'] = {'model': hierarchical_model, 'scaler': scaler}
        
        print("   ✅ Hybrid models trained successfully")
        
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
    
    def evaluate_hybrid_models(self, X_test, y_test, models, test_df=None):
        """하이브리드 모델 평가"""
        print("📊 Evaluating hybrid models...")
        
        results = {}
        
        for model_name, model_info in models.items():
            print(f"   🔍 Evaluating {model_name}...")
            
            # 예측
            X_test_scaled = model_info['scaler'].transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            y_pred = model_info['model'].predict(X_test_scaled, verbose=0).flatten()
            
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
    
    def run_advanced_experiment(self):
        """고급 실험 실행"""
        print("🚀 Starting Advanced Integration Models Experiment")
        print("=" * 80)
        
        # 데이터 로드
        df = self.load_data()
        
        # 특성 준비
        price_features, basic_reddit, advanced_reddit = self.prepare_feature_sets()
        
        # 시계열 분할
        train_df, val_df, test_df = self.strict_time_series_split()
        
        # 실험할 특성 세트들
        feature_sets = {
            'basic_reddit': price_features + basic_reddit,
            'advanced_reddit': price_features + advanced_reddit,
            'all_reddit': price_features + basic_reddit + advanced_reddit
        }
        
        # 결과 저장
        all_results = {}
        
        for feature_set_name, all_features in feature_sets.items():
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {feature_set_name.upper()}")
            print(f"{'='*60}")
            
            # 하이브리드 시퀀스 데이터 준비
            X_train, y_train = self.prepare_hybrid_sequence_data(train_df, price_features, all_features[len(price_features):])
            X_val, y_val = self.prepare_hybrid_sequence_data(val_df, price_features, all_features[len(price_features):])
            X_test, y_test = self.prepare_hybrid_sequence_data(test_df, price_features, all_features[len(price_features):])
            
            # 하이브리드 모델 훈련
            hybrid_models = self.train_hybrid_models(X_train, y_train, X_val, y_val)
            
            # 하이브리드 모델 평가
            hybrid_results = self.evaluate_hybrid_models(X_test, y_test, hybrid_models, test_df)
            
            all_results[feature_set_name] = hybrid_results
        
        self.results = all_results
        return all_results
    
    def create_comparison_table(self):
        """비교 표 생성"""
        print("📋 Creating comparison table...")
        
        comparison_data = []
        
        for feature_set_name, results in self.results.items():
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Feature_Set': feature_set_name,
                    'Model': model_name,
                    'IC': f"{metrics['IC']:.4f}",
                    'Hit_Rate': f"{metrics['Hit_Rate']:.4f}",
                    'ICIR': f"{metrics['ICIR']:.4f}",
                    'QSR': f"{metrics['QSR']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*100)
        print("ADVANCED INTEGRATION MODELS: 4 KEY INDICATORS")
        print("="*100)
        print(comparison_df.to_string(index=False))
        print("="*100)
        
        return comparison_df
    
    def create_visualization(self):
        """시각화 생성"""
        print("📈 Creating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Integration Models: 4 Key Indicators', 
                     fontsize=16, fontweight='bold')
        
        # 데이터 준비
        feature_sets = list(self.results.keys())
        model_names = list(self.results[feature_sets[0]].keys())
        
        # IC 비교
        ax1 = axes[0, 0]
        ic_data = []
        for feature_set in feature_sets:
            ic_values = [self.results[feature_set][model]['IC'] for model in model_names]
            ic_data.append(ic_values)
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, (feature_set, ic_values) in enumerate(zip(feature_sets, ic_data)):
            ax1.bar(x + i*width, ic_values, width, label=feature_set, alpha=0.8)
        
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
        for feature_set in feature_sets:
            hit_rate_values = [self.results[feature_set][model]['Hit_Rate'] for model in model_names]
            hit_rate_data.append(hit_rate_values)
        
        for i, (feature_set, hit_rate_values) in enumerate(zip(feature_sets, hit_rate_data)):
            ax2.bar(x + i*width, hit_rate_values, width, label=feature_set, alpha=0.8)
        
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
        for feature_set in feature_sets:
            icir_values = [self.results[feature_set][model]['ICIR'] for model in model_names]
            icir_data.append(icir_values)
        
        for i, (feature_set, icir_values) in enumerate(zip(feature_sets, icir_data)):
            ax3.bar(x + i*width, icir_values, width, label=feature_set, alpha=0.8)
        
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
        for feature_set in feature_sets:
            qsr_values = [self.results[feature_set][model]['QSR'] for model in model_names]
            qsr_data.append(qsr_values)
        
        for i, (feature_set, qsr_values) in enumerate(zip(feature_sets, qsr_data)):
            ax4.bar(x + i*width, qsr_values, width, label=feature_set, alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('QSR')
        ax4.set_title('Quintile Spread Return', fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/advanced_integration_models.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Visualization saved to results/advanced_integration_models.png")
    
    def generate_report(self, comparison_df):
        """리포트 생성"""
        print("📝 Generating report...")
        
        report = []
        report.append("=" * 150)
        report.append("ADVANCED INTEGRATION MODELS: 4 KEY INDICATORS")
        report.append("=" * 150)
        report.append("")
        report.append("Experiment Design:")
        report.append("- Target Stocks: AMC, BB, GME (Meme Stocks Only)")
        report.append("- Models: Hybrid LSTM, GRU, CNN-LSTM, Transformer, Hierarchical Integration")
        report.append("- Features: Price + Basic Reddit, Price + Advanced Reddit, Price + All Reddit")
        report.append("- Architecture: Advanced Hybrid with Hierarchical Integration")
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
        
        for feature_set_name, results in self.results.items():
            report.append(f"\n{feature_set_name.upper()} RESULTS:")
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
            for feature_set_name, results in self.results.items():
                values = [results[model][metric] for model in results.keys()]
                avg_metrics[metric][feature_set_name] = np.mean(values)
        
        report.append("Average Performance Across All Models:")
        for metric, feature_sets in avg_metrics.items():
            report.append(f"\n{metric}:")
            for feature_set, avg_value in feature_sets.items():
                report.append(f"  {feature_set}: {avg_value:.4f}")
        
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
        with open('results/advanced_integration_models_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Report saved to results/advanced_integration_models_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Advanced Integration Models Analysis")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = AdvancedIntegrationModels()
    
    # 1. 고급 실험 실행
    results = experiment.run_advanced_experiment()
    
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
    
    print("\n🎉 Advanced integration models analysis completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df

if __name__ == "__main__":
    experiment, comparison_df = main()
