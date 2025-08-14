#!/usr/bin/env python3
"""
🚀 Meme Stock Price Prediction with Deep Learning (A100 GPU Optimized)

Fixed version for Colab A100 GPU with robust LSTM data loading
"""

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages for A100 GPU"""
    packages = [
        "pytorch-tabnet",
        "transformers", 
        "optuna",
        "plotly",
        "seaborn"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except:
            print(f"⚠️ Failed to install {package}")

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, classification_report
from scipy.stats import spearmanr, pearsonr
import optuna
from pytorch_tabnet.tab_model import TabNetRegressor

# A100 GPU 최적화
from torch.cuda.amp import autocast, GradScaler

def setup_a100_optimization():
    """A100 GPU 최적화 설정"""
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # A100 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 메모리 효율성 (A100 메모리 40GB+ 활용)
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
    else:
        print("💻 Using CPU")
    
    return device

def upload_data_colab():
    """Colab에서 데이터 업로드"""
    try:
        from google.colab import files
        
        print("📤 Upload the following files from your local machine:")
        print("   - tabular_train_YYYYMMDD_HHMMSS.csv")
        print("   - tabular_val_YYYYMMDD_HHMMSS.csv") 
        print("   - tabular_test_YYYYMMDD_HHMMSS.csv")
        print("   - sequences_YYYYMMDD_HHMMSS.npz")
        print("   - dataset_metadata_YYYYMMDD_HHMMSS.json")
        
        uploaded = files.upload()
        
        # Show uploaded files
        import os
        print("\n📁 Uploaded files:")
        for filename in os.listdir('.'):
            if any(filename.startswith(prefix) for prefix in ['tabular_', 'sequences_', 'dataset_']):
                print(f"   {filename}")
                
        return True
        
    except ImportError:
        print("⚠️ Not running in Colab - skipping file upload")
        return False

def load_data_robust():
    """강화된 데이터 로딩 (오류 처리 포함)"""
    
    import json
    import glob
    
    try:
        # Find metadata file
        metadata_files = glob.glob('dataset_metadata_*.json')
        if not metadata_files:
            raise FileNotFoundError("No metadata file found!")
        
        metadata_file = metadata_files[0]
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        timestamp = metadata['timestamp']
        print(f"📊 Loading datasets with timestamp: {timestamp}")
        
        # Load tabular data
        train_df = pd.read_csv(f'tabular_train_{timestamp}.csv')
        val_df = pd.read_csv(f'tabular_val_{timestamp}.csv')
        test_df = pd.read_csv(f'tabular_test_{timestamp}.csv')
        
        # Convert dates
        train_df['date'] = pd.to_datetime(train_df['date'])
        val_df['date'] = pd.to_datetime(val_df['date'])
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        print(f"\n📈 Tabular data loaded successfully!")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Validation: {len(val_df)} samples")
        print(f"   Test: {len(test_df)} samples")
        print(f"   Features: {len(metadata['tabular_features'])}")
        
        return train_df, val_df, test_df, metadata
        
    except Exception as e:
        print(f"❌ Error loading tabular data: {e}")
        raise

def prepare_sequence_data_fixed(metadata):
    """A100 GPU 최적화된 시퀀스 데이터 준비 (문자열 오류 해결)"""
    
    try:
        # Load sequence data
        timestamp = metadata['timestamp']
        sequences_data = np.load(f'sequences_{timestamp}.npz')
        
        print(f"🔍 Loading sequence data: {timestamp}")
        
        all_sequences = []
        all_targets = []
        all_dates = []
        
        # 데이터 타입 강제 변환 및 문자열 제거
        for ticker in metadata['tickers']:
            if f'{ticker}_sequences' in sequences_data:
                sequences = sequences_data[f'{ticker}_sequences']
                targets = sequences_data[f'{ticker}_targets_1d']
                dates = sequences_data[f'{ticker}_dates']
                
                print(f"   Processing {ticker}: {sequences.shape}, dtype: {sequences.dtype}")
                
                # 문자열이 포함된 경우 숫자 컬럼만 선택
                if sequences.dtype == object:
                    print(f"   ⚠️ {ticker} has object dtype, cleaning...")
                    
                    numeric_cols = []
                    for i in range(sequences.shape[2]):
                        try:
                            # 각 컬럼을 float로 변환 시도
                            test_col = sequences[:, :, i].astype(float)
                            numeric_cols.append(i)
                        except:
                            continue
                    
                    if len(numeric_cols) > 0:
                        sequences = sequences[:, :, numeric_cols].astype(np.float32)
                        print(f"   ✅ {ticker}: {len(numeric_cols)} numeric columns extracted")
                    else:
                        print(f"   ❌ {ticker}: No numeric columns found, skipping")
                        continue
                else:
                    sequences = sequences.astype(np.float32)
                
                # NaN 값 처리
                if np.any(np.isnan(sequences)) or np.any(np.isinf(sequences)):
                    print(f"   🧹 {ticker}: Cleaning NaN/Inf values...")
                    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
                
                all_sequences.append(sequences)
                all_targets.extend(targets)
                all_dates.extend(dates)
        
        if not all_sequences:
            raise ValueError("❌ No valid numeric sequences found!")
        
        # A100 최적화: float32 사용
        X_seq = np.vstack(all_sequences).astype(np.float32)
        y_seq = np.array(all_targets, dtype=np.float32)
        
        print(f"✅ Sequence data prepared: {X_seq.shape}, dtype: {X_seq.dtype}")
        print(f"   Total sequences: {len(all_sequences)}")
        print(f"   Total targets: {len(y_seq)}")
        
        return X_seq, y_seq, all_dates
        
    except Exception as e:
        print(f"❌ Error preparing sequence data: {e}")
        print("🔧 Fallback to tabular models only...")
        return None, None, None

def validate_sequence_dimensions(X_seq, y_seq):
    """시퀀스 차원 검증"""
    
    if X_seq is None or y_seq is None:
        return False
    
    print(f"🔍 Sequence validation:")
    print(f"   X shape: {X_seq.shape}")
    print(f"   y shape: {y_seq.shape}")
    print(f"   X dtype: {X_seq.dtype}")
    print(f"   y dtype: {y_seq.dtype}")
    
    # 차원 검증
    if len(X_seq.shape) != 3:
        print(f"❌ Expected 3D array, got {len(X_seq.shape)}D")
        return False
    
    if X_seq.shape[0] != len(y_seq):
        print(f"❌ Sample count mismatch: X={X_seq.shape[0]}, y={len(y_seq)}")
        return False
    
    # NaN/Inf 검사
    if np.any(np.isnan(X_seq)) or np.any(np.isinf(X_seq)):
        print("⚠️ Warning: NaN/Inf detected in sequences, cleaning...")
        X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("✅ Sequence validation passed!")
    return True

def create_train_val_test_split(X_seq, y_seq, dates_seq):
    """시퀀스 데이터를 훈련/검증/테스트로 분할"""
    
    if X_seq is None:
        return None, None, None, None, None, None
    
    # 날짜 기반 분할
    dates_array = np.array([pd.to_datetime(d) for d in dates_seq])
    
    train_end = pd.to_datetime('2023-02-02')
    val_end = pd.to_datetime('2023-07-15')
    
    train_mask = dates_array <= train_end
    val_mask = (dates_array > train_end) & (dates_array <= val_end)
    test_mask = dates_array > val_end
    
    X_train_seq = X_seq[train_mask]
    X_val_seq = X_seq[val_mask]
    X_test_seq = X_seq[test_mask]
    
    y_train_seq = y_seq[train_mask]
    y_val_seq = y_seq[val_mask]
    y_test_seq = y_seq[test_mask]
    
    print(f"📊 Sequence data split:")
    print(f"   Train: {X_train_seq.shape}")
    print(f"   Val: {X_val_seq.shape}")
    print(f"   Test: {X_test_seq.shape}")
    
    return X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq

def prepare_tabular_data(train_df, val_df, test_df, target='y1d'):
    """테이블 데이터 준비"""
    
    # Feature columns (exclude metadata and targets)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['date', 'ticker', 'ticker_type', 'y1d', 'y5d', 
                                 'alpha_1d', 'alpha_5d', 'direction_1d', 'direction_5d']]
    
    # Prepare features and targets
    X_train = train_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values  
    X_test = test_df[feature_cols].fillna(0).values
    
    y_train = train_df[target].values
    y_val = val_df[target].values
    y_test = test_df[target].values
    
    # Scale features
    scaler = RobustScaler()  # More robust to outliers
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, feature_cols, scaler)

def calculate_ic_metrics(y_true, y_pred):
    """Information Coefficient 메트릭 계산"""
    
    # Remove NaN values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {'ic': 0, 'rank_ic': 0, 'hit_rate': 0.5}
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Calculate correlations
    ic, ic_p = pearsonr(y_pred_clean, y_true_clean) if len(y_true_clean) > 2 else (0, 1)
    rank_ic, rank_p = spearmanr(y_pred_clean, y_true_clean)
    
    # Hit rate (directional accuracy)
    hit_rate = np.mean(np.sign(y_pred_clean) == np.sign(y_true_clean))
    
    return {
        'ic': ic if not np.isnan(ic) else 0,
        'rank_ic': rank_ic if not np.isnan(rank_ic) else 0,
        'ic_p_value': ic_p,
        'rank_ic_p_value': rank_p,
        'hit_rate': hit_rate,
        'n_samples': len(y_true_clean)
    }

def evaluate_model(model, X_test, y_test, model_name, device):
    """모델 평가"""
    
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            y_pred = model(X_tensor).cpu().numpy().flatten()
    
    # Calculate metrics
    ic_metrics = calculate_ic_metrics(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results = {
        'model': model_name,
        'rmse': rmse,
        **ic_metrics
    }
    
    return results, y_pred

# A100 최적화된 모델 클래스들
class DeepMLP(nn.Module):
    """A100 GPU 최적화된 Deep MLP"""
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128], dropout=0.3):
        super(DeepMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class LSTMModel(nn.Module):
    """A100 GPU 최적화된 LSTM"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

def train_mlp_a100(X_train, y_train, X_val, y_val, device, epochs=300, lr=0.001):
    """A100 GPU 최적화된 MLP 훈련"""
    
    model = DeepMLP(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # A100 최적화: 혼합 정밀도 훈련
    scaler = GradScaler()
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    train_losses = []
    val_losses = []
    val_ics = []
    
    best_ic = -float('inf')
    best_model = None
    patience_counter = 0
    
    print("🧠 Training MLP with A100 optimization...")
    
    for epoch in range(epochs):
        # Training with mixed precision
        model.train()
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(X_train_tensor).squeeze()
            train_loss = criterion(outputs, y_train_tensor)
        
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            
            # Calculate IC
            val_pred_np = val_outputs.cpu().numpy()
            val_ic_metrics = calculate_ic_metrics(y_val, val_pred_np)
            val_ic = val_ic_metrics['rank_ic']
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        val_ics.append(val_ic)
        
        scheduler.step(val_loss)
        
        # Early stopping based on IC
        if val_ic > best_ic:
            best_ic = val_ic
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IC: {val_ic:.4f}")
        
        if patience_counter >= 40:  # Early stopping
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model, train_losses, val_losses, val_ics

def train_lstm_a100(X_train, y_train, X_val, y_val, device, epochs=150, batch_size=256, lr=0.001):
    """A100 GPU 최적화된 LSTM 훈련"""
    
    input_size = X_train.shape[2]
    model = LSTMModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # A100 최적화: 혼합 정밀도 훈련
    scaler = GradScaler()
    
    # Create data loaders with larger batch size
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    train_losses = []
    val_ics = []
    best_ic = -float('inf')
    best_model = None
    
    print(f"🔄 Training LSTM with A100 optimization (batch_size={batch_size})...")
    
    for epoch in range(epochs):
        # Training with mixed precision
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X).squeeze()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_actuals.extend(batch_y.numpy())
        
        val_ic_metrics = calculate_ic_metrics(np.array(val_actuals), np.array(val_predictions))
        val_ic = val_ic_metrics['rank_ic']
        
        train_losses.append(train_loss / len(train_loader))
        val_ics.append(val_ic)
        
        if val_ic > best_ic:
            best_ic = val_ic
            best_model = model.state_dict().copy()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val IC: {val_ic:.4f}")
    
    # Load best model
    model.load_state_dict(best_model)
    return model, train_losses, val_ics

def main():
    """메인 실행 함수"""
    
    print("🚀 Starting Meme Stock Deep Learning (A100 GPU Optimized)")
    
    # 1. A100 GPU 최적화 설정
    device = setup_a100_optimization()
    
    # 2. 패키지 설치 (Colab에서만)
    if 'google.colab' in sys.modules:
        install_packages()
    
    # 3. 데이터 업로드 (Colab에서만)
    if 'google.colab' in sys.modules:
        upload_data_colab()
    
    # 4. 데이터 로딩
    try:
        train_df, val_df, test_df, metadata = load_data_robust()
        print("✅ Tabular data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tabular data: {e}")
        return
    
    # 5. 시퀀스 데이터 준비 (오류 처리 포함)
    try:
        X_seq, y_seq, dates_seq = prepare_sequence_data_fixed(metadata)
        
        if X_seq is not None:
            # 차원 검증
            if validate_sequence_dimensions(X_seq, y_seq):
                # 데이터 분할
                X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = create_train_val_test_split(
                    X_seq, y_seq, dates_seq
                )
                USE_SEQUENCE_MODELS = True
                print("✅ Sequence models enabled")
            else:
                USE_SEQUENCE_MODELS = False
                print("⚠️ Sequence models disabled due to validation failure")
        else:
            USE_SEQUENCE_MODELS = False
            print("⚠️ Sequence models disabled - no sequence data")
            
    except Exception as e:
        print(f"⚠️ Sequence data preparation failed: {e}")
        USE_SEQUENCE_MODELS = False
        print("⚠️ Sequence models disabled")
    
    # 6. 테이블 데이터 준비
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler = prepare_tabular_data(
        train_df, val_df, test_df, target='y1d'
    )
    
    print(f"📊 Tabular data prepared: {X_train.shape[1]} features")
    
    # 7. 모델 훈련 및 평가
    results = []
    
    # MLP 모델
    try:
        print("\n🧠 Training MLP...")
        mlp_model, mlp_train_losses, mlp_val_losses, mlp_val_ics = train_mlp_a100(
            X_train, y_train, X_val, y_val, device
        )
        
        mlp_results, mlp_predictions = evaluate_model(mlp_model, X_test, y_test, 'MLP', device)
        results.append(mlp_results)
        
        print(f"✅ MLP Results: IC={mlp_results['ic']:.4f}, Rank IC={mlp_results['rank_ic']:.4f}")
        
    except Exception as e:
        print(f"❌ MLP training failed: {e}")
    
    # LSTM 모델 (시퀀스 데이터가 있는 경우)
    if USE_SEQUENCE_MODELS and X_train_seq is not None:
        try:
            print("\n🔄 Training LSTM...")
            lstm_model, lstm_train_losses, lstm_val_ics = train_lstm_a100(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, device
            )
            
            # LSTM 평가
            lstm_model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
                lstm_predictions = lstm_model(X_test_tensor).cpu().numpy().flatten()
            
            lstm_results = calculate_ic_metrics(y_test_seq, lstm_predictions)
            lstm_results['model'] = 'LSTM'
            lstm_results['rmse'] = np.sqrt(mean_squared_error(y_test_seq, lstm_predictions))
            
            results.append(lstm_results)
            print(f"✅ LSTM Results: IC={lstm_results['ic']:.4f}, Rank IC={lstm_results['rank_ic']:.4f}")
            
        except Exception as e:
            print(f"❌ LSTM training failed: {e}")
    
    # 8. 결과 요약
    if results:
        print(f"\n🏆 FINAL RESULTS SUMMARY")
        print("=" * 50)
        
        for result in results:
            print(f"{result['model']}: IC={result['ic']:.4f}, Rank IC={result['rank_ic']:.4f}, Hit Rate={result['hit_rate']:.3%}")
        
        # 최고 성능 모델 찾기
        best_result = max(results, key=lambda x: x['rank_ic'])
        print(f"\n🥇 BEST MODEL: {best_result['model']}")
        print(f"   Rank IC: {best_result['rank_ic']:.4f}")
        print(f"   Hit Rate: {best_result['hit_rate']:.3%}")
        
        # Go/No-Go 판정
        ic_improvement = best_result['rank_ic'] - 0.0  # Random walk baseline
        meets_threshold = ic_improvement >= 0.03 and best_result['hit_rate'] > 0.55
        
        if meets_threshold:
            print(f"\n🚀 GO DECISION: Model meets success criteria!")
        else:
            print(f"\n🔄 CONTINUE: Model close to threshold but needs improvement")
            
    else:
        print("❌ No models trained successfully")
    
    print("\n✅ Deep Learning pipeline completed!")

if __name__ == "__main__":
    main()
