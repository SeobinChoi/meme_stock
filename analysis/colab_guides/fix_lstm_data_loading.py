#!/usr/bin/env python3
"""
🚀 LSTM 데이터 로딩 문제 해결 스크립트
A100 GPU 환경에서 안정적으로 실행되도록 최적화
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Optional

def fix_sequence_data_loading(sequences_data: np.lib.npyio.NpzFile, 
                             metadata: dict) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    시퀀스 데이터 로딩 문제를 해결하는 강화된 함수
    
    Args:
        sequences_data: np.load()로 로드된 .npz 파일
        metadata: 데이터셋 메타데이터
        
    Returns:
        X_seq: (samples, timesteps, features) 형태의 시퀀스 데이터
        y_seq: 타겟 값들
        dates_seq: 날짜 정보
    """
    
    print("🔧 Fixing sequence data loading...")
    
    all_sequences = []
    all_targets = []
    all_dates = []
    
    # 각 티커별로 데이터 처리
    for ticker in metadata['tickers']:
        print(f"   Processing {ticker}...")
        
        if f'{ticker}_sequences' in sequences_data:
            sequences = sequences_data[f'{ticker}_sequences']
            targets = sequences_data[f'{ticker}_targets_1d']
            dates = sequences_data[f'{ticker}_dates']
            
            print(f"     Raw shape: {sequences.shape}, dtype: {sequences.dtype}")
            
            # 문자열 데이터 타입 문제 해결
            if sequences.dtype == object:
                print(f"     ⚠️ Object dtype detected, cleaning...")
                sequences = clean_object_sequences(sequences, ticker)
                
                if sequences is None:
                    print(f"     ❌ Skipping {ticker} - no valid numeric data")
                    continue
            
            # NaN/Inf 값 정리
            sequences = clean_nan_inf(sequences)
            
            # float32로 변환 (A100 최적화)
            sequences = sequences.astype(np.float32)
            
            all_sequences.append(sequences)
            all_targets.extend(targets)
            all_dates.extend(dates)
            
            print(f"     ✅ Cleaned: {sequences.shape}, dtype: {sequences.dtype}")
        else:
            print(f"     ⚠️ {ticker} sequences not found")
    
    if not all_sequences:
        raise ValueError("❌ No valid sequences found after cleaning!")
    
    # 모든 시퀀스 결합
    X_seq = np.vstack(all_sequences).astype(np.float32)
    y_seq = np.array(all_targets, dtype=np.float32)
    
    print(f"✅ Final data prepared:")
    print(f"   X_seq: {X_seq.shape}, dtype: {X_seq.dtype}")
    print(f"   y_seq: {y_seq.shape}, dtype: {y_seq.dtype}")
    
    return X_seq, y_seq, all_dates

def clean_object_sequences(sequences: np.ndarray, ticker: str) -> Optional[np.ndarray]:
    """
    object dtype 시퀀스에서 숫자 컬럼만 추출
    """
    
    print(f"     🔍 Cleaning {ticker} sequences...")
    
    # 각 컬럼을 float로 변환 시도
    numeric_cols = []
    cleaned_sequences = []
    
    for i in range(sequences.shape[2]):
        try:
            # 해당 컬럼을 float로 변환
            test_col = sequences[:, :, i].astype(np.float64)
            
            # NaN이 아닌 값이 있는지 확인
            if np.any(np.isfinite(test_col)):
                numeric_cols.append(i)
                cleaned_sequences.append(test_col)
            else:
                print(f"       Column {i}: all NaN, skipping")
                
        except (ValueError, TypeError) as e:
            print(f"       Column {i}: conversion failed ({e}), skipping")
            continue
    
    if len(numeric_cols) == 0:
        print(f"     ❌ No numeric columns found in {ticker}")
        return None
    
    # 유효한 컬럼만 선택하여 3D 배열 재구성
    cleaned = np.stack(cleaned_sequences, axis=2)
    
    print(f"       Kept {len(numeric_cols)}/{sequences.shape[2]} columns")
    print(f"       New shape: {cleaned.shape}")
    
    return cleaned

def clean_nan_inf(sequences: np.ndarray) -> np.ndarray:
    """
    NaN과 Inf 값을 안전하게 처리
    """
    
    # NaN을 0으로, Inf를 0으로 변환
    cleaned = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 극단값 클리핑 (표준편차의 5배)
    for i in range(cleaned.shape[2]):
        col_data = cleaned[:, :, i]
        if np.std(col_data) > 0:
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            lower_bound = mean_val - 5 * std_val
            upper_bound = mean_val + 5 * std_val
            col_data = np.clip(col_data, lower_bound, upper_bound)
            cleaned[:, :, i] = col_data
    
    return cleaned

def validate_sequence_dimensions(X_seq: np.ndarray, y_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    시퀀스 차원과 데이터 품질 검증
    """
    
    print("🔍 Validating sequence dimensions...")
    
    # 차원 검증
    if len(X_seq.shape) != 3:
        raise ValueError(f"❌ Expected 3D array, got {len(X_seq.shape)}D: {X_seq.shape}")
    
    if X_seq.shape[0] != len(y_seq):
        raise ValueError(f"❌ Sample count mismatch: X={X_seq.shape[0]}, y={len(y_seq)}")
    
    # 데이터 타입 검증
    if not np.issubdtype(X_seq.dtype, np.floating):
        raise ValueError(f"❌ X_seq must be float, got {X_seq.dtype}")
    
    if not np.issubdtype(y_seq.dtype, np.floating):
        raise ValueError(f"❌ y_seq must be float, got {y_seq.dtype}")
    
    # NaN/Inf 최종 검사
    if np.any(np.isnan(X_seq)) or np.any(np.isinf(X_seq)):
        print("⚠️ Final cleanup: removing remaining NaN/Inf...")
        X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("✅ Sequence validation passed!")
    print(f"   Final X_seq: {X_seq.shape}, dtype: {X_seq.dtype}")
    print(f"   Final y_seq: {y_seq.shape}, dtype: {y_seq.dtype}")
    
    return X_seq, y_seq

def prepare_train_val_test_split(X_seq: np.ndarray, y_seq: np.ndarray, 
                                dates_seq: List, train_ratio: float = 0.7, 
                                val_ratio: float = 0.15) -> Tuple:
    """
    훈련/검증/테스트 세트 분할
    """
    
    print("📊 Preparing train/val/test split...")
    
    # 날짜 기반 분할
    dates = pd.to_datetime(dates_seq)
    
    # 날짜 정렬
    sorted_indices = np.argsort(dates)
    X_sorted = X_seq[sorted_indices]
    y_sorted = y_seq[sorted_indices]
    dates_sorted = dates[sorted_indices]
    
    # 분할 지점 계산
    n_samples = len(X_sorted)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # 분할
    X_train = X_sorted[:train_end]
    X_val = X_sorted[train_end:val_end]
    X_test = X_sorted[val_end:]
    
    y_train = y_sorted[:train_end]
    y_val = y_sorted[train_end:val_end]
    y_test = y_sorted[val_end:]
    
    dates_train = dates_sorted[:train_end]
    dates_val = dates_sorted[train_end:val_end]
    dates_test = dates_sorted[val_end:]
    
    print(f"✅ Split completed:")
    print(f"   Train: {X_train.shape[0]} samples ({dates_train.min().date()} to {dates_train.max().date()})")
    print(f"   Val: {X_val.shape[0]} samples ({dates_val.min().date()} to {dates_val.max().date()})")
    print(f"   Test: {X_test.shape[0]} samples ({dates_test.min().date()} to {dates_test.max().date()})")
    
    return (X_train, X_val, X_test, 
            y_train, y_val, y_test,
            dates_train, dates_val, dates_test)

def main():
    """
    메인 실행 함수 (테스트용)
    """
    print("🚀 LSTM Data Loading Fix Script")
    print("=" * 50)
    
    # 사용 예시
    print("📖 Usage:")
    print("1. Import this script in your Colab notebook")
    print("2. Replace prepare_sequence_data() with fix_sequence_data_loading()")
    print("3. Add validation and error handling")
    
    print("\n🔧 Key fixes implemented:")
    print("   ✅ Object dtype string removal")
    print("   ✅ NaN/Inf cleaning")
    print("   ✅ Dimension validation")
    print("   ✅ A100 GPU optimization (float32)")
    print("   ✅ Robust error handling")


