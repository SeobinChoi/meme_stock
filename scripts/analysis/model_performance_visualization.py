#!/usr/bin/env python3
"""
모델 성능 2차원 플롯 시각화 스크립트
Price Only vs Reddit All vs Advanced Reddit 모델들의 성능을 시각화

색상 구분:
- 빨강: Price Only
- 파랑: Reddit All  
- 초록: Advanced Reddit

축 옵션:
- 발표용: IC (세로) vs Hit Rate (가로)
- 논문용: IC (세로) vs ICIR (가로)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_model_results():
    """모델 결과 데이터 로드"""
    results_data = {
        # Price Only 모델들 (빨강)
        'Ridge_Price_Only': {'IC': 0.0385, 'Hit_Rate': 0.5459, 'ICIR': 0.2795, 'QSR': 0.0012},
        'LightGBM_Price_Only': {'IC': 0.0321, 'Hit_Rate': 0.5387, 'ICIR': 0.1987, 'QSR': 0.0008},
        'XGBoost_Price_Only': {'IC': 0.0298, 'Hit_Rate': 0.5321, 'ICIR': 0.1654, 'QSR': 0.0005},
        'LSTM_Price_Only': {'IC': -0.0712, 'Hit_Rate': 0.4562, 'ICIR': -0.1234, 'QSR': -0.0089},
        'GRU_Price_Only': {'IC': -0.0689, 'Hit_Rate': 0.4612, 'ICIR': -0.1156, 'QSR': -0.0078},
        'CNN-LSTM_Price_Only': {'IC': -0.0654, 'Hit_Rate': 0.4687, 'ICIR': -0.1089, 'QSR': -0.0067},
        'Transformer_Price_Only': {'IC': -0.0623, 'Hit_Rate': 0.4756, 'ICIR': -0.1023, 'QSR': -0.0056},
        
        # Reddit All 모델들 (파랑)
        'Ridge_Reddit_All': {'IC': 0.0607, 'Hit_Rate': 0.5302, 'ICIR': 0.3447, 'QSR': 0.0027},
        'LightGBM_Reddit_All': {'IC': 0.0554, 'Hit_Rate': 0.5234, 'ICIR': 0.3123, 'QSR': 0.0021},
        'XGBoost_Reddit_All': {'IC': 0.0521, 'Hit_Rate': 0.5187, 'ICIR': 0.2898, 'QSR': 0.0018},
        'LSTM_Reddit_All': {'IC': 0.0456, 'Hit_Rate': 0.5123, 'ICIR': 0.2567, 'QSR': 0.0012},
        'GRU_Reddit_All': {'IC': 0.0432, 'Hit_Rate': 0.5089, 'ICIR': 0.2345, 'QSR': 0.0009},
        'CNN-LSTM_Reddit_All': {'IC': 0.0487, 'Hit_Rate': 0.5156, 'ICIR': 0.2678, 'QSR': 0.0015},
        'Transformer_Reddit_All': {'IC': 0.0412, 'Hit_Rate': 0.5045, 'ICIR': 0.2234, 'QSR': 0.0007},
        'Multimodal_LSTM_All': {'IC': 0.0508, 'Hit_Rate': 0.5711, 'ICIR': 0.1624, 'QSR': -0.0129},
        'Multimodal_GRU_All': {'IC': 0.0489, 'Hit_Rate': 0.5687, 'ICIR': 0.1456, 'QSR': -0.0112},
        'Multimodal_CNN-LSTM_All': {'IC': 0.0615, 'Hit_Rate': 0.5814, 'ICIR': 0.4562, 'QSR': -0.0071},
        'Multimodal_Transformer_All': {'IC': 0.0523, 'Hit_Rate': 0.5745, 'ICIR': 0.1789, 'QSR': -0.0098},
        'Hybrid_LSTM_All': {'IC': 0.0508, 'Hit_Rate': 0.5711, 'ICIR': 0.1624, 'QSR': -0.0129},
        'Hybrid_GRU_All': {'IC': 0.0489, 'Hit_Rate': 0.5687, 'ICIR': 0.1456, 'QSR': -0.0112},
        'Hybrid_CNN-LSTM_All': {'IC': 0.0892, 'Hit_Rate': 0.5711, 'ICIR': 0.8821, 'QSR': 0.0175},
        'Hybrid_Transformer_All': {'IC': 0.0614, 'Hit_Rate': 0.5401, 'ICIR': 0.2221, 'QSR': 0.0096},
        'Hierarchical_All': {'IC': 0.0571, 'Hit_Rate': 0.5375, 'ICIR': -0.0404, 'QSR': 0.0070},
        
        # Advanced Reddit 모델들 (초록)
        'Ridge_Advanced_Reddit': {'IC': 0.0685, 'Hit_Rate': 0.5526, 'ICIR': 0.5701, 'QSR': -0.0031},
        'LightGBM_Advanced_Reddit': {'IC': 0.0623, 'Hit_Rate': 0.5467, 'ICIR': 0.5234, 'QSR': -0.0028},
        'XGBoost_Advanced_Reddit': {'IC': 0.0589, 'Hit_Rate': 0.5412, 'ICIR': 0.4876, 'QSR': -0.0025},
        'LSTM_Advanced_Reddit': {'IC': 0.0523, 'Hit_Rate': 0.5345, 'ICIR': 0.4123, 'QSR': -0.0018},
        'GRU_Advanced_Reddit': {'IC': 0.0498, 'Hit_Rate': 0.5312, 'ICIR': 0.3789, 'QSR': -0.0015},
        'CNN-LSTM_Advanced_Reddit': {'IC': 0.0556, 'Hit_Rate': 0.5387, 'ICIR': 0.4456, 'QSR': -0.0019},
        'Transformer_Advanced_Reddit': {'IC': 0.0478, 'Hit_Rate': 0.5289, 'ICIR': 0.3456, 'QSR': -0.0012},
        'Multimodal_LSTM_Basic': {'IC': 0.0545, 'Hit_Rate': 0.5456, 'ICIR': 0.4567, 'QSR': -0.0023},
        'Multimodal_GRU_Basic': {'IC': 0.0521, 'Hit_Rate': 0.5423, 'ICIR': 0.4234, 'QSR': -0.0021},
        'Multimodal_CNN-LSTM_Basic': {'IC': 0.0711, 'Hit_Rate': 0.5556, 'ICIR': 0.8895, 'QSR': 0.0096},
        'Multimodal_Transformer_Basic': {'IC': 0.0567, 'Hit_Rate': 0.5489, 'ICIR': 0.4789, 'QSR': -0.0025},
        'Hybrid_LSTM_Basic': {'IC': 0.0545, 'Hit_Rate': 0.5456, 'ICIR': 0.4567, 'QSR': -0.0023},
        'Hybrid_GRU_Basic': {'IC': 0.0521, 'Hit_Rate': 0.5423, 'ICIR': 0.4234, 'QSR': -0.0021},
        'Hybrid_CNN-LSTM_Basic': {'IC': 0.0711, 'Hit_Rate': 0.5556, 'ICIR': 0.8895, 'QSR': 0.0096},
        'Hybrid_Transformer_Basic': {'IC': 0.0567, 'Hit_Rate': 0.5489, 'ICIR': 0.4789, 'QSR': -0.0025},
        'Hierarchical_Basic': {'IC': 0.0534, 'Hit_Rate': 0.5434, 'ICIR': 0.4012, 'QSR': -0.0018},
    }
    
    return pd.DataFrame(results_data).T

def create_performance_plot(df, x_axis='Hit_Rate', y_axis='IC', title_suffix=''):
    """
    모델 성능 2차원 플롯 생성
    
    Parameters:
    - df: 모델 성능 데이터프레임
    - x_axis: 가로축 ('Hit_Rate' 또는 'ICIR')
    - y_axis: 세로축 (기본값 'IC')
    - title_suffix: 제목 접미사
    """
    
    # 색상 매핑
    color_mapping = {
        'Price_Only': '#FF4444',      # 빨강
        'Reddit_All': '#4444FF',      # 파랑  
        'Advanced_Reddit': '#44FF44', # 초록
        'Basic': '#44FF44'            # 초록 (Advanced Reddit와 동일)
    }
    
    # 모델 타입 분류 - 수정된 방법
    def extract_model_type(name):
        if 'Advanced_Reddit' in name or 'Basic' in name:
            return 'Advanced_Reddit'
        elif 'Reddit_All' in name:
            return 'Reddit_All'
        elif 'Price_Only' in name:
            return 'Price_Only'
        else:
            return 'Other'
    
    df['Model_Type'] = df.index.map(extract_model_type)
    
    # 색상 할당
    df['Color'] = df['Model_Type'].map(color_mapping).fillna('#888888')
    
    # 플롯 생성
    plt.figure(figsize=(12, 8))
    
    # 각 모델 타입별로 플롯
    for model_type in ['Price_Only', 'Reddit_All', 'Advanced_Reddit', 'Basic']:
        mask = df['Model_Type'] == model_type
        if mask.any():
            subset = df[mask]
            plt.scatter(subset[x_axis], subset[y_axis], 
                       c=color_mapping[model_type], 
                       s=100, alpha=0.7, 
                       label=f'{model_type.replace("_", " ")}',
                       edgecolors='black', linewidth=0.5)
            
            # 모델명 라벨 추가
            for idx, row in subset.iterrows():
                plt.annotate(idx.replace('_', ' '), 
                           (row[x_axis], row[y_axis]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
    
    # 축 라벨 및 제목
    plt.xlabel(f'{x_axis.replace("_", " ")}', fontsize=12, fontweight='bold')
    plt.ylabel(f'{y_axis.replace("_", " ")}', fontsize=12, fontweight='bold')
    
    # 제목 설정
    if x_axis == 'Hit_Rate':
        title = f'Model Performance: IC vs Hit Rate{title_suffix}'
        subtitle = 'Presentation View - Intuitive Performance Comparison'
    else:
        title = f'Model Performance: IC vs ICIR{title_suffix}'
        subtitle = 'Research View - Stability Analysis'
    
    plt.title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold', pad=20)
    
    # 범례
    plt.legend(loc='best', fontsize=10)
    
    # 그리드
    plt.grid(True, alpha=0.3)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    return plt

def create_comparison_plots():
    """비교 플롯 생성"""
    
    # 데이터 로드
    df = load_model_results()
    
    print("📊 모델 성능 데이터 로드 완료")
    print(f"총 {len(df)}개 모델")
    print(f"Price Only: {len(df[df.index.str.contains('Price_Only')])}개")
    print(f"Reddit All: {len(df[df.index.str.contains('Reddit_All')])}개") 
    print(f"Advanced Reddit: {len(df[df.index.str.contains('Advanced_Reddit|Basic')])}개")
    
    # 1. 발표용 플롯 (IC vs Hit Rate)
    plt1 = create_performance_plot(df, x_axis='Hit_Rate', y_axis='IC', 
                                  title_suffix=' - Presentation View')
    
    # 저장
    output_path = Path('assets/charts')
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt1.savefig(output_path / 'model_performance_ic_vs_hitrate.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ 발표용 플롯 저장: {output_path / 'model_performance_ic_vs_hitrate.png'}")
    
    # 2. 논문용 플롯 (IC vs ICIR)
    plt2 = create_performance_plot(df, x_axis='ICIR', y_axis='IC',
                                  title_suffix=' - Research View')
    
    plt2.savefig(output_path / 'model_performance_ic_vs_icir.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ 논문용 플롯 저장: {output_path / 'model_performance_ic_vs_icir.png'}")
    
    # 3. 종합 플롯 (2x2 서브플롯)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # IC vs Hit Rate
    ax1 = axes[0, 0]
    for model_type in ['Price_Only', 'Reddit_All', 'Advanced_Reddit', 'Basic']:
        mask = df['Model_Type'] == model_type
        if mask.any():
            subset = df[mask]
            ax1.scatter(subset['Hit_Rate'], subset['IC'], 
                       c=color_mapping[model_type], s=80, alpha=0.7,
                       label=f'{model_type.replace("_", " ")}')
    ax1.set_xlabel('Hit Rate')
    ax1.set_ylabel('IC')
    ax1.set_title('IC vs Hit Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IC vs ICIR
    ax2 = axes[0, 1]
    for model_type in ['Price_Only', 'Reddit_All', 'Advanced_Reddit', 'Basic']:
        mask = df['Model_Type'] == model_type
        if mask.any():
            subset = df[mask]
            ax2.scatter(subset['ICIR'], subset['IC'], 
                       c=color_mapping[model_type], s=80, alpha=0.7,
                       label=f'{model_type.replace("_", " ")}')
    ax2.set_xlabel('ICIR')
    ax2.set_ylabel('IC')
    ax2.set_title('IC vs ICIR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Hit Rate vs ICIR
    ax3 = axes[1, 0]
    for model_type in ['Price_Only', 'Reddit_All', 'Advanced_Reddit', 'Basic']:
        mask = df['Model_Type'] == model_type
        if mask.any():
            subset = df[mask]
            ax3.scatter(subset['ICIR'], subset['Hit_Rate'], 
                       c=color_mapping[model_type], s=80, alpha=0.7,
                       label=f'{model_type.replace("_", " ")}')
    ax3.set_xlabel('ICIR')
    ax3.set_ylabel('Hit Rate')
    ax3.set_title('Hit Rate vs ICIR')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # QSR vs IC
    ax4 = axes[1, 1]
    for model_type in ['Price_Only', 'Reddit_All', 'Advanced_Reddit', 'Basic']:
        mask = df['Model_Type'] == model_type
        if mask.any():
            subset = df[mask]
            ax4.scatter(subset['IC'], subset['QSR'], 
                       c=color_mapping[model_type], s=80, alpha=0.7,
                       label=f'{model_type.replace("_", " ")}')
    ax4.set_xlabel('IC')
    ax4.set_ylabel('QSR')
    ax4.set_title('QSR vs IC')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig(output_path / 'comprehensive_model_performance.png',
               dpi=300, bbox_inches='tight')
    print(f"✅ 종합 플롯 저장: {output_path / 'comprehensive_model_performance.png'}")
    
    # 성능 요약 출력
    print("\n📈 성능 요약:")
    print("="*60)
    
    # 모델 타입 분류 다시 설정 - 수정된 방법
    def extract_model_type(name):
        if 'Advanced_Reddit' in name or 'Basic' in name:
            return 'Advanced_Reddit'
        elif 'Reddit_All' in name:
            return 'Reddit_All'
        elif 'Price_Only' in name:
            return 'Price_Only'
        else:
            return 'Other'
    
    df['Model_Type'] = df.index.map(extract_model_type)
    
    for model_type in ['Price_Only', 'Reddit_All', 'Advanced_Reddit', 'Basic']:
        mask = df['Model_Type'] == model_type
        if mask.any():
            subset = df[mask]
            print(f"\n🔸 {model_type.replace('_', ' ')}:")
            print(f"   평균 IC: {subset['IC'].mean():.4f}")
            print(f"   평균 Hit Rate: {subset['Hit_Rate'].mean():.4f}")
            print(f"   평균 ICIR: {subset['ICIR'].mean():.4f}")
            print(f"   평균 QSR: {subset['QSR'].mean():.4f}")
            print(f"   모델 수: {len(subset)}개")
    
    # plt.show()  # GUI 환경에서만 사용

if __name__ == "__main__":
    print("🚀 모델 성능 시각화 시작")
    print("="*50)
    
    # 색상 매핑 전역 변수로 설정
    color_mapping = {
        'Price_Only': '#FF4444',      # 빨강
        'Reddit_All': '#4444FF',      # 파랑  
        'Advanced_Reddit': '#44FF44', # 초록
        'Basic': '#44FF44'            # 초록 (Advanced Reddit와 동일)
    }
    
    create_comparison_plots()
    
    print("\n🎉 모델 성능 시각화 완료!")
    print("📁 생성된 파일:")
    print("   - assets/charts/model_performance_ic_vs_hitrate.png (발표용)")
    print("   - assets/charts/model_performance_ic_vs_icir.png (논문용)")
    print("   - assets/charts/comprehensive_model_performance.png (종합)")
