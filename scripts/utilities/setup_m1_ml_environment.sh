#!/bin/bash
# M1 Mac ML 환경 설정 스크립트
# GME/AMC/BB 가격 예측을 위한 최적화된 환경

echo "🍎 M1 Mac ML 환경 설정 시작..."

# 1. Conda 환경 생성 (M1 최적화)
echo "📦 가상환경 생성 중..."
conda create -n meme_stock_ml python=3.9 -y
conda activate meme_stock_ml

# 2. M1 최적화 ML 라이브러리 설치
echo "🔧 M1 최적화 ML 라이브러리 설치 중..."

# 기본 데이터 처리
conda install -c conda-forge pandas numpy -y
conda install -c conda-forge matplotlib seaborn -y

# ML 라이브러리 (M1 네이티브 지원)
conda install -c conda-forge scikit-learn -y
pip install lightgbm  # M1 네이티브 지원
pip install xgboost   # M1 최적화됨

# 딥러닝 (선택사항 - 메모리 8GB로도 충분)
# pip install tensorflow-metal  # M1 GPU 가속
# pip install torch torchvision  # PyTorch M1 지원

# 금융 데이터 처리
pip install yfinance ta-lib-binary

# 기타 유용한 라이브러리
pip install optuna  # 하이퍼파라미터 최적화 (가벼움)
pip install joblib  # 병렬 처리

echo "✅ M1 ML 환경 설정 완료!"
echo ""
echo "🚀 사용법:"
echo "conda activate meme_stock_ml"
echo "python scripts/enhanced_contrarian_model.py"
echo ""
echo "💡 M1 8GB 최적화 팁:"
echo "- LightGBM/XGBoost 우선 사용 (메모리 효율적)"
echo "- 배치 크기 작게 설정"
echo "- 데이터 청킹 활용"
echo "- GPU 메모리 대신 통합 메모리 활용"
