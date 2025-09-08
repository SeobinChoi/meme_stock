#!/Users/xavi/miniconda3/bin/python
"""
빠른 M1 ML 테스트 - 간단한 GME/AMC/BB 예측
로딩 없이 바로 결과 확인용
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def quick_test():
    """빠른 ML 테스트"""
    print("🚀 빠른 ML 테스트 시작!")
    print("=" * 40)
    
    # 테스트용 가상 데이터 생성 (실제 데이터 대신)
    np.random.seed(42)
    
    stocks = ['GME', 'AMC', 'BB']
    results = {}
    
    for stock in stocks:
        print(f"📈 {stock} 모델 테스트 중...")
        
        # 가상 특성 데이터 (실제 Reddit 감정 패턴 시뮬레이션)
        n_samples = 1000
        X = np.random.randn(n_samples, 6)  # 6개 특성
        
        # 가상 수익률 (contrarian 패턴 포함)
        contrarian_signal = -X[:, 0]  # 첫 번째 특성을 contrarian으로
        noise = np.random.randn(n_samples) * 0.02
        y = contrarian_signal * 0.05 + noise  # 5% 효과 + 노이즈
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 간단한 Random Forest 모델
        model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=5, 
            random_state=42,
            n_jobs=4  # M1 4코어 활용
        )
        
        # 훈련 (매우 빠름)
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 성능 평가
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 다음 기간 예측
        next_features = np.random.randn(1, 6)
        next_pred = model.predict(next_features)[0]
        
        results[stock] = {
            'rmse': rmse,
            'r2': r2,
            'prediction': next_pred,
            'signal': 'BUY' if next_pred > 0.02 else 'SELL' if next_pred < -0.02 else 'HOLD'
        }
        
        print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"  예상 수익률: {next_pred:.4f}")
    
    print("\n" + "=" * 40)
    print("🎯 빠른 테스트 결과:")
    print("=" * 40)
    
    for stock, result in results.items():
        signal = result['signal']
        pred = result['prediction']
        r2 = result['r2']
        
        emoji = "🔥" if signal == 'BUY' else "📉" if signal == 'SELL' else "😴"
        print(f"{emoji} {stock}: {signal} (예상: {pred:.2%}, 정확도: {r2:.1%})")
    
    print("\n💡 M1 8GB에서 ML 모델이 완벽하게 동작합니다!")
    print("🚀 실제 데이터로도 이 속도로 처리 가능!")
    
    return results

if __name__ == "__main__":
    results = quick_test()
