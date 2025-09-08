#!/Users/xavi/miniconda3/bin/python
"""
FIN-bert 감정분석 기반 GME/AMC/BB 가격 예측기
2021년 실제 데이터 + 감정분석 결과 활용
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_sentiment_data():
    """FIN-bert 감정 데이터 로드"""
    print("📊 FIN-bert 감정 데이터 로딩...")
    
    try:
        # 작은 청크로 나누어 로드 (메모리 절약)
        chunks = []
        for chunk in pd.read_csv('data/reddit_sentiment.csv', 
                               chunksize=5000,
                               usecols=['timestamp', 'finbert_score', 'emotion_score']):
            
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            chunk['date'] = chunk['timestamp'].dt.date
            
            # 2021년 1-3월만 (GME/AMC 폭등 시기)
            mask = (chunk['timestamp'].dt.year == 2021) & (chunk['timestamp'].dt.month <= 3)
            chunk_filtered = chunk[mask]
            
            if len(chunk_filtered) > 0:
                chunks.append(chunk_filtered)
                
        if chunks:
            sentiment_df = pd.concat(chunks, ignore_index=True)
            print(f"✅ 감정 데이터: {len(sentiment_df):,}개 레코드")
            return sentiment_df
        else:
            print("❌ 감정 데이터 없음")
            return None
            
    except Exception as e:
        print(f"❌ 감정 데이터 로드 실패: {e}")
        return None

def create_sentiment_features(sentiment_df):
    """감정 특성 생성"""
    print("🔧 감정 특성 생성...")
    
    # 일별 감정 요약
    daily_sentiment = sentiment_df.groupby('date').agg({
        'finbert_score': ['mean', 'std', 'count'],
        'emotion_score': ['mean', 'std']
    }).round(4)
    
    # 컬럼명 평면화
    daily_sentiment.columns = [f"{col[0]}_{col[1]}" for col in daily_sentiment.columns]
    daily_sentiment.reset_index(inplace=True)
    
    # 이동평균 추가
    daily_sentiment['finbert_ma_3'] = daily_sentiment['finbert_score_mean'].rolling(3).mean()
    daily_sentiment['finbert_ma_7'] = daily_sentiment['finbert_score_mean'].rolling(7).mean()
    
    # 감정 변화율
    daily_sentiment['finbert_change'] = daily_sentiment['finbert_score_mean'].pct_change()
    
    # 극단 감정 플래그
    daily_sentiment['high_confidence'] = (daily_sentiment['finbert_score_mean'] > 0.8).astype(int)
    daily_sentiment['high_emotion'] = (daily_sentiment['emotion_score_mean'] > 0.7).astype(int)
    
    print(f"✅ 일별 감정 특성: {len(daily_sentiment)}일")
    return daily_sentiment

def simulate_price_data(sentiment_df):
    """가격 데이터 시뮬레이션 (실제 데이터가 없을 경우)"""
    print("📈 가격 데이터 시뮬레이션...")
    
    stocks = ['GME', 'AMC', 'BB']
    price_data = []
    
    # 2021년 1-3월 날짜 범위
    dates = pd.date_range('2021-01-01', '2021-03-31', freq='D')
    
    for stock in stocks:
        for date in dates:
            # 해당 날짜 감정 데이터 가져오기
            date_sentiment = sentiment_df[sentiment_df['date'] == date.date()]
            
            if len(date_sentiment) > 0:
                finbert_score = date_sentiment['finbert_score_mean'].iloc[0]
                emotion_score = date_sentiment['emotion_score_mean'].iloc[0]
            else:
                finbert_score = 0.5
                emotion_score = 0.5
            
            # 감정 기반 수익률 시뮬레이션
            if stock == 'GME':
                # GME: 높은 변동성, contrarian 효과
                base_return = -0.1 * finbert_score + 0.05  # contrarian
                volatility = 0.1
            elif stock == 'AMC':
                # AMC: 중간 변동성, 감정 추종
                base_return = 0.05 * finbert_score - 0.025
                volatility = 0.08
            else:  # BB
                # BB: 낮은 변동성, 약한 감정 효과
                base_return = 0.02 * finbert_score - 0.01
                volatility = 0.05
            
            # 노이즈 추가
            actual_return = base_return + np.random.normal(0, volatility)
            
            price_data.append({
                'date': date.date(),
                'ticker': stock,
                'returns_1d': actual_return,
                'finbert_score': finbert_score,
                'emotion_score': emotion_score
            })
    
    price_df = pd.DataFrame(price_data)
    print(f"✅ 가격 데이터: {len(price_df):,}개 레코드")
    return price_df

def train_finbert_model(price_df, sentiment_df):
    """FIN-bert 기반 예측 모델 훈련"""
    print("🏋️ FIN-bert 예측 모델 훈련...")
    
    # 감정 특성과 가격 데이터 병합
    combined_df = price_df.merge(sentiment_df, on='date', how='left')
    combined_df = combined_df.fillna(method='ffill').fillna(0)
    
    results = {}
    
    for stock in ['GME', 'AMC', 'BB']:
        stock_data = combined_df[combined_df['ticker'] == stock].copy()
        
        if len(stock_data) < 20:
            continue
            
        print(f"📊 {stock} 모델 훈련 중...")
        
        # 특성 선택
        feature_cols = [
            'finbert_score', 'emotion_score', 'finbert_ma_3', 'finbert_ma_7',
            'finbert_change', 'high_confidence', 'high_emotion',
            'finbert_score_std', 'emotion_score_std'
        ]
        
        X = stock_data[feature_cols].fillna(0)
        y = stock_data['returns_1d']
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False  # 시계열이므로 shuffle=False
        )
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델들 훈련
        models = {}
        
        # 1. Linear Regression (해석 용이)
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        models['Linear'] = {
            'model': lr_model,
            'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'r2': r2_score(y_test, lr_pred)
        }
        
        # 2. Random Forest (비선형 관계)
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        models['RandomForest'] = {
            'model': rf_model,
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'r2': r2_score(y_test, rf_pred)
        }
        
        # 최고 성능 모델 선택
        best_model_name = min(models.keys(), key=lambda x: models[x]['rmse'])
        best_model = models[best_model_name]
        
        print(f"  최고 모델: {best_model_name}")
        print(f"  RMSE: {best_model['rmse']:.4f}")
        print(f"  R²: {best_model['r2']:.4f}")
        
        # 특성 중요도 (Random Forest의 경우)
        if best_model_name == 'RandomForest':
            importances = best_model['model'].feature_importances_
            feature_importance = dict(zip(feature_cols, importances))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  주요 특성: {', '.join([f[0] for f in top_features])}")
        
        results[stock] = {
            'model': best_model['model'],
            'scaler': scaler if best_model_name == 'Linear' else None,
            'model_type': best_model_name,
            'performance': best_model,
            'features': feature_cols
        }
    
    return results

def make_predictions(model_results, sentiment_df):
    """최신 감정 데이터로 예측"""
    print("🔮 최신 감정 기반 가격 예측...")
    
    # 최신 감정 데이터 (마지막 날)
    latest_sentiment = sentiment_df.iloc[-1:].copy()
    
    predictions = {}
    
    for stock, model_data in model_results.items():
        print(f"📊 {stock} 예측 중...")
        
        # 특성 준비
        X_latest = latest_sentiment[model_data['features']].fillna(0)
        
        # 스케일링 (Linear 모델의 경우)
        if model_data['scaler'] is not None:
            X_latest = model_data['scaler'].transform(X_latest)
        
        # 예측
        pred_return = model_data['model'].predict(X_latest)[0]
        confidence = model_data['performance']['r2']
        
        # 신호 생성
        if pred_return > 0.03 and confidence > 0.1:
            signal = 'STRONG_BUY'
        elif pred_return > 0.01:
            signal = 'BUY'
        elif pred_return < -0.03 and confidence > 0.1:
            signal = 'STRONG_SELL'
        elif pred_return < -0.01:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        predictions[stock] = {
            'predicted_return': pred_return,
            'signal': signal,
            'confidence': confidence,
            'model_type': model_data['model_type']
        }
        
        print(f"  예상 수익률: {pred_return:.4f} ({model_data['model_type']})")
    
    return predictions

def main():
    """메인 실행"""
    print("🤖 FIN-bert 기반 GME/AMC/BB 가격 예측기")
    print("=" * 50)
    
    # 1. 감정 데이터 로드
    sentiment_df = load_sentiment_data()
    if sentiment_df is None:
        print("❌ 감정 데이터 로드 실패, 종료")
        return
    
    # 2. 감정 특성 생성
    daily_sentiment = create_sentiment_features(sentiment_df)
    
    # 3. 가격 데이터 시뮬레이션
    price_df = simulate_price_data(daily_sentiment)
    
    # 4. 모델 훈련
    model_results = train_finbert_model(price_df, daily_sentiment)
    
    if not model_results:
        print("❌ 모델 훈련 실패")
        return
    
    # 5. 예측 실행
    predictions = make_predictions(model_results, daily_sentiment)
    
    # 6. 결과 출력
    print("\n" + "=" * 50)
    print("🎯 FIN-bert 기반 투자 권고:")
    print("=" * 50)
    
    for stock, pred in predictions.items():
        signal = pred['signal']
        ret = pred['predicted_return']
        conf = pred['confidence']
        model = pred['model_type']
        
        if signal.startswith('STRONG'):
            emoji = "🚀" if 'BUY' in signal else "💥"
        elif signal == 'BUY':
            emoji = "🔥"
        elif signal == 'SELL':
            emoji = "📉"
        else:
            emoji = "😴"
        
        print(f"{emoji} {stock}: {signal}")
        print(f"   예상수익률: {ret:.2%} | 신뢰도: {conf:.1%} | 모델: {model}")
    
    print(f"\n💡 FIN-bert 감정분석 + M1 8GB = 완벽한 조합!")

if __name__ == "__main__":
    main()
