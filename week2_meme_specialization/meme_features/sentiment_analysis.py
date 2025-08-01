"""
Advanced Sentiment Analysis for Meme Stock Prediction
Week 2 Implementation - BERT-based Emotion Classification
"""

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers for BERT (will use fallback if not available)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ Transformers not available. Using TextBlob fallback.")

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.sentiment_models = {}
        self.meme_keywords = {
            'bullish': ['moon', 'rocket', 'bull', 'buy', 'hold', 'diamond', 'hands', 'tendies', 'gains'],
            'bearish': ['bear', 'sell', 'crash', 'dump', 'paper', 'hands', 'loss', 'bagholder'],
            'fomo': ['fomo', 'missing out', 'late', 'buy now', 'last chance'],
            'squeeze': ['squeeze', 'short', 'gamma', 'options', 'call'],
            'diamond_hands': ['diamond hands', 'hodl', 'hold', 'not selling', 'to the moon'],
            'paper_hands': ['paper hands', 'sell', 'panic', 'weak hands']
        }
        
        if BERT_AVAILABLE:
            self._initialize_bert_models()
        else:
            print("📝 Using TextBlob for sentiment analysis")
    
    def _initialize_bert_models(self):
        """Initialize BERT models for sentiment analysis"""
        try:
            # Financial sentiment model
            self.sentiment_models['financial'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            
            # Emotion classification model
            self.sentiment_models['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            print("✅ BERT models initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing BERT models: {e}")
            BERT_AVAILABLE = False
    
    def analyze_meme_sentiment(self, reddit_df):
        """
        Perform comprehensive sentiment analysis on Reddit posts
        Target: 20 sentiment-specific features
        """
        print("🧠 Analyzing meme sentiment...")
        
        # Ensure we have text data
        if 'title' not in reddit_df.columns or 'body' not in reddit_df.columns:
            print("❌ Missing title or body columns")
            return pd.DataFrame()
        
        # Combine title and body for analysis
        reddit_df['combined_text'] = reddit_df['title'].fillna('') + ' ' + reddit_df['body'].fillna('')
        
        # Clean text
        reddit_df['cleaned_text'] = reddit_df['combined_text'].apply(self._clean_text)
        
        # Generate sentiment features
        sentiment_features = self._generate_sentiment_features(reddit_df)
        
        # Aggregate by date
        daily_sentiment = self._aggregate_daily_sentiment(sentiment_features)
        
        print(f"✅ Generated {len(daily_sentiment.columns)} sentiment features")
        return daily_sentiment
    
    def _clean_text(self, text):
        """Clean text for sentiment analysis"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep emojis
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def _generate_sentiment_features(self, reddit_df):
        """Generate sentiment features for each post"""
        features = reddit_df.copy()
        
        # Basic TextBlob sentiment
        features['textblob_polarity'] = features['cleaned_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        features['textblob_subjectivity'] = features['cleaned_text'].apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        
        # Meme-specific keyword analysis
        features['bullish_keyword_count'] = features['cleaned_text'].apply(
            lambda x: self._count_keywords(x, self.meme_keywords['bullish'])
        )
        features['bearish_keyword_count'] = features['cleaned_text'].apply(
            lambda x: self._count_keywords(x, self.meme_keywords['bearish'])
        )
        features['fomo_keyword_count'] = features['cleaned_text'].apply(
            lambda x: self._count_keywords(x, self.meme_keywords['fomo'])
        )
        features['squeeze_keyword_count'] = features['cleaned_text'].apply(
            lambda x: self._count_keywords(x, self.meme_keywords['squeeze'])
        )
        features['diamond_hands_count'] = features['cleaned_text'].apply(
            lambda x: self._count_keywords(x, self.meme_keywords['diamond_hands'])
        )
        features['paper_hands_count'] = features['cleaned_text'].apply(
            lambda x: self._count_keywords(x, self.meme_keywords['paper_hands'])
        )
        
        # Advanced sentiment features
        if BERT_AVAILABLE:
            bert_features = self._extract_bert_features(features['cleaned_text'])
            for key, value in bert_features.items():
                features[key] = value
        else:
            # Fallback features using TextBlob
            features = self._generate_fallback_features(features)
        
        return features
    
    def _count_keywords(self, text, keywords):
        """Count occurrences of keywords in text"""
        if pd.isna(text):
            return 0
        return sum(1 for keyword in keywords if keyword in text.lower())
    
    def _extract_bert_features(self, texts):
        """Extract BERT-based sentiment features"""
        features = {}
        
        # Process in batches to avoid memory issues
        batch_size = 100
        all_financial_scores = []
        all_emotion_scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts.iloc[i:i+batch_size].tolist()
            
            # Financial sentiment
            try:
                financial_results = self.sentiment_models['financial'](batch_texts)
                all_financial_scores.extend(financial_results)
            except:
                all_financial_scores.extend([{'label': 'neutral', 'score': 0.5}] * len(batch_texts))
            
            # Emotion classification
            try:
                emotion_results = self.sentiment_models['emotion'](batch_texts)
                all_emotion_scores.extend(emotion_results)
            except:
                all_emotion_scores.extend([{'label': 'neutral', 'score': 0.5}] * len(batch_texts))
        
        # Extract financial sentiment scores
        features['finbert_bullish_score'] = [
            self._extract_financial_score(result, 'positive') for result in all_financial_scores
        ]
        features['finbert_bearish_score'] = [
            self._extract_financial_score(result, 'negative') for result in all_financial_scores
        ]
        features['finbert_neutral_score'] = [
            self._extract_financial_score(result, 'neutral') for result in all_financial_scores
        ]
        
        # Extract emotion scores
        features['emotion_joy_intensity'] = [
            self._extract_emotion_score(result, 'joy') for result in all_emotion_scores
        ]
        features['emotion_fear_intensity'] = [
            self._extract_emotion_score(result, 'fear') for result in all_emotion_scores
        ]
        features['emotion_anger_intensity'] = [
            self._extract_emotion_score(result, 'anger') for result in all_emotion_scores
        ]
        features['emotion_surprise_intensity'] = [
            self._extract_emotion_score(result, 'surprise') for result in all_emotion_scores
        ]
        
        return features
    
    def _extract_financial_score(self, result, target_label):
        """Extract financial sentiment score for specific label"""
        try:
            for item in result:
                if item['label'] == target_label:
                    return item['score']
            return 0.0
        except:
            return 0.0
    
    def _extract_emotion_score(self, result, target_emotion):
        """Extract emotion score for specific emotion"""
        try:
            for item in result:
                if item['label'] == target_emotion:
                    return item['score']
            return 0.0
        except:
            return 0.0
    
    def _generate_fallback_features(self, features):
        """Generate fallback features using TextBlob"""
        
        # Enhanced TextBlob features
        features['sentiment_confidence'] = 1 - features['textblob_subjectivity']
        features['sentiment_polarization'] = abs(features['textblob_polarity'])
        
        # Meme-specific sentiment
        features['meme_bullish_sentiment'] = (
            features['bullish_keyword_count'] - features['bearish_keyword_count']
        ) / (features['bullish_keyword_count'] + features['bearish_keyword_count'] + 1)
        
        features['diamond_hands_sentiment'] = features['diamond_hands_count'] / (
            features['diamond_hands_count'] + features['paper_hands_count'] + 1
        )
        
        features['fomo_intensity'] = features['fomo_keyword_count'] / (
            features['fomo_keyword_count'] + 1
        )
        
        features['squeeze_expectation'] = features['squeeze_keyword_count'] / (
            features['squeeze_keyword_count'] + 1
        )
        
        # Simulate BERT-like features
        features['finbert_bullish_score'] = np.where(
            features['textblob_polarity'] > 0.1,
            features['textblob_polarity'],
            0.0
        )
        features['finbert_bearish_score'] = np.where(
            features['textblob_polarity'] < -0.1,
            -features['textblob_polarity'],
            0.0
        )
        features['finbert_neutral_score'] = np.where(
            abs(features['textblob_polarity']) <= 0.1,
            1.0,
            0.0
        )
        
        # Simulate emotion features
        features['emotion_joy_intensity'] = np.where(
            features['textblob_polarity'] > 0.3,
            features['textblob_polarity'],
            0.0
        )
        features['emotion_fear_intensity'] = np.where(
            features['textblob_polarity'] < -0.3,
            -features['textblob_polarity'],
            0.0
        )
        features['emotion_anger_intensity'] = np.where(
            features['bearish_keyword_count'] > 0,
            0.5 + features['bearish_keyword_count'] * 0.1,
            0.0
        )
        features['emotion_surprise_intensity'] = np.where(
            features['squeeze_keyword_count'] > 0,
            0.5 + features['squeeze_keyword_count'] * 0.1,
            0.0
        )
        
        return features
    
    def _aggregate_daily_sentiment(self, sentiment_features):
        """Aggregate sentiment features by date"""
        
        # Convert timestamp to date
        sentiment_features['timestamp'] = pd.to_datetime(sentiment_features['timestamp'])
        sentiment_features['date'] = sentiment_features['timestamp'].dt.date
        
        # Aggregate by date
        daily_agg = sentiment_features.groupby('date').agg({
            # Basic sentiment
            'textblob_polarity': ['mean', 'std', 'count'],
            'textblob_subjectivity': ['mean', 'std'],
            'sentiment_confidence': ['mean', 'std'],
            'sentiment_polarization': ['mean', 'std'],
            
            # Meme keywords
            'bullish_keyword_count': ['sum', 'mean'],
            'bearish_keyword_count': ['sum', 'mean'],
            'fomo_keyword_count': ['sum', 'mean'],
            'squeeze_keyword_count': ['sum', 'mean'],
            'diamond_hands_count': ['sum', 'mean'],
            'paper_hands_count': ['sum', 'mean'],
            
            # Meme sentiment
            'meme_bullish_sentiment': ['mean', 'std'],
            'diamond_hands_sentiment': ['mean', 'std'],
            'fomo_intensity': ['mean', 'std'],
            'squeeze_expectation': ['mean', 'std'],
            
            # BERT/Financial sentiment
            'finbert_bullish_score': ['mean', 'std'],
            'finbert_bearish_score': ['mean', 'std'],
            'finbert_neutral_score': ['mean', 'std'],
            
            # Emotion features
            'emotion_joy_intensity': ['mean', 'std'],
            'emotion_fear_intensity': ['mean', 'std'],
            'emotion_anger_intensity': ['mean', 'std'],
            'emotion_surprise_intensity': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = ['date'] + [f'sentiment_{col[0]}_{col[1]}' for col in daily_agg.columns[1:]]
        
        # Add derived features
        daily_agg = self._add_derived_sentiment_features(daily_agg)
        
        return daily_agg
    
    def _add_derived_sentiment_features(self, daily_agg):
        """Add derived sentiment features"""
        
        # 1. Sentiment consensus (low std = high consensus)
        daily_agg['sentiment_consensus'] = 1 / (daily_agg['sentiment_textblob_polarity_std'] + 1)
        
        # 2. Sentiment momentum (change in sentiment over time)
        daily_agg['sentiment_momentum'] = daily_agg['sentiment_textblob_polarity_mean'].pct_change()
        
        # 3. Emotional contagion (emotion spreading)
        daily_agg['emotional_contagion'] = (
            daily_agg['sentiment_emotion_joy_intensity_mean'] + 
            daily_agg['sentiment_emotion_fear_intensity_mean']
        ) / 2
        
        # 4. Diamond hands vs Paper hands ratio
        daily_agg['diamond_vs_paper_ratio'] = (
            daily_agg['sentiment_diamond_hands_count_sum'] / 
            (daily_agg['sentiment_paper_hands_count_sum'] + 1)
        )
        
        # 5. Bullish vs Bearish dominance
        daily_agg['bullish_bearish_ratio'] = (
            daily_agg['sentiment_bullish_keyword_count_sum'] / 
            (daily_agg['sentiment_bearish_keyword_count_sum'] + 1)
        )
        
        # 6. Moon expectation level
        daily_agg['moon_expectation_level'] = (
            daily_agg['sentiment_bullish_keyword_count_mean'] * 
            daily_agg['sentiment_finbert_bullish_score_mean']
        )
        
        # 7. Squeeze anticipation
        daily_agg['squeeze_anticipation'] = (
            daily_agg['sentiment_squeeze_keyword_count_mean'] * 
            daily_agg['sentiment_emotion_surprise_intensity_mean']
        )
        
        # 8. Retail vs Institutional sentiment (simulated)
        daily_agg['retail_vs_institutional'] = (
            daily_agg['sentiment_meme_bullish_sentiment_mean'] - 
            daily_agg['sentiment_finbert_neutral_score_mean']
        )
        
        # 9. Weekend sentiment buildup
        daily_agg['weekend_sentiment_buildup'] = self._calculate_weekend_sentiment_buildup(daily_agg)
        
        # 10. FUD detection score
        daily_agg['fud_detection_score'] = (
            daily_agg['sentiment_emotion_fear_intensity_mean'] * 
            daily_agg['sentiment_bearish_keyword_count_mean']
        )
        
        return daily_agg
    
    def _calculate_weekend_sentiment_buildup(self, daily_agg):
        """Calculate weekend sentiment buildup"""
        weekend_buildup = []
        
        for i, date in enumerate(daily_agg['date']):
            if i < 3:  # Need at least 3 days of history
                weekend_buildup.append(0)
                continue
                
            # Check if current day is Monday
            if pd.to_datetime(date).weekday() == 0:  # Monday
                # Compare Monday sentiment with Friday sentiment
                friday_sentiment = daily_agg.iloc[i-3]['sentiment_textblob_polarity_mean'] if i >= 3 else 0
                monday_sentiment = daily_agg.iloc[i]['sentiment_textblob_polarity_mean']
                
                # Calculate weekend buildup
                buildup = monday_sentiment - friday_sentiment
                weekend_buildup.append(buildup)
            else:
                weekend_buildup.append(0)
        
        return weekend_buildup
    
    def save_sentiment_features(self, features, filename='sentiment_features.csv'):
        """Save sentiment features to file"""
        features.to_csv(f'../data/{filename}', index=False)
        print(f"✅ Sentiment features saved to ../data/{filename}")

if __name__ == "__main__":
    # Test sentiment analysis
    analyzer = AdvancedSentimentAnalyzer()
    
    # Load sample data
    try:
        reddit_data = pd.read_csv('../data/reddit_wsb.csv')
        
        sentiment_features = analyzer.analyze_meme_sentiment(reddit_data)
        analyzer.save_sentiment_features(sentiment_features)
        
    except FileNotFoundError:
        print("❌ Data files not found. Please run Week 1 pipeline first.") 