#!/usr/bin/env python3
"""
Sentiment analysis for Reddit posts using VADER and FinBERT.

Implements sentiment scoring as described in the manual for contrarian effect prediction.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Warning: vaderSentiment not installed. Install with: pip install vaderSentiment")
    SentimentIntensityAnalyzer = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers torch")
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Comprehensive sentiment analyzer for financial text."""
    
    def __init__(self, use_finbert: bool = True):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_finbert: Whether to use FinBERT for financial sentiment
        """
        self.use_finbert = use_finbert
        
        # Initialize VADER
        if SentimentIntensityAnalyzer:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
            logger.warning("VADER not available")
        
        # Initialize FinBERT
        self.finbert_model = None
        self.finbert_tokenizer = None
        if use_finbert and pipeline:
            try:
                self.finbert_model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    return_all_scores=True
                )
                logger.info("FinBERT model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}")
                self.finbert_model = None
        
        # Financial sentiment lexicon
        self.bullish_terms = [
            "bull", "bullish", "moon", "rocket", "squeeze", "breakout",
            "rally", "surge", "explosion", "melt up", "bull run",
            "diamond hands", "hodl", "buy", "long", "calls", "puts",
            "green", "up", "rise", "gain", "profit", "win", "beat"
        ]
        
        self.bearish_terms = [
            "bear", "bearish", "crash", "dump", "sell", "short", "puts",
            "red", "down", "fall", "drop", "loss", "lose", "miss",
            "paper hands", "sell", "exit", "liquidate", "panic"
        ]
    
    def clean_text_for_sentiment(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get VADER sentiment scores."""
        if not self.vader or not text:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        clean_text = self.clean_text_for_sentiment(text)
        scores = self.vader.polarity_scores(clean_text)
        return scores
    
    def finbert_sentiment(self, text: str) -> Dict[str, float]:
        """Get FinBERT sentiment scores."""
        if not self.finbert_model or not text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        clean_text = self.clean_text_for_sentiment(text)
        
        # Truncate if too long
        if len(clean_text) > 512:
            clean_text = clean_text[:512]
        
        try:
            results = self.finbert_model(clean_text)
            
            # Extract scores
            scores = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                scores[label] = score
            
            return scores
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def lexicon_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment using financial lexicon."""
        if not text:
            return {'bullish': 0.0, 'bearish': 0.0, 'neutral': 1.0}
        
        clean_text = self.clean_text_for_sentiment(text).lower()
        
        bullish_count = sum(1 for term in self.bullish_terms if term in clean_text)
        bearish_count = sum(1 for term in self.bearish_terms if term in clean_text)
        
        total_terms = bullish_count + bearish_count
        if total_terms == 0:
            return {'bullish': 0.0, 'bearish': 0.0, 'neutral': 1.0}
        
        bullish_score = bullish_count / total_terms
        bearish_score = bearish_count / total_terms
        neutral_score = 1.0 - bullish_score - bearish_score
        
        return {
            'bullish': bullish_score,
            'bearish': bearish_score, 
            'neutral': max(0, neutral_score)
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis.
        
        Returns:
            Dictionary with all sentiment scores and derived features
        """
        if not text:
            return self._empty_sentiment_result()
        
        result = {}
        
        # VADER sentiment
        vader_scores = self.vader_sentiment(text)
        result.update({
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_neutral': vader_scores['neu'],
            'vader_negative': vader_scores['neg']
        })
        
        # FinBERT sentiment
        if self.use_finbert:
            finbert_scores = self.finbert_sentiment(text)
            result.update({
                'finbert_positive': finbert_scores.get('positive', 0.0),
                'finbert_negative': finbert_scores.get('negative', 0.0),
                'finbert_neutral': finbert_scores.get('neutral', 1.0)
            })
        
        # Lexicon sentiment
        lexicon_scores = self.lexicon_sentiment(text)
        result.update({
            'lexicon_bullish': lexicon_scores['bullish'],
            'lexicon_bearish': lexicon_scores['bearish'],
            'lexicon_neutral': lexicon_scores['neutral']
        })
        
        # Derived features
        result.update(self._calculate_derived_features(result))
        
        return result
    
    def _calculate_derived_features(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived sentiment features."""
        features = {}
        
        # Overall sentiment direction
        vader_compound = scores.get('vader_compound', 0.0)
        features['sentiment_direction'] = 1 if vader_compound > 0.05 else (-1 if vader_compound < -0.05 else 0)
        
        # Sentiment intensity
        features['sentiment_intensity'] = abs(vader_compound)
        
        # Financial sentiment (bullish vs bearish)
        bullish_score = scores.get('lexicon_bullish', 0.0)
        bearish_score = scores.get('lexicon_bearish', 0.0)
        if bullish_score + bearish_score > 0:
            features['financial_sentiment'] = (bullish_score - bearish_score) / (bullish_score + bearish_score)
        else:
            features['financial_sentiment'] = 0.0
        
        # Sentiment confidence
        features['sentiment_confidence'] = max(
            scores.get('vader_positive', 0.0),
            scores.get('vader_negative', 0.0)
        )
        
        return features
    
    def _empty_sentiment_result(self) -> Dict[str, float]:
        """Return empty sentiment result."""
        return {
            'vader_compound': 0.0,
            'vader_positive': 0.0,
            'vader_neutral': 1.0,
            'vader_negative': 0.0,
            'finbert_positive': 0.0,
            'finbert_negative': 0.0,
            'finbert_neutral': 1.0,
            'lexicon_bullish': 0.0,
            'lexicon_bearish': 0.0,
            'lexicon_neutral': 1.0,
            'sentiment_direction': 0,
            'sentiment_intensity': 0.0,
            'financial_sentiment': 0.0,
            'sentiment_confidence': 0.0
        }

def calculate_sentiment_features(df: pd.DataFrame, 
                                text_col: str = 'text',
                                use_finbert: bool = True) -> pd.DataFrame:
    """
    Calculate comprehensive sentiment features for a dataframe.
    
    Args:
        df: DataFrame with text data
        text_col: Column name containing text
        use_finbert: Whether to use FinBERT
        
    Returns:
        DataFrame with additional sentiment features
    """
    logger.info(f"Calculating sentiment features for {len(df)} records")
    
    analyzer = SentimentAnalyzer(use_finbert=use_finbert)
    
    # Calculate sentiment for each text
    sentiment_results = []
    for idx, text in enumerate(df[text_col]):
        if idx % 1000 == 0:
            logger.info(f"Processing sentiment for record {idx}/{len(df)}")
        
        result = analyzer.analyze_sentiment(text)
        sentiment_results.append(result)
    
    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiment_results)
    
    # Combine with original data
    result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    
    logger.info(f"Sentiment features calculated. Mean VADER compound: {result_df['vader_compound'].mean():.3f}")
    
    return result_df

def aggregate_daily_sentiment(df: pd.DataFrame, 
                             group_cols: List[str] = ['date', 'ticker']) -> pd.DataFrame:
    """
    Aggregate sentiment features by date and ticker.
    
    Args:
        df: DataFrame with sentiment features
        group_cols: Columns to group by
        
    Returns:
        Aggregated DataFrame with daily sentiment metrics
    """
    logger.info(f"Aggregating sentiment features by {group_cols}")
    
    sentiment_cols = [col for col in df.columns if any(x in col for x in 
                    ['vader', 'finbert', 'lexicon', 'sentiment', 'financial'])]
    
    agg_dict = {}
    for col in sentiment_cols:
        if col in df.columns:
            agg_dict[col] = ['mean', 'std', 'min', 'max']
    
    # Add count of posts
    agg_dict['text'] = 'count'
    
    # Aggregate
    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns]
    
    # Rename key columns
    rename_dict = {
        f"{group_cols[0]}_": group_cols[0],
        f"{group_cols[1]}_": group_cols[1] if len(group_cols) > 1 else None
    }
    agg_df = agg_df.rename(columns={k: v for k, v in rename_dict.items() if v})
    
    logger.info(f"Aggregated to {len(agg_df)} daily records")
    
    return agg_df

if __name__ == "__main__":
    # Test sentiment analysis
    test_texts = [
        "GME to the moon! 🚀💎🙌 Diamond hands forever!",
        "AMC is crashing hard, sell everything!",
        "BB might be a good investment, but I'm not sure.",
        "Market is looking bullish today, calls are printing!",
        "This stock is going to zero, puts are the way."
    ]
    
    analyzer = SentimentAnalyzer(use_finbert=False)  # Skip FinBERT for testing
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"VADER Compound: {result['vader_compound']:.3f}")
        print(f"Financial Sentiment: {result['financial_sentiment']:.3f}")
        print(f"Sentiment Direction: {result['sentiment_direction']}")
        print("-" * 50)

