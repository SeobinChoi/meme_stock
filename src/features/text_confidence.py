#!/usr/bin/env python3
"""
Text confidence feature engineering for meme stock analysis.

Implements confidence scoring based on over-assertiveness vs hedging language
as described in the manual for contrarian effect prediction.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Confidence lexicon - over-assertive terms
CONFIDENCE_TERMS = [
    # English confidence terms
    "guaranteed", "definitely", "surely", "certainly", "absolutely", 
    "100%", "will", "going to", "to the moon", "rocket", "moon",
    "diamond hands", "hodl", "yolo", "all in", "max out",
    "squeeze", "short squeeze", "gamma squeeze", "melt up",
    "breakout", "explosion", "surge", "rally", "bull run",
    "never sell", "hold forever", "buy the dip", "load up",
    
    # Korean confidence terms
    "확실", "반드시", "꼭", "분명", "틀림없", "완전", "진짜",
    "대박", "폭등", "급등", "상승", "매수", "보유", "홀드",
    "달나라", "로켓", "폭발", "급상승", "돌파", "돌풍"
]

# Hedging lexicon - uncertain terms  
HEDGING_TERMS = [
    # English hedging terms
    "maybe", "perhaps", "might", "could", "possibly", "potentially",
    "guess", "think", "believe", "hope", "wish", "if", "unless",
    "doubt", "uncertain", "unsure", "not sure", "maybe not",
    "probably", "likely", "unlikely", "chance", "risk",
    
    # Korean hedging terms  
    "아마", "불확실", "모르", "생각", "추측", "가능", "아닐",
    "의심", "걱정", "우려", "위험", "조심", "신중"
]

# Meme-specific terms that indicate extreme confidence
MEME_CONFIDENCE_TERMS = [
    "🚀", "💎", "🙌", "🌙", "📈", "🔥", "💪", "🎯",
    "diamond hands", "paper hands", "apes", "retard", "autist",
    "tendies", "banana", "smooth brain", "wrinkle brain"
]

def clean_text(text: str) -> str:
    """Clean text for confidence analysis."""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs but keep the text
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def calculate_confidence_score(text: str) -> float:
    """
    Calculate confidence score based on lexicon matching.
    
    Score = count(confidence_terms) - count(hedging_terms)
    Higher scores indicate more over-assertive language.
    """
    if not text:
        return 0.0
    
    clean_txt = clean_text(text)
    
    # Count confidence terms
    conf_count = sum(1 for term in CONFIDENCE_TERMS if term in clean_txt)
    
    # Count hedging terms  
    hedge_count = sum(1 for term in HEDGING_TERMS if term in clean_txt)
    
    # Count meme confidence terms (weighted higher)
    meme_count = sum(1 for term in MEME_CONFIDENCE_TERMS if term in clean_txt)
    
    # Calculate final score
    score = conf_count + (meme_count * 2) - hedge_count
    
    # Normalize by text length to avoid bias toward longer posts
    word_count = len(clean_txt.split())
    if word_count > 0:
        score = score / word_count * 100  # Scale to 0-100 range
    
    return max(0, score)  # Ensure non-negative

def calculate_confidence_features(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Calculate comprehensive confidence features for a dataframe.
    
    Args:
        df: DataFrame with text data
        text_col: Column name containing text
        
    Returns:
        DataFrame with additional confidence features
    """
    logger.info(f"Calculating confidence features for {len(df)} records")
    
    # Basic confidence score
    df['confidence_score'] = df[text_col].apply(calculate_confidence_score)
    
    # Confidence categories
    df['confidence_level'] = pd.cut(
        df['confidence_score'], 
        bins=[0, 10, 25, 50, float('inf')], 
        labels=['low', 'medium', 'high', 'extreme']
    )
    
    # Binary high confidence indicator
    df['is_high_confidence'] = (df['confidence_score'] > 25).astype(int)
    
    # Text length features
    df['text_length'] = df[text_col].str.len()
    df['word_count'] = df[text_col].str.split().str.len()
    
    # Confidence per word (normalized)
    df['confidence_per_word'] = df['confidence_score'] / df['word_count'].clip(lower=1)
    
    # Emoji count (proxy for meme intensity)
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F0F5\U0001F200-\U0001F2FF]'
    df['emoji_count'] = df[text_col].str.count(emoji_pattern)
    
    # Meme intensity score
    df['meme_intensity'] = df['emoji_count'] + df['confidence_score']
    
    logger.info(f"Confidence features calculated. Mean confidence score: {df['confidence_score'].mean():.2f}")
    
    return df

def aggregate_daily_confidence(df: pd.DataFrame, 
                              group_cols: List[str] = ['date', 'ticker']) -> pd.DataFrame:
    """
    Aggregate confidence features by date and ticker.
    
    Args:
        df: DataFrame with confidence features
        group_cols: Columns to group by
        
    Returns:
        Aggregated DataFrame with daily confidence metrics
    """
    logger.info(f"Aggregating confidence features by {group_cols}")
    
    agg_dict = {
        'confidence_score': ['mean', 'std', 'max', 'min', 'sum'],
        'is_high_confidence': ['sum', 'mean'],
        'text_length': ['mean', 'std'],
        'word_count': ['mean', 'sum'],
        'confidence_per_word': ['mean', 'std'],
        'emoji_count': ['sum', 'mean'],
        'meme_intensity': ['mean', 'std', 'max']
    }
    
    # Flatten column names
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
    # Test the confidence scoring
    test_texts = [
        "GME to the moon! 🚀💎🙌 Diamond hands forever!",
        "Maybe GME will go up, but I'm not sure...",
        "AMC is guaranteed to squeeze! 100% certain!",
        "I think BB might be a good investment, perhaps.",
        "BBBY to the moon! Never selling! All in! 🚀🚀🚀"
    ]
    
    for text in test_texts:
        score = calculate_confidence_score(text)
        print(f"Text: {text}")
        print(f"Confidence Score: {score:.2f}")
        print("-" * 50)

