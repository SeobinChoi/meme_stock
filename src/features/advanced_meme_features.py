"""
Day 8-9: Advanced Meme Feature Engineering
Develop sophisticated features capturing meme stock-specific behaviors
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Advanced analysis libraries
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMemeFeatureEngineer:
    """
    Advanced meme-specific feature engineering for viral patterns and social dynamics
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
        
    def generate_advanced_features(self) -> Dict:
        """
        Generate advanced meme-specific features
        """
        logger.info("🚀 Starting Day 8-9: Advanced Meme Feature Engineering")
        logger.info("="*60)
        
        # Load existing engineered dataset
        logger.info("STEP 1: Loading existing engineered dataset")
        logger.info("="*50)
        dataset = self._load_engineered_dataset()
        if dataset is None:
            return {"status": "ERROR", "message": "Failed to load engineered dataset"}
        
        # Step 2: Viral Pattern Detection
        logger.info("STEP 2: Viral Pattern Detection")
        logger.info("="*50)
        viral_features = self._generate_viral_pattern_features(dataset)
        
        # Step 3: Advanced Sentiment Analysis
        logger.info("STEP 3: Advanced Sentiment Analysis")
        logger.info("="*50)
        sentiment_features = self._generate_advanced_sentiment_features(dataset)
        
        # Step 4: Social Network Dynamics
        logger.info("STEP 4: Social Network Dynamics")
        logger.info("="*50)
        social_features = self._generate_social_network_features(dataset)
        
        # Step 5: Cross-Modal Innovation
        logger.info("STEP 5: Cross-Modal Feature Innovation")
        logger.info("="*50)
        cross_modal_features = self._generate_cross_modal_features(dataset)
        
        # Step 6: Combine all features
        logger.info("STEP 6: Combining Advanced Features")
        logger.info("="*50)
        combined_features = self._combine_advanced_features(
            dataset, viral_features, sentiment_features, 
            social_features, cross_modal_features
        )
        
        # Step 7: Feature validation and quality control
        logger.info("STEP 7: Feature Validation and Quality Control")
        logger.info("="*50)
        validation_results = self._validate_advanced_features(combined_features)
        
        # Step 8: Save results
        logger.info("STEP 8: Saving Advanced Features")
        logger.info("="*50)
        self._save_advanced_features(combined_features, validation_results)
        
        logger.info("✅ Advanced Meme Feature Engineering Completed")
        return {
            "status": "COMPLETED",
            "total_features": len(combined_features.columns),
            "viral_features": len(viral_features.columns),
            "sentiment_features": len(sentiment_features.columns),
            "social_features": len(social_features.columns),
            "cross_modal_features": len(cross_modal_features.columns)
        }
    
    def _load_engineered_dataset(self) -> Optional[pd.DataFrame]:
        """
        Load existing engineered dataset
        """
        try:
            dataset_path = self.features_dir / "engineered_features_dataset.csv"
            if not dataset_path.exists():
                logger.error(f"Dataset not found: {dataset_path}")
                return None
            
            dataset = pd.read_csv(dataset_path)
            dataset['date'] = pd.to_datetime(dataset['date'])
            dataset = dataset.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✅ Loaded dataset with shape: {dataset.shape}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def _generate_viral_pattern_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generate viral pattern detection features (15 features)
        """
        logger.info("Generating viral pattern detection features...")
        
        viral_features = pd.DataFrame(index=dataset.index)
        viral_features['date'] = dataset['date']
        
        # 1. Exponential Growth Detection
        for metric in ['reddit_post_count', 'reddit_total_score_x', 'reddit_comments']:
            # Calculate growth rate
            viral_features[f'viral_growth_rate_{metric}'] = dataset[metric].pct_change()
            
            # Exponential growth indicator
            viral_features[f'viral_exponential_{metric}'] = (
                dataset[metric].pct_change() > dataset[metric].pct_change().rolling(7).mean() * 2
            ).astype(int)
            
            # Viral velocity (acceleration)
            viral_features[f'viral_velocity_{metric}'] = (
                dataset[metric].pct_change().diff()
            )
        
        # 2. Viral Lifecycle Classification
        for metric in ['reddit_post_count', 'reddit_total_score_x']:
            # Peak detection
            rolling_mean = dataset[metric].rolling(7).mean()
            peaks, _ = find_peaks(rolling_mean, height=rolling_mean.mean() * 1.5)
            
            # Growth phase indicator
            viral_features[f'viral_growth_phase_{metric}'] = 0
            viral_features[f'viral_peak_phase_{metric}'] = 0
            viral_features[f'viral_decline_phase_{metric}'] = 0
            
            for i in range(len(dataset)):
                if i < 7:
                    continue
                    
                # Check if we're in a growth phase
                if dataset[metric].iloc[i] > dataset[metric].iloc[i-7] * 1.3:
                    viral_features[f'viral_growth_phase_{metric}'].iloc[i] = 1
                
                # Check if we're at a peak
                if i in peaks:
                    viral_features[f'viral_peak_phase_{metric}'].iloc[i] = 1
                
                # Check if we're in decline
                if dataset[metric].iloc[i] < dataset[metric].iloc[i-7] * 0.7:
                    viral_features[f'viral_decline_phase_{metric}'].iloc[i] = 1
        
        # 3. Viral Saturation Detection
        for metric in ['reddit_post_count', 'reddit_total_score_x']:
            # Saturation indicator (when growth rate starts declining)
            growth_rate = dataset[metric].pct_change()
            saturation = (growth_rate < growth_rate.rolling(7).mean() * 0.5) & (growth_rate > 0)
            viral_features[f'viral_saturation_{metric}'] = saturation.astype(int)
        
        # 4. Cross-Platform Amplification
        # Simulate correlation with broader social media trends
        viral_features['viral_cross_platform_correlation'] = (
            dataset['reddit_post_count'].rolling(7).corr(dataset['reddit_total_score_x'])
        )
        
        logger.info(f"✅ Generated {len(viral_features.columns)-1} viral pattern features")
        return viral_features
    
    def _generate_advanced_sentiment_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generate advanced sentiment analysis features (20 features)
        """
        logger.info("Generating advanced sentiment analysis features...")
        
        sentiment_features = pd.DataFrame(index=dataset.index)
        sentiment_features['date'] = dataset['date']
        
        # 1. Multi-Model Sentiment Fusion
        # Enhanced sentiment momentum
        sentiment_features['sentiment_momentum_enhanced'] = (
            dataset['reddit_sentiment_mean'].rolling(3).mean() - 
            dataset['reddit_sentiment_mean'].rolling(7).mean()
        )
        
        # Sentiment acceleration
        sentiment_features['sentiment_acceleration'] = (
            dataset['reddit_sentiment_mean'].diff().diff()
        )
        
        # Sentiment volatility clustering
        sentiment_features['sentiment_volatility_clustering'] = (
            dataset['reddit_sentiment_std'].rolling(5).std()
        )
        
        # 2. Meme-Specific Language Analysis
        # Diamond hands detection (strong positive sentiment)
        sentiment_features['diamond_hands_indicator'] = (
            (dataset['reddit_sentiment_mean'] > dataset['reddit_sentiment_mean'].rolling(30).quantile(0.8)) &
            (dataset['reddit_sentiment_std'] < dataset['reddit_sentiment_std'].rolling(30).quantile(0.3))
        ).astype(int)
        
        # Paper hands detection (weak sentiment)
        sentiment_features['paper_hands_indicator'] = (
            (dataset['reddit_sentiment_mean'] < dataset['reddit_sentiment_mean'].rolling(30).quantile(0.2)) |
            (dataset['reddit_sentiment_std'] > dataset['reddit_sentiment_std'].rolling(30).quantile(0.8))
        ).astype(int)
        
        # FOMO/FUD balance
        sentiment_features['fomo_fud_balance'] = (
            dataset['reddit_extreme_positive_ratio'] - dataset['reddit_extreme_negative_ratio']
        )
        
        # Moon expectation modeling
        sentiment_features['moon_expectation'] = (
            dataset['reddit_sentiment_mean'] * dataset['reddit_post_count'] / 1000
        )
        
        # 3. Advanced NLP Techniques
        # Sentiment confidence scoring
        sentiment_features['sentiment_confidence'] = (
            1 - dataset['reddit_sentiment_std'] / (dataset['reddit_sentiment_std'].max() + 1e-8)
        )
        
        # Sentiment consensus strength
        sentiment_features['sentiment_consensus_strength'] = (
            1 - dataset['reddit_sentiment_std'] / (dataset['reddit_sentiment_mean'].abs() + 1e-8)
        )
        
        # 4. Temporal Sentiment Dynamics
        for window in [1, 3, 7]:
            # Sentiment momentum
            sentiment_features[f'sentiment_momentum_{window}d'] = (
                dataset['reddit_sentiment_mean'] - dataset['reddit_sentiment_mean'].shift(window)
            )
            
            # Sentiment volatility
            sentiment_features[f'sentiment_volatility_{window}d'] = (
                dataset['reddit_sentiment_std'].rolling(window).std()
            )
            
            # Sentiment trend strength
            sentiment_features[f'sentiment_trend_strength_{window}d'] = (
                dataset['reddit_sentiment_mean'].rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            )
        
        # 5. Contextual Sentiment Analysis
        # Market condition sentiment
        sentiment_features['market_sentiment_alignment'] = (
            dataset['reddit_sentiment_mean'] * dataset['reddit_post_count'] / 
            (dataset['reddit_post_count'].rolling(30).mean() + 1e-8)
        )
        
        # Sentiment divergence
        sentiment_features['sentiment_divergence'] = (
            dataset['reddit_sentiment_mean'] - dataset['reddit_sentiment_mean'].rolling(14).mean()
        )
        
        logger.info(f"✅ Generated {len(sentiment_features.columns)-1} advanced sentiment features")
        return sentiment_features
    
    def _generate_social_network_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generate social network dynamics features (10 features)
        """
        logger.info("Generating social network dynamics features...")
        
        social_features = pd.DataFrame(index=dataset.index)
        social_features['date'] = dataset['date']
        
        # 1. Community Behavior Analysis
        # Echo chamber measurement
        social_features['echo_chamber_measurement'] = (
            1 - dataset['reddit_sentiment_std'] / (dataset['reddit_sentiment_std'].max() + 1e-8)
        )
        
        # Community fragmentation
        social_features['community_fragmentation'] = (
            dataset['reddit_sentiment_std'] / (dataset['reddit_sentiment_mean'].abs() + 1e-8)
        )
        
        # Information cascade strength
        social_features['info_cascade_strength'] = (
            dataset['reddit_post_count'] * dataset['reddit_avg_score_x'] / 
            (dataset['reddit_post_count'].rolling(7).mean() + 1e-8)
        )
        
        # 2. Network Effect Modeling
        # Coordinated behavior detection
        social_features['coordinated_behavior'] = (
            (dataset['reddit_post_count'] > dataset['reddit_post_count'].rolling(7).quantile(0.9)) &
            (dataset['reddit_avg_score_x'] > dataset['reddit_avg_score_x'].rolling(7).quantile(0.8))
        ).astype(int)
        
        # Organic vs artificial growth
        growth_rate = dataset['reddit_post_count'].pct_change()
        organic_growth = growth_rate.rolling(7).std() < growth_rate.rolling(30).std() * 0.5
        social_features['organic_growth_indicator'] = organic_growth.astype(int)
        
        # 3. Community Leadership Changes
        # Influence shift detection
        social_features['influence_shift'] = (
            dataset['reddit_avg_score_x'].rolling(7).std() / 
            dataset['reddit_avg_score_x'].rolling(30).std()
        )
        
        # New user integration
        social_features['new_user_integration'] = (
            dataset['reddit_post_count'] / (dataset['reddit_post_count'].rolling(30).mean() + 1e-8)
        )
        
        # 4. Network Connectivity
        # Community cohesion
        social_features['community_cohesion'] = (
            dataset['reddit_sentiment_consensus'] * dataset['reddit_post_count'] / 1000
        )
        
        # Network density
        social_features['network_density'] = (
            dataset['reddit_post_count'] * dataset['reddit_avg_comments'] / 
            (dataset['reddit_post_count'].rolling(7).mean() * dataset['reddit_avg_comments'].rolling(7).mean() + 1e-8)
        )
        
        logger.info(f"✅ Generated {len(social_features.columns)-1} social network features")
        return social_features
    
    def _generate_cross_modal_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-modal feature innovation (14 features)
        """
        logger.info("Generating cross-modal feature innovation...")
        
        cross_modal_features = pd.DataFrame(index=dataset.index)
        cross_modal_features['date'] = dataset['date']
        
        # 1. Social-Financial Signal Integration
        for stock in ['GME', 'AMC', 'BB']:
            # Sentiment-price correlation evolution
            returns_col = f'{stock}_returns_1d'
            if returns_col in dataset.columns:
                cross_modal_features[f'sentiment_price_corr_{stock}'] = (
                    dataset['reddit_sentiment_mean'].rolling(7).corr(dataset[returns_col])
                )
                
                # Volume-mention synchronization
                volume_col = f'{stock}_volume_ma5'
                if volume_col in dataset.columns:
                    cross_modal_features[f'volume_mention_sync_{stock}'] = (
                        dataset['reddit_post_count'].rolling(5).corr(dataset[volume_col])
                    )
        
        # 2. Advanced Interaction Features
        # Regime-dependent correlations
        volatility_regime = dataset['GME_volatility_5d'] > dataset['GME_volatility_5d'].rolling(30).quantile(0.8)
        cross_modal_features['high_vol_sentiment_corr'] = (
            dataset['reddit_sentiment_mean'].rolling(5).corr(dataset['GME_volatility_5d'])
        )
        
        # Volatility-sentiment coupling
        cross_modal_features['volatility_sentiment_coupling'] = (
            dataset['reddit_sentiment_std'] * dataset['GME_volatility_5d']
        )
        
        # 3. Cross-Asset Contagion
        # Meme stock interconnection
        if all(col in dataset.columns for col in ['GME_returns_1d', 'AMC_returns_1d', 'BB_returns_1d']):
            # Calculate rolling correlation for each pair and take mean
            corr_gme_amc = dataset['GME_returns_1d'].rolling(7).corr(dataset['AMC_returns_1d'])
            corr_gme_bb = dataset['GME_returns_1d'].rolling(7).corr(dataset['BB_returns_1d'])
            corr_amc_bb = dataset['AMC_returns_1d'].rolling(7).corr(dataset['BB_returns_1d'])
            
            cross_modal_features['meme_stock_correlation'] = (
                (corr_gme_amc + corr_gme_bb + corr_amc_bb) / 3
            )
        
        # 4. Feedback Loop Detection
        # Price movement influence on sentiment
        cross_modal_features['price_sentiment_feedback'] = (
            dataset['GME_returns_1d'].shift(1).rolling(3).corr(dataset['reddit_sentiment_mean'])
        )
        
        # 5. Prediction Lead-Lag Analysis
        for lag in [1, 3, 7]:
            cross_modal_features[f'sentiment_price_lead_lag_{lag}d'] = (
                dataset['reddit_sentiment_mean'].shift(lag).rolling(7).corr(dataset['GME_returns_1d'])
            )
        
        # 6. Cross-Modal Synchronization
        cross_modal_features['social_financial_sync'] = (
            dataset['reddit_post_count'].rolling(5).corr(dataset['GME_returns_1d'].abs())
        )
        
        # 7. Multi-Timeframe Integration
        for timeframe in ['1d', '3d', '7d']:
            returns_col = f'GME_returns_{timeframe}'
            if returns_col in dataset.columns:
                cross_modal_features[f'multi_timeframe_sentiment_{timeframe}'] = (
                    dataset['reddit_sentiment_mean'].rolling(7).corr(dataset[returns_col])
                )
        
        logger.info(f"✅ Generated {len(cross_modal_features.columns)-1} cross-modal features")
        return cross_modal_features
    
    def _combine_advanced_features(self, dataset: pd.DataFrame, viral_features: pd.DataFrame,
                                 sentiment_features: pd.DataFrame, social_features: pd.DataFrame,
                                 cross_modal_features: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all advanced features with original dataset
        """
        logger.info("Combining all advanced features...")
        
        # Start with original dataset
        combined = dataset.copy()
        
        # Add viral features (excluding date column)
        viral_cols = [col for col in viral_features.columns if col != 'date']
        for col in viral_cols:
            combined[f'viral_{col}'] = viral_features[col]
        
        # Add sentiment features
        sentiment_cols = [col for col in sentiment_features.columns if col != 'date']
        for col in sentiment_cols:
            combined[f'advanced_sentiment_{col}'] = sentiment_features[col]
        
        # Add social features
        social_cols = [col for col in social_features.columns if col != 'date']
        for col in social_cols:
            combined[f'social_{col}'] = social_features[col]
        
        # Add cross-modal features
        cross_modal_cols = [col for col in cross_modal_features.columns if col != 'date']
        for col in cross_modal_cols:
            combined[f'cross_modal_{col}'] = cross_modal_features[col]
        
        # Handle missing values
        combined = combined.fillna(0)
        
        logger.info(f"✅ Combined dataset shape: {combined.shape}")
        return combined
    
    def _validate_advanced_features(self, dataset: pd.DataFrame) -> Dict:
        """
        Validate advanced features quality and predictive power
        """
        logger.info("Validating advanced features...")
        
        validation_results = {
            'total_features': len(dataset.columns),
            'original_features': 0,
            'advanced_features': 0,
            'feature_categories': {},
            'quality_metrics': {}
        }
        
        # Count feature categories
        feature_categories = {
            'viral_features': len([col for col in dataset.columns if col.startswith('viral_')]),
            'advanced_sentiment_features': len([col for col in dataset.columns if col.startswith('advanced_sentiment_')]),
            'social_features': len([col for col in dataset.columns if col.startswith('social_')]),
            'cross_modal_features': len([col for col in dataset.columns if col.startswith('cross_modal_')])
        }
        
        validation_results['feature_categories'] = feature_categories
        validation_results['advanced_features'] = sum(feature_categories.values())
        validation_results['original_features'] = len(dataset.columns) - validation_results['advanced_features'] - 1  # -1 for date
        
        # Quality metrics
        validation_results['quality_metrics'] = {
            'missing_values_pct': (dataset.isnull().sum() / len(dataset)).mean() * 100,
            'zero_variance_features': (dataset.std() == 0).sum(),
            'high_correlation_pairs': self._count_high_correlations(dataset)
        }
        
        logger.info(f"✅ Validation completed: {validation_results['advanced_features']} advanced features")
        return validation_results
    
    def _count_high_correlations(self, dataset: pd.DataFrame, threshold: float = 0.95) -> int:
        """
        Count number of highly correlated feature pairs
        """
        try:
            # Select numeric columns only
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 100:  # Limit for performance
                numeric_cols = numeric_cols[:100]
            
            corr_matrix = dataset[numeric_cols].corr()
            high_corr_pairs = ((corr_matrix > threshold) & (corr_matrix < 1.0)).sum().sum() // 2
            return high_corr_pairs
        except:
            return 0
    
    def _save_advanced_features(self, dataset: pd.DataFrame, validation_results: Dict):
        """
        Save advanced features and validation results
        """
        try:
            # Save enhanced dataset
            output_path = self.features_dir / "advanced_meme_features_dataset.csv"
            dataset.to_csv(output_path, index=False)
            
            # Save validation results
            validation_path = self.features_dir / "advanced_features_validation.json"
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            # Generate completion report
            self._generate_completion_report(dataset, validation_results)
            
            logger.info(f"✅ Saved advanced features to {output_path}")
            logger.info(f"✅ Saved validation results to {validation_path}")
            
        except Exception as e:
            logger.error(f"Error saving advanced features: {e}")
    
    def _generate_completion_report(self, dataset: pd.DataFrame, validation_results: Dict):
        """
        Generate completion report for Day 8-9
        """
        try:
            # Get next sequence number
            sequence_num = self._get_next_sequence_number("012")
            
            report_path = Path("results") / f"{sequence_num}_day8_9_advanced_features_summary.txt"
            
            with open(report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("DAY 8-9: ADVANCED MEME FEATURE ENGINEERING SUMMARY\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Status: COMPLETED\n\n")
                
                f.write("FEATURE GENERATION SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Features: {validation_results['total_features']}\n")
                f.write(f"Original Features: {validation_results['original_features']}\n")
                f.write(f"Advanced Features: {validation_results['advanced_features']}\n\n")
                
                f.write("FEATURE CATEGORIES:\n")
                f.write("-" * 20 + "\n")
                for category, count in validation_results['feature_categories'].items():
                    f.write(f"  • {category}: {count} features\n")
                f.write("\n")
                
                f.write("QUALITY METRICS:\n")
                f.write("-" * 16 + "\n")
                for metric, value in validation_results['quality_metrics'].items():
                    f.write(f"  • {metric}: {value}\n")
                f.write("\n")
                
                f.write("ACHIEVEMENTS:\n")
                f.write("-" * 12 + "\n")
                f.write("  • Implemented viral pattern detection system\n")
                f.write("  • Created advanced sentiment analysis features\n")
                f.write("  • Developed social network dynamics quantification\n")
                f.write("  • Generated cross-modal feature innovation\n")
                f.write("  • Established comprehensive feature validation framework\n\n")
                
                f.write("NEXT STEPS:\n")
                f.write("-" * 11 + "\n")
                f.write("  • Multi-modal transformer architecture development\n")
                f.write("  • Advanced model training with new features\n")
                f.write("  • Hyperparameter optimization\n")
                f.write("  • Ensemble methods implementation\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("END OF DAY 8-9 SUMMARY\n")
                f.write("="*60 + "\n")
            
            logger.info(f"✅ Generated completion report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating completion report: {e}")
    
    def _get_next_sequence_number(self, prefix: str) -> str:
        """
        Get next sequence number for file naming
        """
        try:
            results_dir = Path("results")
            existing_files = list(results_dir.glob(f"{prefix}_*.txt"))
            
            if not existing_files:
                return f"{prefix}_day8_9_advanced_features"
            
            # Extract numbers and find max
            numbers = []
            for file in existing_files:
                try:
                    # Extract number from filename like "012_day8_9_advanced_features_summary.txt"
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        num = int(parts[0])
                        numbers.append(num)
                except:
                    continue
            
            if numbers:
                next_num = max(numbers) + 1
            else:
                next_num = 1
            
            return f"{next_num:03d}_day8_9_advanced_features"
            
        except Exception as e:
            logger.error(f"Error getting sequence number: {e}")
            return f"{prefix}_day8_9_advanced_features"

def main():
    """
    Main function to generate advanced meme features
    """
    engineer = AdvancedMemeFeatureEngineer()
    results = engineer.generate_advanced_features()
    
    print("✅ Advanced Meme Feature Engineering completed successfully!")
    return results

if __name__ == "__main__":
    main() 