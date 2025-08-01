"""
Viral Pattern Detection for Meme Stock Prediction
Week 2 Implementation - Advanced Meme Features
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ViralDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def detect_viral_breakouts(self, reddit_df, stock_data):
        """
        Detect when posts are going viral and create viral-specific features
        Target: 15 viral-specific features
        """
        print("🔥 Detecting viral patterns...")
        
        # Ensure we have the required columns
        required_cols = ['timestamp', 'score', 'comms_num', 'title', 'body']
        if not all(col in reddit_df.columns for col in required_cols):
            print("❌ Missing required columns for viral detection")
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        reddit_df['timestamp'] = pd.to_datetime(reddit_df['timestamp'])
        reddit_df['date'] = reddit_df['timestamp'].dt.date
        
        # Aggregate daily metrics
        daily_metrics = self._aggregate_daily_metrics(reddit_df)
        
        # Generate viral features
        viral_features = self._generate_viral_features(daily_metrics, stock_data)
        
        print(f"✅ Generated {len(viral_features.columns)} viral features")
        return viral_features
    
    def _aggregate_daily_metrics(self, reddit_df):
        """Aggregate Reddit metrics by day"""
        daily_agg = reddit_df.groupby('date').agg({
            'score': ['sum', 'mean', 'count', 'std'],
            'comms_num': ['sum', 'mean', 'count', 'std'],
            'title': 'count',
            'body': 'count'
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = ['date'] + [f'{col[0]}_{col[1]}' for col in daily_agg.columns[1:]]
        
        return daily_agg
    
    def _generate_viral_features(self, daily_metrics, stock_data):
        """Generate 15 viral-specific features"""
        features = daily_metrics.copy()
        
        # 1. Viral Acceleration - Rate of exponential growth in mentions
        features['viral_acceleration'] = self._calculate_exponential_growth(
            features['title_count']
        )
        
        # 2. Cascade Coefficient - New user participation rate
        features['cascade_coefficient'] = self._calculate_cascade_coefficient(
            features['comms_num_count']
        )
        
        # 3. Content Virality Score - Unique content spreading speed
        features['content_virality_score'] = self._calculate_content_virality(
            features['title_count'], features['score_sum']
        )
        
        # 4. Engagement Explosion - Sudden spike in comments/upvotes
        features['engagement_explosion'] = self._detect_engagement_spikes(
            features['score_sum'], features['comms_num_sum']
        )
        
        # 5. Hashtag Momentum - Trending keyword velocity
        features['hashtag_momentum'] = self._calculate_hashtag_momentum(
            features['title_count']
        )
        
        # 6. Influencer Participation - High-karma user involvement
        features['influencer_participation'] = self._detect_influencer_activity(
            features['score_mean'], features['comms_num_mean']
        )
        
        # 7. Cross Platform Sync - Multi-platform mention alignment
        features['cross_platform_sync'] = self._calculate_cross_platform_sync(
            features['title_count'], features['score_sum']
        )
        
        # 8. Viral Saturation Point - Peak virality detection
        features['viral_saturation_point'] = self._detect_saturation_point(
            features['title_count']
        )
        
        # 9. Meme Lifecycle Stage - Birth/growth/peak/decline phases
        features['meme_lifecycle_stage'] = self._classify_lifecycle_stage(
            features['title_count']
        )
        
        # 10. Echo Chamber Strength - Community consensus intensity
        features['echo_chamber_strength'] = self._calculate_echo_chamber(
            features['score_std'], features['comms_num_std']
        )
        
        # 11. Contrarian Signal - Counter-narrative emergence
        features['contrarian_signal'] = self._detect_contrarian_signals(
            features['score_mean'], features['comms_num_mean']
        )
        
        # 12. FOMO Fear Index - Fear of missing out indicators
        features['fomo_fear_index'] = self._calculate_fomo_index(
            features['title_count'], features['score_sum']
        )
        
        # 13. Weekend Viral Buildup - Weekend accumulation patterns
        features['weekend_viral_buildup'] = self._detect_weekend_buildup(
            features['date'], features['title_count']
        )
        
        # 14. Afterhours Buzz - Post-market discussion intensity
        features['afterhours_buzz'] = self._calculate_afterhours_activity(
            features['date'], features['comms_num_sum']
        )
        
        # 15. Volatility Anticipation - Pre-movement social signals
        features['volatility_anticipation'] = self._anticipate_volatility(
            features, stock_data
        )
        
        return features
    
    def _calculate_exponential_growth(self, series, window=7):
        """Calculate exponential growth rate"""
        growth_rates = []
        for i in range(window, len(series)):
            recent_values = series.iloc[i-window:i+1]
            if recent_values.sum() > 0:
                # Calculate growth rate using log-linear regression
                x = np.arange(len(recent_values))
                y = np.log(recent_values + 1)  # Add 1 to avoid log(0)
                slope, _, _, _, _ = stats.linregress(x, y)
                growth_rates.append(slope)
            else:
                growth_rates.append(0)
        
        # Pad with zeros for initial periods
        return pd.Series([0] * window + growth_rates, index=series.index)
    
    def _calculate_cascade_coefficient(self, series, window=5):
        """Calculate cascade coefficient (new user participation)"""
        cascade_rates = []
        for i in range(window, len(series)):
            recent = series.iloc[i-window:i+1]
            if recent.iloc[-1] > recent.iloc[:-1].mean():
                # Calculate participation growth rate
                growth_rate = (recent.iloc[-1] - recent.iloc[:-1].mean()) / recent.iloc[:-1].mean()
                cascade_rates.append(growth_rate)
            else:
                cascade_rates.append(0)
        
        return pd.Series([0] * window + cascade_rates, index=series.index)
    
    def _calculate_content_virality(self, post_count, engagement):
        """Calculate content virality score"""
        # Normalize by post count to get engagement per post
        engagement_per_post = engagement / (post_count + 1)
        
        # Calculate rolling z-score to identify viral content
        rolling_mean = engagement_per_post.rolling(window=7).mean()
        rolling_std = engagement_per_post.rolling(window=7).std()
        virality_score = (engagement_per_post - rolling_mean) / (rolling_std + 1e-8)
        
        return virality_score
    
    def _detect_engagement_spikes(self, score_sum, comms_sum, threshold=2.0):
        """Detect sudden spikes in engagement"""
        # Combine score and comments
        total_engagement = score_sum + comms_sum * 10  # Weight comments
        
        # Calculate rolling statistics
        rolling_mean = total_engagement.rolling(window=7).mean()
        rolling_std = total_engagement.rolling(window=7).std()
        
        # Detect spikes (values > mean + threshold * std)
        spikes = (total_engagement > (rolling_mean + threshold * rolling_std)).astype(int)
        
        return spikes
    
    def _calculate_hashtag_momentum(self, post_count, window=3):
        """Calculate hashtag/keyword momentum"""
        momentum = []
        for i in range(window, len(post_count)):
            recent = post_count.iloc[i-window:i+1]
            # Calculate momentum as rate of change
            momentum_val = (recent.iloc[-1] - recent.iloc[0]) / (recent.iloc[0] + 1)
            momentum.append(momentum_val)
        
        return pd.Series([0] * window + momentum, index=post_count.index)
    
    def _detect_influencer_activity(self, score_mean, comms_mean):
        """Detect high-karma user involvement"""
        # High engagement per post indicates influencer activity
        influencer_score = (score_mean * 0.7 + comms_mean * 0.3) / 100
        
        # Normalize to 0-1 scale
        influencer_score = (influencer_score - influencer_score.min()) / (influencer_score.max() - influencer_score.min() + 1e-8)
        
        return influencer_score
    
    def _calculate_cross_platform_sync(self, post_count, engagement):
        """Calculate cross-platform synchronization"""
        # Simulate cross-platform sync using engagement/post ratio
        sync_ratio = engagement / (post_count + 1)
        
        # Calculate correlation with previous day (simulating platform alignment)
        sync_score = sync_ratio.rolling(window=2).corr(sync_ratio.shift(1))
        
        return sync_score.fillna(0)
    
    def _detect_saturation_point(self, post_count, window=14):
        """Detect viral saturation point"""
        saturation_signals = []
        for i in range(window, len(post_count)):
            recent = post_count.iloc[i-window:i+1]
            
            # Check for peak followed by decline
            peak_idx = recent.idxmax()
            if peak_idx < recent.index[-1]:
                # Calculate decline rate after peak
                decline_rate = (recent.iloc[-1] - recent.max()) / (recent.max() + 1)
                saturation_signals.append(decline_rate)
            else:
                saturation_signals.append(0)
        
        return pd.Series([0] * window + saturation_signals, index=post_count.index)
    
    def _classify_lifecycle_stage(self, post_count, window=7):
        """Classify meme lifecycle stage (0=birth, 1=growth, 2=peak, 3=decline)"""
        stages = []
        for i in range(window, len(post_count)):
            recent = post_count.iloc[i-window:i+1]
            
            # Calculate growth trend
            x = np.arange(len(recent))
            y = recent.values
            slope, _, _, _, _ = stats.linregress(x, y)
            
            # Classify based on slope and current value
            if slope > 0.1 and recent.iloc[-1] < recent.max() * 0.8:
                stage = 1  # Growth
            elif slope > 0.1 and recent.iloc[-1] >= recent.max() * 0.8:
                stage = 2  # Peak
            elif slope < -0.1:
                stage = 3  # Decline
            else:
                stage = 0  # Birth/stable
            
            stages.append(stage)
        
        return pd.Series([0] * window + stages, index=post_count.index)
    
    def _calculate_echo_chamber(self, score_std, comms_std):
        """Calculate echo chamber strength (low variance = high echo chamber)"""
        # Combine standard deviations
        combined_std = (score_std + comms_std) / 2
        
        # Invert so higher values = stronger echo chamber
        echo_strength = 1 / (combined_std + 1)
        
        return echo_strength
    
    def _detect_contrarian_signals(self, score_mean, comms_mean):
        """Detect contrarian signals (unusual patterns)"""
        # Calculate expected engagement based on recent history
        expected_score = score_mean.rolling(window=7).mean()
        expected_comms = comms_mean.rolling(window=7).mean()
        
        # Detect deviations from expected patterns
        score_deviation = abs(score_mean - expected_score) / (expected_score + 1)
        comms_deviation = abs(comms_mean - expected_comms) / (expected_comms + 1)
        
        contrarian_signal = (score_deviation + comms_deviation) / 2
        
        return contrarian_signal
    
    def _calculate_fomo_index(self, post_count, engagement):
        """Calculate FOMO (Fear of Missing Out) index"""
        # FOMO increases with rapid growth and high engagement
        growth_rate = post_count.pct_change().rolling(window=3).mean()
        engagement_intensity = engagement / (post_count + 1)
        
        # Combine growth and engagement for FOMO score
        fomo_score = growth_rate * engagement_intensity
        
        return fomo_score.fillna(0)
    
    def _detect_weekend_buildup(self, dates, post_count):
        """Detect weekend viral buildup patterns"""
        weekend_buildup = []
        
        for i, date in enumerate(dates):
            if i < 2:  # Need at least 2 days of history
                weekend_buildup.append(0)
                continue
                
            # Check if current day is Monday
            if pd.to_datetime(date).weekday() == 0:  # Monday
                # Compare with previous Friday
                friday_posts = post_count.iloc[i-3] if i >= 3 else 0
                monday_posts = post_count.iloc[i]
                
                # Calculate weekend buildup
                buildup = (monday_posts - friday_posts) / (friday_posts + 1)
                weekend_buildup.append(buildup)
            else:
                weekend_buildup.append(0)
        
        return pd.Series(weekend_buildup, index=dates)
    
    def _calculate_afterhours_activity(self, dates, comms_sum):
        """Calculate afterhours discussion intensity"""
        afterhours_buzz = []
        
        for i, date in enumerate(dates):
            if i < 1:  # Need at least 1 day of history
                afterhours_buzz.append(0)
                continue
                
            # Check if current day is Monday (after weekend)
            if pd.to_datetime(date).weekday() == 0:  # Monday
                # Compare Monday morning activity with Friday evening
                friday_comms = comms_sum.iloc[i-3] if i >= 3 else 0
                monday_comms = comms_sum.iloc[i]
                
                # Afterhours buzz = weekend discussion intensity
                buzz = monday_comms / (friday_comms + 1)
                afterhours_buzz.append(buzz)
            else:
                afterhours_buzz.append(0)
        
        return pd.Series(afterhours_buzz, index=dates)
    
    def _anticipate_volatility(self, features, stock_data):
        """Anticipate volatility based on social signals"""
        volatility_anticipation = []
        
        # Combine multiple viral signals
        viral_signals = (
            features['viral_acceleration'] +
            features['engagement_explosion'] +
            features['fomo_fear_index']
        ) / 3
        
        # Calculate rolling volatility of viral signals
        for i in range(7, len(viral_signals)):
            recent_signals = viral_signals.iloc[i-7:i+1]
            volatility = recent_signals.std()
            volatility_anticipation.append(volatility)
        
        # Pad with zeros for initial periods
        return pd.Series([0] * 7 + volatility_anticipation, index=viral_signals.index)
    
    def save_viral_features(self, features, filename='viral_features.csv'):
        """Save viral features to file"""
        features.to_csv(f'../data/{filename}', index=False)
        print(f"✅ Viral features saved to ../data/{filename}")

if __name__ == "__main__":
    # Test viral detection
    detector = ViralDetector()
    
    # Load sample data
    try:
        reddit_data = pd.read_csv('../data/reddit_wsb.csv')
        stock_data = pd.read_csv('../data/processed_data.csv')
        
        viral_features = detector.detect_viral_breakouts(reddit_data, stock_data)
        detector.save_viral_features(viral_features)
        
    except FileNotFoundError:
        print("❌ Data files not found. Please run Week 1 pipeline first.") 