"""
Social Dynamics Analysis for Meme Stock Prediction
Week 2 Implementation - Community Behavior Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SocialDynamicsAnalyzer:
    def __init__(self):
        self.community_patterns = {}
        
    def analyze_community_behavior(self, reddit_df):
        """
        Analyze WSB community dynamics and behavior patterns
        Target: 10 social dynamics features
        """
        print("👥 Analyzing social dynamics...")
        
        # Ensure we have required columns
        required_cols = ['timestamp', 'score', 'comms_num', 'title', 'body']
        if not all(col in reddit_df.columns for col in required_cols):
            print("❌ Missing required columns for social dynamics analysis")
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        reddit_df['timestamp'] = pd.to_datetime(reddit_df['timestamp'])
        reddit_df['date'] = reddit_df['timestamp'].dt.date
        
        # Aggregate daily metrics
        daily_metrics = self._aggregate_daily_metrics(reddit_df)
        
        # Generate social dynamics features
        social_features = self._generate_social_features(daily_metrics, reddit_df)
        
        print(f"✅ Generated {len(social_features.columns)} social dynamics features")
        return social_features
    
    def _aggregate_daily_metrics(self, reddit_df):
        """Aggregate Reddit metrics by day for social analysis"""
        daily_agg = reddit_df.groupby('date').agg({
            'score': ['sum', 'mean', 'count', 'std', 'max'],
            'comms_num': ['sum', 'mean', 'count', 'std'],
            'title': 'count',
            'body': 'count'
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = ['date'] + [f'{col[0]}_{col[1]}' for col in daily_agg.columns[1:]]
        
        return daily_agg
    
    def _generate_social_features(self, daily_metrics, reddit_df):
        """Generate 10 social dynamics features"""
        features = daily_metrics.copy()
        
        # 1. Influential User Participation - Top 1% user activity
        features['influential_user_participation'] = self._track_high_karma_users(
            reddit_df, daily_metrics
        )
        
        # 2. New User Conversion Rate - Lurkers becoming posters
        features['new_user_conversion_rate'] = self._measure_new_member_participation(
            reddit_df, daily_metrics
        )
        
        # 3. Echo Chamber Coefficient - Opinion homogeneity
        features['echo_chamber_coefficient'] = self._calculate_opinion_homogeneity(
            features['score_std'], features['comms_num_std']
        )
        
        # 4. Dissent Emergence Rate - Contrarian opinion growth
        features['dissent_emergence_rate'] = self._detect_dissent_emergence(
            features['score_mean'], features['comms_num_mean']
        )
        
        # 5. Community Fragmentation - Sub-group formation
        features['community_fragmentation'] = self._calculate_community_fragmentation(
            features['score_count'], features['comms_num_count']
        )
        
        # 6. Information Cascade Strength - Follow-the-leader behavior
        features['information_cascade_strength'] = self._measure_cascade_behavior(
            features['score_sum'], features['comms_num_sum']
        )
        
        # 7. Mod Intervention Frequency - Moderation activity
        features['mod_intervention_frequency'] = self._detect_mod_intervention(
            reddit_df, daily_metrics
        )
        
        # 8. Brigading Detection - External influence detection
        features['brigading_detection'] = self._detect_brigading(
            features['score_count'], features['comms_num_count']
        )
        
        # 9. Coordinated Behavior Score - Synchronized posting
        features['coordinated_behavior_score'] = self._detect_coordinated_behavior(
            reddit_df, daily_metrics
        )
        
        # 10. Tribal Identity Strength - "Ape" community cohesion
        features['tribal_identity_strength'] = self._calculate_tribal_cohesion(
            reddit_df, daily_metrics
        )
        
        return features
    
    def _track_high_karma_users(self, reddit_df, daily_metrics):
        """Track high-karma user participation"""
        # Simulate high-karma users based on post scores
        high_karma_threshold = reddit_df['score'].quantile(0.95)  # Top 5%
        
        daily_high_karma = reddit_df[reddit_df['score'] >= high_karma_threshold].groupby('date').agg({
            'score': 'count'
        }).reset_index()
        
        # Merge with daily metrics
        daily_high_karma.columns = ['date', 'high_karma_posts']
        merged = daily_metrics.merge(daily_high_karma, on='date', how='left').fillna(0)
        
        # Calculate participation rate
        participation_rate = merged['high_karma_posts'] / (merged['score_count'] + 1)
        
        return participation_rate
    
    def _measure_new_member_participation(self, reddit_df, daily_metrics):
        """Measure new member participation (simulated)"""
        # Simulate new users based on low-karma posts
        low_karma_threshold = reddit_df['score'].quantile(0.25)  # Bottom 25%
        
        daily_new_users = reddit_df[reddit_df['score'] <= low_karma_threshold].groupby('date').agg({
            'score': 'count'
        }).reset_index()
        
        # Merge with daily metrics
        daily_new_users.columns = ['date', 'new_user_posts']
        merged = daily_metrics.merge(daily_new_users, on='date', how='left').fillna(0)
        
        # Calculate conversion rate
        conversion_rate = merged['new_user_posts'] / (merged['score_count'] + 1)
        
        return conversion_rate
    
    def _calculate_opinion_homogeneity(self, score_std, comms_std):
        """Calculate opinion homogeneity (echo chamber strength)"""
        # Combine standard deviations
        combined_std = (score_std + comms_std) / 2
        
        # Invert so higher values = more homogeneous (stronger echo chamber)
        homogeneity = 1 / (combined_std + 1)
        
        return homogeneity
    
    def _detect_dissent_emergence(self, score_mean, comms_mean):
        """Detect emergence of contrarian opinions"""
        dissent_signals = []
        
        for i in range(7, len(score_mean)):
            # Calculate recent trend
            recent_score_trend = score_mean.iloc[i-7:i+1].pct_change().mean()
            recent_comms_trend = comms_mean.iloc[i-7:i+1].pct_change().mean()
            
            # Detect if engagement is increasing while sentiment is decreasing
            if recent_comms_trend > 0.1 and recent_score_trend < -0.05:
                dissent_signals.append(1)  # Dissent detected
            else:
                dissent_signals.append(0)
        
        # Pad with zeros for initial periods
        return pd.Series([0] * 7 + dissent_signals, index=score_mean.index)
    
    def _calculate_community_fragmentation(self, post_count, comms_count):
        """Calculate community fragmentation"""
        fragmentation_scores = []
        
        for i in range(7, len(post_count)):
            # Calculate variance in engagement patterns
            recent_posts = post_count.iloc[i-7:i+1]
            recent_comms = comms_count.iloc[i-7:i+1]
            
            # High variance indicates fragmentation
            post_variance = recent_posts.var() / (recent_posts.mean() + 1)
            comms_variance = recent_comms.var() / (recent_comms.mean() + 1)
            
            fragmentation = (post_variance + comms_variance) / 2
            fragmentation_scores.append(fragmentation)
        
        # Pad with zeros for initial periods
        return pd.Series([0] * 7 + fragmentation_scores, index=post_count.index)
    
    def _measure_cascade_behavior(self, score_sum, comms_sum):
        """Measure information cascade strength"""
        cascade_strength = []
        
        for i in range(5, len(score_sum)):
            # Calculate autocorrelation in engagement
            recent_scores = score_sum.iloc[i-5:i+1]
            recent_comms = comms_sum.iloc[i-5:i+1]
            
            # High autocorrelation indicates cascade behavior
            score_autocorr = recent_scores.autocorr() if len(recent_scores) > 1 else 0
            comms_autocorr = recent_comms.autocorr() if len(recent_comms) > 1 else 0
            
            cascade = (score_autocorr + comms_autocorr) / 2
            cascade_strength.append(cascade if not pd.isna(cascade) else 0)
        
        # Pad with zeros for initial periods
        return pd.Series([0] * 5 + cascade_strength, index=score_sum.index)
    
    def _detect_mod_intervention(self, reddit_df, daily_metrics):
        """Detect moderation intervention (simulated)"""
        # Simulate mod intervention based on unusual activity patterns
        intervention_signals = []
        
        for i, date in enumerate(daily_metrics['date']):
            if i < 7:  # Need history for comparison
                intervention_signals.append(0)
                continue
            
            # Check for unusual spikes in activity
            recent_posts = daily_metrics.iloc[i-7:i+1]['score_count']
            recent_comms = daily_metrics.iloc[i-7:i+1]['comms_num_count']
            
            # Calculate z-scores
            post_zscore = (recent_posts.iloc[-1] - recent_posts.mean()) / (recent_posts.std() + 1e-8)
            comms_zscore = (recent_comms.iloc[-1] - recent_comms.mean()) / (recent_comms.std() + 1e-8)
            
            # High z-scores might trigger mod intervention
            if abs(post_zscore) > 3 or abs(comms_zscore) > 3:
                intervention_signals.append(1)
            else:
                intervention_signals.append(0)
        
        return pd.Series(intervention_signals, index=daily_metrics.index)
    
    def _detect_brigading(self, post_count, comms_count):
        """Detect brigading (coordinated external influence)"""
        brigading_signals = []
        
        for i in range(7, len(post_count)):
            # Look for sudden coordinated activity
            recent_posts = post_count.iloc[i-7:i+1]
            recent_comms = comms_count.iloc[i-7:i+1]
            
            # Calculate activity patterns
            post_trend = recent_posts.pct_change().std()
            comms_trend = recent_comms.pct_change().std()
            
            # Low variance in activity might indicate coordinated behavior
            if post_trend < 0.1 and comms_trend < 0.1:
                # Check for sudden spike
                if recent_posts.iloc[-1] > recent_posts.iloc[:-1].mean() * 2:
                    brigading_signals.append(1)
                else:
                    brigading_signals.append(0)
            else:
                brigading_signals.append(0)
        
        # Pad with zeros for initial periods
        return pd.Series([0] * 7 + brigading_signals, index=post_count.index)
    
    def _detect_coordinated_behavior(self, reddit_df, daily_metrics):
        """Detect coordinated behavior patterns"""
        coordinated_signals = []
        
        for i, date in enumerate(daily_metrics['date']):
            if i < 3:  # Need at least 3 days of history
                coordinated_signals.append(0)
                continue
            
            # Check for synchronized posting patterns
            recent_posts = daily_metrics.iloc[i-3:i+1]['score_count']
            recent_comms = daily_metrics.iloc[i-3:i+1]['comms_num_count']
            
            # Calculate correlation between posts and comments
            correlation = recent_posts.corr(recent_comms)
            
            # High correlation might indicate coordination
            if correlation > 0.8:
                coordinated_signals.append(1)
            else:
                coordinated_signals.append(0)
        
        return pd.Series(coordinated_signals, index=daily_metrics.index)
    
    def _calculate_tribal_cohesion(self, reddit_df, daily_metrics):
        """Calculate tribal identity strength ("Ape" community cohesion)"""
        cohesion_scores = []
        
        # Define "ape" keywords that indicate tribal identity
        ape_keywords = ['ape', 'apes', 'diamond hands', 'hodl', 'to the moon', 'tendies', 'gains']
        
        for i, date in enumerate(daily_metrics['date']):
            if i < 1:  # Need at least 1 day of history
                cohesion_scores.append(0)
                continue
            
            # Get posts for this date
            date_posts = reddit_df[reddit_df['date'] == date]
            
            if len(date_posts) == 0:
                cohesion_scores.append(0)
                continue
            
            # Count ape-related keywords
            ape_count = 0
            total_posts = len(date_posts)
            
            for _, post in date_posts.iterrows():
                text = str(post['title']) + ' ' + str(post['body'])
                text = text.lower()
                
                for keyword in ape_keywords:
                    if keyword in text:
                        ape_count += 1
                        break
            
            # Calculate cohesion as percentage of posts with ape keywords
            cohesion = ape_count / total_posts if total_posts > 0 else 0
            cohesion_scores.append(cohesion)
        
        return pd.Series(cohesion_scores, index=daily_metrics.index)
    
    def save_social_features(self, features, filename='social_features.csv'):
        """Save social dynamics features to file"""
        features.to_csv(f'../data/{filename}', index=False)
        print(f"✅ Social dynamics features saved to ../data/{filename}")

if __name__ == "__main__":
    # Test social dynamics analysis
    analyzer = SocialDynamicsAnalyzer()
    
    # Load sample data
    try:
        reddit_data = pd.read_csv('../data/reddit_wsb.csv')
        
        social_features = analyzer.analyze_community_behavior(reddit_data)
        analyzer.save_social_features(social_features)
        
    except FileNotFoundError:
        print("❌ Data files not found. Please run Week 1 pipeline first.") 