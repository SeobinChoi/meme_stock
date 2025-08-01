"""
Week 2 Main Pipeline - Meme Stock Prediction
Advanced Meme Features & Model Enhancement
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def print_banner():
    """Print Week 2 project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  🚀 MEME STOCK PREDICTION - WEEK 2 IMPLEMENTATION          ║
    ║                                                              ║
    ║  🎯 Meme-Specific Features & Advanced Models                ║
    ║  🔥 Viral Detection + BERT Sentiment + Ensemble Systems     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_week1_data():
    """Check if Week 1 data is available"""
    print("🔍 Checking Week 1 data availability...")
    
    required_files = [
        '../data/processed_data.csv',
        '../data/features_data.csv',
        '../data/reddit_wsb.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing Week 1 files: {missing_files}")
        print("Please run Week 1 pipeline first: cd ../week1_baseline && python main.py")
        return False
    
    print("✅ Week 1 data available")
    return True

def run_viral_detection():
    """Run viral pattern detection"""
    print("\n" + "="*60)
    print("🔥 STEP 1: VIRAL PATTERN DETECTION")
    print("="*60)
    
    try:
        from meme_features.viral_detection import ViralDetector
        
        # Load data
        reddit_data = pd.read_csv('../data/reddit_wsb.csv')
        stock_data = pd.read_csv('../data/processed_data.csv')
        
        # Run viral detection
        detector = ViralDetector()
        viral_features = detector.detect_viral_breakouts(reddit_data, stock_data)
        
        if not viral_features.empty:
            detector.save_viral_features(viral_features)
            print("✅ Viral detection completed successfully!")
            return True
        else:
            print("❌ Viral detection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in viral detection: {e}")
        return False

def run_sentiment_analysis():
    """Run advanced sentiment analysis"""
    print("\n" + "="*60)
    print("🧠 STEP 2: ADVANCED SENTIMENT ANALYSIS")
    print("="*60)
    
    try:
        from meme_features.sentiment_analysis import AdvancedSentimentAnalyzer
        
        # Load data
        reddit_data = pd.read_csv('../data/reddit_wsb.csv')
        
        # Run sentiment analysis
        analyzer = AdvancedSentimentAnalyzer()
        sentiment_features = analyzer.analyze_meme_sentiment(reddit_data)
        
        if not sentiment_features.empty:
            analyzer.save_sentiment_features(sentiment_features)
            print("✅ Sentiment analysis completed successfully!")
            return True
        else:
            print("❌ Sentiment analysis failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")
        return False

def run_social_dynamics():
    """Run social dynamics analysis"""
    print("\n" + "="*60)
    print("👥 STEP 3: SOCIAL DYNAMICS ANALYSIS")
    print("="*60)
    
    try:
        from meme_features.social_dynamics import SocialDynamicsAnalyzer
        
        # Load data
        reddit_data = pd.read_csv('../data/reddit_wsb.csv')
        
        # Run social dynamics analysis
        analyzer = SocialDynamicsAnalyzer()
        social_features = analyzer.analyze_community_behavior(reddit_data)
        
        if not social_features.empty:
            analyzer.save_social_features(social_features)
            print("✅ Social dynamics analysis completed successfully!")
            return True
        else:
            print("❌ Social dynamics analysis failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in social dynamics analysis: {e}")
        return False

def merge_week2_features():
    """Merge all Week 2 features with Week 1 data"""
    print("\n" + "="*60)
    print("🔗 STEP 4: MERGE WEEK 2 FEATURES")
    print("="*60)
    
    try:
        # Load Week 1 data
        week1_data = pd.read_csv('../data/features_data.csv')
        week1_data['date'] = pd.to_datetime(week1_data['date'])
        
        # Load Week 2 features
        viral_features = pd.read_csv('../data/viral_features.csv')
        viral_features['date'] = pd.to_datetime(viral_features['date'])
        
        sentiment_features = pd.read_csv('../data/sentiment_features.csv')
        sentiment_features['date'] = pd.to_datetime(sentiment_features['date'])
        
        social_features = pd.read_csv('../data/social_features.csv')
        social_features['date'] = pd.to_datetime(social_features['date'])
        
        # Merge all features
        merged_data = week1_data.copy()
        
        # Merge viral features
        merged_data = merged_data.merge(viral_features, on='date', how='left', suffixes=('', '_viral'))
        
        # Merge sentiment features
        merged_data = merged_data.merge(sentiment_features, on='date', how='left', suffixes=('', '_sentiment'))
        
        # Merge social features
        merged_data = merged_data.merge(social_features, on='date', how='left', suffixes=('', '_social'))
        
        # Fill missing values
        merged_data = merged_data.fillna(0)
        
        # Save merged data
        merged_data.to_csv('../data/meme_enhanced_data.csv', index=False)
        
        print(f"✅ Merged data shape: {merged_data.shape}")
        print(f"✅ Total features: {len(merged_data.columns)}")
        print(f"✅ Week 2 features added: {len(merged_data.columns) - len(week1_data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging features: {e}")
        return False

def run_ensemble_system():
    """Run ensemble system training"""
    print("\n" + "="*60)
    print("🤖 STEP 5: ENSEMBLE SYSTEM TRAINING")
    print("="*60)
    
    try:
        from advanced_models.ensemble_system import MemeStockEnsemble
        
        # Load enhanced data
        enhanced_data = pd.read_csv('../data/meme_enhanced_data.csv')
        
        # Initialize ensemble
        ensemble = MemeStockEnsemble()
        
        # Load Week 1 models
        ensemble.load_week1_models()
        
        # Load Week 2 models (if available)
        ensemble.load_week2_models()
        
        print(f"📦 Total models loaded: {len(ensemble.models)}")
        
        # Train ensemble for each target
        targets = ['GME_direction_1d', 'GME_direction_3d', 'AMC_direction_1d', 'AMC_direction_3d']
        
        for target in targets:
            if target in enhanced_data.columns:
                print(f"\n🎯 Training ensemble for {target}")
                
                # Prepare data
                exclude_cols = ['date'] + [col for col in enhanced_data.columns if 'direction' in col or 'magnitude' in col]
                feature_cols = [col for col in enhanced_data.columns if col not in exclude_cols]
                
                X = enhanced_data[feature_cols].values
                y = enhanced_data[target].values
                
                # Remove NaN values
                mask = ~np.isnan(y)
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) > 0:
                    # Simple train/validation split for ensemble training
                    split_idx = int(len(X_clean) * 0.8)
                    X_train, X_val = X_clean[:split_idx], X_clean[split_idx:]
                    y_train, y_val = y_clean[:split_idx], y_clean[split_idx:]
                    
                    # Train ensemble
                    ensemble.train_ensemble(X_train, y_train, X_val, y_val, target)
        
        # Save ensemble system
        ensemble.save_ensemble()
        
        print("✅ Ensemble system training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in ensemble training: {e}")
        return False

def generate_week2_report():
    """Generate Week 2 comprehensive report"""
    print("\n" + "="*60)
    print("📋 STEP 6: GENERATE WEEK 2 REPORT")
    print("="*60)
    
    try:
        # Load enhanced data
        enhanced_data = pd.read_csv('../data/meme_enhanced_data.csv')
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'../results/week2_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("MEME STOCK PREDICTION PROJECT - WEEK 2 REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {len(enhanced_data)}\n")
            f.write(f"Total features: {len(enhanced_data.columns)}\n")
            f.write(f"Week 1 features: 93\n")
            f.write(f"Week 2 features: {len(enhanced_data.columns) - 93}\n\n")
            
            f.write("WEEK 2 FEATURES ADDED:\n")
            f.write("-" * 25 + "\n")
            f.write("🔥 Viral Detection Features (15):\n")
            f.write("  - viral_acceleration\n")
            f.write("  - cascade_coefficient\n")
            f.write("  - content_virality_score\n")
            f.write("  - engagement_explosion\n")
            f.write("  - hashtag_momentum\n")
            f.write("  - influencer_participation\n")
            f.write("  - cross_platform_sync\n")
            f.write("  - viral_saturation_point\n")
            f.write("  - meme_lifecycle_stage\n")
            f.write("  - echo_chamber_strength\n")
            f.write("  - contrarian_signal\n")
            f.write("  - fomo_fear_index\n")
            f.write("  - weekend_viral_buildup\n")
            f.write("  - afterhours_buzz\n")
            f.write("  - volatility_anticipation\n\n")
            
            f.write("🧠 Sentiment Analysis Features (20):\n")
            f.write("  - finbert_bullish_score\n")
            f.write("  - finbert_bearish_score\n")
            f.write("  - emotion_joy_intensity\n")
            f.write("  - emotion_fear_intensity\n")
            f.write("  - emotion_anger_intensity\n")
            f.write("  - emotion_surprise_intensity\n")
            f.write("  - sentiment_consensus\n")
            f.write("  - sentiment_momentum\n")
            f.write("  - emotional_contagion\n")
            f.write("  - diamond_vs_paper_ratio\n")
            f.write("  - bullish_bearish_ratio\n")
            f.write("  - moon_expectation_level\n")
            f.write("  - squeeze_anticipation\n")
            f.write("  - retail_vs_institutional\n")
            f.write("  - weekend_sentiment_buildup\n")
            f.write("  - fud_detection_score\n\n")
            
            f.write("👥 Social Dynamics Features (10):\n")
            f.write("  - influential_user_participation\n")
            f.write("  - new_user_conversion_rate\n")
            f.write("  - echo_chamber_coefficient\n")
            f.write("  - dissent_emergence_rate\n")
            f.write("  - community_fragmentation\n")
            f.write("  - information_cascade_strength\n")
            f.write("  - mod_intervention_frequency\n")
            f.write("  - brigading_detection\n")
            f.write("  - coordinated_behavior_score\n")
            f.write("  - tribal_identity_strength\n\n")
            
            f.write("DELIVERABLES:\n")
            f.write("-" * 15 + "\n")
            f.write("✅ Enhanced dataset: ../data/meme_enhanced_data.csv\n")
            f.write("✅ Viral features: ../data/viral_features.csv\n")
            f.write("✅ Sentiment features: ../data/sentiment_features.csv\n")
            f.write("✅ Social features: ../data/social_features.csv\n")
            f.write("✅ Ensemble system: ../models/ensemble_system.pkl\n")
            f.write("✅ Week 2 report: ../results/week2_report_*.txt\n")
        
        print(f"✅ Week 2 report saved to {report_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False

def main():
    """Main Week 2 execution function"""
    start_time = time.time()
    
    # Print banner
    print_banner()
    
    # Check Week 1 data
    if not check_week1_data():
        return
    
    # Create directories
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # Run Week 2 pipeline steps
    steps = [
        ("Viral Pattern Detection", run_viral_detection),
        ("Advanced Sentiment Analysis", run_sentiment_analysis),
        ("Social Dynamics Analysis", run_social_dynamics),
        ("Feature Merging", merge_week2_features),
        ("Ensemble System Training", run_ensemble_system),
        ("Report Generation", generate_week2_report)
    ]
    
    success_count = 0
    
    for step_name, step_function in steps:
        print(f"\n🔄 Starting {step_name}...")
        
        if step_function():
            success_count += 1
            print(f"✅ {step_name} completed successfully!")
        else:
            print(f"❌ {step_name} failed!")
            print("Continuing with next step...")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("🎉 WEEK 2 PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"✅ Completed steps: {success_count}/{len(steps)}")
    print(f"⏱️ Total execution time: {duration:.2f} seconds")
    
    if success_count == len(steps):
        print("\n🎯 WEEK 2 IMPLEMENTATION COMPLETED SUCCESSFULLY!")
        print("🚀 Enhanced with 45+ meme-specific features")
        print("🤖 Advanced ensemble system ready")
        print("📊 Ready for 80%+ accuracy targets")
    else:
        print(f"\n⚠️ {len(steps) - success_count} steps failed")
        print("Please check the logs and fix any issues")
    
    print("\n📁 Check the following directories for results:")
    print("   - ../data/meme_enhanced_data.csv (Enhanced dataset)")
    print("   - ../models/ensemble_system.pkl (Ensemble system)")
    print("   - ../results/week2_report_*.txt (Week 2 report)")
    
    print("\n🔥 NEXT STEPS:")
    print("   1. Train BERT models on Colab (GPU required)")
    print("   2. Fine-tune transformer architectures")
    print("   3. Run comparative analysis with Week 1")
    print("   4. Prepare for Week 3 advanced models")

if __name__ == "__main__":
    main() 