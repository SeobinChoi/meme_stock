"""
Main Pipeline for Meme Stock Prediction - Week 1 Implementation
Academic Competition Project
"""

import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  📊 MEME STOCK PREDICTION PROJECT - WEEK 1 IMPLEMENTATION   ║
    ║                                                              ║
    ║  🎯 Academic Competition-Winning Baseline Model             ║
    ║  🚀 Data Preprocessing + Feature Engineering + ML Models     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required libraries are available"""
    print("🔍 Checking dependencies...")
    
    required_libs = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn'
    ]
    
    optional_libs = [
        'lightgbm', 'xgboost', 'tensorflow', 'transformers'
    ]
    
    missing_required = []
    missing_optional = []
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✅ {lib}")
        except ImportError:
            missing_required.append(lib)
            print(f"❌ {lib}")
    
    for lib in optional_libs:
        try:
            __import__(lib)
            print(f"✅ {lib} (optional)")
        except ImportError:
            missing_optional.append(lib)
            print(f"⚠️ {lib} (optional) - not available")
    
    if missing_required:
        print(f"\n❌ Missing required libraries: {missing_required}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional libraries: {missing_optional}")
        print("Some advanced features may not be available.")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        '../data',
        '../models',
        '../results',
        '../logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}")

def run_data_preprocessing():
    """Run data preprocessing pipeline"""
    print("\n" + "="*60)
    print("🚀 STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    try:
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.run_full_pipeline()
        
        if processed_data is not None:
            print("✅ Data preprocessing completed successfully!")
            return True
        else:
            print("❌ Data preprocessing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return False

def run_feature_engineering():
    """Run feature engineering pipeline"""
    print("\n" + "="*60)
    print("🚀 STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    try:
        from feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        features_data = engineer.run_full_pipeline()
        
        if features_data is not None:
            print("✅ Feature engineering completed successfully!")
            return True
        else:
            print("❌ Feature engineering failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return False

def run_model_training():
    """Run model training pipeline"""
    print("\n" + "="*60)
    print("🚀 STEP 3: MODEL TRAINING")
    print("="*60)
    
    try:
        from models import BaselineModels
        
        models = BaselineModels()
        if models.load_data():
            models.train_all_models()
            models.save_models()
            models.generate_performance_report()
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False

def run_evaluation():
    """Run evaluation pipeline"""
    print("\n" + "="*60)
    print("🚀 STEP 4: MODEL EVALUATION")
    print("="*60)
    
    try:
        from evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        report = evaluator.run_full_evaluation()
        
        if report is not None:
            print("✅ Model evaluation completed successfully!")
            return True
        else:
            print("❌ Model evaluation failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in model evaluation: {e}")
        return False

def generate_final_report():
    """Generate final comprehensive report"""
    print("\n" + "="*60)
    print("📋 GENERATING FINAL REPORT")
    print("="*60)
    
    try:
        # Read performance data
        import pandas as pd
        
        performance_file = '../data/baseline_performance.csv'
        if os.path.exists(performance_file):
            performance_df = pd.read_csv(performance_file)
            
            print("\n📊 BASELINE MODEL PERFORMANCE SUMMARY")
            print("-" * 50)
            print(performance_df.to_string(index=False))
            
            # Save final report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f'../results/week1_final_report_{timestamp}.txt'
            
            with open(report_file, 'w') as f:
                f.write("MEME STOCK PREDICTION PROJECT - WEEK 1 FINAL REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("MODEL PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                f.write(performance_df.to_string(index=False))
                f.write("\n\n")
                f.write("DELIVERABLES:\n")
                f.write("-" * 15 + "\n")
                f.write("✅ Preprocessed data: ../data/processed_data.csv\n")
                f.write("✅ Feature-engineered data: ../data/features_data.csv\n")
                f.write("✅ Trained models: ../models/\n")
                f.write("✅ Performance report: ../data/baseline_performance.csv\n")
                f.write("✅ Feature importance: ../data/feature_importance.png\n")
                f.write("✅ Model comparison: ../data/model_comparison.png\n")
                f.write("✅ Predictions plots: ../data/predictions_*.png\n")
            
            print(f"\n✅ Final report saved to {report_file}")
            
        else:
            print("❌ Performance data not found!")
            
    except Exception as e:
        print(f"❌ Error generating final report: {e}")

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install required libraries.")
        return
    
    # Create directories
    create_directories()
    
    # Run pipeline steps
    steps = [
        ("Data Preprocessing", run_data_preprocessing),
        ("Feature Engineering", run_feature_engineering),
        ("Model Training", run_model_training),
        ("Model Evaluation", run_evaluation)
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
    
    # Generate final report
    generate_final_report()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("🎉 PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"✅ Completed steps: {success_count}/{len(steps)}")
    print(f"⏱️ Total execution time: {duration:.2f} seconds")
    
    if success_count == len(steps):
        print("\n🎯 WEEK 1 IMPLEMENTATION COMPLETED SUCCESSFULLY!")
        print("📊 Ready for academic competition presentation")
        print("🚀 Foundation set for Week 2 meme-specific enhancements")
    else:
        print(f"\n⚠️ {len(steps) - success_count} steps failed")
        print("Please check the logs and fix any issues")
    
    print("\n📁 Check the following directories for results:")
    print("   - ../data/ (processed data and features)")
    print("   - ../models/ (trained models)")
    print("   - ../results/ (final reports)")

if __name__ == "__main__":
    main() 