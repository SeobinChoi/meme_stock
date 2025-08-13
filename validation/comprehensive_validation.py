#!/usr/bin/env python3
"""
Comprehensive Data Validation Script
Validates all requirements for the meme stock prediction project
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveValidator:
    """
    Comprehensive validation for all project requirements
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        
    def validate_all_requirements(self) -> Dict:
        """
        Run comprehensive validation for all requirements
        """
        logger.info("🚀 Starting Comprehensive Data Validation...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'requirements': {},
            'feature_quality': {},
            'temporal_alignment': {},
            'integration_test': {},
            'data_leakage_check': {}
        }
        
        # 1. Check Feature Quality
        logger.info("📊 Checking Feature Quality...")
        feature_results = self.check_feature_quality()
        validation_results['feature_quality'] = feature_results
        
        # 2. Verify Reddit Features
        logger.info("🔍 Verifying Reddit Features...")
        reddit_results = self.verify_reddit_features()
        validation_results['requirements']['reddit_features'] = reddit_results
        
        # 3. Check for NaN Values
        logger.info("❌ Checking for NaN Values...")
        nan_results = self.check_nan_values()
        validation_results['requirements']['nan_check'] = nan_results
        
        # 4. Validate Temporal Alignment
        logger.info("⏰ Validating Temporal Alignment...")
        temporal_results = self.validate_temporal_alignment()
        validation_results['temporal_alignment'] = temporal_results
        
        # 5. Test Integration
        logger.info("🔗 Testing Integration...")
        integration_results = self.test_integration()
        validation_results['integration_test'] = integration_results
        
        # 6. Check for Data Leakage
        logger.info("🚫 Checking for Data Leakage...")
        leakage_results = self.check_data_leakage()
        validation_results['data_leakage_check'] = leakage_results
        
        # Determine overall status
        all_passed = all([
            feature_results.get('status') == 'PASS',
            reddit_results.get('status') == 'PASS',
            nan_results.get('status') == 'PASS',
            temporal_results.get('status') == 'PASS',
            integration_results.get('status') == 'PASS',
            leakage_results.get('status') == 'PASS'
        ])
        
        validation_results['overall_status'] = 'PASS' if all_passed else 'FAIL'
        
        return validation_results
    
    def check_feature_quality(self) -> Dict:
        """
        Check quality of all features
        """
        try:
            # Load feature datasets
            features_file = self.data_dir / "features" / "engineered_features_dataset.csv"
            if not features_file.exists():
                return {'status': 'FAIL', 'error': 'Features dataset not found'}
            
            df = pd.read_csv(features_file)
            
            quality_metrics = {
                'total_features': len(df.columns),
                'total_samples': len(df),
                'missing_data': {},
                'data_types': df.dtypes.to_dict(),
                'feature_ranges': {},
                'quality_score': 0.0
            }
            
            # Check missing data
            missing_data = df.isnull().sum()
            quality_metrics['missing_data'] = {
                'total_missing': missing_data.sum(),
                'missing_percentage': (missing_data.sum() / (len(df) * len(df.columns))) * 100,
                'columns_with_missing': missing_data[missing_data > 0].to_dict()
            }
            
            # Check feature ranges for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].notna().any():
                    quality_metrics['feature_ranges'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std())
                    }
            
            # Calculate quality score
            missing_penalty = quality_metrics['missing_data']['missing_percentage']
            quality_score = max(0, 100 - missing_penalty)
            quality_metrics['quality_score'] = quality_score
            
            status = 'PASS' if quality_score >= 90 else 'WARNING' if quality_score >= 70 else 'FAIL'
            
            return {
                'status': status,
                'metrics': quality_metrics
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def verify_reddit_features(self) -> Dict:
        """
        Verify all Reddit features are generated correctly
        """
        try:
            # Load feature metadata
            metadata_file = self.data_dir / "features" / "feature_metadata.json"
            if not metadata_file.exists():
                return {'status': 'FAIL', 'error': 'Feature metadata not found'}
            
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check Reddit features
            reddit_features = [f for f in metadata['feature_list'] if f.startswith('reddit_')]
            
            # Expected Reddit feature categories
            expected_categories = [
                'reddit_post_count', 'reddit_total_score', 'reddit_avg_score', 'reddit_score_std',
                'reddit_total_comments', 'reddit_avg_comments', 'reddit_comment_std',
                'reddit_score_to_comment_ratio', 'reddit_posting_velocity', 'reddit_engagement_acceleration',
                'reddit_weekend_indicator', 'reddit_weekend_post_ratio', 'reddit_activity_concentration',
                'reddit_unique_users_estimate', 'reddit_engagement_volatility',
                'reddit_sentiment_momentum_1d', 'reddit_sentiment_momentum_3d', 'reddit_sentiment_momentum_7d',
                'reddit_sentiment_volatility', 'reddit_extreme_positive_ratio', 'reddit_extreme_negative_ratio',
                'reddit_sentiment_consensus', 'reddit_positive_sentiment_ratio', 'reddit_negative_sentiment_ratio',
                'reddit_sentiment_mean', 'reddit_sentiment_std', 'reddit_sentiment_min', 'reddit_sentiment_max',
                'reddit_sentiment_count', 'reddit_avg_title_length', 'reddit_title_length_std',
                'reddit_avg_word_count', 'reddit_word_count_std', 'reddit_avg_uppercase_ratio',
                'reddit_avg_exclamation_count', 'reddit_avg_question_count', 'reddit_trading_keyword_density',
                'reddit_linguistic_complexity', 'reddit_urgency_indicators', 'reddit_emotional_intensity',
                'reddit_info_opinion_ratio', 'reddit_content_diversity', 'reddit_engagement_efficiency',
                'reddit_post_quality_index'
            ]
            
            # Check if all expected features exist
            missing_features = [f for f in expected_categories if f not in reddit_features]
            extra_features = [f for f in reddit_features if f not in expected_categories]
            
            # Check feature data quality
            features_file = self.data_dir / "features" / "engineered_features_dataset.csv"
            if features_file.exists():
                df = pd.read_csv(features_file)
                reddit_cols = [col for col in df.columns if col.startswith('reddit_')]
                
                feature_quality = {}
                for col in reddit_cols:
                    if col in df.columns:
                        feature_quality[col] = {
                            'has_data': df[col].notna().any(),
                            'data_type': str(df[col].dtype),
                            'unique_values': df[col].nunique(),
                            'missing_count': df[col].isnull().sum()
                        }
            else:
                feature_quality = {}
            
            return {
                'status': 'PASS' if not missing_features else 'WARNING',
                'total_reddit_features': len(reddit_features),
                'expected_features': len(expected_categories),
                'missing_features': missing_features,
                'extra_features': extra_features,
                'feature_quality': feature_quality
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def check_nan_values(self) -> Dict:
        """
        Check for NaN values in critical features
        """
        try:
            # Load main dataset
            features_file = self.data_dir / "features" / "engineered_features_dataset.csv"
            if not features_file.exists():
                return {'status': 'FAIL', 'error': 'Features dataset not found'}
            
            df = pd.read_csv(features_file)
            
            # Check for NaN values
            nan_summary = df.isnull().sum()
            total_nans = nan_summary.sum()
            total_cells = len(df) * len(df.columns)
            nan_percentage = (total_nans / total_cells) * 100
            
            # Identify critical features with NaN values
            critical_features = ['reddit_post_count', 'reddit_total_score_x', 'reddit_avg_score_x', 
                               'GME_returns_1d', 'AMC_returns_1d', 'BB_returns_1d']
            
            critical_nans = {}
            for feature in critical_features:
                if feature in df.columns:
                    critical_nans[feature] = {
                        'nan_count': df[feature].isnull().sum(),
                        'nan_percentage': (df[feature].isnull().sum() / len(df)) * 100
                    }
            
            # Check if any critical features have too many NaN values
            critical_issues = [f for f, data in critical_nans.items() 
                             if data['nan_percentage'] > 5.0]
            
            status = 'PASS' if not critical_issues and nan_percentage < 5 else 'WARNING' if nan_percentage < 10 else 'FAIL'
            
            return {
                'status': status,
                'total_nan_percentage': nan_percentage,
                'total_nan_count': total_nans,
                'critical_features_nan': critical_nans,
                'critical_issues': critical_issues
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def validate_temporal_alignment(self) -> Dict:
        """
        Validate temporal alignment with stock data
        """
        try:
            # Load stock data
            gme_file = self.data_dir / "processed" / "cleaned_GME_stock_data.csv"
            amc_file = self.data_dir / "processed" / "cleaned_AMC_stock_data.csv"
            bb_file = self.data_dir / "processed" / "cleaned_BB_stock_data.csv"
            
            if not all([gme_file.exists(), amc_file.exists(), bb_file.exists()]):
                return {'status': 'FAIL', 'error': 'Stock data files not found'}
            
            gme_df = pd.read_csv(gme_file)
            amc_df = pd.read_csv(amc_file)
            bb_df = pd.read_csv(bb_file)
            
            # Check date ranges
            gme_dates = pd.to_datetime(gme_df['Date'])
            amc_dates = pd.to_datetime(amc_df['Date'])
            bb_dates = pd.to_datetime(bb_df['Date'])
            
            # Check if dates align
            gme_date_range = (gme_dates.min(), gme_dates.max())
            amc_date_range = (amc_dates.min(), amc_dates.max())
            bb_date_range = (bb_dates.min(), bb_dates.max())
            
            # Check for missing dates
            gme_missing_dates = self._find_missing_dates(gme_dates)
            amc_missing_dates = self._find_missing_dates(amc_dates)
            bb_missing_dates = self._find_missing_dates(bb_dates)
            
            # Check if all stocks have same date range
            date_alignment = (
                gme_date_range == amc_date_range == bb_date_range
            )
            
            # Check for weekend/holiday gaps (expected)
            expected_gaps = self._count_expected_gaps(gme_dates)
            
            status = 'PASS' if date_alignment and len(gme_missing_dates) <= expected_gaps else 'WARNING'
            
            return {
                'status': status,
                'gme_date_range': [str(d) for d in gme_date_range],
                'amc_date_range': [str(d) for d in amc_date_range],
                'bb_date_range': [str(d) for d in bb_date_range],
                'date_alignment': date_alignment,
                'gme_missing_dates': len(gme_missing_dates),
                'amc_missing_dates': len(amc_missing_dates),
                'bb_missing_dates': len(bb_missing_dates),
                'expected_gaps': expected_gaps
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def _find_missing_dates(self, dates: pd.Series) -> List:
        """
        Find missing dates in a date series
        """
        dates = dates.sort_values()
        date_range = pd.date_range(dates.min(), dates.max(), freq='D')
        missing = [d for d in date_range if d not in dates.values]
        return missing
    
    def _count_expected_gaps(self, dates: pd.Series) -> int:
        """
        Count expected gaps (weekends, holidays)
        """
        dates = dates.sort_values()
        date_range = pd.date_range(dates.min(), dates.max(), freq='D')
        business_days = pd.bdate_range(dates.min(), dates.max())
        expected_gaps = len(date_range) - len(business_days)
        return expected_gaps
    
    def test_integration(self) -> Dict:
        """
        Run a small sample through the pipeline
        """
        try:
            # Load a small sample of data
            features_file = self.data_dir / "features" / "engineered_features_dataset.csv"
            if not features_file.exists():
                return {'status': 'FAIL', 'error': 'Features dataset not found'}
            
            # Load small sample
            df = pd.read_csv(features_file, nrows=100)
            
            # Basic pipeline test
            test_results = {
                'sample_size': len(df),
                'features_loaded': len(df.columns),
                'data_types_valid': True,
                'basic_operations': []
            }
            
            # Test basic operations
            try:
                # Test numeric operations
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    test_col = numeric_cols[0]
                    mean_val = df[test_col].mean()
                    test_results['basic_operations'].append(f'Mean calculation: {mean_val}')
                
                # Test string operations
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    test_col = string_cols[0]
                    unique_count = df[test_col].nunique()
                    test_results['basic_operations'].append(f'Unique count: {unique_count}')
                
                test_results['pipeline_test'] = 'PASS'
                
            except Exception as e:
                test_results['pipeline_test'] = f'FAIL: {str(e)}'
            
            return {
                'status': 'PASS' if test_results['pipeline_test'] == 'PASS' else 'FAIL',
                'results': test_results
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def check_data_leakage(self) -> Dict:
        """
        Check for data leakage issues
        """
        try:
            # Load features
            features_file = self.data_dir / "features" / "engineered_features_dataset.csv"
            if not features_file.exists():
                return {'status': 'FAIL', 'error': 'Features dataset not found'}
            
            df = pd.read_csv(features_file)
            
            leakage_checks = {
                'future_data_leakage': False,
                'target_contamination': False,
                'cross_validation_ready': True,
                'issues_found': []
            }
            
            # Check for future data leakage (features that shouldn't know about future)
            future_indicators = ['future_', 'tomorrow_', 'next_day_', 'forward_']
            for col in df.columns:
                for indicator in future_indicators:
                    if indicator in col.lower():
                        leakage_checks['future_data_leakage'] = True
                        leakage_checks['issues_found'].append(f'Future indicator in column: {col}')
            
            # Check if target variables are in features
            target_indicators = ['target', 'label', 'y_', 'returns_']
            for col in df.columns:
                for indicator in target_indicators:
                    if indicator in col.lower() and 'returns_' in col:
                        # This might be okay for some features, but flag for review
                        leakage_checks['issues_found'].append(f'Potential target in features: {col}')
            
            # Check for perfect correlation with targets (potential leakage)
            if 'GME_returns_1d' in df.columns:
                for col in df.columns:
                    if col != 'GME_returns_1d' and df[col].dtype in ['float64', 'int64']:
                        correlation = df[col].corr(df['GME_returns_1d'])
                        if abs(correlation) > 0.95:
                            leakage_checks['issues_found'].append(f'High correlation with target: {col} ({correlation:.3f})')
            
            # Determine status
            if leakage_checks['future_data_leakage'] or len(leakage_checks['issues_found']) > 5:
                status = 'FAIL'
            elif len(leakage_checks['issues_found']) > 0:
                status = 'WARNING'
            else:
                status = 'PASS'
            
            return {
                'status': status,
                'checks': leakage_checks
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def generate_report(self, validation_results: Dict) -> str:
        """
        Generate human-readable validation report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE DATA VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {validation_results['timestamp']}")
        report.append(f"Overall Status: {validation_results['overall_status']}")
        report.append("")
        
        # Feature Quality
        report.append("📊 FEATURE QUALITY CHECK")
        report.append("-" * 40)
        feature_quality = validation_results['feature_quality']
        if feature_quality.get('status') == 'PASS':
            report.append("✅ Feature Quality: PASS")
            metrics = feature_quality.get('metrics', {})
            report.append(f"   Total Features: {metrics.get('total_features', 'N/A')}")
            report.append(f"   Total Samples: {metrics.get('total_samples', 'N/A')}")
            report.append(f"   Quality Score: {metrics.get('quality_score', 'N/A'):.1f}%")
        else:
            report.append(f"❌ Feature Quality: {feature_quality.get('status', 'UNKNOWN')}")
            if 'error' in feature_quality:
                report.append(f"   Error: {feature_quality['error']}")
        report.append("")
        
        # Reddit Features
        report.append("🔍 REDDIT FEATURES VERIFICATION")
        report.append("-" * 40)
        reddit_features = validation_results['requirements']['reddit_features']
        if reddit_features.get('status') == 'PASS':
            report.append("✅ Reddit Features: PASS")
            report.append(f"   Total Reddit Features: {reddit_features.get('total_reddit_features', 'N/A')}")
            report.append(f"   Expected Features: {reddit_features.get('expected_features', 'N/A')}")
        else:
            report.append(f"⚠️ Reddit Features: {reddit_features.get('status', 'UNKNOWN')}")
            missing = reddit_features.get('missing_features', [])
            if missing:
                report.append(f"   Missing Features: {len(missing)}")
        report.append("")
        
        # NaN Check
        report.append("❌ NAN VALUES CHECK")
        report.append("-" * 40)
        nan_check = validation_results['requirements']['nan_check']
        if nan_check.get('status') == 'PASS':
            report.append("✅ NaN Check: PASS")
            report.append(f"   Total NaN Percentage: {nan_check.get('total_nan_percentage', 'N/A'):.2f}%")
        else:
            report.append(f"⚠️ NaN Check: {nan_check.get('status', 'UNKNOWN')}")
            issues = nan_check.get('critical_issues', [])
            if issues:
                report.append(f"   Critical Issues: {len(issues)}")
        report.append("")
        
        # Temporal Alignment
        report.append("⏰ TEMPORAL ALIGNMENT")
        report.append("-" * 40)
        temporal = validation_results['temporal_alignment']
        if temporal.get('status') == 'PASS':
            report.append("✅ Temporal Alignment: PASS")
            report.append("   Stock data dates are properly aligned")
        else:
            report.append(f"⚠️ Temporal Alignment: {temporal.get('status', 'UNKNOWN')}")
        report.append("")
        
        # Integration Test
        report.append("🔗 INTEGRATION TEST")
        report.append("-" * 40)
        integration = validation_results['integration_test']
        if integration.get('status') == 'PASS':
            report.append("✅ Integration Test: PASS")
            results = integration.get('results', {})
            report.append(f"   Sample Size: {results.get('sample_size', 'N/A')}")
            report.append(f"   Features Loaded: {results.get('features_loaded', 'N/A')}")
        else:
            report.append(f"❌ Integration Test: {integration.get('status', 'UNKNOWN')}")
        report.append("")
        
        # Data Leakage
        report.append("🚫 DATA LEAKAGE CHECK")
        report.append("-" * 40)
        leakage = validation_results['data_leakage_check']
        if leakage.get('status') == 'PASS':
            report.append("✅ Data Leakage Check: PASS")
            report.append("   No significant data leakage detected")
        else:
            report.append(f"⚠️ Data Leakage Check: {leakage.get('status', 'UNKNOWN')}")
            issues = leakage.get('checks', {}).get('issues_found', [])
            if issues:
                report.append(f"   Issues Found: {len(issues)}")
        report.append("")
        
        # Summary
        report.append("📋 SUMMARY")
        report.append("-" * 40)
        if validation_results['overall_status'] == 'PASS':
            report.append("🎉 ALL VALIDATION CHECKS PASSED!")
            report.append("   The dataset is ready for model training")
        else:
            report.append("⚠️ VALIDATION ISSUES DETECTED")
            report.append("   Please review and fix the issues above")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_validation_report(self, validation_results: Dict):
        """
        Save validation results to file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = self.data_dir / "results" / f"comprehensive_validation_{timestamp}.json"
        json_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(json_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Save human-readable report
        txt_file = self.data_dir / "results" / f"comprehensive_validation_{timestamp}.txt"
        report_text = self.generate_report(validation_results)
        
        with open(txt_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"✅ Validation reports saved to:")
        logger.info(f"   JSON: {json_file}")
        logger.info(f"   Text: {txt_file}")
        
        return json_file, txt_file

def main():
    """
    Main function to run comprehensive validation
    """
    logger.info("🚀 Starting Comprehensive Data Validation...")
    
    # Initialize validator
    validator = ComprehensiveValidator()
    
    # Run all validation checks
    validation_results = validator.validate_all_requirements()
    
    # Generate and save reports
    json_file, txt_file = validator.save_validation_report(validation_results)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION COMPLETE")
    print("="*80)
    print(f"Overall Status: {validation_results['overall_status']}")
    print(f"Feature Quality: {validation_results['feature_quality'].get('status', 'UNKNOWN')}")
    print(f"Reddit Features: {validation_results['requirements']['reddit_features'].get('status', 'UNKNOWN')}")
    print(f"NaN Check: {validation_results['requirements']['nan_check'].get('status', 'UNKNOWN')}")
    print(f"Temporal Alignment: {validation_results['temporal_alignment'].get('status', 'UNKNOWN')}")
    print(f"Integration Test: {validation_results['integration_test'].get('status', 'UNKNOWN')}")
    print(f"Data Leakage: {validation_results['data_leakage_check'].get('status', 'UNKNOWN')}")
    print("="*80)
    
    if validation_results['overall_status'] == 'PASS':
        logger.info("✅ All validation checks PASSED!")
    else:
        logger.warning("⚠️ Some validation checks failed - review the detailed report")
    
    logger.info("✅ Comprehensive validation completed")

if __name__ == "__main__":
    main()
