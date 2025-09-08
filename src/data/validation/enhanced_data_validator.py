"""
Enhanced Data Quality Validator
Advanced data validation with ML-based quality assessment and multi-source validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports for quality assessment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Data validation libraries
import pandera as pa
from pandera import DataFrameSchema, Column, Check
import cerberus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataValidator:
    """
    Advanced data validator with ML-based quality assessment
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.quality_metrics = {}
        self.validation_results = {}
        self.spam_classifier = None
        
    def create_data_schemas(self):
        """
        Create comprehensive data validation schemas
        """
        # Reddit data schema
        self.reddit_schema = DataFrameSchema({
            'title': Column(str, checks=[
                Check.str_length(min_value=5, max_value=500)
            ]),
            'score': Column(int, checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100000)
            ]),
            'comms_num': Column(int, checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(10000)
            ]),
            'created': Column(str)
        })
        
        # Stock data schema
        self.stock_schema = DataFrameSchema({
            'Date': Column(str),
            'Open': Column(float, checks=[
                Check.greater_than(0)
            ]),
            'High': Column(float, checks=[
                Check.greater_than(0)
            ]),
            'Low': Column(float, checks=[
                Check.greater_than(0)
            ]),
            'Close': Column(float, checks=[
                Check.greater_than(0)
            ]),
            'Volume': Column(int, checks=[
                Check.greater_than_or_equal_to(0)
            ])
        })
        
        logger.info("✅ Data validation schemas created")
    
    def train_spam_classifier(self, reddit_data: pd.DataFrame):
        """
        Train ML-based spam classifier
        """
        logger.info("🤖 Training ML-based spam classifier...")
        
        # Create features for spam detection
        features = self._extract_spam_features(reddit_data)
        
        # Create labels (basic heuristic for now)
        labels = self._create_spam_labels(reddit_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.spam_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.spam_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.spam_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✅ Spam classifier trained with {accuracy:.2%} accuracy")
        
        # Save model
        model_path = self.data_dir / "models" / "spam_classifier.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.spam_classifier, model_path)
        
        return accuracy
    
    def _extract_spam_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for spam detection
        """
        features = pd.DataFrame()
        
        # Text-based features
        features['title_length'] = df['title'].str.len()
        features['word_count'] = df['title'].str.split().str.len()
        features['uppercase_ratio'] = df['title'].str.count(r'[A-Z]') / df['title'].str.len()
        features['exclamation_count'] = df['title'].str.count(r'!')
        features['question_count'] = df['title'].str.count(r'\?')
        features['number_count'] = df['title'].str.count(r'\d')
        
        # Engagement features
        features['score'] = df['score']
        features['comments'] = df['comms_num']
        features['engagement_ratio'] = df['score'] / (df['comms_num'] + 1)
        
        # Spam indicators
        spam_keywords = ['buy', 'sell', 'profit', 'money', 'cash', 'rich', 'free', 'click', 'link']
        for keyword in spam_keywords:
            features[f'has_{keyword}'] = df['title'].str.lower().str.contains(keyword).astype(int)
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def _create_spam_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create spam labels using heuristic rules
        """
        labels = np.zeros(len(df))
        
        # Spam indicators
        spam_conditions = (
            (df['score'] <= 1) |  # Very low score
            (df['title'].str.len() < 10) |  # Very short title
            (df['title'].str.count(r'[A-Z]') / df['title'].str.len() > 0.7) |  # Excessive caps
            (df['title'].str.contains(r'\b(buy|sell|profit|money|cash|rich|free|click|link)\b', case=False))  # Spam keywords
        )
        
        labels[spam_conditions] = 1
        
        return labels
    
    def assess_data_quality(self, dataset_name: str, data: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality assessment
        """
        logger.info(f"📊 Assessing quality for {dataset_name}...")
        
        quality_metrics = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'total_columns': len(data.columns),
            'missing_values': {},
            'data_types': {},
            'quality_scores': {},
            'validation_results': {}
        }
        
        # Missing value analysis
        missing_data = data.isnull().sum()
        quality_metrics['missing_values'] = {
            'total_missing': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': missing_data[missing_data > 0].to_dict()
        }
        
        # Data type analysis
        quality_metrics['data_types'] = data.dtypes.to_dict()
        
        # Quality scores
        quality_metrics['quality_scores'] = {
            'completeness': self._calculate_completeness(data),
            'consistency': self._calculate_consistency(data),
            'accuracy': self._calculate_accuracy(data, dataset_name),
            'timeliness': self._calculate_timeliness(data, dataset_name)
        }
        
        # Overall quality score
        overall_score = np.mean(list(quality_metrics['quality_scores'].values()))
        quality_metrics['overall_quality_score'] = overall_score
        
        # Schema validation
        try:
            if dataset_name == 'reddit':
                self.reddit_schema.validate(data)
                quality_metrics['validation_results']['schema_validation'] = 'PASS'
            elif 'stock' in dataset_name:
                self.stock_schema.validate(data)
                quality_metrics['validation_results']['schema_validation'] = 'PASS'
        except Exception as e:
            quality_metrics['validation_results']['schema_validation'] = f'FAIL: {str(e)}'
        
        self.quality_metrics[dataset_name] = quality_metrics
        return quality_metrics
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """
        Calculate data completeness score
        """
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        return completeness * 100
    
    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """
        Calculate data consistency score
        """
        consistency_checks = []
        
        # Check for duplicate rows
        duplicate_ratio = data.duplicated().sum() / len(data)
        consistency_checks.append(1 - duplicate_ratio)
        
        # Check for consistent data types
        type_consistency = 1.0  # Assume consistent for now
        consistency_checks.append(type_consistency)
        
        # Check for logical consistency (if applicable)
        if 'High' in data.columns and 'Low' in data.columns:
            logical_consistency = (data['High'] >= data['Low']).mean()
            consistency_checks.append(logical_consistency)
        
        return np.mean(consistency_checks) * 100
    
    def _calculate_accuracy(self, data: pd.DataFrame, dataset_name: str) -> float:
        """
        Calculate data accuracy score
        """
        accuracy_checks = []
        
        # Check for reasonable value ranges
        if 'score' in data.columns:
            score_accuracy = ((data['score'] >= 0) & (data['score'] <= 100000)).mean()
            accuracy_checks.append(score_accuracy)
        
        if 'Volume' in data.columns:
            volume_accuracy = (data['Volume'] >= 0).mean()
            accuracy_checks.append(volume_accuracy)
        
        # Check for outliers (basic IQR method)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_ratios = []
        
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_ratio = outliers / len(data)
            outlier_ratios.append(1 - outlier_ratio)
        
        if outlier_ratios:
            accuracy_checks.append(np.mean(outlier_ratios))
        
        return np.mean(accuracy_checks) * 100 if accuracy_checks else 80.0
    
    def _calculate_timeliness(self, data: pd.DataFrame, dataset_name: str) -> float:
        """
        Calculate data timeliness score
        """
        # For now, assume good timeliness if data exists
        # In real implementation, would check against expected update frequency
        return 85.0
    
    def cross_validate_sources(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Cross-validate data across multiple sources
        """
        logger.info("🔍 Cross-validating data across sources...")
        
        validation_results = {
            'temporal_alignment': {},
            'data_consistency': {},
            'quality_comparison': {}
        }
        
        # Temporal alignment check
        if 'reddit' in datasets and any('stock' in k for k in datasets.keys()):
            reddit_dates = pd.to_datetime(datasets['reddit']['created']).dt.date
            stock_dates = pd.to_datetime(datasets[list(datasets.keys())[1]]['Date']).dt.date
            
            common_dates = set(reddit_dates) & set(stock_dates)
            alignment_score = len(common_dates) / len(set(stock_dates))
            
            validation_results['temporal_alignment'] = {
                'alignment_score': alignment_score,
                'common_dates': len(common_dates),
                'total_stock_dates': len(set(stock_dates))
            }
        
        # Quality comparison
        quality_scores = {}
        for name, data in datasets.items():
            quality = self.assess_data_quality(name, data)
            quality_scores[name] = quality['overall_quality_score']
        
        validation_results['quality_comparison'] = quality_scores
        
        return validation_results
    
    def generate_quality_report(self) -> Dict:
        """
        Generate comprehensive quality report
        """
        logger.info("📋 Generating comprehensive quality report...")
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'overall_quality_score': 0,
            'dataset_quality': self.quality_metrics,
            'recommendations': [],
            'quality_improvements': {}
        }
        
        # Calculate overall quality score
        if self.quality_metrics:
            scores = [metrics['overall_quality_score'] for metrics in self.quality_metrics.values()]
            report['overall_quality_score'] = np.mean(scores)
        
        # Generate recommendations
        report['recommendations'] = self._generate_quality_recommendations()
        
        # Quality improvement suggestions
        report['quality_improvements'] = self._suggest_quality_improvements()
        
        return report
    
    def _generate_quality_recommendations(self) -> List[str]:
        """
        Generate quality improvement recommendations
        """
        recommendations = []
        
        for dataset_name, metrics in self.quality_metrics.items():
            overall_score = metrics['overall_quality_score']
            
            if overall_score < 50:
                recommendations.append(f"🚨 {dataset_name}: Critical quality issues - immediate attention required")
            elif overall_score < 70:
                recommendations.append(f"⚠️ {dataset_name}: Quality improvements needed - consider data cleaning")
            elif overall_score < 85:
                recommendations.append(f"✅ {dataset_name}: Good quality - minor improvements possible")
            else:
                recommendations.append(f"🎉 {dataset_name}: Excellent quality - maintain current standards")
        
        return recommendations
    
    def _suggest_quality_improvements(self) -> Dict:
        """
        Suggest specific quality improvements
        """
        improvements = {
            'data_sources': [
                "Add Alpha Vantage API for more complete financial data",
                "Integrate Twitter API for additional social signals",
                "Include StockTwits for financial social media data",
                "Add news sentiment data via NewsAPI"
            ],
            'validation': [
                "Implement real-time data validation",
                "Add automated quality monitoring",
                "Create quality alert system",
                "Build quality dashboard"
            ],
            'cleaning': [
                "Enhance spam detection with ML models",
                "Improve outlier detection algorithms",
                "Add data consistency checks",
                "Implement automated data repair"
            ]
        }
        
        return improvements
    
    def save_quality_report(self, report: Dict):
        """
        Save quality report to file with sequential numbering
        """
        import json
        
        # Get next sequence number
        sequence_num = self._get_next_sequence_number("quality_report")
        
        # Save detailed report (JSON for internal use only)
        internal_file = Path("results") / f"{sequence_num:03d}_quality_report_internal.json"
        internal_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(internal_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = Path("results") / f"{sequence_num:03d}_quality_report.txt"
        with open(summary_file, 'w') as f:
            f.write("=== ENHANCED DATA QUALITY REPORT ===\n\n")
            f.write(f"Report #: {sequence_num:03d}\n")
            f.write(f"Report Generated: {report['report_timestamp']}\n")
            f.write(f"Overall Quality Score: {report['overall_quality_score']:.1f}%\n\n")
            
            f.write("DATASET QUALITY:\n")
            for dataset_name, metrics in report['dataset_quality'].items():
                f.write(f"  {dataset_name}: {metrics['overall_quality_score']:.1f}%\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS:\n")
            for rec in report['recommendations']:
                f.write(f"  • {rec}\n")
            f.write("\n")
            
            f.write("QUALITY IMPROVEMENTS:\n")
            for category, suggestions in report['quality_improvements'].items():
                f.write(f"  {category.upper()}:\n")
                for suggestion in suggestions:
                    f.write(f"    • {suggestion}\n")
                f.write("\n")
        
        logger.info(f"✅ Quality report #{sequence_num:03d} saved to {summary_file}")
    
    def _get_next_sequence_number(self, prefix: str) -> int:
        """
        Get next sequence number for file naming
        """
        results_dir = Path("results")
        if not results_dir.exists():
            return 1
        
        # Find existing files with the prefix
        existing_files = list(results_dir.glob(f"*_{prefix}.txt"))
        if not existing_files:
            return 1
        
        # Extract sequence numbers and find the highest
        sequence_numbers = []
        for file in existing_files:
            try:
                # Extract number from filename like "001_quality_report.txt"
                num_str = file.stem.split('_')[0]
                sequence_numbers.append(int(num_str))
            except (ValueError, IndexError):
                continue
        
        return max(sequence_numbers) + 1 if sequence_numbers else 1


def main():
    """
    Test enhanced data validator
    """
    logger.info("🚀 Testing Enhanced Data Validator...")
    
    validator = EnhancedDataValidator()
    validator.create_data_schemas()
    
    # Load test data
    reddit_file = validator.data_dir / "raw" / "reddit_wsb.csv"
    if reddit_file.exists():
        reddit_data = pd.read_csv(reddit_file)
        
        # Train spam classifier
        accuracy = validator.train_spam_classifier(reddit_data)
        
        # Assess quality
        quality_metrics = validator.assess_data_quality('reddit', reddit_data)
        
        # Generate report
        report = validator.generate_quality_report()
        validator.save_quality_report(report)
        
        logger.info(f"✅ Enhanced validation completed. Overall quality: {report['overall_quality_score']:.1f}%")
    else:
        logger.error("❌ Reddit data not found for testing")


if __name__ == "__main__":
    main() 