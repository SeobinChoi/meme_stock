"""
Quality Monitoring Dashboard
Real-time data quality monitoring and alerting system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import time
from threading import Thread, Event
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityMonitor:
    """
    Real-time data quality monitoring and alerting system
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.quality_thresholds = {
            'completeness': 80.0,
            'consistency': 85.0,
            'accuracy': 90.0,
            'timeliness': 95.0
        }
        self.alert_history = []
        self.monitoring_active = False
        self.stop_event = Event()
        
    def set_quality_thresholds(self, thresholds: Dict[str, float]):
        """
        Set custom quality thresholds
        """
        self.quality_thresholds.update(thresholds)
        logger.info("✅ Quality thresholds updated")
    
    def start_monitoring(self, interval_seconds: int = 300):
        """
        Start real-time quality monitoring
        """
        logger.info(f"🚀 Starting quality monitoring (interval: {interval_seconds}s)")
        self.monitoring_active = True
        self.stop_event.clear()
        
        # Start monitoring thread
        monitor_thread = Thread(target=self._monitoring_loop, args=(interval_seconds,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    
    def stop_monitoring(self):
        """
        Stop quality monitoring
        """
        logger.info("🛑 Stopping quality monitoring")
        self.monitoring_active = False
        self.stop_event.set()
    
    def _monitoring_loop(self, interval_seconds: int):
        """
        Main monitoring loop
        """
        while not self.stop_event.is_set():
            try:
                # Check data quality
                quality_report = self.check_current_quality()
                
                # Process alerts
                alerts = self.process_alerts(quality_report)
                
                # Log results
                if alerts:
                    logger.warning(f"⚠️ Quality alerts detected: {len(alerts)} issues")
                    for alert in alerts:
                        logger.warning(f"  {alert['severity']}: {alert['message']}")
                else:
                    logger.info("✅ All quality metrics within thresholds")
                
                # Save monitoring results
                self.save_monitoring_results(quality_report, alerts)
                
                # Wait for next check
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def check_current_quality(self) -> Dict:
        """
        Check current data quality
        """
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'overall_quality': 0.0,
            'alerts': []
        }
        
        # Check each dataset
        datasets = self._get_available_datasets()
        
        for dataset_name, dataset_path in datasets.items():
            try:
                data = pd.read_csv(dataset_path)
                quality_metrics = self._assess_dataset_quality(dataset_name, data)
                quality_report['datasets'][dataset_name] = quality_metrics
            except Exception as e:
                logger.error(f"❌ Error assessing {dataset_name}: {e}")
                quality_report['datasets'][dataset_name] = {
                    'error': str(e),
                    'quality_score': 0.0
                }
        
        # Calculate overall quality
        scores = [metrics.get('quality_score', 0) for metrics in quality_report['datasets'].values() 
                 if isinstance(metrics, dict) and 'quality_score' in metrics]
        
        if scores:
            quality_report['overall_quality'] = np.mean(scores)
        
        return quality_report
    
    def _get_available_datasets(self) -> Dict[str, Path]:
        """
        Get list of available datasets
        """
        datasets = {}
        
        # Check raw data directory
        raw_dir = self.data_dir / "raw"
        if raw_dir.exists():
            for file in raw_dir.glob("*.csv"):
                if "reddit" in file.name.lower():
                    datasets['reddit'] = file
                elif "stock" in file.name.lower():
                    symbol = file.stem.split('_')[0]
                    datasets[f'stock_{symbol}'] = file
        
        # Check processed data directory
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            for file in processed_dir.glob("*.csv"):
                if "unified" in file.name.lower():
                    datasets['unified'] = file
        
        return datasets
    
    def _assess_dataset_quality(self, dataset_name: str, data: pd.DataFrame) -> Dict:
        """
        Assess quality of a specific dataset
        """
        quality_metrics = {
            'dataset_name': dataset_name,
            'total_records': len(data),
            'total_columns': len(data.columns),
            'quality_scores': {},
            'quality_score': 0.0
        }
        
        # Completeness
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        completeness = (1 - missing_ratio) * 100
        quality_metrics['quality_scores']['completeness'] = completeness
        
        # Consistency
        duplicate_ratio = data.duplicated().sum() / len(data)
        consistency = (1 - duplicate_ratio) * 100
        quality_metrics['quality_scores']['consistency'] = consistency
        
        # Accuracy (basic checks)
        accuracy_checks = []
        
        # Check for reasonable value ranges
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['score', 'comms_num']:
                accuracy_check = ((data[col] >= 0) & (data[col] <= 100000)).mean()
                accuracy_checks.append(accuracy_check)
            elif col in ['Open', 'High', 'Low', 'Close']:
                accuracy_check = ((data[col] > 0) & (data[col] < 10000)).mean()
                accuracy_checks.append(accuracy_check)
            elif col == 'Volume':
                accuracy_check = (data[col] >= 0).mean()
                accuracy_checks.append(accuracy_check)
        
        accuracy = np.mean(accuracy_checks) * 100 if accuracy_checks else 85.0
        quality_metrics['quality_scores']['accuracy'] = accuracy
        
        # Timeliness (assume good if data exists)
        timeliness = 85.0
        quality_metrics['quality_scores']['timeliness'] = timeliness
        
        # Overall quality score
        overall_score = np.mean(list(quality_metrics['quality_scores'].values()))
        quality_metrics['quality_score'] = overall_score
        
        return quality_metrics
    
    def process_alerts(self, quality_report: Dict) -> List[Dict]:
        """
        Process quality alerts based on thresholds
        """
        alerts = []
        
        for dataset_name, metrics in quality_report['datasets'].items():
            if not isinstance(metrics, dict) or 'quality_score' not in metrics:
                continue
            
            quality_score = metrics['quality_score']
            
            # Check overall quality
            if quality_score < 50:
                alerts.append({
                    'severity': 'CRITICAL',
                    'dataset': dataset_name,
                    'message': f'Critical quality issues: {quality_score:.1f}%',
                    'timestamp': datetime.now().isoformat()
                })
            elif quality_score < 70:
                alerts.append({
                    'severity': 'WARNING',
                    'dataset': dataset_name,
                    'message': f'Quality below threshold: {quality_score:.1f}%',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check individual metrics
            for metric_name, score in metrics.get('quality_scores', {}).items():
                threshold = self.quality_thresholds.get(metric_name, 80.0)
                
                if score < threshold:
                    alerts.append({
                        'severity': 'WARNING',
                        'dataset': dataset_name,
                        'metric': metric_name,
                        'message': f'{metric_name} below threshold: {score:.1f}% < {threshold}%',
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Check overall quality
        overall_quality = quality_report.get('overall_quality', 0)
        if overall_quality < 60:
            alerts.append({
                'severity': 'CRITICAL',
                'dataset': 'OVERALL',
                'message': f'Overall quality critical: {overall_quality:.1f}%',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def save_monitoring_results(self, quality_report: Dict, alerts: List[Dict]):
        """
        Save monitoring results to file
        """
        # Save detailed report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.data_dir / "results" / f"quality_monitor_{timestamp}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        monitoring_results = {
            'quality_report': quality_report,
            'alerts': alerts,
            'monitoring_timestamp': datetime.now().isoformat()
        }
        
        with open(report_file, 'w') as f:
            json.dump(monitoring_results, f, indent=2, default=str)
        
        # Update alert history
        self.alert_history.extend(alerts)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # Save alert history
        history_file = self.data_dir / "results" / "alert_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.alert_history, f, indent=2, default=str)
    
    def get_quality_dashboard(self) -> Dict:
        """
        Get current quality dashboard data
        """
        quality_report = self.check_current_quality()
        
        dashboard = {
            'current_time': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'overall_quality': quality_report['overall_quality'],
            'datasets': {},
            'recent_alerts': self.alert_history[-10:],  # Last 10 alerts
            'quality_trends': self._get_quality_trends()
        }
        
        # Add dataset summaries
        for dataset_name, metrics in quality_report['datasets'].items():
            if isinstance(metrics, dict) and 'quality_score' in metrics:
                dashboard['datasets'][dataset_name] = {
                    'quality_score': metrics['quality_score'],
                    'total_records': metrics.get('total_records', 0),
                    'status': 'GOOD' if metrics['quality_score'] >= 80 else 'WARNING' if metrics['quality_score'] >= 60 else 'CRITICAL'
                }
        
        return dashboard
    
    def _get_quality_trends(self) -> Dict:
        """
        Get quality trends over time (placeholder for future implementation)
        """
        return {
            'trend_period': '24h',
            'overall_trend': 'stable',
            'dataset_trends': {}
        }
    
    def generate_quality_report(self) -> Dict:
        """
        Generate comprehensive quality report
        """
        dashboard = self.get_quality_dashboard()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'monitoring_summary': {
                'active': self.monitoring_active,
                'total_alerts': len(self.alert_history),
                'recent_alerts': len([a for a in self.alert_history if 
                                   datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)])
            },
            'quality_summary': {
                'overall_quality': dashboard['overall_quality'],
                'dataset_count': len(dashboard['datasets']),
                'quality_distribution': self._get_quality_distribution(dashboard['datasets'])
            },
            'recommendations': self._generate_quality_recommendations(dashboard)
        }
        
        return report
    
    def _get_quality_distribution(self, datasets: Dict) -> Dict:
        """
        Get quality score distribution
        """
        scores = [metrics['quality_score'] for metrics in datasets.values()]
        
        return {
            'excellent': len([s for s in scores if s >= 90]),
            'good': len([s for s in scores if 80 <= s < 90]),
            'fair': len([s for s in scores if 60 <= s < 80]),
            'poor': len([s for s in scores if s < 60])
        }
    
    def _generate_quality_recommendations(self, dashboard: Dict) -> List[str]:
        """
        Generate quality improvement recommendations
        """
        recommendations = []
        
        overall_quality = dashboard['overall_quality']
        
        if overall_quality < 60:
            recommendations.append("🚨 CRITICAL: Overall data quality is poor - immediate action required")
            recommendations.append("   • Review data sources and collection methods")
            recommendations.append("   • Implement data validation checks")
            recommendations.append("   • Consider alternative data sources")
        elif overall_quality < 80:
            recommendations.append("⚠️ WARNING: Data quality needs improvement")
            recommendations.append("   • Enhance data cleaning procedures")
            recommendations.append("   • Add data validation rules")
            recommendations.append("   • Monitor quality metrics regularly")
        else:
            recommendations.append("✅ GOOD: Data quality is acceptable")
            recommendations.append("   • Continue monitoring quality metrics")
            recommendations.append("   • Consider incremental improvements")
        
        # Dataset-specific recommendations
        for dataset_name, metrics in dashboard['datasets'].items():
            if metrics['status'] == 'CRITICAL':
                recommendations.append(f"   • {dataset_name}: Critical quality issues - review immediately")
            elif metrics['status'] == 'WARNING':
                recommendations.append(f"   • {dataset_name}: Quality improvements recommended")
        
        return recommendations


def main():
    """
    Test quality monitor
    """
    logger.info("🚀 Testing Quality Monitor...")
    
    monitor = QualityMonitor()
    
    # Set custom thresholds
    monitor.set_quality_thresholds({
        'completeness': 85.0,
        'consistency': 90.0,
        'accuracy': 95.0,
        'timeliness': 98.0
    })
    
    # Check current quality
    quality_report = monitor.check_current_quality()
    
    # Process alerts
    alerts = monitor.process_alerts(quality_report)
    
    # Generate dashboard
    dashboard = monitor.get_quality_dashboard()
    
    # Generate report
    report = monitor.generate_quality_report()
    
    # Save results
    monitor.save_monitoring_results(quality_report, alerts)
    
    # Print summary
    print(f"\n=== QUALITY MONITOR SUMMARY ===")
    print(f"Overall Quality: {dashboard['overall_quality']:.1f}%")
    print(f"Datasets Monitored: {len(dashboard['datasets'])}")
    print(f"Recent Alerts: {len(dashboard['recent_alerts'])}")
    print(f"Monitoring Active: {dashboard['monitoring_active']}")
    
    if alerts:
        print(f"\n⚠️ ALERTS DETECTED:")
        for alert in alerts:
            print(f"  {alert['severity']}: {alert['message']}")
    else:
        print(f"\n✅ No quality alerts")
    
    logger.info("✅ Quality monitor test completed")


if __name__ == "__main__":
    main() 