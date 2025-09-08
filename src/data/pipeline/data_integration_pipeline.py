"""
Data Integration Pipeline - Main Orchestrator
Runs comprehensive data exploration, cleaning, and integration pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List

# Import data processing modules using correct path structure
from data.processing.data_exploration import DataExplorer
from data.processing.data_cleaning import DataCleaner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIntegrationPipeline:
    """
    Main orchestrator for Data Quality Assessment & Integration Pipeline
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.exploration_results = {}
        self.cleaning_results = {}
        self.pipeline_results = {}
        
    def run_integration_pipeline(self) -> Dict:
        """
        Run complete data integration pipeline: exploration + cleaning + integration
        """
        logger.info("🚀 Starting Data Integration Pipeline")
        
        # Step 1: Data Exploration
        logger.info("="*50)
        logger.info("STEP 1: Comprehensive Data Exploration")
        logger.info("="*50)
        
        explorer = DataExplorer(self.data_dir)
        exploration_report = explorer.generate_exploration_report()
        explorer.save_exploration_report(exploration_report)
        self.exploration_results = exploration_report
        
        # Step 2: Data Cleaning
        logger.info("="*50)
        logger.info("STEP 2: Data Cleaning and Preprocessing")
        logger.info("="*50)
        
        cleaner = DataCleaner(self.data_dir)
        cleaning_report = cleaner.generate_cleaning_report()
        cleaner.save_cleaning_report(cleaning_report)
        self.cleaning_results = cleaning_report
        
        # Step 3: Generate integration completion report
        logger.info("="*50)
        logger.info("STEP 3: Integration Completion Report")
        logger.info("="*50)
        
        integration_report = self._generate_integration_completion_report()
        self._save_integration_report(integration_report)
        
        self.pipeline_results = integration_report
        return integration_report
    
    def _generate_integration_completion_report(self) -> Dict:
        """
        Generate comprehensive integration completion report
        """
        logger.info("📊 Generating integration completion report...")
        
        # Compile comprehensive report
        report = {
            "integration_timestamp": datetime.now().isoformat(),
            "overall_status": "PASS",
            "exploration_summary": {
                "datasets_analyzed": self.exploration_results.get("datasets_analyzed", 0),
                "data_quality_score": self.exploration_results.get("overall_assessment", {}).get("data_quality_score", 0),
                "critical_issues": self.exploration_results.get("overall_assessment", {}).get("critical_issues", []),
                "recommendations": self.exploration_results.get("overall_assessment", {}).get("recommendations", [])
            },
            "cleaning_summary": {
                "reddit_retention_rate": self.cleaning_results.get("cleaning_metrics", {}).get("reddit_retention_rate", 0),
                "stock_retention_rate": self.cleaning_results.get("cleaning_metrics", {}).get("stock_retention_rate", 0),
                "unified_records": self.cleaning_results.get("cleaned_data", {}).get("unified_records", 0),
                "quality_improvements": self.cleaning_results.get("data_quality_improvements", {})
            },
            "deliverables": self._assess_deliverables(),
            "next_steps": self._generate_next_steps()
        }
        
        # Determine overall status
        exploration_score = report["exploration_summary"]["data_quality_score"]
        reddit_retention = report["cleaning_summary"]["reddit_retention_rate"]
        unified_records = report["cleaning_summary"]["unified_records"]
        
        if exploration_score >= 90 and reddit_retention >= 80 and unified_records > 0:
            report["overall_status"] = "PASS"
        elif exploration_score >= 70 and reddit_retention >= 60:
            report["overall_status"] = "WARNING"
        else:
            report["overall_status"] = "FAIL"
        
        return report
    
    def _assess_deliverables(self) -> Dict:
        """
        Assess integration deliverables completion
        """
        deliverables = {
            "comprehensive_data_exploration": False,
            "data_cleaning_pipeline": False,
            "unified_dataset_creation": False,
            "data_quality_report": False,
            "initial_insights": False
        }
        
        # Check exploration completion
        if self.exploration_results.get("datasets_analyzed", 0) > 0:
            deliverables["comprehensive_data_exploration"] = True
        
        # Check cleaning completion
        if self.cleaning_results.get("cleaning_metrics", {}).get("reddit_retention_rate", 0) > 0:
            deliverables["data_cleaning_pipeline"] = True
        
        # Check unified dataset
        if self.cleaning_results.get("cleaned_data", {}).get("unified_records", 0) > 0:
            deliverables["unified_dataset_creation"] = True
        
        # Check quality report
        if self.exploration_results.get("overall_assessment", {}).get("data_quality_score", 0) > 0:
            deliverables["data_quality_report"] = True
        
        # Check insights (basic assessment)
        if len(self.exploration_results.get("overall_assessment", {}).get("strengths", [])) > 0:
            deliverables["initial_insights"] = True
        
        return deliverables
    
    def _generate_next_steps(self) -> List[str]:
        """
        Generate next steps for feature engineering
        """
        next_steps = []
        
        # Based on exploration results
        exploration_score = self.exploration_results.get("overall_assessment", {}).get("data_quality_score", 0)
        if exploration_score < 90:
            next_steps.append("Address data quality issues identified in exploration")
        
        # Based on cleaning results
        reddit_retention = self.cleaning_results.get("cleaning_metrics", {}).get("reddit_retention_rate", 0)
        if reddit_retention < 80:
            next_steps.append("Review and optimize data cleaning pipeline")
        
        # Standard next steps
        next_steps.extend([
            "Proceed to Feature Engineering Pipeline",
            "Implement Reddit-based features (25 features)",
            "Implement Financial market features (35 features per stock)",
            "Implement Temporal and cross-modal features (19 features)",
            "Create feature engineering pipeline"
        ])
        
        return next_steps
    
    def _save_integration_report(self, report: Dict):
        """
        Save integration completion report
        """
        import json
        from pathlib import Path
        
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        report_file = results_dir / "data_integration_completion_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Integration completion report saved to {report_file}")
        
        # Save human-readable summary
        summary_file = results_dir / "data_integration_completion_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== DATA INTEGRATION COMPLETION SUMMARY ===\n\n")
            f.write(f"Completion Timestamp: {report['integration_timestamp']}\n")
            f.write(f"Overall Status: {report['overall_status']}\n\n")
            
            # Exploration summary
            f.write("DATA EXPLORATION:\n")
            exploration = report['exploration_summary']
            f.write(f"  Datasets Analyzed: {exploration['datasets_analyzed']}\n")
            f.write(f"  Data Quality Score: {exploration['data_quality_score']}%\n")
            if exploration['critical_issues']:
                f.write(f"  Critical Issues: {len(exploration['critical_issues'])}\n")
            if exploration['recommendations']:
                f.write(f"  Recommendations: {len(exploration['recommendations'])}\n")
            f.write("\n")
            
            # Cleaning summary
            f.write("DATA CLEANING:\n")
            cleaning = report['cleaning_summary']
            f.write(f"  Reddit Retention: {cleaning['reddit_retention_rate']}%\n")
            f.write(f"  Stock Retention: {cleaning['stock_retention_rate']}%\n")
            f.write(f"  Unified Records: {cleaning['unified_records']:,}\n")
            f.write("\n")
            
            # Deliverables
            f.write("DELIVERABLES:\n")
            deliverables = report['deliverables']
            for deliverable, completed in deliverables.items():
                status = "✅" if completed else "❌"
                f.write(f"  {status} {deliverable.replace('_', ' ').title()}\n")
            f.write("\n")
            
            # Next steps
            f.write("NEXT STEPS:\n")
            for step in report['next_steps']:
                f.write(f"  • {step}\n")
        
        logger.info(f"Integration completion summary saved to {summary_file}")
    
    def print_completion_summary(self):
        """
        Print integration completion summary to console
        """
        if not self.pipeline_results:
            logger.error("No integration results available")
            return
        
        print("\n" + "="*60)
        print("🎉 DATA INTEGRATION PIPELINE COMPLETE")
        print("="*60)
        
        # Overall status
        status = self.pipeline_results["overall_status"]
        status_emoji = "✅" if status == "PASS" else "⚠️" if status == "WARNING" else "❌"
        print(f"{status_emoji} Overall Status: {status}")
        
        # Key metrics
        exploration = self.pipeline_results["exploration_summary"]
        cleaning = self.pipeline_results["cleaning_summary"]
        
        print(f"\n📊 EXPLORATION METRICS:")
        print(f"   Datasets Analyzed: {exploration['datasets_analyzed']}")
        print(f"   Data Quality Score: {exploration['data_quality_score']}%")
        
        print(f"\n🧹 CLEANING METRICS:")
        print(f"   Reddit Retention: {cleaning['reddit_retention_rate']}%")
        print(f"   Stock Retention: {cleaning['stock_retention_rate']}%")
        print(f"   Unified Records: {cleaning['unified_records']:,}")
        
        # Deliverables
        print(f"\n📋 DELIVERABLES:")
        deliverables = self.pipeline_results["deliverables"]
        completed_count = sum(deliverables.values())
        total_count = len(deliverables)
        print(f"   Completed: {completed_count}/{total_count}")
        
        for deliverable, completed in deliverables.items():
            status_icon = "✅" if completed else "❌"
            deliverable_name = deliverable.replace('_', ' ').title()
            print(f"   {status_icon} {deliverable_name}")
        
        print("\n" + "="*60)
        
        if status == "PASS":
            print("🎯 Integration objectives achieved! Ready for Feature Engineering")
        elif status == "WARNING":
            print("⚠️ Integration completed with warnings. Review recommendations before proceeding.")
        else:
            print("❌ Integration failed. Address critical issues before proceeding.")


def main():
    """
    Main function to run data integration pipeline
    """
    logger.info("Starting Data Integration Pipeline")
    
    # Initialize pipeline
    pipeline = DataIntegrationPipeline()
    
    # Run complete integration pipeline
    report = pipeline.run_integration_pipeline()
    
    # Print completion summary
    pipeline.print_completion_summary()
    
    return report


if __name__ == "__main__":
    main() 