"""
Main Data Processing Orchestrator
Entry point for running the complete data processing pipeline
"""

import sys
import os
from pathlib import Path

# Import using the correct path structure
from data.pipeline.data_integration_pipeline import DataIntegrationPipeline
from data.validation.data_validation import DataValidator
from data.processing.historical_data_downloader import HistoricalDataDownloader

def run_data_validation():
    """
    Run data validation pipeline
    """
    print("🔍 Running Data Validation Pipeline...")
    validator = DataValidator()
    validation_report = validator.generate_validation_report()
    validator.save_validation_report(validation_report)
    return validation_report

def run_data_integration():
    """
    Run data integration pipeline
    """
    print("🔄 Running Data Integration Pipeline...")
    pipeline = DataIntegrationPipeline()
    integration_report = pipeline.run_integration_pipeline()
    pipeline.print_completion_summary()
    return integration_report

def download_historical_data():
    """
    Download historical stock data
    """
    print("📈 Downloading Historical Stock Data...")
    downloader = HistoricalDataDownloader()
    downloader.download_historical_stock_data()
    downloader.create_dataset_description()
    print("✅ Historical data download completed")

def main():
    """
    Main function to orchestrate data processing
    """
    print("🚀 Starting Meme Stock Data Processing Pipeline")
    print("="*60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "validate":
            run_data_validation()
        elif command == "integrate":
            run_data_integration()
        elif command == "download":
            download_historical_data()
        elif command == "all":
            # Run complete pipeline
            download_historical_data()
            run_data_validation()
            run_data_integration()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: validate, integrate, download, all")
    else:
        # Default: run complete pipeline
        print("No command specified. Running complete pipeline...")
        download_historical_data()
        run_data_validation()
        run_data_integration()
    
    print("🎉 Data processing pipeline completed!")

if __name__ == "__main__":
    main() 