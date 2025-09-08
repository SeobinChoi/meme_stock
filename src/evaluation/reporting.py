#!/usr/bin/env python3
"""
Comprehensive reporting and visualization for meme stock contrarian effect prediction.

Generates results, visualizations, and reports as described in the manual.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
    plt = None
    sns = None

logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """Generate comprehensive visualizations for results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        """
        Initialize results visualizer.
        
        Args:
            figsize: Default figure size
            style: Seaborn style
        """
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required for visualization")
        
        self.figsize = figsize
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_model_performance_comparison(self, results: Dict[str, Dict[str, float]], 
                                        save_path: Optional[str] = None):
        """
        Plot model performance comparison.
        
        Args:
            results: Dictionary mapping model names to metrics
            save_path: Optional path to save plot
        """
        logger.info("Creating model performance comparison plot...")
        
        # Prepare data
        metrics_data = []
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': value
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots for different metric types
        metric_types = {
            'Correlation': ['pearson_corr', 'spearman_corr', 'rank_ic', 'ic'],
            'Accuracy': ['hit_rate', 'r2'],
            'Error': ['rmse', 'mae', 'mse']
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (metric_type, metric_list) in enumerate(metric_types.items()):
            ax = axes[i]
            
            # Filter data for this metric type
            type_data = df[df['Metric'].isin(metric_list)]
            
            if not type_data.empty:
                # Create bar plot
                sns.barplot(data=type_data, x='Metric', y='Value', hue='Model', ax=ax)
                ax.set_title(f'{metric_type} Metrics')
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importance_data: Dict[str, Dict[str, float]], 
                              top_n: int = 15, save_path: Optional[str] = None):
        """
        Plot feature importance comparison.
        
        Args:
            importance_data: Dictionary mapping model names to feature importance
            top_n: Number of top features to show
            save_path: Optional path to save plot
        """
        logger.info("Creating feature importance plot...")
        
        # Get top features across all models
        all_features = set()
        for model_importance in importance_data.values():
            all_features.update(model_importance.keys())
        
        # Calculate average importance
        avg_importance = {}
        for feature in all_features:
            values = [model_importance.get(feature, 0) for model_importance in importance_data.values()]
            avg_importance[feature] = np.mean(values)
        
        # Get top N features
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_feature_names = [f[0] for f in top_features]
        
        # Create DataFrame for plotting
        plot_data = []
        for model_name, model_importance in importance_data.items():
            for feature in top_feature_names:
                plot_data.append({
                    'Feature': feature,
                    'Model': model_name,
                    'Importance': model_importance.get(feature, 0)
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Importance', y='Feature', hue='Model')
        plt.title(f'Top {top_n} Feature Importance Comparison')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_strategy_performance(self, strategy_results: Dict[str, Any], 
                                save_path: Optional[str] = None):
        """
        Plot trading strategy performance.
        
        Args:
            strategy_results: Dictionary with strategy results
            save_path: Optional path to save plot
        """
        logger.info("Creating strategy performance plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        ax1 = axes[0, 0]
        if 'returns' in strategy_results:
            cumulative_returns = np.cumprod(1 + strategy_results['returns'])
            ax1.plot(cumulative_returns)
            ax1.set_title('Cumulative Returns')
            ax1.set_ylabel('Cumulative Return')
            ax1.grid(True)
        
        # Drawdown
        ax2 = axes[0, 1]
        if 'returns' in strategy_results:
            cumulative_returns = np.cumprod(1 + strategy_results['returns'])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            ax2.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown')
            ax2.grid(True)
        
        # Position changes
        ax3 = axes[1, 0]
        if 'positions' in strategy_results:
            ax3.plot(strategy_results['positions'])
            ax3.set_title('Position Changes')
            ax3.set_ylabel('Position')
            ax3.set_xlabel('Time')
            ax3.grid(True)
        
        # Signal vs Returns
        ax4 = axes[1, 1]
        if 'signals' in strategy_results and 'returns' in strategy_results:
            ax4.scatter(strategy_results['signals'], strategy_results['returns'], alpha=0.5)
            ax4.set_title('Signal vs Returns')
            ax4.set_xlabel('Signal')
            ax4.set_ylabel('Returns')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Strategy performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_stability_analysis(self, stability_results: Dict[str, Any], 
                               save_path: Optional[str] = None):
        """
        Plot stability analysis results.
        
        Args:
            stability_results: Dictionary with stability results
            save_path: Optional path to save plot
        """
        logger.info("Creating stability analysis plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Walk-forward performance
        ax1 = axes[0, 0]
        if 'walk_forward' in stability_results and 'rolling_metrics' in stability_results['walk_forward']:
            rolling_metrics = stability_results['walk_forward']['rolling_metrics']
            window_starts = stability_results['walk_forward']['window_starts']
            
            # Plot rolling correlation
            correlations = [m.get('spearman_corr', 0) for m in rolling_metrics]
            ax1.plot(window_starts, correlations)
            ax1.set_title('Rolling Spearman Correlation')
            ax1.set_ylabel('Correlation')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True)
        
        # Yearly performance
        ax2 = axes[0, 1]
        if 'yearly' in stability_results and 'yearly_metrics' in stability_results['yearly']:
            yearly_metrics = stability_results['yearly']['yearly_metrics']
            years = list(yearly_metrics.keys())
            correlations = [metrics.get('spearman_corr', 0) for metrics in yearly_metrics.values()]
            
            ax2.bar(years, correlations)
            ax2.set_title('Yearly Performance')
            ax2.set_ylabel('Spearman Correlation')
            ax2.set_xlabel('Year')
            ax2.grid(True)
        
        # Regime performance
        ax3 = axes[1, 0]
        if 'regime' in stability_results and 'regime_metrics' in stability_results['regime']:
            regime_metrics = stability_results['regime']['regime_metrics']
            regimes = list(regime_metrics.keys())
            correlations = [metrics.get('spearman_corr', 0) for metrics in regime_metrics.values()]
            
            ax3.bar(regimes, correlations)
            ax3.set_title('Regime Performance')
            ax3.set_ylabel('Spearman Correlation')
            ax3.set_xlabel('Regime')
            ax3.grid(True)
        
        # Performance distribution
        ax4 = axes[1, 1]
        if 'walk_forward' in stability_results and 'rolling_metrics' in stability_results['walk_forward']:
            rolling_metrics = stability_results['walk_forward']['rolling_metrics']
            correlations = [m.get('spearman_corr', 0) for m in rolling_metrics]
            
            ax4.hist(correlations, bins=20, alpha=0.7)
            ax4.set_title('Performance Distribution')
            ax4.set_xlabel('Spearman Correlation')
            ax4.set_ylabel('Frequency')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Stability analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_placebo_test_results(self, placebo_results: Dict[str, Any], 
                                 save_path: Optional[str] = None):
        """
        Plot placebo test results.
        
        Args:
            placebo_results: Dictionary with placebo test results
            save_path: Optional path to save plot
        """
        logger.info("Creating placebo test results plot...")
        
        # Extract data for plotting
        plot_data = []
        
        for model_name, model_results in placebo_results.items():
            for test_name, test_results in model_results.items():
                if test_results and 'spearman_corr' in test_results:
                    corr_results = test_results['spearman_corr']
                    plot_data.append({
                        'Model': model_name,
                        'Test': test_name,
                        'Original': corr_results.get('original', 0),
                        'Placebo_Mean': corr_results.get('placebo_mean', 0),
                        'Placebo_Std': corr_results.get('placebo_std', 0),
                        'P_Value': corr_results.get('p_value', 1)
                    })
        
        if not plot_data:
            logger.warning("No placebo test data available for plotting")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original vs Placebo comparison
        ax1 = axes[0]
        x_pos = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x_pos - width/2, df['Original'], width, label='Original', alpha=0.8)
        ax1.bar(x_pos + width/2, df['Placebo_Mean'], width, label='Placebo Mean', alpha=0.8)
        ax1.errorbar(x_pos + width/2, df['Placebo_Mean'], yerr=df['Placebo_Std'], 
                    fmt='none', color='black', capsize=5)
        
        ax1.set_xlabel('Model-Test Combination')
        ax1.set_ylabel('Spearman Correlation')
        ax1.set_title('Original vs Placebo Performance')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{row['Model']}-{row['Test']}" for _, row in df.iterrows()], 
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # P-values
        ax2 = axes[1]
        colors = ['red' if p < 0.05 else 'green' for p in df['P_Value']]
        ax2.bar(x_pos, df['P_Value'], color=colors, alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold')
        ax2.set_xlabel('Model-Test Combination')
        ax2.set_ylabel('P-Value')
        ax2.set_title('Placebo Test P-Values')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{row['Model']}-{row['Test']}" for _, row in df.iterrows()], 
                           rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Placebo test results plot saved to {save_path}")
        
        plt.show()

class ResultsReporter:
    """Generate comprehensive results reports."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize results reporter.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.visualizer = ResultsVisualizer()
    
    def generate_model_report(self, model_results: Dict[str, Any], 
                            feature_importance: Dict[str, Dict[str, float]],
                            save_path: Optional[str] = None) -> str:
        """
        Generate model performance report.
        
        Args:
            model_results: Dictionary with model results
            feature_importance: Dictionary with feature importance
            save_path: Optional path to save report
            
        Returns:
            Report content as string
        """
        logger.info("Generating model performance report...")
        
        report_lines = []
        report_lines.append("# Model Performance Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model comparison
        report_lines.append("## Model Performance Comparison")
        report_lines.append("")
        
        # Create performance table
        performance_data = []
        for model_name, metrics in model_results.items():
            row = {'Model': model_name}
            row.update(metrics)
            performance_data.append(row)
        
        df_performance = pd.DataFrame(performance_data)
        report_lines.append(df_performance.to_markdown(index=False))
        report_lines.append("")
        
        # Feature importance
        report_lines.append("## Feature Importance Analysis")
        report_lines.append("")
        
        for model_name, importance in feature_importance.items():
            report_lines.append(f"### {model_name}")
            report_lines.append("")
            
            # Top 10 features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            report_lines.append("Top 10 Features:")
            for i, (feature, score) in enumerate(top_features, 1):
                report_lines.append(f"{i:2d}. {feature}: {score:.4f}")
            report_lines.append("")
        
        # Best performing model
        if model_results:
            best_model = max(model_results.items(), key=lambda x: x[1].get('spearman_corr', 0))
            report_lines.append("## Best Performing Model")
            report_lines.append(f"**{best_model[0]}** with Spearman correlation: {best_model[1].get('spearman_corr', 0):.4f}")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Model report saved to {save_path}")
        
        return report_content
    
    def generate_strategy_report(self, strategy_results: Dict[str, Any], 
                               save_path: Optional[str] = None) -> str:
        """
        Generate trading strategy report.
        
        Args:
            strategy_results: Dictionary with strategy results
            save_path: Optional path to save report
            
        Returns:
            Report content as string
        """
        logger.info("Generating trading strategy report...")
        
        report_lines = []
        report_lines.append("# Trading Strategy Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Strategy parameters
        if 'parameters' in strategy_results:
            report_lines.append("## Strategy Parameters")
            report_lines.append("")
            for param, value in strategy_results['parameters'].items():
                report_lines.append(f"- **{param}**: {value}")
            report_lines.append("")
        
        # Performance metrics
        if 'metrics' in strategy_results:
            report_lines.append("## Performance Metrics")
            report_lines.append("")
            
            metrics = strategy_results['metrics']
            report_lines.append(f"- **Total Return**: {metrics.get('total_return', 0):.2%}")
            report_lines.append(f"- **Annualized Return**: {metrics.get('annualized_return', 0):.2%}")
            report_lines.append(f"- **Volatility**: {metrics.get('volatility', 0):.2%}")
            report_lines.append(f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.3f}")
            report_lines.append(f"- **Hit Rate**: {metrics.get('hit_rate', 0):.2%}")
            report_lines.append(f"- **Maximum Drawdown**: {metrics.get('max_drawdown', 0):.2%}")
            report_lines.append(f"- **Calmar Ratio**: {metrics.get('calmar_ratio', 0):.3f}")
            report_lines.append(f"- **Turnover**: {metrics.get('turnover', 0):.3f}")
            report_lines.append("")
        
        # Information Coefficient
        if 'ic' in strategy_results.get('metrics', {}):
            ic_metrics = strategy_results['metrics']['ic']
            report_lines.append("## Information Coefficient")
            report_lines.append("")
            report_lines.append(f"- **IC**: {ic_metrics.get('ic', 0):.4f}")
            report_lines.append(f"- **Rank IC**: {ic_metrics.get('rank_ic', 0):.4f}")
            report_lines.append("")
        
        # Strategy analysis
        report_lines.append("## Strategy Analysis")
        report_lines.append("")
        
        if strategy_results.get('strategy_type') == 'contrarian':
            report_lines.append("This is a **contrarian strategy** that bets against Reddit sentiment.")
            report_lines.append("The strategy assumes that high Reddit activity and confidence")
            report_lines.append("predict negative future returns (contrarian effect).")
        else:
            report_lines.append(f"This is a **{strategy_results.get('strategy_type', 'unknown')} strategy**.")
        
        report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Strategy report saved to {save_path}")
        
        return report_content
    
    def generate_robustness_report(self, robustness_results: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> str:
        """
        Generate robustness analysis report.
        
        Args:
            robustness_results: Dictionary with robustness results
            save_path: Optional path to save report
            
        Returns:
            Report content as string
        """
        logger.info("Generating robustness report...")
        
        report_lines = []
        report_lines.append("# Robustness Analysis Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Placebo tests
        if 'placebo_tests' in robustness_results:
            report_lines.append("## Placebo Tests")
            report_lines.append("")
            
            for model_name, model_results in robustness_results['placebo_tests'].items():
                report_lines.append(f"### {model_name}")
                report_lines.append("")
                
                for test_name, test_results in model_results.items():
                    if test_results and 'spearman_corr' in test_results:
                        corr_results = test_results['spearman_corr']
                        report_lines.append(f"**{test_name}**:")
                        report_lines.append(f"- Original: {corr_results.get('original', 0):.4f}")
                        report_lines.append(f"- Placebo Mean: {corr_results.get('placebo_mean', 0):.4f}")
                        report_lines.append(f"- P-value: {corr_results.get('p_value', 1):.4f}")
                        report_lines.append(f"- Significant: {'Yes' if corr_results.get('significant', False) else 'No'}")
                        report_lines.append("")
        
        # Stability analysis
        if 'stability_analysis' in robustness_results:
            report_lines.append("## Stability Analysis")
            report_lines.append("")
            
            for model_name, model_results in robustness_results['stability_analysis'].items():
                report_lines.append(f"### {model_name}")
                report_lines.append("")
                
                # Walk-forward stability
                if 'walk_forward' in model_results and 'stability_stats' in model_results['walk_forward']:
                    stability_stats = model_results['walk_forward']['stability_stats']
                    if 'spearman_corr' in stability_stats:
                        corr_stats = stability_stats['spearman_corr']
                        report_lines.append("**Walk-forward Stability:**")
                        report_lines.append(f"- Mean: {corr_stats.get('mean', 0):.4f}")
                        report_lines.append(f"- Std: {corr_stats.get('std', 0):.4f}")
                        report_lines.append(f"- CV: {corr_stats.get('cv', 0):.4f}")
                        report_lines.append("")
                
                # Yearly stability
                if 'yearly' in model_results and 'yearly_metrics' in model_results['yearly']:
                    yearly_metrics = model_results['yearly']['yearly_metrics']
                    report_lines.append("**Yearly Performance:**")
                    for year, metrics in yearly_metrics.items():
                        report_lines.append(f"- {year}: {metrics.get('spearman_corr', 0):.4f}")
                    report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Robustness report saved to {save_path}")
        
        return report_content
    
    def generate_comprehensive_report(self, all_results: Dict[str, Any], 
                                    save_dir: Optional[str] = None) -> str:
        """
        Generate comprehensive report combining all analyses.
        
        Args:
            all_results: Dictionary with all analysis results
            save_dir: Optional directory to save report
            
        Returns:
            Report content as string
        """
        logger.info("Generating comprehensive report...")
        
        if save_dir:
            save_path = Path(save_dir) / "comprehensive_report.md"
        else:
            save_path = self.output_dir / "comprehensive_report.md"
        
        report_lines = []
        report_lines.append("# Meme Stock Contrarian Effect Prediction - Comprehensive Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This report presents a comprehensive analysis of meme stock contrarian effects")
        report_lines.append("using Reddit sentiment data and machine learning models. The analysis includes:")
        report_lines.append("- Feature engineering from Reddit text and technical indicators")
        report_lines.append("- Multiple machine learning models (Ridge, LightGBM, TCN, TFT)")
        report_lines.append("- Ensemble methods and meta-learning")
        report_lines.append("- Trading strategy implementation with costs")
        report_lines.append("- Robustness testing and interpretability analysis")
        report_lines.append("")
        
        # Model Performance
        if 'model_results' in all_results:
            report_lines.append("## Model Performance")
            report_lines.append("")
            
            model_results = all_results['model_results']
            best_model = max(model_results.items(), key=lambda x: x[1].get('spearman_corr', 0))
            
            report_lines.append(f"**Best Model**: {best_model[0]}")
            report_lines.append(f"**Spearman Correlation**: {best_model[1].get('spearman_corr', 0):.4f}")
            report_lines.append(f"**Hit Rate**: {best_model[1].get('hit_rate', 0):.2%}")
            report_lines.append("")
        
        # Trading Strategy Results
        if 'strategy_results' in all_results:
            report_lines.append("## Trading Strategy Results")
            report_lines.append("")
            
            strategy_results = all_results['strategy_results']
            if 'metrics' in strategy_results:
                metrics = strategy_results['metrics']
                report_lines.append(f"**Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.3f}")
                report_lines.append(f"**Maximum Drawdown**: {metrics.get('max_drawdown', 0):.2%}")
                report_lines.append(f"**Hit Rate**: {metrics.get('hit_rate', 0):.2%}")
                report_lines.append("")
        
        # Robustness Analysis
        if 'robustness_results' in all_results:
            report_lines.append("## Robustness Analysis")
            report_lines.append("")
            
            robustness_results = all_results['robustness_results']
            
            # Check placebo tests
            placebo_passed = True
            if 'placebo_tests' in robustness_results:
                for model_results in robustness_results['placebo_tests'].values():
                    for test_results in model_results.values():
                        if test_results and 'spearman_corr' in test_results:
                            if test_results['spearman_corr'].get('significant', False):
                                placebo_passed = False
            
            report_lines.append(f"**Placebo Tests**: {'Passed' if placebo_passed else 'Failed'}")
            
            # Check stability
            stability_passed = True
            if 'stability_analysis' in robustness_results:
                for model_results in robustness_results['stability_analysis'].values():
                    if 'walk_forward' in model_results and 'stability_stats' in model_results['walk_forward']:
                        stability_stats = model_results['walk_forward']['stability_stats']
                        if 'spearman_corr' in stability_stats:
                            cv = stability_stats['spearman_corr'].get('cv', 0)
                            if cv > 0.5:
                                stability_passed = False
            
            report_lines.append(f"**Stability Tests**: {'Passed' if stability_passed else 'Failed'}")
            report_lines.append("")
        
        # Conclusions
        report_lines.append("## Conclusions")
        report_lines.append("")
        
        if 'model_results' in all_results:
            model_results = all_results['model_results']
            best_corr = max(metrics.get('spearman_corr', 0) for metrics in model_results.values())
            
            if best_corr > 0.1:
                report_lines.append("✅ **Contrarian hypothesis supported**: Models show positive predictive power")
            elif best_corr > 0.05:
                report_lines.append("⚠️ **Weak contrarian effect**: Models show modest predictive power")
            else:
                report_lines.append("❌ **Contrarian hypothesis not supported**: Models show limited predictive power")
        
        report_lines.append("")
        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("1. **Data Quality**: Ensure high-quality Reddit data collection")
        report_lines.append("2. **Feature Engineering**: Continue refining text and technical features")
        report_lines.append("3. **Model Selection**: Use ensemble methods for improved robustness")
        report_lines.append("4. **Risk Management**: Implement proper position sizing and risk controls")
        report_lines.append("5. **Monitoring**: Continuously monitor model performance and market conditions")
        report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        with open(save_path, 'w') as f:
            f.write(report_content)
        logger.info(f"Comprehensive report saved to {save_path}")
        
        return report_content

def generate_all_visualizations(all_results: Dict[str, Any], 
                              output_dir: str = "reports/figures") -> Dict[str, str]:
    """
    Generate all visualizations for the analysis.
    
    Args:
        all_results: Dictionary with all analysis results
        output_dir: Output directory for figures
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    logger.info("Generating all visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = ResultsVisualizer()
    figure_paths = {}
    
    # Model performance comparison
    if 'model_results' in all_results:
        fig_path = output_path / "model_performance_comparison.png"
        visualizer.plot_model_performance_comparison(all_results['model_results'], str(fig_path))
        figure_paths['model_performance'] = str(fig_path)
    
    # Feature importance
    if 'feature_importance' in all_results:
        fig_path = output_path / "feature_importance.png"
        visualizer.plot_feature_importance(all_results['feature_importance'], save_path=str(fig_path))
        figure_paths['feature_importance'] = str(fig_path)
    
    # Strategy performance
    if 'strategy_results' in all_results:
        fig_path = output_path / "strategy_performance.png"
        visualizer.plot_strategy_performance(all_results['strategy_results'], str(fig_path))
        figure_paths['strategy_performance'] = str(fig_path)
    
    # Stability analysis
    if 'stability_analysis' in all_results:
        fig_path = output_path / "stability_analysis.png"
        visualizer.plot_stability_analysis(all_results['stability_analysis'], str(fig_path))
        figure_paths['stability_analysis'] = str(fig_path)
    
    # Placebo test results
    if 'placebo_tests' in all_results:
        fig_path = output_path / "placebo_test_results.png"
        visualizer.plot_placebo_test_results(all_results['placebo_tests'], str(fig_path))
        figure_paths['placebo_tests'] = str(fig_path)
    
    logger.info(f"Generated {len(figure_paths)} visualizations")
    
    return figure_paths

if __name__ == "__main__":
    # Test reporting functionality
    np.random.seed(42)
    
    # Create sample results
    model_results = {
        'ridge': {'spearman_corr': 0.15, 'hit_rate': 0.55, 'rmse': 0.02},
        'lgb': {'spearman_corr': 0.18, 'hit_rate': 0.58, 'rmse': 0.019},
        'tcn': {'spearman_corr': 0.12, 'hit_rate': 0.52, 'rmse': 0.021}
    }
    
    feature_importance = {
        'ridge': {'reddit_surprise': 0.3, 'confidence_score': 0.25, 'sentiment': 0.2},
        'lgb': {'reddit_surprise': 0.35, 'confidence_score': 0.22, 'sentiment': 0.18}
    }
    
    strategy_results = {
        'strategy_type': 'contrarian',
        'parameters': {'threshold': 0.001, 'cost_half': 0.001},
        'metrics': {'sharpe_ratio': 1.2, 'max_drawdown': -0.15, 'hit_rate': 0.55},
        'returns': np.random.randn(252) * 0.01,
        'positions': np.random.choice([-1, 0, 1], 252),
        'signals': np.random.randn(252) * 0.02
    }
    
    all_results = {
        'model_results': model_results,
        'feature_importance': feature_importance,
        'strategy_results': strategy_results
    }
    
    # Generate reports
    reporter = ResultsReporter()
    
    # Model report
    model_report = reporter.generate_model_report(model_results, feature_importance)
    print("Model Report Generated")
    
    # Strategy report
    strategy_report = reporter.generate_strategy_report(strategy_results)
    print("Strategy Report Generated")
    
    # Comprehensive report
    comprehensive_report = reporter.generate_comprehensive_report(all_results)
    print("Comprehensive Report Generated")
    
    # Generate visualizations
    try:
        figure_paths = generate_all_visualizations(all_results)
        print(f"Generated {len(figure_paths)} visualizations")
    except Exception as e:
        print(f"Visualization generation failed: {e}")

