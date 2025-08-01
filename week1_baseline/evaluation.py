"""
Evaluation Framework for Meme Stock Prediction
Week 1 Implementation - Academic Competition Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, data_path='../data/features_data.csv', results_path='../models/model_results.pkl'):
        self.data_path = data_path
        self.results_path = results_path
        self.data = None
        self.results = None
        
    def load_data(self):
        """Load data and results"""
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"✅ Loaded data: {self.data.shape}")
        except FileNotFoundError:
            print("❌ Features data not found.")
            return False
        
        try:
            import joblib
            self.results = joblib.load(self.results_path)
            print(f"✅ Loaded results for {len(self.results)} models")
        except FileNotFoundError:
            print("❌ Model results not found. Please run models.py first.")
            return False
        
        return True
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio for trading strategy"""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy for price predictions"""
        if len(y_true) != len(y_pred):
            return 0
        
        correct_directions = 0
        total_predictions = 0
        
        for i in range(1, len(y_true)):
            true_direction = 1 if y_true[i] > y_true[i-1] else 0
            pred_direction = 1 if y_pred[i] > y_pred[i-1] else 0
            
            if true_direction == pred_direction:
                correct_directions += 1
            total_predictions += 1
        
        return correct_directions / total_predictions if total_predictions > 0 else 0
    
    def evaluate_model_performance(self, model_name, target_col):
        """Evaluate individual model performance"""
        print(f"\n📊 Evaluating {model_name} for {target_col}")
        
        # Get feature columns
        exclude_cols = ['date'] + [col for col in self.data.columns if 'direction' in col or 'magnitude' in col]
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Prepare data
        clean_data = self.data.dropna(subset=[target_col])
        X = clean_data[feature_cols].values
        y_true = clean_data[target_col].values
        
        # Load model and predict
        try:
            import joblib
            model = joblib.load(f"../models/{model_name}.pkl")
            y_pred = model.predict(X)
        except:
            print(f"❌ Could not load model {model_name}")
            return None
        
        # Calculate metrics
        metrics = {}
        
        if 'direction' in target_col:
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            # For binary classification, calculate AUC
            if len(np.unique(y_true)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X)[:, 1]
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                except:
                    metrics['auc_roc'] = 0
        else:
            # Regression metrics
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['directional_accuracy'] = self.calculate_directional_accuracy(y_true, y_pred)
        
        return metrics
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        print("📋 Generating Comprehensive Evaluation Report")
        print("=" * 60)
        
        report_data = []
        
        for model_name, results in self.results.items():
            # Extract target from model name
            target_col = model_name.split('_', 1)[1]
            
            # Get basic metrics from cross-validation
            basic_metrics = {
                'Model': model_name,
                'Target': target_col,
                'Mean CV Score': f"{results['mean_score']:.4f}",
                'Std CV Score': f"{results['std_score']:.4f}",
                'CV Scores': results['cv_scores']
            }
            
            # Get additional evaluation metrics
            eval_metrics = self.evaluate_model_performance(model_name, target_col)
            if eval_metrics:
                basic_metrics.update(eval_metrics)
            
            report_data.append(basic_metrics)
        
        # Create comprehensive report
        report_df = pd.DataFrame(report_data)
        
        # Save detailed report
        report_df.to_csv('../data/comprehensive_evaluation.csv', index=False)
        print(f"✅ Comprehensive evaluation saved to ../data/comprehensive_evaluation.csv")
        
        # Print summary
        print("\n📊 Model Performance Summary")
        print("-" * 40)
        summary_cols = ['Model', 'Target', 'Mean CV Score', 'Std CV Score']
        if 'accuracy' in report_df.columns:
            summary_cols.append('accuracy')
        if 'rmse' in report_df.columns:
            summary_cols.append('rmse')
        
        print(report_df[summary_cols].to_string(index=False))
        
        return report_df
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance for all models"""
        print("\n📈 Generating Feature Importance Plots")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        plot_idx = 0
        
        for model_name, results in self.results.items():
            if 'feature_importance' in results and 'feature_names' in results:
                # Get top features
                importance = results['feature_importance']
                feature_names = results['feature_names']
                
                # Create DataFrame for plotting
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True).tail(top_n)
                
                # Plot
                ax = axes[plot_idx]
                importance_df.plot(kind='barh', x='feature', y='importance', ax=ax)
                ax.set_title(f'Feature Importance - {model_name}')
                ax.set_xlabel('Importance')
                
                plot_idx += 1
                
                if plot_idx >= 4:  # Limit to 4 plots
                    break
        
        # Hide empty subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('../data/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✅ Feature importance plot saved to ../data/feature_importance.png")
        plt.show()
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        print("\n📊 Generating Model Comparison Plot")
        
        # Prepare data for plotting
        plot_data = []
        
        for model_name, results in self.results.items():
            plot_data.append({
                'Model': model_name.split('_')[0],  # Extract model type
                'Target': results.get('target', 'Unknown'),
                'Score': results['mean_score'],
                'Std': results['std_score']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # Group by model type
        model_groups = plot_df.groupby('Model')
        
        for i, (model_type, group) in enumerate(model_groups):
            plt.errorbar(
                range(len(group)), 
                group['Score'], 
                yerr=group['Std'],
                marker='o',
                label=model_type.upper(),
                capsize=5
            )
        
        plt.xlabel('Target Variables')
        plt.ylabel('Mean CV Score')
        plt.title('Model Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels
        targets = plot_df['Target'].unique()
        plt.xticks(range(len(targets)), targets, rotation=45)
        
        plt.tight_layout()
        plt.savefig('../data/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Model comparison plot saved to ../data/model_comparison.png")
        plt.show()
    
    def plot_predictions_vs_actual(self, model_name, target_col):
        """Plot predictions vs actual values"""
        print(f"\n📈 Plotting predictions vs actual for {model_name}")
        
        # Get feature columns
        exclude_cols = ['date'] + [col for col in self.data.columns if 'direction' in col or 'magnitude' in col]
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Prepare data
        clean_data = self.data.dropna(subset=[target_col])
        X = clean_data[feature_cols].values
        y_true = clean_data[target_col].values
        
        # Load model and predict
        try:
            import joblib
            model = joblib.load(f"../models/{model_name}.pkl")
            y_pred = model.predict(X)
        except:
            print(f"❌ Could not load model {model_name}")
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot predictions vs actual
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actual - {model_name}')
        
        # Plot residuals
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Plot time series
        plt.subplot(2, 2, 3)
        dates = clean_data['date'].values
        plt.plot(dates, y_true, label='Actual', alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title('Time Series Comparison')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot distribution
        plt.subplot(2, 2, 4)
        plt.hist(y_true, alpha=0.7, label='Actual', bins=30)
        plt.hist(y_pred, alpha=0.7, label='Predicted', bins=30)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Distribution Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'../data/predictions_{model_name}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Predictions plot saved to ../data/predictions_{model_name}.png")
        plt.show()
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 Starting Complete Evaluation Pipeline")
        print("=" * 60)
        
        if not self.load_data():
            return None
        
        # Generate comprehensive report
        report_df = self.generate_comprehensive_report()
        
        # Generate plots
        self.plot_feature_importance()
        self.plot_model_comparison()
        
        # Plot predictions for best model
        if len(self.results) > 0:
            best_model = max(self.results.items(), key=lambda x: x[1]['mean_score'])[0]
            target_col = best_model.split('_', 1)[1]
            self.plot_predictions_vs_actual(best_model, target_col)
        
        print("\n🎉 Evaluation Pipeline Completed!")
        return report_df

if __name__ == "__main__":
    # Run evaluation
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation() 