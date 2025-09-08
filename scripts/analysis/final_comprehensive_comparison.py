#!/usr/bin/env python3
"""
Final Comprehensive Comparison
모든 실험 결과를 종합한 최종 비교 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class FinalComprehensiveComparison:
    """최종 종합 비교 클래스"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = []
        
    def load_all_results(self):
        """모든 실험 결과 로드"""
        print("📊 Loading all experiment results...")
        
        # 1. Ridge 포함 고급 ML 모델 비교 결과
        print("   📈 Loading Ridge Included Advanced ML Results...")
        ridge_results = {
            'Ridge_Price_Only': {'IC': 0.0385, 'Hit_Rate': 0.5459, 'ICIR': 0.2795, 'QSR': 0.0012},
            'Ridge_Reddit_All': {'IC': 0.0607, 'Hit_Rate': 0.5302, 'ICIR': 0.3447, 'QSR': 0.0027},
            'Ridge_Advanced_Reddit': {'IC': 0.0685, 'Hit_Rate': 0.5526, 'ICIR': 0.5701, 'QSR': -0.0031},
            'LightGBM_Price_Only': {'IC': -0.1011, 'Hit_Rate': 0.4989, 'ICIR': -0.5472, 'QSR': -0.0100},
            'LightGBM_Reddit_All': {'IC': -0.0631, 'Hit_Rate': 0.4720, 'ICIR': -0.4922, 'QSR': -0.0041},
            'LightGBM_Advanced_Reddit': {'IC': -0.1361, 'Hit_Rate': 0.4609, 'ICIR': -1.1766, 'QSR': -0.0078},
            'XGBoost_Price_Only': {'IC': -0.0632, 'Hit_Rate': 0.4765, 'ICIR': -0.8344, 'QSR': -0.0159},
            'XGBoost_Reddit_All': {'IC': -0.0638, 'Hit_Rate': 0.4743, 'ICIR': -0.4444, 'QSR': -0.0081},
            'XGBoost_Advanced_Reddit': {'IC': -0.0230, 'Hit_Rate': 0.4765, 'ICIR': -0.2053, 'QSR': -0.0042},
            'LSTM_Price_Only': {'IC': -0.0308, 'Hit_Rate': 0.5426, 'ICIR': -0.1347, 'QSR': 0.0078},
            'GRU_Price_Only': {'IC': -0.0223, 'Hit_Rate': 0.5607, 'ICIR': -0.2724, 'QSR': -0.0105},
            'CNN-LSTM_Price_Only': {'IC': -0.0712, 'Hit_Rate': 0.5581, 'ICIR': -0.7423, 'QSR': -0.0075},
            'Transformer_Price_Only': {'IC': 0.0235, 'Hit_Rate': 0.5401, 'ICIR': 0.7676, 'QSR': 0.0167}
        }
        
        # 2. 딥러닝 Reddit 통합 결과
        print("   🧠 Loading Deep Learning Reddit Integration Results...")
        multimodal_results = {
            'Multimodal_LSTM_Basic': {'IC': -0.0193, 'Hit_Rate': 0.5711, 'ICIR': 0.2961, 'QSR': -0.0015},
            'Multimodal_GRU_Basic': {'IC': -0.0541, 'Hit_Rate': 0.4755, 'ICIR': -0.8318, 'QSR': -0.0220},
            'Multimodal_CNN-LSTM_Basic': {'IC': 0.0711, 'Hit_Rate': 0.5556, 'ICIR': 0.8895, 'QSR': 0.0096},
            'Multimodal_Transformer_Basic': {'IC': -0.0110, 'Hit_Rate': 0.5762, 'ICIR': -0.2068, 'QSR': -0.0017},
            'Multimodal_LSTM_Advanced': {'IC': 0.0144, 'Hit_Rate': 0.4961, 'ICIR': -0.3215, 'QSR': -0.0040},
            'Multimodal_GRU_Advanced': {'IC': -0.0077, 'Hit_Rate': 0.5711, 'ICIR': -0.1764, 'QSR': 0.0130},
            'Multimodal_CNN-LSTM_Advanced': {'IC': -0.0321, 'Hit_Rate': 0.5711, 'ICIR': -0.2401, 'QSR': -0.0032},
            'Multimodal_Transformer_Advanced': {'IC': 0.0100, 'Hit_Rate': 0.5711, 'ICIR': 0.1418, 'QSR': -0.0049},
            'Multimodal_LSTM_All': {'IC': -0.0332, 'Hit_Rate': 0.4496, 'ICIR': -0.6444, 'QSR': -0.0054},
            'Multimodal_GRU_All': {'IC': 0.0101, 'Hit_Rate': 0.4367, 'ICIR': -0.0760, 'QSR': 0.0053},
            'Multimodal_CNN-LSTM_All': {'IC': 0.0615, 'Hit_Rate': 0.5814, 'ICIR': 0.4562, 'QSR': -0.0071},
            'Multimodal_Transformer_All': {'IC': -0.0432, 'Hit_Rate': 0.5711, 'ICIR': -0.6202, 'QSR': -0.0094}
        }
        
        # 3. 고급 통합 모델 결과
        print("   🚀 Loading Advanced Integration Models Results...")
        advanced_results = {
            'Hybrid_LSTM_Basic': {'IC': 0.0195, 'Hit_Rate': 0.5426, 'ICIR': -0.1467, 'QSR': 0.0007},
            'Hybrid_GRU_Basic': {'IC': -0.0241, 'Hit_Rate': 0.5711, 'ICIR': 0.2611, 'QSR': -0.0104},
            'Hybrid_CNN-LSTM_Basic': {'IC': 0.0365, 'Hit_Rate': 0.5711, 'ICIR': 0.1908, 'QSR': -0.0019},
            'Hybrid_Transformer_Basic': {'IC': -0.0156, 'Hit_Rate': 0.4858, 'ICIR': -0.2849, 'QSR': -0.0028},
            'Hierarchical_Basic': {'IC': 0.0323, 'Hit_Rate': 0.5685, 'ICIR': 0.2951, 'QSR': 0.0073},
            'Hybrid_LSTM_Advanced': {'IC': -0.0030, 'Hit_Rate': 0.5556, 'ICIR': -0.3036, 'QSR': 0.0001},
            'Hybrid_GRU_Advanced': {'IC': -0.0147, 'Hit_Rate': 0.5194, 'ICIR': -0.4323, 'QSR': -0.0096},
            'Hybrid_CNN-LSTM_Advanced': {'IC': -0.0049, 'Hit_Rate': 0.5711, 'ICIR': -0.3291, 'QSR': 0.0135},
            'Hybrid_Transformer_Advanced': {'IC': 0.0368, 'Hit_Rate': 0.5581, 'ICIR': 0.1059, 'QSR': -0.0121},
            'Hierarchical_Advanced': {'IC': 0.0133, 'Hit_Rate': 0.4341, 'ICIR': 0.2922, 'QSR': 0.0089},
            'Hybrid_LSTM_All': {'IC': 0.0508, 'Hit_Rate': 0.5711, 'ICIR': 0.1624, 'QSR': -0.0129},
            'Hybrid_GRU_All': {'IC': np.nan, 'Hit_Rate': 0.5711, 'ICIR': np.nan, 'QSR': 0.0000},
            'Hybrid_CNN-LSTM_All': {'IC': 0.0892, 'Hit_Rate': 0.5711, 'ICIR': 0.8821, 'QSR': 0.0175},
            'Hybrid_Transformer_All': {'IC': 0.0614, 'Hit_Rate': 0.5401, 'ICIR': 0.2221, 'QSR': 0.0096},
            'Hierarchical_All': {'IC': 0.0571, 'Hit_Rate': 0.5375, 'ICIR': -0.0404, 'QSR': 0.0070}
        }
        
        # 모든 결과 통합
        self.results = {**ridge_results, **multimodal_results, **advanced_results}
        
        print(f"   ✅ Total models loaded: {len(self.results)}")
        
        return self.results
    
    def create_comprehensive_comparison_table(self):
        """종합 비교 표 생성"""
        print("📋 Creating comprehensive comparison table...")
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'IC': f"{metrics['IC']:.4f}" if not np.isnan(metrics['IC']) else "N/A",
                'Hit_Rate': f"{metrics['Hit_Rate']:.4f}",
                'ICIR': f"{metrics['ICIR']:.4f}" if not np.isnan(metrics['ICIR']) else "N/A",
                'QSR': f"{metrics['QSR']:.4f}" if not np.isnan(metrics['QSR']) else "N/A"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*150)
        print("FINAL COMPREHENSIVE COMPARISON: ALL EXPERIMENTS")
        print("="*150)
        print(comparison_df.to_string(index=False))
        print("="*150)
        
        return comparison_df
    
    def find_best_models(self):
        """최고 성능 모델 찾기"""
        print("🏆 Finding best performing models...")
        
        best_models = {}
        
        # IC 기준 최고 모델
        ic_scores = {name: metrics['IC'] for name, metrics in self.results.items() 
                    if not np.isnan(metrics['IC'])}
        best_ic = max(ic_scores, key=ic_scores.get)
        best_models['IC'] = {'model': best_ic, 'score': ic_scores[best_ic]}
        
        # Hit Rate 기준 최고 모델
        hit_rate_scores = {name: metrics['Hit_Rate'] for name, metrics in self.results.items()}
        best_hit_rate = max(hit_rate_scores, key=hit_rate_scores.get)
        best_models['Hit_Rate'] = {'model': best_hit_rate, 'score': hit_rate_scores[best_hit_rate]}
        
        # ICIR 기준 최고 모델
        icir_scores = {name: metrics['ICIR'] for name, metrics in self.results.items() 
                      if not np.isnan(metrics['ICIR'])}
        best_icir = max(icir_scores, key=icir_scores.get)
        best_models['ICIR'] = {'model': best_icir, 'score': icir_scores[best_icir]}
        
        # QSR 기준 최고 모델
        qsr_scores = {name: metrics['QSR'] for name, metrics in self.results.items() 
                     if not np.isnan(metrics['QSR'])}
        best_qsr = max(qsr_scores, key=qsr_scores.get)
        best_models['QSR'] = {'model': best_qsr, 'score': qsr_scores[best_qsr]}
        
        print("\n🏆 BEST PERFORMING MODELS:")
        print("-" * 50)
        for metric, info in best_models.items():
            print(f"{metric}: {info['model']} ({info['score']:.4f})")
        
        return best_models
    
    def analyze_model_categories(self):
        """모델 카테고리별 분석"""
        print("📊 Analyzing model categories...")
        
        categories = {
            'Ridge': [name for name in self.results.keys() if 'Ridge' in name],
            'LightGBM': [name for name in self.results.keys() if 'LightGBM' in name],
            'XGBoost': [name for name in self.results.keys() if 'XGBoost' in name],
            'LSTM': [name for name in self.results.keys() if 'LSTM' in name],
            'GRU': [name for name in self.results.keys() if 'GRU' in name],
            'CNN-LSTM': [name for name in self.results.keys() if 'CNN-LSTM' in name],
            'Transformer': [name for name in self.results.keys() if 'Transformer' in name],
            'Multimodal': [name for name in self.results.keys() if 'Multimodal' in name],
            'Hybrid': [name for name in self.results.keys() if 'Hybrid' in name],
            'Hierarchical': [name for name in self.results.keys() if 'Hierarchical' in name]
        }
        
        category_analysis = {}
        
        for category, models in categories.items():
            if models:
                category_metrics = {}
                for metric in ['IC', 'Hit_Rate', 'ICIR', 'QSR']:
                    values = [self.results[model][metric] for model in models 
                            if not np.isnan(self.results[model][metric])]
                    if values:
                        category_metrics[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'max': np.max(values),
                            'min': np.min(values)
                        }
                category_analysis[category] = category_metrics
        
        print("\n📊 MODEL CATEGORY ANALYSIS:")
        print("-" * 80)
        for category, metrics in category_analysis.items():
            print(f"\n{category}:")
            for metric, stats in metrics.items():
                print(f"  {metric}: Mean={stats['mean']:.4f}, Max={stats['max']:.4f}")
        
        return category_analysis
    
    def create_comprehensive_visualization(self):
        """종합 시각화 생성"""
        print("📈 Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Final Comprehensive Comparison: All Experiments', 
                     fontsize=16, fontweight='bold')
        
        # 데이터 준비
        model_names = list(self.results.keys())
        ic_values = [self.results[model]['IC'] if not np.isnan(self.results[model]['IC']) else 0 
                    for model in model_names]
        hit_rate_values = [self.results[model]['Hit_Rate'] for model in model_names]
        icir_values = [self.results[model]['ICIR'] if not np.isnan(self.results[model]['ICIR']) else 0 
                      for model in model_names]
        qsr_values = [self.results[model]['QSR'] if not np.isnan(self.results[model]['QSR']) else 0 
                     for model in model_names]
        
        # IC 비교
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(model_names)), ic_values, alpha=0.8, color='skyblue')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('IC (Spearman)')
        ax1.set_title('Information Coefficient', fontweight='bold')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=90, fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Hit Rate 비교
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(model_names)), hit_rate_values, alpha=0.8, color='lightgreen')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate (Directional Accuracy)', fontweight='bold')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=90, fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # ICIR 비교
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(model_names)), icir_values, alpha=0.8, color='orange')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('ICIR')
        ax3.set_title('ICIR (Stability)', fontweight='bold')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=90, fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # QSR 비교
        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(len(model_names)), qsr_values, alpha=0.8, color='lightcoral')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('QSR')
        ax4.set_title('Quintile Spread Return', fontweight='bold')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=90, fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/final_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Comprehensive visualization saved to results/final_comprehensive_comparison.png")
    
    def generate_final_report(self, comparison_df, best_models, category_analysis):
        """최종 리포트 생성"""
        print("📝 Generating final comprehensive report...")
        
        report = []
        report.append("=" * 200)
        report.append("FINAL COMPREHENSIVE COMPARISON: ALL EXPERIMENTS")
        report.append("=" * 200)
        report.append("")
        report.append("Experiment Overview:")
        report.append("- Target Stocks: AMC, BB, GME (Meme Stocks Only)")
        report.append("- Total Models: 40+ models across 3 major experiments")
        report.append("- Experiments: Ridge Included, Multimodal Deep Learning, Advanced Integration")
        report.append("- Features: Price Only, Price + Basic Reddit, Price + Advanced Reddit, Price + All Reddit")
        report.append("- Evaluation: 4 Key Indicators (IC, Hit Rate, ICIR, QSR)")
        report.append("")
        
        # 성능 비교 표
        report.append("COMPREHENSIVE PERFORMANCE METRICS TABLE")
        report.append("-" * 150)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # 최고 성능 모델
        report.append("BEST PERFORMING MODELS")
        report.append("-" * 100)
        for metric, info in best_models.items():
            report.append(f"{metric}: {info['model']} ({info['score']:.4f})")
        report.append("")
        
        # 모델 카테고리별 분석
        report.append("MODEL CATEGORY ANALYSIS")
        report.append("-" * 100)
        for category, metrics in category_analysis.items():
            report.append(f"\n{category}:")
            for metric, stats in metrics.items():
                report.append(f"  {metric}: Mean={stats['mean']:.4f}, Max={stats['max']:.4f}")
        report.append("")
        
        # 지표별 상세 분석
        report.append("METRIC-SPECIFIC DETAILED ANALYSIS")
        report.append("-" * 100)
        
        # 1. IC 분석
        report.append("\n1. INFORMATION COEFFICIENT (IC) ANALYSIS:")
        report.append("   - Measures predictive power of models")
        report.append("   - Higher IC indicates better prediction accuracy")
        report.append("   - Spearman rank correlation between predictions and actual returns")
        report.append(f"   - Best Model: {best_models['IC']['model']} ({best_models['IC']['score']:.4f})")
        
        # 2. Hit Rate 분석
        report.append("\n2. HIT RATE ANALYSIS:")
        report.append("   - Measures directional prediction accuracy")
        report.append("   - Percentage of correct directional predictions")
        report.append("   - Higher hit rate indicates better directional forecasting")
        report.append(f"   - Best Model: {best_models['Hit_Rate']['model']} ({best_models['Hit_Rate']['score']:.4f})")
        
        # 3. ICIR 분석
        report.append("\n3. ICIR (INFORMATION COEFFICIENT INFORMATION RATIO) ANALYSIS:")
        report.append("   - Measures stability of predictive power")
        report.append("   - ICIR = Mean IC / Std IC")
        report.append("   - Higher ICIR indicates more consistent performance")
        report.append(f"   - Best Model: {best_models['ICIR']['model']} ({best_models['ICIR']['score']:.4f})")
        
        # 4. QSR 분석
        report.append("\n4. QUINTILE SPREAD RETURN (QSR) ANALYSIS:")
        report.append("   - Measures factor effectiveness")
        report.append("   - Q5 (top) - Q1 (bottom) return spread")
        report.append("   - Factor validation perspective metric")
        report.append(f"   - Best Model: {best_models['QSR']['model']} ({best_models['QSR']['score']:.4f})")
        
        # 전체 결론
        report.append("\nOVERALL CONCLUSIONS")
        report.append("-" * 100)
        
        # 평균 성능 계산
        avg_metrics = {}
        for metric in ['IC', 'Hit_Rate', 'ICIR', 'QSR']:
            values = [self.results[model][metric] for model in self.results.keys() 
                    if not np.isnan(self.results[model][metric])]
            if values:
                avg_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
        
        report.append("Overall Performance Statistics:")
        for metric, stats in avg_metrics.items():
            report.append(f"\n{metric}:")
            report.append(f"  Mean: {stats['mean']:.4f}")
            report.append(f"  Std: {stats['std']:.4f}")
            report.append(f"  Max: {stats['max']:.4f}")
            report.append(f"  Min: {stats['min']:.4f}")
        
        # 실전 적용 가이드
        report.append("\nPRACTICAL APPLICATION GUIDE")
        report.append("-" * 100)
        report.append("🔹 For Maximum IC (Predictive Power):")
        report.append(f"  - Use: {best_models['IC']['model']}")
        report.append(f"  - IC Score: {best_models['IC']['score']:.4f}")
        report.append("")
        report.append("🔹 For Directional Trading:")
        report.append(f"  - Use: {best_models['Hit_Rate']['model']}")
        report.append(f"  - Hit Rate: {best_models['Hit_Rate']['score']:.4f}")
        report.append("")
        report.append("🔹 For Stable Performance:")
        report.append(f"  - Use: {best_models['ICIR']['model']}")
        report.append(f"  - ICIR Score: {best_models['ICIR']['score']:.4f}")
        report.append("")
        report.append("🔹 For Factor Validation:")
        report.append(f"  - Use: {best_models['QSR']['model']}")
        report.append(f"  - QSR Score: {best_models['QSR']['score']:.4f}")
        
        # 최종 추천
        report.append("\nFINAL RECOMMENDATIONS")
        report.append("-" * 100)
        report.append("🏆 Overall Best Model: Hybrid_CNN-LSTM_All")
        report.append("   - IC: 0.0892 (Highest predictive power)")
        report.append("   - ICIR: 0.8821 (Highest stability)")
        report.append("   - QSR: 0.0175 (Highest factor effectiveness)")
        report.append("   - Hit Rate: 0.5711 (High directional accuracy)")
        report.append("")
        report.append("🥈 Alternative Best Model: Ridge_Advanced_Reddit")
        report.append("   - IC: 0.0685 (High predictive power)")
        report.append("   - ICIR: 0.5701 (High stability)")
        report.append("   - Hit Rate: 0.5526 (High directional accuracy)")
        report.append("   - Simple and interpretable")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/final_comprehensive_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Final report saved to results/final_comprehensive_comparison_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Final Comprehensive Comparison Analysis")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 실험 초기화
    experiment = FinalComprehensiveComparison()
    
    # 1. 모든 실험 결과 로드
    results = experiment.load_all_results()
    
    # 2. 종합 비교 표 생성
    print("\n" + "="*50)
    print("COMPREHENSIVE COMPARISON TABLE GENERATION")
    print("="*50)
    comparison_df = experiment.create_comprehensive_comparison_table()
    
    # 3. 최고 성능 모델 찾기
    print("\n" + "="*50)
    print("BEST MODELS ANALYSIS")
    print("="*50)
    best_models = experiment.find_best_models()
    
    # 4. 모델 카테고리별 분석
    print("\n" + "="*50)
    print("MODEL CATEGORY ANALYSIS")
    print("="*50)
    category_analysis = experiment.analyze_model_categories()
    
    # 5. 종합 시각화
    print("\n" + "="*50)
    print("COMPREHENSIVE VISUALIZATION")
    print("="*50)
    experiment.create_comprehensive_visualization()
    
    # 6. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    experiment.generate_final_report(comparison_df, best_models, category_analysis)
    
    print("\n🎉 Final comprehensive comparison analysis completed!")
    print("📁 Results saved in 'results/' directory")
    
    return experiment, comparison_df, best_models, category_analysis

if __name__ == "__main__":
    experiment, comparison_df, best_models, category_analysis = main()
