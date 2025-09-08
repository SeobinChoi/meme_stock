#!/usr/bin/env python3
"""
Contrarian Strategy Mathematical Explanation
===========================================
Contrarian 전략의 수식적 구현과 이론적 배경
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def explain_contrarian_mathematics():
    """Contrarian 전략의 수학적 설명"""
    print("📊 CONTRARIAN STRATEGY: MATHEMATICAL EXPLANATION")
    print("=" * 60)
    
    print("\n🔢 1. BASIC MATHEMATICAL FORMULATION")
    print("-" * 40)
    
    print("Given:")
    print("   • f(X) = trained model prediction function")
    print("   • X = feature vector")
    print("   • y = actual return")
    print("   • ŷ = f(X) = predicted return")
    print("")
    
    print("Standard Strategy:")
    print("   ŷ_standard = f(X)")
    print("   IC_standard = corr(y, ŷ_standard)")
    print("")
    
    print("Contrarian Strategy:")
    print("   ŷ_contrarian = -f(X)")
    print("   IC_contrarian = corr(y, ŷ_contrarian)")
    print("                 = corr(y, -ŷ_standard)")
    print("                 = -corr(y, ŷ_standard)")
    print("                 = -IC_standard")
    print("")
    
    print("Key Insight:")
    print("   |IC_contrarian| = |IC_standard|")
    print("   Therefore: Contrarian doesn't improve prediction power,")
    print("            it only reverses the correlation sign!")

def demonstrate_contrarian_effect():
    """Contrarian 효과 시연"""
    print("\n🎯 2. PRACTICAL DEMONSTRATION")
    print("-" * 40)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 100
    
    # 실제 수익률 (정규분포)
    y_actual = np.random.normal(0, 0.05, n_samples)
    
    # 모델 예측 (음의 상관관계 가정)
    noise = np.random.normal(0, 0.02, n_samples)
    y_pred_standard = -0.3 * y_actual + noise  # 음의 상관관계
    
    # Contrarian 예측
    y_pred_contrarian = -y_pred_standard
    
    # IC 계산
    ic_standard, _ = pearsonr(y_actual, y_pred_standard)
    ic_contrarian, _ = pearsonr(y_actual, y_pred_contrarian)
    
    print(f"Example with synthetic data:")
    print(f"   Actual returns: μ={np.mean(y_actual):.4f}, σ={np.std(y_actual):.4f}")
    print(f"   Standard predictions: μ={np.mean(y_pred_standard):.4f}, σ={np.std(y_pred_standard):.4f}")
    print(f"   Contrarian predictions: μ={np.mean(y_pred_contrarian):.4f}, σ={np.std(y_pred_contrarian):.4f}")
    print("")
    print(f"Results:")
    print(f"   IC_standard = {ic_standard:.4f}")
    print(f"   IC_contrarian = {ic_contrarian:.4f}")
    print(f"   IC_contrarian = -IC_standard? {np.isclose(ic_contrarian, -ic_standard)}")
    print(f"   |IC| improvement = {abs(ic_contrarian) - abs(ic_standard):.6f}")

def trading_signal_mathematics():
    """거래 신호의 수학적 표현"""
    print("\n💰 3. TRADING SIGNAL MATHEMATICS")
    print("-" * 40)
    
    print("Position Generation:")
    print("   Standard Strategy:")
    print("     Position_t = sign(ŷ_t)")
    print("     where ŷ_t = f(X_t)")
    print("")
    print("   Contrarian Strategy:")
    print("     Position_t = sign(-ŷ_t)")
    print("                = -sign(ŷ_t)")
    print("                = -Position_standard_t")
    print("")
    
    print("Strategy Returns:")
    print("   R_standard_t = Position_standard_t × y_t")
    print("   R_contrarian_t = Position_contrarian_t × y_t")
    print("                  = -Position_standard_t × y_t")
    print("                  = -R_standard_t")
    print("")
    
    print("Key Mathematical Property:")
    print("   If IC_standard < 0 (negative correlation):")
    print("     → Standard strategy loses money on average")
    print("     → Contrarian strategy makes money on average")
    print("   This is because: E[R_contrarian] = -E[R_standard]")

def information_coefficient_theory():
    """Information Coefficient 이론"""
    print("\n📊 4. INFORMATION COEFFICIENT THEORY")
    print("-" * 40)
    
    print("Definition:")
    print("   IC = corr(y_actual, y_predicted)")
    print("   IC ∈ [-1, 1]")
    print("")
    
    print("Properties under Contrarian transformation:")
    print("   Let ŷ_c = -ŷ_s (contrarian = negative standard)")
    print("")
    print("   IC_c = corr(y, ŷ_c)")
    print("        = corr(y, -ŷ_s)")
    print("        = (1/n-1) Σ[(y_i - μ_y) × (-ŷ_s_i - μ_{-ŷ_s})] / (σ_y × σ_{-ŷ_s})")
    print("        = (1/n-1) Σ[(y_i - μ_y) × (-(ŷ_s_i - μ_ŷ_s))] / (σ_y × σ_ŷ_s)")
    print("        = -(1/n-1) Σ[(y_i - μ_y) × (ŷ_s_i - μ_ŷ_s)] / (σ_y × σ_ŷ_s)")
    print("        = -IC_s")
    print("")
    print("Therefore: IC_contrarian = -IC_standard")
    print("          |IC_contrarian| = |IC_standard|")

def hit_rate_mathematics():
    """Hit Rate 수학적 분석"""
    print("\n🎯 5. HIT RATE MATHEMATICS")
    print("-" * 40)
    
    print("Hit Rate Definition:")
    print("   HR = (1/n) Σ I[sign(y_i) = sign(ŷ_i)]")
    print("   where I[·] is indicator function")
    print("")
    
    print("Under Contrarian Strategy:")
    print("   HR_contrarian = (1/n) Σ I[sign(y_i) = sign(-ŷ_i)]")
    print("                 = (1/n) Σ I[sign(y_i) = -sign(ŷ_i)]")
    print("                 = (1/n) Σ I[sign(y_i) ≠ sign(ŷ_i)]")
    print("                 = 1 - HR_standard")
    print("")
    print("Key Insight:")
    print("   If HR_standard < 0.5:")
    print("     → HR_contrarian > 0.5")
    print("   The model is directionally wrong, but contrarian fixes it!")

def practical_implementation():
    """실제 구현 방법"""
    print("\n💻 6. PRACTICAL IMPLEMENTATION")
    print("-" * 40)
    
    print("Code Implementation:")
    print("""
    # Standard Strategy
    def standard_strategy(model, X_test):
        predictions = model.predict(X_test)
        return predictions
    
    # Contrarian Strategy  
    def contrarian_strategy(model, X_test):
        predictions = model.predict(X_test)
        return -predictions  # Simple negation!
    
    # Target-Flipped Training
    def target_flipped_training(model, X_train, y_train):
        y_train_flipped = -y_train  # Flip targets
        model.fit(X_train, y_train_flipped)
        return model
    """)
    
    print("\nThree Contrarian Approaches:")
    print("   1. Prediction Flip: ŷ_contrarian = -ŷ_standard")
    print("   2. Target Flip: Train on -y, predict normally")
    print("   3. Sign Flip: Position = -sign(ŷ_standard)")

def when_contrarian_works():
    """Contrarian이 언제 효과적인가"""
    print("\n🔍 7. WHEN DOES CONTRARIAN WORK?")
    print("-" * 40)
    
    print("Theoretical Conditions:")
    print("   1. IC_standard < 0 (negative correlation)")
    print("   2. Model captures inverse relationship")
    print("   3. Market exhibits contrarian behavior")
    print("")
    
    print("Meme Stock Context:")
    print("   • High social media attention → Price reversal")
    print("   • Hype peaks → Subsequent decline")
    print("   • Fear peaks → Subsequent recovery")
    print("   • Momentum exhaustion → Mean reversion")
    print("")
    
    print("Mathematical Expectation:")
    print("   If E[y_t × ŷ_t] < 0:")
    print("     → Standard strategy: E[return] < 0")
    print("     → Contrarian strategy: E[return] > 0")

def limitations_and_caveats():
    """한계점과 주의사항"""
    print("\n⚠️ 8. LIMITATIONS AND CAVEATS")
    print("-" * 40)
    
    print("Mathematical Limitations:")
    print("   1. |IC_contrarian| = |IC_standard| (no improvement in correlation)")
    print("   2. Same prediction variance: Var(ŷ_c) = Var(ŷ_s)")
    print("   3. Noise amplification: If model is noisy, contrarian is equally noisy")
    print("")
    
    print("Practical Considerations:")
    print("   • Transaction costs not accounted for")
    print("   • Market regime changes")
    print("   • Model overfitting to historical patterns")
    print("   • Assumption of consistent contrarian effect")
    print("")
    
    print("Statistical Validity:")
    print("   • Contrarian effectiveness should be cross-validated")
    print("   • Out-of-sample testing essential")
    print("   • Multiple time periods required")

def main():
    """메인 실행 함수"""
    explain_contrarian_mathematics()
    demonstrate_contrarian_effect()
    trading_signal_mathematics()
    information_coefficient_theory()
    hit_rate_mathematics()
    practical_implementation()
    when_contrarian_works()
    limitations_and_caveats()
    
    print("\n" + "="*60)
    print("📋 SUMMARY: CONTRARIAN MATHEMATICS")
    print("="*60)
    print("🔢 Core Formula: ŷ_contrarian = -ŷ_standard")
    print("📊 IC Property: IC_contrarian = -IC_standard")
    print("🎯 Hit Rate: HR_contrarian = 1 - HR_standard")
    print("💰 Returns: R_contrarian = -R_standard")
    print("⚠️ Key Insight: Contrarian reverses direction, not prediction quality")
    print("✅ Effective when: IC_standard < 0 (negative correlation)")

if __name__ == "__main__":
    main()
    print(f"\n📚 Use this mathematical framework in your paper!")

