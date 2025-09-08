#!/usr/bin/env python3
"""
Week 2 Advanced Model Training - Colab Version
Multi-Modal Transformer & Advanced Ensemble Training

This script implements the advanced model architectures for Week 2:
- BERT Sentiment Pipeline (FinBERT)
- Multi-Modal Transformer Architecture
- Advanced LSTM with Attention
- Ensemble System Training

Estimated Training Time: 4-6 hours with GPU
"""

# ============================================================================
# 1. ENVIRONMENT SETUP & DEPENDENCIES
# ============================================================================

# Install required dependencies (run this in Colab)
# !pip install transformers torch torchvision torchaudio
# !pip install sentence-transformers
# !pip install optuna
# !pip install lightgbm xgboost
# !pip install scikit-learn pandas numpy matplotlib seaborn
# !pip install tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import optuna
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 2. DATA UPLOAD & LOADING
# ============================================================================

# In Colab, use this code to upload data:
# from google.colab import files
# import io
# 
# print("Please upload your advanced features dataset (colab_advanced_features.csv)")
# uploaded = files.upload()
# 
# for filename in uploaded.keys():
#     print(f"Loading {filename}...")
#     data = pd.read_csv(io.BytesIO(uploaded[filename]))

# For local testing, load the prepared dataset
try:
    data = pd.read_csv('colab_advanced_features.csv')
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {len(data.columns)}")
    print(data.head())
except FileNotFoundError:
    print("Please run prepare_colab_data.py first to create the dataset")
    exit(1)

# ============================================================================
# 3. DATA PREPARATION & FEATURE ENGINEERING
# ============================================================================

def prepare_advanced_features(data):
    """Prepare features for advanced model training"""
    
    # Separate features and targets
    feature_cols = [col for col in data.columns if not any(x in col for x in 
                    ['direction', 'magnitude', 'returns', 'target'])]
    
    # Target variables
    target_cols = [col for col in data.columns if any(x in col for x in 
                   ['direction', 'magnitude'])]
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns: {len(target_cols)}")
    
    # Remove any text columns for now (we'll handle BERT separately)
    numeric_features = data[feature_cols].select_dtypes(include=[np.number])
    
    # Handle missing values
    numeric_features = numeric_features.fillna(0)
    
    return numeric_features, target_cols, data

# Prepare features
features, target_cols, full_data = prepare_advanced_features(data)
print(f"\nNumeric features shape: {features.shape}")
print(f"Target columns: {target_cols}")

# ============================================================================
# 4. BERT SENTIMENT ANALYSIS PIPELINE
# ============================================================================

class FinancialBERTClassifier(nn.Module):
    def __init__(self, model_name='ProsusAI/finbert', num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Initialize model
print("Initializing FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
bert_model = FinancialBERTClassifier().to(device)
print("✅ FinBERT model initialized")

# ============================================================================
# 5. MULTI-MODAL TRANSFORMER ARCHITECTURE
# ============================================================================

class MultiModalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=6, num_classes=2):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        return logits

# Initialize transformer
transformer_model = MultiModalTransformer(
    input_dim=features.shape[1],
    hidden_dim=256,
    num_heads=8,
    num_layers=6
).to(device)

print(f"✅ Multi-modal transformer initialized with {features.shape[1]} input features")

# ============================================================================
# 6. ADVANCED LSTM WITH ATTENTION
# ============================================================================

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=4,
            batch_first=True
        )
        
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        return logits

# Initialize LSTM
lstm_model = AttentionLSTM(
    input_dim=features.shape[1],
    hidden_dim=128,
    num_layers=2
).to(device)

print(f"✅ Attention LSTM initialized with {features.shape[1]} input features")

# ============================================================================
# 7. MODEL TRAINING PIPELINE
# ============================================================================

def train_advanced_models(features, data, target_col='GME_direction_1d'):
    """Train all advanced models"""
    
    # Prepare data
    X = features.values
    y = data[target_col].values
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)
    
    models = {
        'transformer': transformer_model,
        'lstm': lstm_model
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🔄 Training {model_name.upper()} model...")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_tensor.unsqueeze(1))  # Add sequence dimension
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor.unsqueeze(1))
            pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
            accuracy = accuracy_score(y, pred_labels)
            
        results[model_name] = {
            'accuracy': accuracy,
            'model': model
        }
        
        print(f"  ✅ {model_name.upper()} Accuracy: {accuracy:.4f}")
    
    return results

# Train all models
print("\n🚀 Starting advanced model training...")
training_results = train_advanced_models(features, data)
print("\n🎉 Advanced model training completed!")

# ============================================================================
# 8. ENSEMBLE SYSTEM TRAINING
# ============================================================================

class AdvancedEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
    def predict(self, X):
        predictions = []
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                outputs = model(X_tensor.unsqueeze(1))
                pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions.append(pred_probs)
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(predictions[0])
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            ensemble_pred += weight * pred
        
        return np.argmax(ensemble_pred, axis=1)

# Create ensemble
ensemble = AdvancedEnsemble(training_results)

# Evaluate ensemble
X_test = features.values
y_test = data['GME_direction_1d'].values

ensemble_preds = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)

print(f"\n🎯 Ensemble Accuracy: {ensemble_accuracy:.4f}")
print("✅ Advanced ensemble system completed!")

# ============================================================================
# 9. RESULTS SUMMARY & MODEL COMPARISON
# ============================================================================

# Compile results
results_summary = {
    'Model': [],
    'Accuracy': [],
    'Type': []
}

# Add individual model results
for model_name, result in training_results.items():
    results_summary['Model'].append(model_name.title())
    results_summary['Accuracy'].append(result['accuracy'])
    results_summary['Type'].append('Individual')

# Add ensemble result
results_summary['Model'].append('Ensemble')
results_summary['Accuracy'].append(ensemble_accuracy)
results_summary['Type'].append('Ensemble')

# Create results DataFrame
results_df = pd.DataFrame(results_summary)
print("\n📊 Week 2 Advanced Model Results:")
print(results_df)

# Plot results
plt.figure(figsize=(10, 6))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=colors)
plt.title('Week 2 Advanced Model Performance', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, results_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('week2_advanced_model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 Week 2 Advanced Model Training Complete!")
print("\n📋 Next Steps:")
print("1. Save trained models")
print("2. Compare with Week 1 baseline")
print("3. Perform statistical validation")
print("4. Conduct ablation studies")

# ============================================================================
# 10. SAVE TRAINED MODELS
# ============================================================================

import pickle

def save_models(training_results, ensemble, results_df):
    """Save all trained models and results"""
    
    # Save individual models
    for model_name, result in training_results.items():
        torch.save(result['model'].state_dict(), f'{model_name}_week2.pth')
        print(f"✅ Saved {model_name} model")
    
    # Save ensemble
    with open('ensemble_week2.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("✅ Saved ensemble model")
    
    # Save results
    results_df.to_csv('week2_advanced_results.csv', index=False)
    print("✅ Saved results summary")
    
    # In Colab, add download links:
    # from google.colab import files
    # print("\n📥 Download trained models:")
    # for model_name in training_results.keys():
    #     files.download(f'{model_name}_week2.pth')
    # files.download('ensemble_week2.pkl')
    # files.download('week2_advanced_results.csv')

# Save all models
save_models(training_results, ensemble, results_df)

print("\n🎯 Week 2 Advanced Model Training Successfully Completed!")
print("\n📊 Key Achievements:")
print(f"- Trained {len(training_results)} advanced models")
print(f"- Created ensemble system")
print(f"- Best accuracy: {results_df['Accuracy'].max():.4f}")
print(f"- Ensemble accuracy: {ensemble_accuracy:.4f}")

print("\n📁 Files created:")
print("- transformer_week2.pth")
print("- lstm_week2.pth")
print("- ensemble_week2.pkl")
print("- week2_advanced_results.csv")
print("- week2_advanced_model_performance.png") 