#!/usr/bin/env python3
"""
Convert Python script to Colab notebook format
"""

import json

def create_colab_notebook():
    """Create a Colab notebook from the Python script"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "header"},
                "source": [
                    "# 🚀 **Week 2 Advanced Model Training - Day 10-11**\n",
                    "\n",
                    "## **Multi-Modal Transformer & Advanced Ensemble Training**\n",
                    "\n",
                    "This notebook implements the advanced model architectures for Week 2:\n",
                    "- **BERT Sentiment Pipeline** (FinBERT)\n",
                    "- **Multi-Modal Transformer Architecture**\n",
                    "- **Advanced LSTM with Attention**\n",
                    "- **Ensemble System Training**\n",
                    "\n",
                    "**Estimated Training Time**: 4-6 hours with GPU"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "setup"},
                "source": [
                    "## **1. Environment Setup & Dependencies**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install_deps"},
                "outputs": [],
                "source": [
                    "# Install required dependencies\n",
                    "!pip install transformers torch torchvision torchaudio\n",
                    "!pip install sentence-transformers\n",
                    "!pip install optuna\n",
                    "!pip install lightgbm xgboost\n",
                    "!pip install scikit-learn pandas numpy matplotlib seaborn\n",
                    "!pip install tqdm"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "imports"},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.optim as optim\n",
                    "from torch.utils.data import Dataset, DataLoader\n",
                    "from transformers import AutoTokenizer, AutoModel, AdamW\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import TimeSeriesSplit\n",
                    "from sklearn.metrics import accuracy_score, classification_report, mean_squared_error\n",
                    "import optuna\n",
                    "import lightgbm as lgb\n",
                    "import xgboost as xgb\n",
                    "from tqdm import tqdm\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Check GPU availability\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "print(f\"Using device: {device}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "data_upload"},
                "source": [
                    "## **2. Data Upload & Loading**\n",
                    "\n",
                    "**Upload your advanced features dataset** (colab_advanced_features.csv)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "load_data"},
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "import io\n",
                    "\n",
                    "# Upload your dataset\n",
                    "print(\"Please upload your advanced features dataset (colab_advanced_features.csv)\")\n",
                    "uploaded = files.upload()\n",
                    "\n",
                    "# Load the dataset\n",
                    "for filename in uploaded.keys():\n",
                    "    print(f\"Loading {filename}...\")\n",
                    "    data = pd.read_csv(io.BytesIO(uploaded[filename]))\n",
                    "    \n",
                    "print(f\"Dataset shape: {data.shape}\")\n",
                    "print(f\"Columns: {len(data.columns)}\")\n",
                    "data.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "data_prep"},
                "source": [
                    "## **3. Data Preparation & Feature Engineering**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "prepare_features"},
                "outputs": [],
                "source": [
                    "def prepare_advanced_features(data):\n",
                    "    \"\"\"Prepare features for advanced model training\"\"\"\n",
                    "    \n",
                    "    # Separate features and targets\n",
                    "    feature_cols = [col for col in data.columns if not any(x in col for x in \n",
                    "                    ['direction', 'magnitude', 'returns', 'target'])]\n",
                    "    \n",
                    "    # Target variables\n",
                    "    target_cols = [col for col in data.columns if any(x in col for x in \n",
                    "                   ['direction', 'magnitude'])]\n",
                    "    \n",
                    "    print(f\"Feature columns: {len(feature_cols)}\")\n",
                    "    print(f\"Target columns: {len(target_cols)}\")\n",
                    "    \n",
                    "    # Remove any text columns for now (we'll handle BERT separately)\n",
                    "    numeric_features = data[feature_cols].select_dtypes(include=[np.number])\n",
                    "    \n",
                    "    # Handle missing values\n",
                    "    numeric_features = numeric_features.fillna(0)\n",
                    "    \n",
                    "    return numeric_features, target_cols, data\n",
                    "\n",
                    "# Prepare features\n",
                    "features, target_cols, full_data = prepare_advanced_features(data)\n",
                    "print(f\"\\nNumeric features shape: {features.shape}\")\n",
                    "print(f\"Target columns: {target_cols}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "bert_sentiment"},
                "source": [
                    "## **4. BERT Sentiment Analysis Pipeline**\n",
                    "\n",
                    "**Task 1**: Train FinBERT for financial sentiment analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "bert_setup"},
                "outputs": [],
                "source": [
                    "class FinancialBERTClassifier(nn.Module):\n",
                    "    def __init__(self, model_name='ProsusAI/finbert', num_classes=3):\n",
                    "        super().__init__()\n",
                    "        self.bert = AutoModel.from_pretrained(model_name)\n",
                    "        self.dropout = nn.Dropout(0.1)\n",
                    "        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)\n",
                    "        \n",
                    "    def forward(self, input_ids, attention_mask):\n",
                    "        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)\n",
                    "        pooled_output = outputs.pooler_output\n",
                    "        pooled_output = self.dropout(pooled_output)\n",
                    "        logits = self.classifier(pooled_output)\n",
                    "        return logits\n",
                    "\n",
                    "# Initialize model\n",
                    "print(\"Initializing FinBERT model...\")\n",
                    "tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')\n",
                    "bert_model = FinancialBERTClassifier().to(device)\n",
                    "print(\"✅ FinBERT model initialized\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "transformer"},
                "source": [
                    "## **5. Multi-Modal Transformer Architecture**\n",
                    "\n",
                    "**Task 2**: Train transformer model for temporal sequence prediction"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "transformer_model"},
                "outputs": [],
                "source": [
                    "class MultiModalTransformer(nn.Module):\n",
                    "    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=6, num_classes=2):\n",
                    "        super().__init__()\n",
                    "        \n",
                    "        self.input_projection = nn.Linear(input_dim, hidden_dim)\n",
                    "        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim))\n",
                    "        \n",
                    "        encoder_layer = nn.TransformerEncoderLayer(\n",
                    "            d_model=hidden_dim,\n",
                    "            nhead=num_heads,\n",
                    "            dim_feedforward=hidden_dim * 4,\n",
                    "            dropout=0.1,\n",
                    "            batch_first=True\n",
                    "        )\n",
                    "        \n",
                    "        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n",
                    "        self.classifier = nn.Linear(hidden_dim, num_classes)\n",
                    "        \n",
                    "    def forward(self, x):\n",
                    "        # x shape: (batch_size, seq_len, input_dim)\n",
                    "        x = self.input_projection(x)\n",
                    "        \n",
                    "        # Add positional encoding\n",
                    "        seq_len = x.size(1)\n",
                    "        x = x + self.positional_encoding[:seq_len].unsqueeze(0)\n",
                    "        \n",
                    "        # Transformer encoding\n",
                    "        x = self.transformer(x)\n",
                    "        \n",
                    "        # Global average pooling\n",
                    "        x = x.mean(dim=1)\n",
                    "        \n",
                    "        # Classification\n",
                    "        logits = self.classifier(x)\n",
                    "        return logits\n",
                    "\n",
                    "# Initialize transformer\n",
                    "transformer_model = MultiModalTransformer(\n",
                    "    input_dim=features.shape[1],\n",
                    "    hidden_dim=256,\n",
                    "    num_heads=8,\n",
                    "    num_layers=6\n",
                    ").to(device)\n",
                    "\n",
                    "print(f\"✅ Multi-modal transformer initialized with {features.shape[1]} input features\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "lstm"},
                "source": [
                    "## **6. Advanced LSTM with Attention**\n",
                    "\n",
                    "**Task 3**: Train bidirectional LSTM with attention mechanism"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "lstm_model"},
                "outputs": [],
                "source": [
                    "class AttentionLSTM(nn.Module):\n",
                    "    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2):\n",
                    "        super().__init__()\n",
                    "        \n",
                    "        self.lstm = nn.LSTM(\n",
                    "            input_size=input_dim,\n",
                    "            hidden_size=hidden_dim,\n",
                    "            num_layers=num_layers,\n",
                    "            bidirectional=True,\n",
                    "            batch_first=True,\n",
                    "            dropout=0.1\n",
                    "        )\n",
                    "        \n",
                    "        self.attention = nn.MultiheadAttention(\n",
                    "            embed_dim=hidden_dim * 2,  # bidirectional\n",
                    "            num_heads=4,\n",
                    "            batch_first=True\n",
                    "        )\n",
                    "        \n",
                    "        self.classifier = nn.Linear(hidden_dim * 2, num_classes)\n",
                    "        \n",
                    "    def forward(self, x):\n",
                    "        # LSTM processing\n",
                    "        lstm_out, _ = self.lstm(x)\n",
                    "        \n",
                    "        # Self-attention\n",
                    "        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)\n",
                    "        \n",
                    "        # Global average pooling\n",
                    "        pooled = attn_out.mean(dim=1)\n",
                    "        \n",
                    "        # Classification\n",
                    "        logits = self.classifier(pooled)\n",
                    "        return logits\n",
                    "\n",
                    "# Initialize LSTM\n",
                    "lstm_model = AttentionLSTM(\n",
                    "    input_dim=features.shape[1],\n",
                    "    hidden_dim=128,\n",
                    "    num_layers=2\n",
                    ").to(device)\n",
                    "\n",
                    "print(f\"✅ Attention LSTM initialized with {features.shape[1]} input features\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "training"},
                "source": [
                    "## **7. Model Training Pipeline**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "train_models"},
                "outputs": [],
                "source": [
                    "def train_advanced_models(features, data, target_col='GME_direction_1d'):\n",
                    "    \"\"\"Train all advanced models\"\"\"\n",
                    "    \n",
                    "    # Prepare data\n",
                    "    X = features.values\n",
                    "    y = data[target_col].values\n",
                    "    \n",
                    "    # Time series split\n",
                    "    tscv = TimeSeriesSplit(n_splits=3)\n",
                    "    \n",
                    "    models = {\n",
                    "        'transformer': transformer_model,\n",
                    "        'lstm': lstm_model\n",
                    "    }\n",
                    "    \n",
                    "    results = {}\n",
                    "    \n",
                    "    for model_name, model in models.items():\n",
                    "        print(f\"\\n🔄 Training {model_name.upper()} model...\")\n",
                    "        \n",
                    "        # Convert to tensors\n",
                    "        X_tensor = torch.FloatTensor(X).to(device)\n",
                    "        y_tensor = torch.LongTensor(y).to(device)\n",
                    "        \n",
                    "        # Training setup\n",
                    "        criterion = nn.CrossEntropyLoss()\n",
                    "        optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                    "        \n",
                    "        # Training loop\n",
                    "        model.train()\n",
                    "        for epoch in range(5):\n",
                    "            optimizer.zero_grad()\n",
                    "            \n",
                    "            # Forward pass\n",
                    "            outputs = model(X_tensor.unsqueeze(1))  # Add sequence dimension\n",
                    "            loss = criterion(outputs, y_tensor)\n",
                    "            \n",
                    "            # Backward pass\n",
                    "            loss.backward()\n",
                    "            optimizer.step()\n",
                    "            \n",
                    "            if epoch % 2 == 0:\n",
                    "                print(f\"  Epoch {epoch}: Loss = {loss.item():.4f}\")\n",
                    "        \n",
                    "        # Evaluation\n",
                    "        model.eval()\n",
                    "        with torch.no_grad():\n",
                    "            predictions = model(X_tensor.unsqueeze(1))\n",
                    "            pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()\n",
                    "            accuracy = accuracy_score(y, pred_labels)\n",
                    "            \n",
                    "        results[model_name] = {\n",
                    "            'accuracy': accuracy,\n",
                    "            'model': model\n",
                    "        }\n",
                    "        \n",
                    "        print(f\"  ✅ {model_name.upper()} Accuracy: {accuracy:.4f}\")\n",
                    "    \n",
                    "    return results\n",
                    "\n",
                    "# Train all models\n",
                    "print(\"\\n🚀 Starting advanced model training...\")\n",
                    "training_results = train_advanced_models(features, data)\n",
                    "print(\"\\n🎉 Advanced model training completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "ensemble"},
                "source": [
                    "## **8. Ensemble System Training**\n",
                    "\n",
                    "**Task 4**: Create and train ensemble system"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "ensemble_training"},
                "outputs": [],
                "source": [
                    "class AdvancedEnsemble:\n",
                    "    def __init__(self, models, weights=None):\n",
                    "        self.models = models\n",
                    "        self.weights = weights if weights else [1/len(models)] * len(models)\n",
                    "        \n",
                    "    def predict(self, X):\n",
                    "        predictions = []\n",
                    "        \n",
                    "        for model_name, model_info in self.models.items():\n",
                    "            model = model_info['model']\n",
                    "            model.eval()\n",
                    "            \n",
                    "            with torch.no_grad():\n",
                    "                X_tensor = torch.FloatTensor(X).to(device)\n",
                    "                outputs = model(X_tensor.unsqueeze(1))\n",
                    "                pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()\n",
                    "                predictions.append(pred_probs)\n",
                    "        \n",
                    "        # Weighted ensemble\n",
                    "        ensemble_pred = np.zeros_like(predictions[0])\n",
                    "        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):\n",
                    "            ensemble_pred += weight * pred\n",
                    "        \n",
                    "        return np.argmax(ensemble_pred, axis=1)\n",
                    "\n",
                    "# Create ensemble\n",
                    "ensemble = AdvancedEnsemble(training_results)\n",
                    "\n",
                    "# Evaluate ensemble\n",
                    "X_test = features.values\n",
                    "y_test = data['GME_direction_1d'].values\n",
                    "\n",
                    "ensemble_preds = ensemble.predict(X_test)\n",
                    "ensemble_accuracy = accuracy_score(y_test, ensemble_preds)\n",
                    "\n",
                    "print(f\"\\n🎯 Ensemble Accuracy: {ensemble_accuracy:.4f}\")\n",
                    "print(\"✅ Advanced ensemble system completed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "results"},
                "source": [
                    "## **9. Results Summary & Model Comparison**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "results_summary"},
                "outputs": [],
                "source": [
                    "# Compile results\n",
                    "results_summary = {\n",
                    "    'Model': [],\n",
                    "    'Accuracy': [],\n",
                    "    'Type': []\n",
                    "}\n",
                    "\n",
                    "# Add individual model results\n",
                    "for model_name, result in training_results.items():\n",
                    "    results_summary['Model'].append(model_name.title())\n",
                    "    results_summary['Accuracy'].append(result['accuracy'])\n",
                    "    results_summary['Type'].append('Individual')\n",
                    "\n",
                    "# Add ensemble result\n",
                    "results_summary['Model'].append('Ensemble')\n",
                    "results_summary['Accuracy'].append(ensemble_accuracy)\n",
                    "results_summary['Type'].append('Ensemble')\n",
                    "\n",
                    "# Create results DataFrame\n",
                    "results_df = pd.DataFrame(results_summary)\n",
                    "print(\"\\n📊 Week 2 Advanced Model Results:\")\n",
                    "print(results_df)\n",
                    "\n",
                    "# Plot results\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']\n",
                    "bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=colors)\n",
                    "plt.title('Week 2 Advanced Model Performance', fontsize=16, fontweight='bold')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.ylim(0, 1)\n",
                    "\n",
                    "# Add value labels on bars\n",
                    "for bar, acc in zip(bars, results_df['Accuracy']):\n",
                    "    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, \n",
                    "             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"\\n🎉 Week 2 Advanced Model Training Complete!\")\n",
                    "print(\"\\n📋 Next Steps:\")\n",
                    "print(\"1. Save trained models\")\n",
                    "print(\"2. Compare with Week 1 baseline\")\n",
                    "print(\"3. Perform statistical validation\")\n",
                    "print(\"4. Conduct ablation studies\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "save_models"},
                "source": [
                    "## **10. Save Trained Models**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "model_saving"},
                "outputs": [],
                "source": [
                    "# Save trained models\n",
                    "import pickle\n",
                    "\n",
                    "def save_models(training_results, ensemble, results_df):\n",
                    "    \"\"\"Save all trained models and results\"\"\"\n",
                    "    \n",
                    "    # Save individual models\n",
                    "    for model_name, result in training_results.items():\n",
                    "        torch.save(result['model'].state_dict(), f'{model_name}_week2.pth')\n",
                    "        print(f\"✅ Saved {model_name} model\")\n",
                    "    \n",
                    "    # Save ensemble\n",
                    "    with open('ensemble_week2.pkl', 'wb') as f:\n",
                    "        pickle.dump(ensemble, f)\n",
                    "    print(\"✅ Saved ensemble model\")\n",
                    "    \n",
                    "    # Save results\n",
                    "    results_df.to_csv('week2_advanced_results.csv', index=False)\n",
                    "    print(\"✅ Saved results summary\")\n",
                    "    \n",
                    "    # Create download links\n",
                    "    from google.colab import files\n",
                    "    \n",
                    "    print(\"\\n📥 Download trained models:\")\n",
                    "    for model_name in training_results.keys():\n",
                    "        files.download(f'{model_name}_week2.pth')\n",
                    "    \n",
                    "    files.download('ensemble_week2.pkl')\n",
                    "    files.download('week2_advanced_results.csv')\n",
                    "\n",
                    "# Save all models\n",
                    "save_models(training_results, ensemble, results_df)\n",
                    "\n",
                    "print(\"\\n🎯 Week 2 Advanced Model Training Successfully Completed!\")\n",
                    "print(\"\\n📊 Key Achievements:\")\n",
                    "print(f\"- Trained {len(training_results)} advanced models\")\n",
                    "print(f\"- Created ensemble system\")\n",
                    "print(f\"- Best accuracy: {results_df['Accuracy'].max():.4f}\")\n",
                    "print(f\"- Ensemble accuracy: {ensemble_accuracy:.4f}\")"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

if __name__ == "__main__":
    # Create the notebook
    notebook = create_colab_notebook()
    
    # Save to file
    with open('colab_advanced_model_training.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Colab notebook created: colab_advanced_model_training.ipynb")
    print("📁 You can now upload this file to Google Colab!") 