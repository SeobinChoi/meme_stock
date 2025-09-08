#!/usr/bin/env python3
"""
Convert a Python script into a well-structured Colab-compatible notebook.

Usage:
  python scripts/py_to_ipynb_improved.py --src meme_stock_deep_learning_colab_fixed.py --dst ../notebooks/meme_stock_deep_learning_colab_fixed_improved.ipynb
"""

import argparse
import json
import re
from pathlib import Path


def extract_functions_and_classes(py_source: str):
    """Extract function and class definitions to create logical cell breaks"""
    
    lines = py_source.splitlines()
    cells = []
    current_cell = []
    current_section = "setup"
    
    # Define section patterns
    section_patterns = {
        "setup": r"^(import|from|def setup_|def install_)",
        "data_loading": r"^(def load_|def prepare_|def validate_)",
        "models": r"^(class |def train_)",
        "utilities": r"^(def calculate_|def evaluate_)",
        "main": r"^(def main|if __name__)"
    }
    
    for line in lines:
        # Check if this line starts a new section
        new_section = None
        for section, pattern in section_patterns.items():
            if re.match(pattern, line):
                new_section = section
                break
        
        # If new section found, save current cell and start new one
        if new_section and new_section != current_section:
            if current_cell:
                cells.append({
                    "type": "code",
                    "title": f"Section: {current_section.replace('_', ' ').title()}",
                    "content": current_cell
                })
            current_cell = [line]
            current_section = new_section
        else:
            current_cell.append(line)
    
    # Add the last cell
    if current_cell:
        cells.append({
            "type": "code",
            "title": f"Section: {current_section.replace('_', ' ').title()}",
            "content": current_cell
        })
    
    return cells


def create_notebook_structure():
    """Create the basic notebook structure with markdown cells"""
    
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Meme Stock Price Prediction with Deep Learning (A100 GPU Optimized)\n",
                "\n",
                "**Fixed version for Colab A100 GPU with robust LSTM data loading**\n",
                "\n",
                "## Overview\n",
                "This notebook implements state-of-the-art deep learning models to predict meme stock price movements using:\n",
                "- **Technical indicators** (price, volume, volatility)\n",
                "- **Reddit sentiment features** (mentions, surprises, market sentiment)\n",
                "- **Time series patterns** (momentum, regimes, interactions)\n",
                "\n",
                "## Key Fixes for A100 GPU\n",
                "1. **LSTM Data Loading**: Robust handling of string data types\n",
                "2. **GPU Optimization**: Mixed precision training (FP16)\n",
                "3. **Error Handling**: Fallback mechanisms for sequence models\n",
                "4. **Memory Management**: A100 40GB+ memory utilization\n",
                "\n",
                "## Success Criteria\n",
                "- **IC improvement** ≥ 0.03 vs price-only baseline\n",
                "- **Information Ratio (IR)** ≥ 0.3\n",
                "- **Hit Rate** > 55%\n",
                "- **Statistical significance** (p < 0.05)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🛠️ Setup and Package Installation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "install_packages"},
            "outputs": [],
            "source": [
                "# Install required packages for A100 GPU\n",
                "import subprocess\n",
                "import sys\n",
                "\n",
                "def install_packages():\n",
                "    \"\"\"Install required packages for A100 GPU\"\"\"\n",
                "    packages = [\n",
                "        \"pytorch-tabnet\",\n",
                "        \"transformers\", \n",
                "        \"optuna\",\n",
                "        \"plotly\",\n",
                "        \"seaborn\"\n",
                "    ]\n",
                "    \n",
                "    for package in packages:\n",
                "        try:\n",
                "            subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", package])\n",
                "            print(f\"✅ {package} installed successfully\")\n",
                "        except:\n",
                "            print(f\"⚠️ Failed to install {package}\")\n",
                "\n",
                "# Install packages\n",
                "install_packages()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📚 Import Libraries and Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "import_libraries"},
            "outputs": [],
            "source": [
                "# Import libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import plotly.graph_objects as go\n",
                "import plotly.express as px\n",
                "from plotly.subplots import make_subplots\n",
                "\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# ML libraries\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import DataLoader, TensorDataset\n",
                "from sklearn.preprocessing import StandardScaler, RobustScaler\n",
                "from sklearn.metrics import mean_squared_error, classification_report\n",
                "from scipy.stats import spearmanr, pearsonr\n",
                "import optuna\n",
                "from pytorch_tabnet.tab_model import TabNetRegressor\n",
                "\n",
                "# A100 GPU 최적화\n",
                "from torch.cuda.amp import autocast, GradScaler\n",
                "\n",
                "# Set random seeds\n",
                "np.random.seed(42)\n",
                "torch.manual_seed(42)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed(42)\n",
                "    torch.cuda.manual_seed_all(42)\n",
                "\n",
                "print(\"✅ Libraries imported successfully\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 A100 GPU Optimization Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "gpu_setup"},
            "outputs": [],
            "source": [
                "def setup_a100_optimization():\n",
                "    \"\"\"A100 GPU 최적화 설정\"\"\"\n",
                "    \n",
                "    # Set device\n",
                "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "    \n",
                "    if device.type == 'cuda':\n",
                "        print(f\"🚀 Using GPU: {torch.cuda.get_device_name(0)}\")\n",
                "        print(f\"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n",
                "        \n",
                "        # A100 최적화 설정\n",
                "        torch.backends.cudnn.benchmark = True\n",
                "        torch.backends.cudnn.deterministic = False\n",
                "        \n",
                "        # 메모리 효율성 (A100 메모리 40GB+ 활용)\n",
                "        torch.cuda.set_per_process_memory_fraction(0.95)\n",
                "        \n",
                "        # GPU 메모리 정리\n",
                "        torch.cuda.empty_cache()\n",
                "    else:\n",
                "        print(\"💻 Using CPU\")\n",
                "    \n",
                "    return device\n",
                "\n",
                "# Setup GPU\n",
                "device = setup_a100_optimization()\n",
                "\n",
                "# Initialize mixed precision training\n",
                "scaler = GradScaler()\n",
                "print(\"✅ A100 GPU optimization setup completed\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📤 Data Upload (Colab)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "upload_data"},
            "outputs": [],
            "source": [
                "def upload_data_colab():\n",
                "    \"\"\"Colab에서 데이터 업로드\"\"\"\n",
                "    try:\n",
                "        from google.colab import files\n",
                "        \n",
                "        print(\"📤 Upload the following files from your local machine:\")\n",
                "        print(\"   - tabular_train_YYYYMMDD_HHMMSS.csv\")\n",
                "        print(\"   - tabular_val_YYYYMMDD_HHMMSS.csv\") \n",
                "        print(\"   - tabular_test_YYYYMMDD_HHMMSS.csv\")\n",
                "        print(\"   - sequences_YYYYMMDD_HHMMSS.npz\")\n",
                "        print(\"   - dataset_metadata_YYYYMMDD_HHMMSS.json\")\n",
                "        \n",
                "        uploaded = files.upload()\n",
                "        \n",
                "        # Show uploaded files\n",
                "        import os\n",
                "        print(\"\\n📁 Uploaded files:\")\n",
                "        for filename in os.listdir('.'):\n",
                "            if any(filename.startswith(prefix) for prefix in ['tabular_', 'sequences_', 'dataset_']):\n",
                "                print(f\"   {filename}\")\n",
                "                \n",
                "        return True\n",
                "        \n",
                "    except ImportError:\n",
                "        print(\"⚠️ Not running in Colab - skipping file upload\")\n",
                "        return False\n",
                "\n",
                "# Upload data (only in Colab)\n",
                "if 'google.colab' in globals():\n",
                "    upload_data_colab()\n",
                "else:\n",
                "    print(\"⚠️ Not in Colab - please upload data files manually\")"
            ]
        }
    ]


def create_code_cells_from_functions(py_source: str):
    """Create code cells from function definitions"""
    
    # Extract main functions and classes
    functions = [
        ("load_data_robust", "강화된 데이터 로딩"),
        ("prepare_sequence_data_fixed", "시퀀스 데이터 준비 (문자열 오류 해결)"),
        ("validate_sequence_dimensions", "시퀀스 차원 검증"),
        ("create_train_val_test_split", "시퀀스 데이터 분할"),
        ("prepare_tabular_data", "테이블 데이터 준비"),
        ("calculate_ic_metrics", "IC 메트릭 계산"),
        ("evaluate_model", "모델 평가"),
        ("DeepMLP", "A100 최적화된 Deep MLP"),
        ("LSTMModel", "A100 최적화된 LSTM"),
        ("train_mlp_a100", "A100 GPU 최적화된 MLP 훈련"),
        ("train_lstm_a100", "A100 GPU 최적화된 LSTM 훈련"),
        ("main", "메인 실행 함수")
    ]
    
    cells = []
    
    for func_name, description in functions:
        # Find function in source code
        pattern = rf"(def {func_name}|class {func_name}).*?(?=def |class |$)"
        match = re.search(pattern, py_source, re.DOTALL)
        
        if match:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {description}"]
            })
            
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": func_name},
                "outputs": [],
                "source": match.group(0).splitlines()
            })
    
    return cells


def convert_to_notebook(src: Path, dst: Path):
    """Convert Python script to well-structured notebook"""
    
    code = src.read_text()
    
    # Create notebook structure
    notebook = {
        "cells": create_notebook_structure() + create_code_cells_from_functions(code),
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "A100",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.5",
                "mimetype": "text/x-python",
                "file_extension": ".py"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Write notebook
    dst.write_text(json.dumps(notebook, indent=2))
    print(f"✅ Created improved notebook: {dst}")
    print(f"   Total cells: {len(notebook['cells'])}")
    print(f"   Markdown cells: {len([c for c in notebook['cells'] if c['cell_type'] == 'markdown'])}")
    print(f"   Code cells: {len([c for c in notebook['cells'] if c['cell_type'] == 'code'])}")


def main():
    parser = argparse.ArgumentParser(description="Convert Python script to well-structured Colab notebook")
    parser.add_argument('--src', type=str, required=True, help="Source Python file")
    parser.add_argument('--dst', type=str, required=True, help="Destination notebook file")
    args = parser.parse_args()

    convert_to_notebook(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
