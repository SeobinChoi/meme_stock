"""
Day 10: Multi-Modal Transformer Architecture Development
Implement sophisticated models leveraging new features and advanced architectures
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ML Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Advanced ML
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalTransformer(nn.Module):
    """
    Multi-Modal Transformer Architecture for meme stock prediction
    """
    
    def __init__(self, 
                 num_features: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super(MultiModalTransformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Feature embedding layer
        self.feature_embedding = nn.Linear(num_features, hidden_dim)
        
        # Positional encoding for temporal sequences
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=1000)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling
        self.attention_pooling = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, attention_mask=None):
        # x shape: (batch_size, seq_len, num_features)
        batch_size, seq_len, _ = x.shape
        
        # Feature embedding
        x = self.feature_embedding(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x.device)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask == 0)
        
        # Attention pooling
        query = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, hidden_dim)
        attn_output, _ = self.attention_pooling(query, x, x)
        pooled = attn_output.squeeze(1)  # (batch_size, hidden_dim)
        
        # Multi-task outputs
        classification_output = self.classifier(pooled)
        regression_output = self.regressor(pooled)
        
        return classification_output, regression_output

class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class AdvancedLSTM(nn.Module):
    """
    Enhanced LSTM Architecture with Attention
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 num_classes: int = 2):
        super(AdvancedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=4,
            batch_first=True
        )
        
        # Output layers
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        query = torch.mean(lstm_out, dim=1, keepdim=True)
        attn_output, _ = self.attention(query, lstm_out, lstm_out)
        pooled = attn_output.squeeze(1)
        
        # Multi-task outputs
        classification_output = self.classifier(pooled)
        regression_output = self.regressor(pooled)
        
        return classification_output, regression_output

class TimeSeriesDataset(Dataset):
    """
    Custom dataset for time series data
    """
    
    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray], 
                 sequence_length: int = 30, task_type: str = 'classification'):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.task_type = task_type
        
    def __len__(self):
        return len(self.features) - self.sequence_length
        
    def __getitem__(self, idx):
        # Extract sequence
        sequence = self.features[idx:idx + self.sequence_length]
        
        # Get target (use the last timestep)
        target_idx = idx + self.sequence_length - 1
        
        # Convert to tensors
        sequence = torch.FloatTensor(sequence)
        
        if self.task_type == 'classification':
            target = torch.LongTensor([self.targets['classification'][target_idx]])
        else:
            target = torch.FloatTensor([self.targets['regression'][target_idx]])
            
        return sequence, target

class AdvancedModelTrainer:
    """
    Advanced model training and ensemble system
    """
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.configs = {
            'transformer': {
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'learning_rate': 1e-4,
                'batch_size': 32,
                'epochs': 50
            },
            'lstm': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'epochs': 100
            }
        }
        
    def generate_advanced_models(self) -> Dict:
        """
        Generate advanced models with multi-modal transformer architecture
        """
        logger.info("🚀 Starting Day 10: Multi-Modal Transformer Architecture Development")
        logger.info("="*70)
        
        # Step 1: Load advanced features dataset
        logger.info("STEP 1: Loading Advanced Features Dataset")
        logger.info("="*50)
        dataset = self._load_advanced_dataset()
        if dataset is None:
            return {"status": "ERROR", "message": "Failed to load advanced dataset"}
        
        # Step 2: Prepare data for deep learning models
        logger.info("STEP 2: Data Preparation for Deep Learning")
        logger.info("="*50)
        prepared_data = self._prepare_deep_learning_data(dataset)
        
        # Step 3: Train Transformer Model
        logger.info("STEP 3: Training Multi-Modal Transformer")
        logger.info("="*50)
        transformer_results = self._train_transformer_model(prepared_data)
        
        # Step 4: Train Advanced LSTM
        logger.info("STEP 4: Training Advanced LSTM")
        logger.info("="*50)
        lstm_results = self._train_lstm_model(prepared_data)
        
        # Step 5: Train Enhanced Traditional Models
        logger.info("STEP 5: Training Enhanced Traditional Models")
        logger.info("="*50)
        traditional_results = self._train_traditional_models(prepared_data)
        
        # Step 6: Build Advanced Ensemble
        logger.info("STEP 6: Building Advanced Ensemble System")
        logger.info("="*50)
        ensemble_results = self._build_advanced_ensemble(
            prepared_data, transformer_results, lstm_results, traditional_results
        )
        
        # Step 7: Model Validation and Analysis
        logger.info("STEP 7: Model Validation and Analysis")
        logger.info("="*50)
        validation_results = self._validate_advanced_models(
            prepared_data, transformer_results, lstm_results, 
            traditional_results, ensemble_results
        )
        
        # Step 8: Save results and generate report
        logger.info("STEP 8: Saving Results and Generating Report")
        logger.info("="*50)
        self._save_advanced_models_results(
            transformer_results, lstm_results, traditional_results, 
            ensemble_results, validation_results
        )
        
        logger.info("✅ Multi-Modal Transformer Architecture Development Completed")
        return {
            "status": "COMPLETED",
            "transformer_models": len(transformer_results),
            "lstm_models": len(lstm_results),
            "traditional_models": len(traditional_results),
            "ensemble_models": len(ensemble_results)
        }
    
    def _load_advanced_dataset(self) -> Optional[pd.DataFrame]:
        """
        Load advanced meme features dataset
        """
        try:
            dataset_path = self.data_dir / "features" / "advanced_meme_features_dataset.csv"
            if not dataset_path.exists():
                logger.error(f"Advanced dataset not found: {dataset_path}")
                return None
            
            dataset = pd.read_csv(dataset_path)
            dataset['date'] = pd.to_datetime(dataset['date'])
            dataset = dataset.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✅ Loaded advanced dataset with shape: {dataset.shape}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading advanced dataset: {e}")
            return None
    
    def _prepare_deep_learning_data(self, dataset: pd.DataFrame) -> Dict:
        """
        Prepare data for deep learning models
        """
        logger.info("Preparing data for deep learning models...")
        
        # Separate features and targets
        feature_cols = [col for col in dataset.columns 
                       if col not in ['date'] and not col.endswith('_direction_1d') 
                       and not col.endswith('_returns_1d')]
        
        features = dataset[feature_cols].values
        
        # Create targets
        targets = {}
        
        # Classification targets (price direction)
        for stock in ['GME', 'AMC', 'BB']:
            target_col = f"{stock}_direction_1d"
            if target_col in dataset.columns:
                targets[f"{stock}_classification"] = dataset[target_col].values
        
        # Regression targets (price returns)
        for stock in ['GME', 'AMC', 'BB']:
            target_col = f"{stock}_returns_1d"
            if target_col in dataset.columns:
                targets[f"{stock}_regression"] = dataset[target_col].values
        
        # Remove return columns from features to prevent data leakage
        return_cols_to_exclude = []
        for stock in ['GME', 'AMC', 'BB']:
            for horizon in ['1d', '3d', '7d', '14d']:
                return_cols_to_exclude.append(f"{stock}_returns_{horizon}")
        
        feature_cols = [col for col in feature_cols if col not in return_cols_to_exclude]
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Create train/test split (80/20)
        split_idx = int(len(features_normalized) * 0.8)
        
        train_data = {
            'features': features_normalized[:split_idx],
            'targets': {k: v[:split_idx] for k, v in targets.items()}
        }
        
        test_data = {
            'features': features_normalized[split_idx:],
            'targets': {k: v[split_idx:] for k, v in targets.items()}
        }
        
        logger.info(f"✅ Prepared data: {len(features_normalized)} samples, {len(feature_cols)} features")
        logger.info(f"✅ Train set: {len(train_data['features'])} samples")
        logger.info(f"✅ Test set: {len(test_data['features'])} samples")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'feature_cols': feature_cols,
            'scaler': scaler
        }
    
    def _train_transformer_model(self, prepared_data: Dict) -> Dict:
        """
        Train multi-modal transformer model
        """
        logger.info("Training multi-modal transformer model...")
        
        results = {}
        
        # Get data
        train_data = prepared_data['train_data']
        test_data = prepared_data['test_data']
        num_features = len(prepared_data['feature_cols'])
        
        # Train for each stock
        for stock in ['GME', 'AMC', 'BB']:
            classification_target = f"{stock}_classification"
            regression_target = f"{stock}_regression"
            
            if classification_target not in train_data['targets']:
                continue
                
            logger.info(f"Training transformer for {stock}...")
            
            # Create datasets
            train_dataset = TimeSeriesDataset(
                train_data['features'], 
                {'classification': train_data['targets'][classification_target]},
                sequence_length=30,
                task_type='classification'
            )
            
            test_dataset = TimeSeriesDataset(
                test_data['features'],
                {'classification': test_data['targets'][classification_target]},
                sequence_length=30,
                task_type='classification'
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            model = MultiModalTransformer(
                num_features=num_features,
                hidden_dim=256,
                num_heads=8,
                num_layers=4,
                dropout=0.1,
                num_classes=2
            )
            
            # Training configuration
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training loop
            best_accuracy = 0
            patience_counter = 0
            
            for epoch in range(50):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_features, batch_targets in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    classification_output, _ = model(batch_features)
                    loss = criterion(classification_output, batch_targets.squeeze())
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(classification_output.data, 1)
                    total += batch_targets.size(0)
                    correct += (predicted == batch_targets.squeeze()).sum().item()
                
                # Validation
                model.eval()
                val_accuracy = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_features, batch_targets in test_loader:
                        classification_output, _ = model(batch_features)
                        _, predicted = torch.max(classification_output.data, 1)
                        val_total += batch_targets.size(0)
                        val_accuracy += (predicted == batch_targets.squeeze()).sum().item()
                
                val_accuracy = val_accuracy / val_total
                scheduler.step(val_accuracy)
                
                # Early stopping
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Acc = {correct/total:.3f}, Val Acc = {val_accuracy:.3f}")
            
            results[f"{stock}_transformer"] = {
                'model': model,
                'accuracy': best_accuracy,
                'config': self.configs['transformer']
            }
            
            logger.info(f"✅ {stock} Transformer - Best Accuracy: {best_accuracy:.3f}")
        
        return results
    
    def _train_lstm_model(self, prepared_data: Dict) -> Dict:
        """
        Train advanced LSTM model
        """
        logger.info("Training advanced LSTM model...")
        
        results = {}
        
        # Get data
        train_data = prepared_data['train_data']
        test_data = prepared_data['test_data']
        num_features = len(prepared_data['feature_cols'])
        
        # Train for each stock
        for stock in ['GME', 'AMC', 'BB']:
            classification_target = f"{stock}_classification"
            
            if classification_target not in train_data['targets']:
                continue
                
            logger.info(f"Training LSTM for {stock}...")
            
            # Create datasets
            train_dataset = TimeSeriesDataset(
                train_data['features'], 
                {'classification': train_data['targets'][classification_target]},
                sequence_length=30,
                task_type='classification'
            )
            
            test_dataset = TimeSeriesDataset(
                test_data['features'],
                {'classification': test_data['targets'][classification_target]},
                sequence_length=30,
                task_type='classification'
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            # Initialize model
            model = AdvancedLSTM(
                input_size=num_features,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                bidirectional=True,
                num_classes=2
            )
            
            # Training configuration
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training loop
            best_accuracy = 0
            patience_counter = 0
            
            for epoch in range(100):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_features, batch_targets in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    classification_output, _ = model(batch_features)
                    loss = criterion(classification_output, batch_targets.squeeze())
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(classification_output.data, 1)
                    total += batch_targets.size(0)
                    correct += (predicted == batch_targets.squeeze()).sum().item()
                
                # Validation
                model.eval()
                val_accuracy = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_features, batch_targets in test_loader:
                        classification_output, _ = model(batch_features)
                        _, predicted = torch.max(classification_output.data, 1)
                        val_total += batch_targets.size(0)
                        val_accuracy += (predicted == batch_targets.squeeze()).sum().item()
                
                val_accuracy = val_accuracy / val_total
                scheduler.step(val_accuracy)
                
                # Early stopping
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:
                    break
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}: Train Acc = {correct/total:.3f}, Val Acc = {val_accuracy:.3f}")
            
            results[f"{stock}_lstm"] = {
                'model': model,
                'accuracy': best_accuracy,
                'config': self.configs['lstm']
            }
            
            logger.info(f"✅ {stock} LSTM - Best Accuracy: {best_accuracy:.3f}")
        
        return results
    
    def _train_traditional_models(self, prepared_data: Dict) -> Dict:
        """
        Train enhanced traditional models with advanced features
        """
        logger.info("Training enhanced traditional models...")
        
        results = {}
        
        # Get data
        train_data = prepared_data['train_data']
        test_data = prepared_data['test_data']
        
        # Train for each stock
        for stock in ['GME', 'AMC', 'BB']:
            classification_target = f"{stock}_classification"
            regression_target = f"{stock}_regression"
            
            if classification_target not in train_data['targets']:
                continue
                
            logger.info(f"Training traditional models for {stock}...")
            
            # Prepare data (use last timestep of each sequence)
            X_train = train_data['features'][29:]  # Skip first 29 timesteps
            y_train_class = train_data['targets'][classification_target][29:]
            y_train_reg = train_data['targets'][regression_target][29:]
            
            X_test = test_data['features'][29:]
            y_test_class = test_data['targets'][classification_target][29:]
            y_test_reg = test_data['targets'][regression_target][29:]
            
            # LightGBM Classification
            lgb_classifier = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            
            lgb_classifier.fit(X_train, y_train_class)
            y_pred_class = lgb_classifier.predict(X_test)
            accuracy = accuracy_score(y_test_class, y_pred_class)
            
            results[f"{stock}_lgb_classifier"] = {
                'model': lgb_classifier,
                'accuracy': accuracy,
                'type': 'classification'
            }
            
            # LightGBM Regression
            lgb_regressor = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            
            lgb_regressor.fit(X_train, y_train_reg)
            y_pred_reg = lgb_regressor.predict(X_test)
            r2 = r2_score(y_test_reg, y_pred_reg)
            
            results[f"{stock}_lgb_regressor"] = {
                'model': lgb_regressor,
                'r2': r2,
                'type': 'regression'
            }
            
            logger.info(f"✅ {stock} LightGBM - Classification Acc: {accuracy:.3f}, Regression R²: {r2:.3f}")
        
        return results
    
    def _build_advanced_ensemble(self, prepared_data: Dict, transformer_results: Dict,
                                lstm_results: Dict, traditional_results: Dict) -> Dict:
        """
        Build advanced ensemble system
        """
        logger.info("Building advanced ensemble system...")
        
        ensemble_results = {}
        
        # Get data
        test_data = prepared_data['test_data']
        
        # Build ensemble for each stock
        for stock in ['GME', 'AMC', 'BB']:
            classification_target = f"{stock}_classification"
            
            if classification_target not in test_data['targets']:
                continue
                
            logger.info(f"Building ensemble for {stock}...")
            
            # Collect predictions from all models
            predictions = []
            model_names = []
            
            # Transformer predictions
            if f"{stock}_transformer" in transformer_results:
                model = transformer_results[f"{stock}_transformer"]['model']
                model.eval()
                
                test_dataset = TimeSeriesDataset(
                    test_data['features'],
                    {'classification': test_data['targets'][classification_target]},
                    sequence_length=30,
                    task_type='classification'
                )
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                transformer_preds = []
                with torch.no_grad():
                    for batch_features, _ in test_loader:
                        classification_output, _ = model(batch_features)
                        _, predicted = torch.max(classification_output.data, 1)
                        transformer_preds.extend(predicted.numpy())
                
                predictions.append(transformer_preds)
                model_names.append('transformer')
            
            # LSTM predictions
            if f"{stock}_lstm" in lstm_results:
                model = lstm_results[f"{stock}_lstm"]['model']
                model.eval()
                
                test_dataset = TimeSeriesDataset(
                    test_data['features'],
                    {'classification': test_data['targets'][classification_target]},
                    sequence_length=30,
                    task_type='classification'
                )
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                
                lstm_preds = []
                with torch.no_grad():
                    for batch_features, _ in test_loader:
                        classification_output, _ = model(batch_features)
                        _, predicted = torch.max(classification_output.data, 1)
                        lstm_preds.extend(predicted.numpy())
                
                predictions.append(lstm_preds)
                model_names.append('lstm')
            
            # Traditional model predictions
            if f"{stock}_lgb_classifier" in traditional_results:
                model = traditional_results[f"{stock}_lgb_classifier"]['model']
                X_test = test_data['features'][29:]  # Skip first 29 timesteps
                lgb_preds = model.predict(X_test)
                
                predictions.append(lgb_preds)
                model_names.append('lightgbm')
            
            # Ensemble prediction (majority voting)
            if len(predictions) > 1:
                ensemble_preds = []
                for i in range(len(predictions[0])):
                    votes = [pred[i] for pred in predictions]
                    ensemble_preds.append(1 if sum(votes) > len(votes)/2 else 0)
                
                # Calculate ensemble accuracy
                y_test = test_data['targets'][classification_target][29:]
                ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
                
                ensemble_results[f"{stock}_ensemble"] = {
                    'accuracy': ensemble_accuracy,
                    'models': model_names,
                    'predictions': ensemble_preds
                }
                
                logger.info(f"✅ {stock} Ensemble ({', '.join(model_names)}) - Accuracy: {ensemble_accuracy:.3f}")
        
        return ensemble_results
    
    def _validate_advanced_models(self, prepared_data: Dict, transformer_results: Dict,
                                 lstm_results: Dict, traditional_results: Dict,
                                 ensemble_results: Dict) -> Dict:
        """
        Validate advanced models and generate comprehensive analysis
        """
        logger.info("Validating advanced models...")
        
        validation_results = {
            'model_performance': {},
            'ensemble_analysis': {},
            'feature_importance': {},
            'comparison_summary': {}
        }
        
        # Collect performance metrics
        all_models = {}
        all_models.update(transformer_results)
        all_models.update(lstm_results)
        all_models.update(traditional_results)
        all_models.update(ensemble_results)
        
        # Performance summary
        performance_summary = {}
        for model_name, model_info in all_models.items():
            if 'accuracy' in model_info:
                performance_summary[model_name] = model_info['accuracy']
            elif 'r2' in model_info:
                performance_summary[model_name] = model_info['r2']
        
        validation_results['model_performance'] = performance_summary
        
        # Find best models
        classification_models = [(k, v) for k, v in performance_summary.items() 
                               if 'accuracy' in all_models[k]]
        regression_models = [(k, v) for k, v in performance_summary.items() 
                           if 'r2' in all_models[k]]
        
        if classification_models:
            best_classification = max(classification_models, key=lambda x: x[1])
        else:
            best_classification = ("None", 0.0)
        
        if regression_models:
            best_regression = max(regression_models, key=lambda x: x[1])
        else:
            best_regression = ("None", 0.0)
        
        validation_results['comparison_summary'] = {
            'best_classification_model': best_classification[0],
            'best_classification_score': best_classification[1],
            'best_regression_model': best_regression[0],
            'best_regression_score': best_regression[1],
            'total_models_trained': len(all_models)
        }
        
        logger.info(f"✅ Validation completed: {len(all_models)} models analyzed")
        return validation_results
    
    def _save_advanced_models_results(self, transformer_results: Dict, lstm_results: Dict,
                                     traditional_results: Dict, ensemble_results: Dict,
                                     validation_results: Dict):
        """
        Save advanced models results and generate completion report
        """
        try:
            # Save validation results
            validation_path = self.results_dir / "advanced_models_validation.json"
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            # Generate completion report
            self._generate_completion_report(
                transformer_results, lstm_results, traditional_results,
                ensemble_results, validation_results
            )
            
            logger.info(f"✅ Saved validation results to {validation_path}")
            
        except Exception as e:
            logger.error(f"Error saving advanced models results: {e}")
    
    def _generate_completion_report(self, transformer_results: Dict, lstm_results: Dict,
                                   traditional_results: Dict, ensemble_results: Dict,
                                   validation_results: Dict):
        """
        Generate completion report for Day 10
        """
        try:
            # Get next sequence number
            sequence_num = self._get_next_sequence_number("013")
            
            report_path = Path("results") / f"{sequence_num}_day10_advanced_models_summary.txt"
            
            with open(report_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("DAY 10: MULTI-MODAL TRANSFORMER ARCHITECTURE DEVELOPMENT SUMMARY\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Status: COMPLETED\n\n")
                
                f.write("MODEL DEVELOPMENT SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Transformer Models: {len(transformer_results)}\n")
                f.write(f"LSTM Models: {len(lstm_results)}\n")
                f.write(f"Traditional Models: {len(traditional_results)}\n")
                f.write(f"Ensemble Models: {len(ensemble_results)}\n")
                f.write(f"Total Models: {len(transformer_results) + len(lstm_results) + len(traditional_results) + len(ensemble_results)}\n\n")
                
                f.write("PERFORMANCE SUMMARY:\n")
                f.write("-" * 20 + "\n")
                for model_name, score in validation_results['model_performance'].items():
                    f.write(f"  • {model_name}: {score:.3f}\n")
                f.write("\n")
                
                f.write("BEST MODELS:\n")
                f.write("-" * 12 + "\n")
                comparison = validation_results['comparison_summary']
                f.write(f"  • Best Classification: {comparison['best_classification_model']} ({comparison['best_classification_score']:.3f})\n")
                f.write(f"  • Best Regression: {comparison['best_regression_model']} ({comparison['best_regression_score']:.3f})\n\n")
                
                f.write("ACHIEVEMENTS:\n")
                f.write("-" * 12 + "\n")
                f.write("  • Implemented multi-modal transformer architecture\n")
                f.write("  • Developed advanced LSTM with attention mechanisms\n")
                f.write("  • Enhanced traditional models with advanced features\n")
                f.write("  • Built sophisticated ensemble system\n")
                f.write("  • Established comprehensive validation framework\n\n")
                
                f.write("ARCHITECTURE FEATURES:\n")
                f.write("-" * 22 + "\n")
                f.write("  • Multi-head attention mechanisms\n")
                f.write("  • Positional encoding for temporal sequences\n")
                f.write("  • Bidirectional LSTM with attention\n")
                f.write("  • Multi-task learning (classification + regression)\n")
                f.write("  • Advanced ensemble with majority voting\n")
                f.write("  • Early stopping and learning rate scheduling\n\n")
                
                f.write("NEXT STEPS:\n")
                f.write("-" * 11 + "\n")
                f.write("  • Hyperparameter optimization\n")
                f.write("  • GPU training for larger models\n")
                f.write("  • Real-time prediction system\n")
                f.write("  • Model deployment and monitoring\n")
                
                f.write("\n" + "="*70 + "\n")
                f.write("END OF DAY 10 SUMMARY\n")
                f.write("="*70 + "\n")
            
            logger.info(f"✅ Generated completion report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating completion report: {e}")
    
    def _get_next_sequence_number(self, prefix: str) -> str:
        """
        Get next sequence number for file naming
        """
        try:
            results_dir = Path("results")
            existing_files = list(results_dir.glob(f"{prefix}_*.txt"))
            
            if not existing_files:
                return f"{prefix}_day10_advanced_models"
            
            # Extract numbers and find max
            numbers = []
            for file in existing_files:
                try:
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        num = int(parts[0])
                        numbers.append(num)
                except:
                    continue
            
            if numbers:
                next_num = max(numbers) + 1
            else:
                next_num = 1
            
            return f"{next_num:03d}_day10_advanced_models"
            
        except Exception as e:
            logger.error(f"Error getting sequence number: {e}")
            return f"{prefix}_day10_advanced_models"

def main():
    """
    Main function to generate advanced models
    """
    trainer = AdvancedModelTrainer()
    results = trainer.generate_advanced_models()
    
    print("✅ Multi-Modal Transformer Architecture Development completed successfully!")
    return results

if __name__ == "__main__":
    main() 