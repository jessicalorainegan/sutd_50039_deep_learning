# RUL Prediction Implementation
## Hybrid Transformer Models for Remaining Useful Life Prediction

### Overview
This folder contains implementations of three hybrid transformer-based models for aircraft engine RUL prediction:
- **LSTM-Transformer**: Bidirectional LSTM + Transformer Encoder
- **GRU-Transformer**: Bidirectional GRU + Transformer Encoder
- **CNN-Transformer**: 1D CNN + Transformer Encoder

### Files

#### Main Notebooks
- **`FD001_hybrid_transformers_FE_AWS_SagemakerAI.ipynb`** 
Complete implementation notebook with:
  - Exploratory Data Analysis (EDA)
  - Data preparation & sequence creation
  - Three model architectures
  - Training framework
  - Examples and next steps

- **`FD001_hyperparameter_tuning.ipynb`** 
After training all models with fixed hyperparameters and saved the models, we are able to identify the best dataset and model combination. We will then select the best model and dataset combination to perform hyperparameter tuning and evaluate the model using classification metrics (accuracy, precision, recall, F1-score, confusion matrix) in the `FD001_hyperparameter_tuning.ipynb` notebook.

#### Supporting Utilities
- **`config.py`** - Configuration file for all model hyperparameters and paths
- **`helpers.py`** - Helper functions (seeding, device management, model I/O)

### Quick Start

#### 1. Run EDA
```python
# All in FD001_hybrid_transformers_FE_AWS_SagemakerAI.ipynb Part 1
# Shows dataset statistics, visualizations, and data quality
```

#### 2. Create Datasets
```python
# Part 2 handles sequence creation and DataLoaders
train_dataset = RULDataset(train_df, SEQUENCE_LENGTH, feature_cols)
test_dataset = RULDataset(test_df, SEQUENCE_LENGTH, feature_cols)
```

#### 3. Define Model
```python
from model_exploration import LSTMTransformer

model = LSTMTransformer(
    num_features=24,
    lstm_hidden=64,
    num_lstm_layers=2,
    d_model=64,
    nhead=4,
    num_transformer_layers=2,
    dropout=0.1
)
```

#### 4. Train Model
```python
trainer = ModelTrainer(model, DEVICE, 'LSTM-Transformer')
history = trainer.train(train_loader, val_loader, epochs=50, lr=0.001)
trainer.plot_history()
```

### Evaluation Instructions

#### A) FD001_hybrid_transformers_FE_AWS_SagemakerAI.ipynb (Evaluate saved models)
1. Open the notebook: `Hybrid/FD001_hybrid_transformers_FE_AWS_SagemakerAI.ipynb`.
2. Ensure your working directory is the repo root:
    - Expected: `.../sutd_50039_deep_learning`
    - If needed, change via the first directory alignment cell.
3. Run the notebook from top to bottom (accept the training sections; evaluation loads saved models).
4. What you will see:
    - Multi-seed evaluation logs with RMSE for train/val/test.
    - A summary table of results per model and dataset.
    - Optional plots comparing model performance.

#### B) FD001_hyperparameter_tuning.ipynb (Classification-based evaluation)
1. Open the notebook: `Hybrid/FD001_hyperparameter_tuning.ipynb`.
2. Ensure your working directory is the repo root:
    - Expected: `.../sutd_50039_deep_learning`
3. Run the last section only (the standalone inference + metrics cells).
4. What you will see:
    - Health-state predictions (Healthy vs Not Healthy).
    - Accuracy, precision, recall, and F1 score.
    - A confusion matrix plot.

### Model Architectures

#### LSTM-Transformer
```
Input (batch, 30, 24)
    ↓
Bidirectional LSTM (2 layers, 64 hidden)
    ↓ (batch, 30, 128)
Linear Projection (128 → 64)
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓ (batch, 30, 64)
Global Average Pooling
    ↓ (batch, 64)
FC Head (64 → 32 → 1)
    ↓
RUL Prediction (batch,)
```

**Parameters:** ~255K

#### GRU-Transformer
```
Similar to LSTM-Transformer but with GRU instead of LSTM
Parameters:** ~219K
```

#### CNN-Transformer
```
Input (batch, 30, 24)
    ↓
1D Conv (24 → 32 filters) + BatchNorm + ReLU
    ↓
1D Conv (32 → 64 filters) + BatchNorm + ReLU
    ↓
Linear Projection (64 → 64)
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓
Global Average Pooling
    ↓
FC Head (64 → 32 → 1)
    ↓
RUL Prediction (batch,)
```

**Parameters:** ~115K

### Dataset Info

**FD001 (Current Default)**
- Training: 100 engines, ~192 cycles each, 21 sensors
- Test: 100 engines, ~31 cycles each, 21 sensors
- Sequence length: 50 cycles
- Features: 24 (21 sensors + 3 operational settings)
- Total sequences: ~27K (train+val+test)

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 50 |
| Batch Size | 32 |
| d_model | 64 |
| Attention Heads | 4 |
| Transformer Layers | 2 |
| Dropout | 0.1 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSE |
| Epochs | 50 |
| Early Stopping Patience | 10 |
| Random Seed | 1234 |

### Hyperparameter Tuning Space

This is the search space used in `FD001_hyperparameter_tuning.ipynb`:

| Parameter | Values |
|-----------|--------|
| Sequence Length | 30, 50, 100 |
| Learning Rate | 1e-4, 5e-5, 1e-3 |
| d_model | 32, 64, 128 |
| Epochs (per trial) | 5 |
| Seeds | 1234, 42, 999 |


### Key Features

✅ **Reproducibility**: Fixed seed (1234) for consistent results
✅ **Flexible**: Easy to modify hyperparameters in config.py
✅ **Modular**: Well-organized, reusable components
✅ **EDA Included**: Understand data before modeling
✅ **Training Utilities**: Early stopping, metrics tracking
✅ **Documentation**: Clear architecture diagrams
✅ **GPU Support**: Automatic device detection

### Troubleshooting

**Q: How do I run the training?**
A: Uncomment the examples in Part 6 of the notebook

**Q: Can I use different sequence length?**
A: Yes, change `SEQUENCE_LENGTH` in the data preparation cell

**Q: How do I save trained models?**
A: Use `torch.save(model.state_dict(), 'model_name.pt')`


### References

- **Architecture**: Based on hybrid attention mechanisms for time series
- **Data**: NASA Turbofan Engine Dataset (CMapss)
- **Seed**: 1234 (consistent with Jessica's transformer.ipynb)
- **Framework**: PyTorch 2.0+

### Contact
Implemented by: Palinya Sengdalavong (Owen)
Team: Deep Learning Project SUTD 50.039
