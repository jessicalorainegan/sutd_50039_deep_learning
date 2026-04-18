# Owen's RUL Prediction Implementation
## Hybrid Transformer Models for Remaining Useful Life Prediction

### Overview
This folder contains implementations of three hybrid transformer-based models for aircraft engine RUL prediction:
- **LSTM-Transformer**: Bidirectional LSTM + Transformer Encoder
- **GRU-Transformer**: Bidirectional GRU + Transformer Encoder
- **CNN-Transformer**: 1D CNN + Transformer Encoder

### Files

#### Main Notebook
- **`model_exploration.ipynb`** - Complete implementation notebook with:
  - Exploratory Data Analysis (EDA)
  - Data preparation & sequence creation
  - Three model architectures
  - Training framework
  - Examples and next steps

#### Supporting Utilities
- **`config.py`** - Configuration file for all model hyperparameters and paths
- **`utils.py`** - Helper functions (seeding, device management, model I/O)

### Quick Start

#### 1. Run EDA
```python
# All in model_exploration.ipynb Part 1
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
- Sequence length: 30 cycles
- Features: 24 (21 sensors + 3 operational settings)
- Total sequences: ~27K (train+val+test)

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 30 |
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

### Training Tips

1. **Start Small**: Begin with CNN-Transformer (fastest training)
2. **Monitor MAE**: Target validation MAE < 10 cycles
3. **Early Stopping**: Prevents overfitting around epoch 20-30
4. **GPU Acceleration**: Automatically uses CUDA if available
5. **Save Best Models**: For ensemble integration with Jessica & Lucas

### Performance Targets

Based on PHM08 competition (FD001 dataset):
- Best competition score: 436.84
- Our target MAE: < 8 cycles
- Ensemble expectation: 5% improvement over best single model

### Next Steps

1. ✅ Individual model training (uncomment examples in Part 6)
2. ⏳ Cross-dataset validation (FD002, FD003, FD004)
3. ⏳ Hyperparameter tuning (grid search)
4. ⏳ Ensemble integration with Jessica & Lucas's models
5. ⏳ Final comparison and best model selection

### Files to Create

After training, organize saved models:
```
trained_models/
├── lstm_transformer_fd001.pt
├── gru_transformer_fd001.pt
├── cnn_transformer_fd001.pt
└── lstm_transformer_fd002.pt  (if trained on FD002)

results/
├── lstm_transformer_metrics.json
├── gru_transformer_metrics.json
├── cnn_transformer_metrics.json
└── ensemble_results.csv
```

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

**Q: Which model should I train first?**
A: CNN-Transformer is fastest; LSTM-Transformer usually best

### References

- **Architecture**: Based on hybrid attention mechanisms for time series
- **Data**: NASA Turbofan Engine Dataset (CMapss)
- **Seed**: 1234 (consistent with Jessica's transformer.ipynb)
- **Framework**: PyTorch 2.0+

### Contact
Implemented by: Owen
Team: Deep Learning Project SUTD 50.039
