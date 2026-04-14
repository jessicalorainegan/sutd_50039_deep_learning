"""
Configuration file for RUL Prediction Models
"""

# ============ DATA CONFIGURATION ============
DATA_DIR = '../data/processed-nasa-data'
DATASETS = {
    'FD001': {
        'train': f'{DATA_DIR}/train_processed_rul_only_fd001.csv',
        'test': f'{DATA_DIR}/test_processed_rul_only_fd001.csv'
    },
    'FD002': {
        'train': f'{DATA_DIR}/train_processed_rul_only_fd002.csv',
        'test': f'{DATA_DIR}/test_processed_rul_only_fd002.csv'
    }
}

SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
TRAIN_VAL_SPLIT = 0.8

# ============ MODEL CONFIGURATION ============
SEED = 1234

# LSTM-Transformer Config
LSTM_TRANSFORMER_CONFIG = {
    'lstm_hidden': 64,
    'num_lstm_layers': 2,
    'd_model': 64,
    'nhead': 4,
    'num_transformer_layers': 2,
    'dropout': 0.1
}

# GRU-Transformer Config
GRU_TRANSFORMER_CONFIG = {
    'gru_hidden': 64,
    'num_gru_layers': 2,
    'd_model': 64,
    'nhead': 4,
    'num_transformer_layers': 2,
    'dropout': 0.1
}

# CNN-Transformer Config
CNN_TRANSFORMER_CONFIG = {
    'num_filters': 32,
    'kernel_size': 3,
    'd_model': 64,
    'nhead': 4,
    'num_transformer_layers': 2,
    'dropout': 0.1
}

# ============ TRAINING CONFIGURATION ============
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# ============ PATHS ============
MODEL_SAVE_DIR = './trained_models'
RESULTS_DIR = './results'
