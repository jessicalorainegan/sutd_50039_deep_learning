"""
Utility functions for RUL prediction models
"""

import os
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path


def seed_everything(seed=1234):
    """Set seeds for reproducibility across all libraries"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """Seed worker processes for DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """Get available device (GPU or CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_dataset(csv_path):
    """Load RUL dataset from CSV"""
    return pd.read_csv(csv_path)


def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model, filepath):
    """Save model state dict"""
    torch.save(model.state_dict(), filepath)
    print(f"✅ Model saved to {filepath}")


def load_model(model, filepath):
    """Load model state dict"""
    model.load_state_dict(torch.load(filepath))
    print(f"✅ Model loaded from {filepath}")
    return model


def save_predictions(predictions, targets, filepath):
    """Save predictions and targets to CSV"""
    results_df = pd.DataFrame({
        'predictions': predictions,
        'targets': targets,
        'error': np.abs(predictions - targets)
    })
    results_df.to_csv(filepath, index=False)
    print(f"✅ Results saved to {filepath}")


def print_model_summary(model, model_name):
    """Print model summary"""
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(model)
    print(f"{'='*60}")
    print(f"Total Parameters: {n_params:,}")
    print(f"{'='*60}\n")
