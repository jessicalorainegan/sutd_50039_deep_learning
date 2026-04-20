# SUTD 50.039 Deep Learning Project

This repository contains experiments and model training workflows for NASA turbofan Remaining Useful Life (RUL) prediction.

## 0) Dataset

This project uses the NASA CMAPSS (C-MAPSS) FD001 dataset.  
They can be downloaded from: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/  
or via Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

## 1) Environment Setup (Do This First)

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (cmd)
.\.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### CUDA Note

This project was developed with a CUDA-enabled setup. You must use a PyTorch/CUDA version that is compatible with your own GPU, CUDA driver, and OS.

If your machine does not support CUDA, use CPU-only PyTorch instead.

## 2) RNN Project Navigation (`RNN-based/`)

Use `RNN-based/` as the main area for the GRU/RNN/LSTM experiments.

- `archive/`: old exploratory files. You can ignore this folder.
- `best_model_and_evaluation/`: best GRU workflow.
  - `best-gru.ipynb`: trains the best GRU model using the selected optimal hyperparameters and produces training/loss curves.
  - `load_and_evaluate.ipynb`: loads the saved model from `best-gru.ipynb`, reproduces testing, and shows example predictions.
- `tuning/`: hyperparameter search and multi-dataset model training used to find the final comparison setup, together with loss curves.

## 3) Hybrid Transformer Project Navigation (`Hybrid/`)

Use `Hybrid/` for hybrid Transformer-based experiments (LSTM/GRU/CNN encoders with Transformer blocks).

- `archive/`: older files and experiments.
- `FD001_hybrid_transformers_FE_AWS_SagemakerAI.ipynb`: baseline end-to-end workflow (EDA, sequence creation, hybrid model definitions, training pipeline).
- `FD001_hyperparamter_tunning.ipynb`: FD001 hyperparameter tuning workflow (focus on sequence length and learning rate, with validation/test RMSE tracking).
- `config.py` and `helpers.py` (at root directory): shared configuration and utility helpers for reproducible runs.

## 4) Transformer-Only Experiment (`Transformer/`)

- `best_transformer_hopt_lowvar_FINAL.pth`: saved best checkpoint weights from hyperparameter optimisation run.
- `hopt_lowvar_results.csv`: saved results from hyperparameter optimisation runs.
- `transformer-with-opt.ipynb`: standalone Transformer model workflow with 7 dataset runs, hyperparameter optimization loop and evaluation logging, with plotted graph of best model's RMSE Loss.
- `recreate-transformer-inference.ipynb`: notebook that takes the best hyperparameters found in `transformer-with-opt.ipynb` and `best_transformer_hopt_lowvar_FINAL.pth` to reproduce metrics.
- `transformer_results_detailed.csv`: saved results (all seeds) from various dataset runs.
- `transformer_results_summary.csv`: saved results (mean average) from various dataset runs.

## 5) Architecture Performance After Fine-tuning

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Model Architecture</th>
      <th>Best Mean RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>RNN-based</strong></td>
      <td>RNN</td>
      <td>38.58</td>
    </tr>
    <tr>
      <td>GRU</td>
      <td>13.17</td>
    </tr>
    <tr>
      <td>LSTM</td>
      <td>39.58</td>
    </tr>
    <tr>
      <td><strong>Transformer</strong></td>
      <td><code>RULTransformer</code></td>
      <td>12.41</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Hybrid</strong></td>
      <td>CNN-Trans</td>
      <td>13.15</td>
    </tr>
    <tr>
      <td>GRU-Trans</td>
      <td>12.05</td>
    </tr>
    <tr>
      <td>LSTM-Trans</td>
      <td>12.20</td>
    </tr>
  </tbody>
</table>

## 6) Data Processing Notebooks (`data_preparation/`)

- `data_cleaning_1.ipynb`: generates cleaned RUL-labeled CSVs (linear and piecewise targets, normalized/non-normalized variants) from raw NASA files.
- `feature_engineering_2.ipynb`: feature engineering workflows (including filtering/selection variants) and exports processed datasets used by training notebooks.
- Processed files are stored under `data/processed-nasa-data/` (not in `data_preparation/output/`).

## 7) Suggested Reading Order

1. `data_preparation/data_cleaning_1.ipynb` and `data_preparation/feature_engineering_2.ipynb` (prepare and export datasets into `data/processed-nasa-data/`).
2. `RNN-based/tuning/` notebooks (RNN hyperparameter and dataset exploration).
3. `RNN-based/best_model_and_evaluation/best-gru.ipynb` and `RNN-based/best_model_and_evaluation/load_and_evaluate.ipynb` (final RNN training and evaluation).
4. `Transformer/transformer-with-opt.ipynb` (standalone Transformer optimization and evaluation).
5. `Hybrid/README.md` to have a rough idea of the Hybrid Transformer workflows and how to run the evaluation, then `Hybrid/FD001_hybrid_transformers_FE_AWS_SagemakerAI.ipynb`, then `Hybrid/FD001_hyperparamter_tunning.ipynb` and FE/AWS notebooks (Hybrid Transformer workflows).

## 8) Reproducing Report Figures

| Figure | Notebook | Instructions |
|--------|----------|--------------|
| Fig 5.1.1–5.1.2 (RNN tuning tables) | `RNN-based/tuning/` | Run all cells; RMSE values auto-logged |
| Fig 5.1.3 (RNN loss curves) | `RNN-based/tuning/rnn-fd001.ipynb` | Run all cells |
| Fig 5.1.4 (GRU loss curves) | `RNN-based/tuning/gru-fd001.ipynb` | Run all cells |
| Fig 5.1.5 (LSTM loss curves) | `RNN-based/tuning/lstm-fd001.ipynb` | Run all cells |
| Fig 5.1.6 (best GRU loss curve) | `RNN-based/best_model_and_evaluation/best-gru.ipynb` | Run all cells |
| Fig 5.2.1 (Transformer dataset results) | `Transformer/transformer-with-opt.ipynb` | Results auto-logged to `hopt_lowvar_results.csv` |
| Fig 5.2.2 (Transformer loss curve) | `Transformer/transformer-with-opt.ipynb` | Run last cell |
| Fig 5.3.1 (Hybrid results table) | ` Hybrid/FD001_hybrid_transformers_FE_AWS_SagemakerAI.ipynb` | Run all cells; results printed per seed |
| Fig 5.3.2 (GRU-Transformer loss curve) | `Hybrid/FD001_hyperparamter_tunning.ipynb` | Best config plotted automatically across all seeds |
| Fig 7.1 (confusion matrix) | `Hybrid/FD001_hyperparamter_tunning.ipynb` | Run the classification threshold cell at the end of the notebook |
