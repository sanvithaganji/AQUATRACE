"""
Training Script for Groundwater Fingerprint ML Model
Executes dataset generation, LSTM fingerprint autoencoder training,
Isolation Forest anomaly detection, and benchmark metrics generation.
"""

import sys
import os
import random
import numpy as np
import torch
from src.ml.pipeline import GroundwaterPipeline

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    pipeline = GroundwaterPipeline(model_dir="models")
    metrics = pipeline.train(
        n_normal=700,
        n_excessive=250,
        n_unregistered=250,
        epochs=35
    )
    print("\nTraining summary saved in models/metrics.json")
    print(f"Overall ROC-AUC Score: {metrics['roc_auc_score']}")

if __name__ == "__main__":
    main()
