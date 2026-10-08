"""
LSTM Time-Series Autoencoder for Groundwater Fingerprint
Learns the site-specific aquifer drawdown-recovery response under normal registered pumping.
Abnormal drawdown curves or slow recoveries produce high reconstruction error.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class LSTMFingerprintAutoencoder(nn.Module):
    """
    Encoder-Decoder LSTM architecture for time-series reconstruction.
    Input shape: (batch_size, seq_len, 1) -> relative water level trajectory only.
    Strictly NO pump_state, NO is_registered, NO start_hour.
    """
    def __init__(self, seq_len: int = 90, input_dim: int = 1, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Encode
        _, (h_n, _) = self.encoder_lstm(x) # h_n: (1, batch_size, hidden_dim)
        latent = torch.relu(self.encoder_fc(h_n.squeeze(0))) # (batch_size, latent_dim)
        
        # Decode
        decoder_input = torch.relu(self.decoder_fc(latent)) # (batch_size, hidden_dim)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch_size, seq_len, hidden_dim)
        
        lstm_out, _ = self.decoder_lstm(decoder_input) # (batch_size, seq_len, hidden_dim)
        reconstruction = self.output_layer(lstm_out) # (batch_size, seq_len, input_dim)
        
        return reconstruction

    def compute_reconstruction_error(self, x: torch.Tensor) -> float:
        """
        Compute Mean Squared Error (MSE) between input and reconstructed sequence.
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Focus error on water level channel (index 0)
            loss = torch.mean((x[:, :, 0] - reconstructed[:, :, 0]) ** 2).item()
        return float(loss)


class FingerprintModelTrainer:
    """
    Manages data preparation, training loop, and model persistence.
    """
    def __init__(self, seq_len: int = 90, input_dim: int = 1, lr: float = 0.003):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model = LSTMFingerprintAutoencoder(seq_len=seq_len, input_dim=input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.mean_stats = np.zeros(input_dim)
        self.std_stats = np.ones(input_dim)
        self.normal_threshold_mse = 0.05

    def fit(self, normal_sequences: np.ndarray, epochs: int = 35, batch_size: int = 32) -> float:
        """
        normal_sequences: numpy array of shape (N, seq_len, 2)
        channel 0 = water level (cm)
        channel 1 = pump state (0 or 1)
        """
        # Normalize
        self.mean_stats = np.mean(normal_sequences, axis=(0, 1))
        self.std_stats = np.std(normal_sequences, axis=(0, 1)) + 1e-6
        norm_seqs = (normal_sequences - self.mean_stats) / self.std_stats

        tensor_data = torch.tensor(norm_seqs, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_x)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            
            final_loss = epoch_loss / len(dataset)

        # Calibrate baseline threshold
        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor_data)
            errors = torch.mean((tensor_data[:, :, 0] - preds[:, :, 0]) ** 2, dim=1).numpy()
            # 97.5th percentile of normal errors as anomaly threshold
            self.normal_threshold_mse = float(np.percentile(errors, 97.5))

        print(f"LSTM Fingerprint training complete. Final loss: {final_loss:.5f}, Threshold MSE: {self.normal_threshold_mse:.5f}")
        return final_loss

    def score_sequence(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Takes raw sequence of shape (seq_len, 2) or (1, seq_len, 2).
        Returns:
          - mse_error: absolute reconstruction error
          - normalized_score: 0.0 to 100.0 score based on calibration threshold
        """
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        # Truncate or pad to seq_len
        curr_len = sequence.shape[1]
        if curr_len > self.seq_len:
            sequence = sequence[:, :self.seq_len, :]
        elif curr_len < self.seq_len:
            pad_len = self.seq_len - curr_len
            last_val = sequence[:, -1:, :]
            padding = np.repeat(last_val, pad_len, axis=1)
            sequence = np.concatenate([sequence, padding], axis=1)

        norm_seq = (sequence - self.mean_stats) / self.std_stats
        t_seq = torch.tensor(norm_seq, dtype=torch.float32)
        mse = self.model.compute_reconstruction_error(t_seq)

        # Scale relative to normal threshold (if mse == threshold, normalized_score ~ 50)
        norm_score = 100.0 * (1.0 / (1.0 + np.exp(-3.0 * (mse / max(1e-5, self.normal_threshold_mse) - 1.2))))
        return mse, float(np.clip(norm_score, 0.0, 100.0))

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "mean_stats": self.mean_stats,
            "std_stats": self.std_stats,
            "normal_threshold_mse": self.normal_threshold_mse,
            "seq_len": self.seq_len,
            "input_dim": self.input_dim
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, weights_only=False)
        self.seq_len = checkpoint["seq_len"]
        self.input_dim = checkpoint["input_dim"]
        self.model = LSTMFingerprintAutoencoder(seq_len=self.seq_len, input_dim=self.input_dim)
        self.model.load_state_dict(checkpoint["model_state"])
        self.mean_stats = checkpoint["mean_stats"]
        self.std_stats = checkpoint["std_stats"]
        self.normal_threshold_mse = checkpoint["normal_threshold_mse"]
        self.model.eval()
