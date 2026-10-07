"""
Groundwater ML Pipeline
End-to-end management: dataset generation, training, validation, serialization,
and live inference for the Groundwater Fingerprint system.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from .hydro_physics import GroundwaterSimulator, AquiferConfig
from .feature_extractor import GroundwaterFeatureExtractor
from .dataset_generator import GroundwaterDatasetGenerator
from .fingerprint_model import FingerprintModelTrainer
from .anomaly_scorer import ExtractionRiskScorer

class GroundwaterPipeline:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.extractor = GroundwaterFeatureExtractor()
        self.lstm_trainer = FingerprintModelTrainer(seq_len=90, input_dim=2)
        self.risk_scorer = ExtractionRiskScorer()
        self.is_loaded = False

    def train(
        self,
        n_normal: int = 600,
        n_excessive: int = 250,
        n_unregistered: int = 250,
        epochs: int = 30
    ) -> Dict[str, Any]:
        """
        Train both the LSTM Fingerprint model and the Risk Scorer on baseline data,
        then evaluate against all test scenarios.
        """
        print("=" * 60)
        print("STARTING GROUNDWATER FINGERPRINT MODEL TRAINING")
        print("=" * 60)

        generator = GroundwaterDatasetGenerator()
        df, raw_events = generator.generate_dataset(
            n_normal=n_normal,
            n_excessive=n_excessive,
            n_unregistered=n_unregistered
        )

        # 1. Prepare normal sequences for LSTM Autoencoder
        normal_events = [ev for ev in raw_events if ev["event_type"] == "normal"]
        normal_seqs = []
        for ev in normal_events:
            w_norm = ev["measured_water_level"]
            p_state = ev["pump_state"]
            # Subsample or interpolate to 90 timesteps
            indices = np.linspace(0, len(w_norm) - 1, 90).astype(int)
            seq = np.column_stack([w_norm[indices], p_state[indices]])
            normal_seqs.append(seq)
        normal_seqs = np.array(normal_seqs, dtype=np.float32)

        print(f"\n[1/3] Training LSTM Autoencoder on {len(normal_seqs)} normal sequences...")
        lstm_loss = self.lstm_trainer.fit(normal_seqs, epochs=epochs, batch_size=32)

        # 2. Train Isolation Forest and baseline reference stats
        print("\n[2/3] Training Isolation Forest and calibrating Hydrogeological Risk Scorer...")
        normal_df = df[df["scenario"] == "normal"].copy()
        self.risk_scorer.fit(normal_df)

        # 3. Comprehensive Evaluation
        print("\n[3/3] Evaluating across Scenarios (Normal, Excessive, Unregistered)...")
        eval_results = []
        y_true_binary = []
        y_pred_risk = []
        y_pred_binary = []

        for idx, row in df.iterrows():
            features_dict = row[self.extractor.feature_names].to_dict()
            features_dict["residual_deficit_cm"] = row["residual_deficit_cm"]
            features_dict["is_registered_window"] = row["is_registered_window"]
            features_dict["pump_duration_sec"] = row["pump_duration_sec"]

            # Compute LSTM score for this event's raw sequence
            ev = raw_events[idx]
            indices = np.linspace(0, len(ev["measured_water_level"]) - 1, 90).astype(int)
            seq = np.column_stack([ev["measured_water_level"][indices], ev["pump_state"][indices]])
            _, lstm_score = self.lstm_trainer.score_sequence(seq)

            eval_out = self.risk_scorer.evaluate(features_dict, lstm_residual_score=lstm_score)
            eval_out["true_scenario"] = row["scenario"]
            eval_out["true_is_anomaly"] = row["is_anomaly"]
            eval_results.append(eval_out)

            risk_score = eval_out["extraction_risk_score"]
            y_pred_risk.append(risk_score)
            # Binary flag: risk >= 50 is flagged as abnormal
            pred_binary = 1 if risk_score >= 50 else 0
            y_pred_binary.append(pred_binary)
            y_true_binary.append(row["is_anomaly"])

        # Calculate metrics
        eval_df = pd.DataFrame(eval_results)
        scenario_metrics = {}
        for sc in ["normal", "excessive", "unregistered"]:
            sc_subset = [r for r in eval_results if r["true_scenario"] == sc]
            scores = [r["extraction_risk_score"] for r in sc_subset]
            priorities = [r["inspection_priority"] for r in sc_subset]
            scenario_metrics[sc] = {
                "count": len(sc_subset),
                "mean_risk_score": float(np.mean(scores)),
                "std_risk_score": float(np.std(scores)),
                "min_risk_score": int(np.min(scores)),
                "max_risk_score": int(np.max(scores)),
                "priority_distribution": {p: priorities.count(p) for p in set(priorities)}
            }

        roc_auc = float(roc_auc_score(y_true_binary, y_pred_risk))
        cm = confusion_matrix(y_true_binary, y_pred_binary).tolist()
        
        metrics = {
            "roc_auc_score": round(roc_auc, 4),
            "confusion_matrix": cm,
            "scenario_benchmarks": scenario_metrics,
            "lstm_final_loss": round(lstm_loss, 5)
        }

        # 4. Save models
        self.save()
        with open(os.path.join(self.model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        self.is_loaded = True
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETE! ROC-AUC: {roc_auc:.4f}")
        for sc, m in scenario_metrics.items():
            print(f"Scenario [{sc.upper()}]: Mean Risk = {m['mean_risk_score']:.1f}/100, Priority = {m['priority_distribution']}")
        print("=" * 60)
        return metrics

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        self.lstm_trainer.save(os.path.join(self.model_dir, "lstm_fingerprint.pt"))
        self.risk_scorer.save(os.path.join(self.model_dir, "risk_scorer.joblib"))
        print(f"Model artifacts saved successfully in '{self.model_dir}/'")

    def load(self):
        lstm_path = os.path.join(self.model_dir, "lstm_fingerprint.pt")
        scorer_path = os.path.join(self.model_dir, "risk_scorer.joblib")
        if not (os.path.exists(lstm_path) and os.path.exists(scorer_path)):
            raise FileNotFoundError("Model files not found. Run training first.")
        self.lstm_trainer.load(lstm_path)
        self.risk_scorer.load(scorer_path)
        self.is_loaded = True
        print(f"Loaded trained Groundwater Fingerprint models from '{self.model_dir}/'")

    def predict_event(
        self,
        time_sec: np.ndarray,
        water_level_cm: np.ndarray,
        pump_state: np.ndarray,
        is_registered: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate a complete or partial pumping event.
        """
        if not self.is_loaded:
            self.load()

        # Extract features
        features = self.extractor.extract_from_event(
            time_sec=time_sec,
            water_level_cm=water_level_cm,
            pump_state=pump_state,
            is_registered=is_registered
        )

        # LSTM sequence score
        indices = np.linspace(0, len(water_level_cm) - 1, 90).astype(int)
        seq = np.column_stack([water_level_cm[indices], pump_state[indices]])
        _, lstm_score = self.lstm_trainer.score_sequence(seq)

        # Full risk evaluation
        result = self.risk_scorer.evaluate(features, lstm_residual_score=lstm_score)
        return result
