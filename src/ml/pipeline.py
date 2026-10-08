"""
Groundwater ML Pipeline
=======================
End-to-end management: dataset generation, training, validation, serialization,
and live inference for the Groundwater Fingerprint system.
Uses unified preprocessing pipeline across training, evaluation, and live inference.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from .hydro_physics import GroundwaterSimulator, AquiferConfig
from .preprocessing import preprocess_event_telemetry
from .feature_extractor import GroundwaterFeatureExtractor
from .dataset_generator import GroundwaterDatasetGenerator
from .fingerprint_model import FingerprintModelTrainer
from .anomaly_scorer import ExtractionRiskScorer

class GroundwaterPipeline:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.extractor = GroundwaterFeatureExtractor()
        self.lstm_trainer = FingerprintModelTrainer(seq_len=90, input_dim=1)
        self.risk_scorer = ExtractionRiskScorer()
        self.is_loaded = False

    def train(
        self,
        n_normal: int = 700,
        n_excessive: int = 250,
        n_unregistered: int = 250,
        epochs: int = 35
    ) -> Dict[str, Any]:
        """
        Train both the LSTM Fingerprint model and the Risk Scorer on baseline data,
        then evaluate against all test scenarios. Preprocessing is applied identically.
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

        # 1. Prepare normal sequences for LSTM Autoencoder (strictly preprocessed relative water level, input_dim=1)
        normal_events = [ev for ev in raw_events if ev["event_type"] == "normal"]
        normal_seqs = []
        for ev in normal_events:
            prep = preprocess_event_telemetry(
                time_sec=ev["time_sec"],
                water_level_cm=ev["measured_water_level"],
                pump_state=ev["pump_state"]
            )
            w_clean = prep["water_level_cm"]
            b_line = prep["baseline_level_cm"]
            rel_w = w_clean - b_line
            # Resample / subsample to 90 timesteps
            indices = np.linspace(0, len(w_clean) - 1, 90).astype(int)
            seq = rel_w[indices, None]  # Shape: (90, 1)
            normal_seqs.append(seq)
        normal_seqs = np.array(normal_seqs, dtype=np.float32)

        print(f"\n[1/3] Training LSTM Autoencoder on {len(normal_seqs)} normal sequences (input_dim=1)...")
        lstm_loss = self.lstm_trainer.fit(normal_seqs, epochs=epochs, batch_size=32)

        # 2. Train Isolation Forest and baseline reference stats on training normal features only
        print("\n[2/3] Training Isolation Forest and calibrating Hydrogeological Risk Scorer...")
        normal_df = df[df["scenario"] == "normal"].copy()
        self.risk_scorer.fit(normal_df)

        # 3. Comprehensive Evaluation
        print("\n[3/3] Evaluating across Scenarios (Normal, Excessive, Outside Pattern)...")
        eval_results = []
        y_true_binary = []
        y_pred_risk = []
        y_pred_binary = []

        for idx, row in df.iterrows():
            features_dict = row[self.extractor.feature_names].to_dict()

            ev = raw_events[idx]
            prep = preprocess_event_telemetry(
                time_sec=ev["time_sec"],
                water_level_cm=ev["measured_water_level"],
                pump_state=ev["pump_state"]
            )
            w_clean = prep["water_level_cm"]
            b_line = prep["baseline_level_cm"]
            rel_w = w_clean - b_line
            indices = np.linspace(0, len(w_clean) - 1, 90).astype(int)
            seq = rel_w[indices, None]
            _, lstm_score = self.lstm_trainer.score_sequence(seq)

            eval_out = self.risk_scorer.evaluate(features_dict, lstm_residual_score=lstm_score)
            eval_out["true_scenario"] = row["scenario"]
            eval_out["true_is_anomaly"] = row["is_anomaly"]
            eval_results.append(eval_out)

            risk_score = eval_out["extraction_risk_score"]
            y_pred_risk.append(risk_score)
            pred_binary = 1 if risk_score >= 50 else 0
            y_pred_binary.append(pred_binary)
            y_true_binary.append(row["is_anomaly"])

        # Calculate metrics
        eval_df = pd.DataFrame(eval_results)
        scenario_metrics = {}
        for sc in ["normal", "excessive", "outside_pattern"]:
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
        water_level_cm: Optional[np.ndarray] = None,
        pump_state: Optional[np.ndarray] = None,
        sensor_distance_cm: Optional[np.ndarray] = None,
        is_registered: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate a complete or partial pumping event using the unified preprocessing pipeline.
        """
        if not self.is_loaded:
            self.load()

        # Step a - g: Preprocessing
        prep = preprocess_event_telemetry(
            time_sec=time_sec,
            sensor_distance_cm=sensor_distance_cm,
            water_level_cm=water_level_cm,
            pump_state=pump_state
        )
        
        if prep.get("is_bad_data", False):
            return {
                "extraction_risk_score": 0,
                "inspection_priority": "DISCARDED",
                "status": "BAD DATA QUALITY",
                "badge_color": "#9ca3af",
                "recommendation": f"Telemetry rejected: {prep.get('discard_reason')}",
                "is_bad_data": True,
                "discard_reason": prep.get("discard_reason"),
                "readings_fixed": prep.get("readings_fixed", 0),
                "repair_pct": prep.get("repair_pct", 0.0)
            }

        # Extract features from preprocessed series
        features = self.extractor.extract_from_event(
            time_sec=prep["time_sec"],
            water_level_cm=prep["water_level_cm"],
            pump_state=prep["pump_state"],
            is_registered=is_registered
        )

        # LSTM sequence score (using preprocessed relative water level)
        clean_levels = prep["water_level_cm"]
        baseline = prep["baseline_level_cm"]
        rel_levels = clean_levels - baseline
        indices = np.linspace(0, len(clean_levels) - 1, 90).astype(int)
        seq = rel_levels[indices, None]
        _, lstm_score = self.lstm_trainer.score_sequence(seq)

        # Full risk evaluation
        result = self.risk_scorer.evaluate(features, lstm_residual_score=lstm_score)
        result["preprocessing"] = {
            "readings_fixed": prep.get("readings_fixed", 0),
            "repair_pct": prep.get("repair_pct", 0.0),
            "is_bad_data": False
        }
        return result
