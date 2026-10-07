"""
Extraction Risk Scorer & Multi-Model Anomaly Detection Engine
Combines Isolation Forest, LSTM Fingerprint Residuals, and Hydrogeological Calibration
to compute the Extraction Risk Score (0-100) and Inspection Priority.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class ExtractionRiskScorer:
    """
    Decision-support model that translates raw hydrogeological and IoT telemetry
    into actionable enforcement priorities for water authorities.
    """
    def __init__(self, contamination: float = 0.05):
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42
        )
        self.feature_names = [
            "baseline_level_cm",
            "max_drawdown_cm",
            "drawdown_rate_cm_s",
            "pump_duration_sec",
            "recovery_duration_sec",
            "recovery_rate_cm_s",
            "drawdown_recovery_ratio",
            "residual_deficit_cm"
        ]
        # Baseline reference statistics from normal calibration runs
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, normal_features_df: pd.DataFrame):
        """
        Fit model on normal calibration operations.
        """
        X = normal_features_df[self.feature_names].values
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.iso_forest.fit(X_scaled)

        # Store baseline calibration references
        for col in self.feature_names:
            self.baseline_stats[col] = {
                "mean": float(normal_features_df[col].mean()),
                "std": float(normal_features_df[col].std() + 1e-6),
                "p95": float(np.percentile(normal_features_df[col], 95)),
                "p99": float(np.percentile(normal_features_df[col], 99))
            }
        print("ExtractionRiskScorer trained on normal baseline fingerprint.")

    def evaluate(
        self,
        features: Dict[str, float],
        lstm_residual_score: float = 20.0
    ) -> Dict[str, Any]:
        """
        Evaluate an event or real-time window and produce the Extraction Risk Score.
        """
        # 1. Isolation Forest Anomaly Score
        vec = np.array([[features.get(k, 0.0) for k in self.feature_names]])
        vec_scaled = self.scaler.transform(vec)
        # Decision function: lower values mean more anomalous (negative = outlier)
        raw_iso_score = self.iso_forest.decision_function(vec_scaled)[0]
        # Map raw decision function [-0.3, 0.3] -> [100, 0]
        iso_risk = float(np.clip(100.0 * (0.5 - raw_iso_score * 1.8), 0.0, 100.0))

        # 2. Hydrogeological Drawdown Sub-Score
        max_dd = features.get("max_drawdown_cm", 0.0)
        dd_p95 = self.baseline_stats.get("max_drawdown_cm", {}).get("p95", 2.5)
        dd_ratio = max_dd / max(0.5, dd_p95)
        if dd_ratio <= 1.1:
            drawdown_subscore = float(np.clip(dd_ratio * 25.0, 5.0, 35.0))
            drawdown_label = "NORMAL"
        elif dd_ratio <= 1.8:
            drawdown_subscore = float(np.clip(40.0 + (dd_ratio - 1.1) * 35.0, 40.0, 70.0))
            drawdown_label = "MEDIUM"
        else:
            drawdown_subscore = float(np.clip(70.0 + (dd_ratio - 1.8) * 20.0, 75.0, 98.0))
            drawdown_label = "HIGH"

        # 3. Recovery Sub-Score (Drawdown-to-recovery ratio & time)
        rec_dur = features.get("recovery_duration_sec", 0.0)
        rec_p95 = self.baseline_stats.get("recovery_duration_sec", {}).get("p95", 38.0)
        deficit = features.get("residual_deficit_cm", 0.0)
        rec_ratio = rec_dur / max(10.0, rec_p95)

        if rec_ratio <= 1.1 and deficit < 0.3:
            recovery_subscore = float(np.clip(rec_ratio * 25.0, 5.0, 35.0))
            recovery_label = "NORMAL"
        elif rec_ratio <= 1.8 and deficit < 0.8:
            recovery_subscore = float(np.clip(45.0 + (rec_ratio - 1.1) * 30.0, 45.0, 72.0))
            recovery_label = "SLOW"
        else:
            recovery_subscore = float(np.clip(75.0 + (rec_ratio - 1.8) * 15.0 + deficit * 10.0, 75.0, 99.0))
            recovery_label = "SEVERE / CRITICAL"

        # 4. Activity / Schedule Mismatch Sub-Score
        is_registered = bool(int(features.get("is_registered_window", 1.0)))
        pump_active = features.get("pump_duration_sec", 0.0) > 3.0 or max_dd > 0.6
        
        if not is_registered and pump_active:
            schedule_subscore = 95.0
            schedule_label = "SUSPECTED UNREGISTERED EXTRACTION"
        elif not is_registered and not pump_active:
            schedule_subscore = 10.0
            schedule_label = "ALIGNED (IDLE)"
        else:
            schedule_subscore = 15.0
            schedule_label = "PERMITTED / REGISTERED"

        # 5. Historical Aquifer Drift / Deficit
        drift_subscore = float(np.clip(deficit * 30.0 + 10.0, 10.0, 90.0))
        historical_label = "STABLE" if drift_subscore < 40.0 else ("ELEVATED STRESS" if drift_subscore < 70 else "CRITICAL DEPLETION")

        # 6. Composite Ensemble Extraction Risk Score
        # Weights:
        # If unregistered event detected -> heavy priority weight on schedule mismatch
        if not is_registered and pump_active:
            composite_risk = (
                0.40 * schedule_subscore +
                0.20 * drawdown_subscore +
                0.20 * recovery_subscore +
                0.10 * iso_risk +
                0.10 * lstm_residual_score
            )
            # Guarantee >= 85 for active extraction outside registration
            composite_risk = max(88.0, composite_risk)
        else:
            composite_risk = (
                0.25 * drawdown_subscore +
                0.25 * recovery_subscore +
                0.20 * iso_risk +
                0.20 * lstm_residual_score +
                0.10 * schedule_subscore
            )

        composite_risk = float(np.clip(round(composite_risk, 1), 0.0, 100.0))

        # 7. Categorization & Inspection Priority
        if composite_risk < 35.0:
            priority = "LOW"
            status = "NORMAL"
            badge_color = "#10b981" # Emerald Green
            recommendation = "No intervention needed. Aquifer response matches normal site fingerprint."
        elif composite_risk < 65.0:
            priority = "MEDIUM"
            status = "ELEVATED CONCERN"
            badge_color = "#f59e0b" # Amber Yellow
            recommendation = "Monitor subsequent recharge cycles. Minor drawdown or recovery elongation observed."
        elif composite_risk < 85.0:
            priority = "HIGH"
            status = "SUSPECTED EXCESSIVE EXTRACTION"
            badge_color = "#ef4444" # Red
            recommendation = "Prioritize field inspection. Excessive drawdown depth with slow recovery indicates extraction exceeding permitted threshold."
        else:
            priority = "CRITICAL"
            status = "SUSPECTED UNREGISTERED EXTRACTION" if not is_registered else "SEVERE OVER-EXTRACTION"
            badge_color = "#dc2626" # Deep Crimson Red
            recommendation = "Urgent field inspection recommended. Pumping activity detected outside registered hours or causing severe localized aquifer cone of depression."

        return {
            "extraction_risk_score": int(round(composite_risk)),
            "inspection_priority": priority,
            "status": status,
            "badge_color": badge_color,
            "recommendation": recommendation,
            "sub_scores": {
                "drawdown_anomaly": {
                    "score": int(round(drawdown_subscore)),
                    "level": drawdown_label,
                    "measured_cm": round(max_dd, 2)
                },
                "recovery_anomaly": {
                    "score": int(round(recovery_subscore)),
                    "level": recovery_label,
                    "duration_sec": round(rec_dur, 1)
                },
                "activity_mismatch": {
                    "score": int(round(schedule_subscore)),
                    "level": schedule_label,
                    "is_registered": is_registered
                },
                "historical_pattern": {
                    "score": int(round(drift_subscore)),
                    "level": historical_label,
                    "residual_deficit_cm": round(deficit, 2)
                },
                "isolation_forest_risk": int(round(iso_risk)),
                "lstm_fingerprint_residual": int(round(lstm_residual_score))
            },
            "features_summary": features
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "iso_forest": self.iso_forest,
            "feature_names": self.feature_names,
            "baseline_stats": self.baseline_stats
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.iso_forest = data["iso_forest"]
        self.feature_names = data["feature_names"]
        self.baseline_stats = data["baseline_stats"]
