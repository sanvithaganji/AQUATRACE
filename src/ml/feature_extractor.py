"""
Feature Extractor for Groundwater Fingerprint
Extracts hydrogeological drawdown-recovery metrics from raw time-series water level data.
Supports both batch dataset processing and real-time streaming buffers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class GroundwaterFeatureExtractor:
    def __init__(self, baseline_window_pts: int = 15, recovery_threshold_ratio: float = 0.85):
        """
        baseline_window_pts: Number of data points before pump starts to estimate baseline
        recovery_threshold_ratio: Fraction of drawdown recovered to consider recovery achieved
        """
        self.baseline_window_pts = baseline_window_pts
        self.recovery_threshold_ratio = recovery_threshold_ratio

    def extract_from_event(
        self,
        time_sec: np.ndarray,
        water_level_cm: np.ndarray,
        pump_state: np.ndarray,
        is_registered: int = 1
    ) -> Dict[str, float]:
        """
        Extract hydrogeological fingerprint features from an individual pumping event trajectory.
        """
        # Smooth high-frequency ultrasonic sensor jitter using moving median
        smoothed_level = pd.Series(water_level_cm).rolling(window=5, min_periods=1, center=True).median().to_numpy()

        # Find pump start and pump end indices
        pump_active_indices = np.where(pump_state > 0)[0]
        
        if len(pump_active_indices) == 0:
            # Baseline-only period without active pump
            baseline = float(np.median(smoothed_level))
            return {
                "baseline_level_cm": baseline,
                "max_drawdown_cm": 0.0,
                "drawdown_rate_cm_s": 0.0,
                "pump_duration_sec": 0.0,
                "recovery_duration_sec": 0.0,
                "recovery_rate_cm_s": 0.0,
                "drawdown_recovery_ratio": 1.0,
                "residual_deficit_cm": 0.0,
                "is_registered_window": float(is_registered),
                "has_pumping_event": 0.0
            }

        start_idx = pump_active_indices[0]
        end_idx = pump_active_indices[-1]

        # 1. Baseline Level Estimation (pre-pump baseline)
        pre_pump_indices = np.arange(max(0, start_idx - self.baseline_window_pts), start_idx)
        if len(pre_pump_indices) > 0:
            baseline = float(np.median(smoothed_level[pre_pump_indices]))
        else:
            baseline = float(smoothed_level[0])

        # 2. Drawdown phase analysis
        # Search for minimum level from start_idx up to a few seconds post-pump
        post_margin = min(len(smoothed_level), end_idx + 10)
        drawdown_window = smoothed_level[start_idx:post_margin]
        min_level_idx = start_idx + int(np.argmin(drawdown_window))
        min_level = float(smoothed_level[min_level_idx])
        
        max_drawdown_cm = max(0.0, baseline - min_level)
        
        pump_duration_sec = float(time_sec[end_idx] - time_sec[start_idx]) + (time_sec[1] - time_sec[0] if len(time_sec) > 1 else 1.0)
        drawdown_time_sec = max(1.0, float(time_sec[min_level_idx] - time_sec[start_idx]))
        drawdown_rate_cm_s = max_drawdown_cm / drawdown_time_sec

        # 3. Recovery phase analysis (from min_level_idx to end of recording)
        post_drawdown_levels = smoothed_level[min_level_idx:]
        post_drawdown_times = time_sec[min_level_idx:]
        
        target_recovery_level = baseline - (1.0 - self.recovery_threshold_ratio) * max_drawdown_cm
        
        # Look for when water level crosses target recovery level
        recovered_indices = np.where(post_drawdown_levels >= target_recovery_level)[0]
        
        if len(recovered_indices) > 0:
            recovery_idx = min_level_idx + recovered_indices[0]
            recovery_duration_sec = max(1.0, float(time_sec[recovery_idx] - time_sec[min_level_idx]))
            recharged_cm = max(0.0, float(smoothed_level[recovery_idx] - min_level))
            recovery_rate_cm_s = recharged_cm / recovery_duration_sec
        else:
            # Did not fully recover within the observation window! (Signs of severe depletion/over-pumping)
            recovery_duration_sec = max(1.0, float(time_sec[-1] - time_sec[min_level_idx]))
            recharged_cm = max(0.0, float(smoothed_level[-1] - min_level))
            recovery_rate_cm_s = recharged_cm / recovery_duration_sec

        # Ratio of drawdown rate to recovery rate (higher indicates stress/slow recharge)
        ratio = drawdown_rate_cm_s / max(0.001, recovery_rate_cm_s)
        
        # Residual level deficit at the end of recording
        final_level = float(smoothed_level[-1])
        residual_deficit_cm = max(0.0, baseline - final_level)

        return {
            "baseline_level_cm": round(baseline, 3),
            "max_drawdown_cm": round(max_drawdown_cm, 3),
            "drawdown_rate_cm_s": round(drawdown_rate_cm_s, 4),
            "pump_duration_sec": round(pump_duration_sec, 2),
            "recovery_duration_sec": round(recovery_duration_sec, 2),
            "recovery_rate_cm_s": round(recovery_rate_cm_s, 4),
            "drawdown_recovery_ratio": round(ratio, 3),
            "residual_deficit_cm": round(residual_deficit_cm, 3),
            "is_registered_window": float(is_registered),
            "has_pumping_event": 1.0
        }

    def extract_features_vector(self, features_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary into a fixed-order numeric vector for ML models.
        """
        feature_order = [
            "baseline_level_cm",
            "max_drawdown_cm",
            "drawdown_rate_cm_s",
            "pump_duration_sec",
            "recovery_duration_sec",
            "recovery_rate_cm_s",
            "drawdown_recovery_ratio",
            "residual_deficit_cm",
            "is_registered_window"
        ]
        return np.array([features_dict.get(k, 0.0) for k in feature_order], dtype=np.float32)

    @property
    def feature_names(self) -> List[str]:
        return [
            "baseline_level_cm",
            "max_drawdown_cm",
            "drawdown_rate_cm_s",
            "pump_duration_sec",
            "recovery_duration_sec",
            "recovery_rate_cm_s",
            "drawdown_recovery_ratio",
            "residual_deficit_cm",
            "is_registered_window"
        ]
