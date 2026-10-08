"""
Feature Extractor for Groundwater Fingerprint
=============================================
Extracts hydrogeological drawdown-recovery metrics from raw time-series water level data.
Integrates with the unified preprocessing pipeline (Hampel filter, short-gap interpolation,
calibrated tank height conversion, and median pre-pump & post-recovery baselines).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from .preprocessing import preprocess_event_telemetry


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
        water_level_cm: Optional[np.ndarray] = None,
        pump_state: Optional[np.ndarray] = None,
        sensor_distance_cm: Optional[np.ndarray] = None,
        is_registered: int = 1,
        calibrated_tank_height_cm: float = 35.0,
        sensor_offset_cm: float = 3.0
    ) -> Dict[str, Any]:
        """
        Extract hydrogeological fingerprint features from an individual pumping event trajectory
        using the unified 8-step preprocessing pipeline.
        """
        # Step a - g: Unified preprocessing
        prep = preprocess_event_telemetry(
            time_sec=time_sec,
            sensor_distance_cm=sensor_distance_cm,
            water_level_cm=water_level_cm,
            pump_state=pump_state,
            calibrated_tank_height_cm=calibrated_tank_height_cm,
            sensor_offset_cm=sensor_offset_cm
        )
        
        if prep.get("is_bad_data", False):
            # Event had excessive repair (>20%) or gap > 3 samples
            return {
                "max_drawdown_cm": 0.0,
                "drawdown_rate_cm_s": 0.0,
                "pump_duration_sec": 0.0,
                "recovery_duration_sec": 0.0,
                "recovery_rate_cm_s": 0.0,
                "drawdown_recovery_ratio": 1.0,
                "residual_deficit_cm": 0.0,
                "recovery_pct": 100.0,
                "is_bad_data": True,
                "discard_reason": prep.get("discard_reason", "Data quality threshold exceeded"),
                "readings_fixed": prep.get("readings_fixed", 0),
                "repair_pct": prep.get("repair_pct", 0.0)
            }

        smoothed_level = prep["water_level_cm"]
        p_state = prep["pump_state"]
        t_sec = prep["time_sec"]

        # Find pump start and pump end indices
        pump_active_indices = np.where(p_state > 0)[0]
        
        if len(pump_active_indices) == 0:
            # Baseline-only period without active pump
            baseline = prep["baseline_level_cm"]
            return {
                "max_drawdown_cm": 0.0,
                "drawdown_rate_cm_s": 0.0,
                "pump_duration_sec": 0.0,
                "recovery_duration_sec": 0.0,
                "recovery_rate_cm_s": 0.0,
                "drawdown_recovery_ratio": 1.0,
                "residual_deficit_cm": 0.0,
                "recovery_pct": 100.0,
                "is_bad_data": False,
                "readings_fixed": prep.get("readings_fixed", 0),
                "repair_pct": prep.get("repair_pct", 0.0)
            }

        start_idx = pump_active_indices[0]
        end_idx = pump_active_indices[-1]

        # Step f: Baseline level estimation (median of pre-pump window)
        baseline = prep["baseline_level_cm"]

        # Drawdown phase analysis
        post_margin = min(len(smoothed_level), end_idx + 10)
        drawdown_window = smoothed_level[start_idx:post_margin]
        min_level_idx = start_idx + int(np.argmin(drawdown_window))
        min_level = float(smoothed_level[min_level_idx])
        
        max_drawdown_cm = max(0.0, baseline - min_level)
        
        dt_val = (t_sec[1] - t_sec[0]) if len(t_sec) > 1 else 1.0
        pump_duration_sec = float(t_sec[end_idx] - t_sec[start_idx]) + dt_val
        drawdown_time_sec = max(1.0, float(t_sec[min_level_idx] - t_sec[start_idx]))
        drawdown_rate_cm_s = max_drawdown_cm / drawdown_time_sec

        # Recovery phase analysis (from min_level_idx to end of recording)
        post_drawdown_levels = smoothed_level[min_level_idx:]
        post_drawdown_times = t_sec[min_level_idx:]
        
        target_recovery_level = baseline - (1.0 - self.recovery_threshold_ratio) * max_drawdown_cm
        recovered_indices = np.where(post_drawdown_levels >= target_recovery_level)[0]
        
        if len(recovered_indices) > 0:
            recovery_idx = min_level_idx + recovered_indices[0]
            recovery_duration_sec = max(1.0, float(t_sec[recovery_idx] - t_sec[min_level_idx]))
            recharged_cm = max(0.0, float(smoothed_level[recovery_idx] - min_level))
            recovery_rate_cm_s = recharged_cm / recovery_duration_sec
        else:
            recovery_duration_sec = max(1.0, float(t_sec[-1] - t_sec[min_level_idx]))
            recharged_cm = max(0.0, float(smoothed_level[-1] - min_level))
            recovery_rate_cm_s = recharged_cm / recovery_duration_sec

        ratio = drawdown_rate_cm_s / max(0.001, recovery_rate_cm_s)
        
        # Step f: Final level from median of last 10 points
        final_level = prep["final_level_cm"]
        residual_deficit_cm = max(0.0, baseline - final_level)

        # Percentage recovered from maximum drawdown
        recovered_cm = max(0.0, final_level - min_level)
        recovery_pct = float(np.clip((recovered_cm / max(0.1, max_drawdown_cm)) * 100.0, 0.0, 100.0))

        return {
            "max_drawdown_cm": round(max_drawdown_cm, 3),
            "drawdown_rate_cm_s": round(drawdown_rate_cm_s, 4),
            "pump_duration_sec": round(pump_duration_sec, 2),
            "recovery_duration_sec": round(recovery_duration_sec, 2),
            "recovery_rate_cm_s": round(recovery_rate_cm_s, 4),
            "drawdown_recovery_ratio": round(ratio, 3),
            "residual_deficit_cm": round(residual_deficit_cm, 3),
            "recovery_pct": round(recovery_pct, 2),
            "is_bad_data": False,
            "readings_fixed": prep.get("readings_fixed", 0),
            "repair_pct": prep.get("repair_pct", 0.0)
        }

    def extract_features_vector(self, features_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary into a fixed-order numeric vector for ML models.
        Strictly hydrological: NO is_registered, NO pump_state, NO start_hour, NO baseline_depth_cm.
        """
        return np.array([features_dict.get(k, 0.0) for k in self.feature_names], dtype=np.float32)

    @property
    def feature_names(self) -> List[str]:
        return [
            "max_drawdown_cm",
            "drawdown_rate_cm_s",
            "pump_duration_sec",
            "recovery_duration_sec",
            "recovery_rate_cm_s",
            "drawdown_recovery_ratio",
            "residual_deficit_cm",
            "recovery_pct"
        ]
