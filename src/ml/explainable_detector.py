"""
Explainable Rule-Based Groundwater Extraction Detector
======================================================
Implements a simple, transparent, and physics-grounded decision engine:
1. Computes three hydrogeological ratios from preprocessed water levels:
   a. ratio_drawdown = drawdown / typical_drawdown
   b. ratio_duration = pump_duration / typical_duration
   c. ratio_recovery = (100 - recovery_pct) / (100 - typical_recovery_pct)
2. ratio = max(ratio_drawdown, ratio_duration, ratio_recovery)
3. Maps ratio smoothly to a 0-100 risk score (1.0 = 50, 1.5 = 85)
4. Assigns status:
   - under 40: Normal
   - 40 - 70: Monitor
   - over 70: Inspect
5. Reports which check caused the score in plain English.
6. Optional second score from ML autoencoder shown beside the rule.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from .preprocessing import preprocess_event_telemetry
from .feature_extractor import GroundwaterFeatureExtractor


def map_ratio_to_risk_score(ratio: float) -> int:
    """
    Smooth, monotonic S-curve mapping:
      ratio = 1.0 -> 50
      ratio = 1.5 -> 85
      ratio <= 0.0 -> 0
    """
    if ratio <= 0.0:
        return 0
    # S(r) = 100 / (1 + exp(-3.47 * (r - 1.0)))
    val = 100.0 / (1.0 + np.exp(-3.47 * (ratio - 1.0)))
    return int(round(np.clip(val, 0.0, 100.0)))


def get_status_from_score(score: int) -> Tuple[str, str]:
    """
    Assign status and UI badge color:
      - Normal: below 50
      - Monitor: 50 - 70
      - Inspect: above 70
    """
    if score < 50:
        return "Normal", "#10b981"   # Green
    elif score <= 70:
        return "Monitor", "#f59e0b"  # Amber Yellow
    else:
        return "Inspect", "#ef4444"  # Red


class ExplainableDetector:
    def __init__(self, calibration_path: str = "calibration.json"):
        self.calibration_path = calibration_path
        self.extractor = GroundwaterFeatureExtractor()
        self.typical_drawdown_cm: float = 2.5
        self.typical_duration_sec: float = 25.0
        self.typical_recovery_pct: float = 90.0
        self.is_calibrated: bool = False
        
        if os.path.exists(calibration_path):
            self.load_calibration(calibration_path)

    def load_calibration(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.typical_drawdown_cm = float(data["typical_drawdown_cm"])
        self.typical_duration_sec = float(data["typical_duration_sec"])
        self.typical_recovery_pct = float(data["typical_recovery_pct"])
        self.is_calibrated = True

    def save_calibration(self, filepath: str, events_count: int, metadata: Optional[Dict] = None):
        cal_data = {
            "typical_drawdown_cm": round(self.typical_drawdown_cm, 3),
            "typical_duration_sec": round(self.typical_duration_sec, 2),
            "typical_recovery_pct": round(self.typical_recovery_pct, 2),
            "events_used": events_count,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S") if "time" in globals() else "",
            "description": "95th percentile of drawdown & duration, 5th percentile of recovery % on normal runs"
        }
        if metadata:
            cal_data.update(metadata)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(cal_data, f, indent=2)
        self.is_calibrated = True

    def score_event(
        self,
        time_sec: np.ndarray,
        water_level_cm: Optional[np.ndarray] = None,
        pump_on: Optional[np.ndarray] = None,
        sensor_distance_cm: Optional[np.ndarray] = None,
        ml_pipeline: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Scores a single pumping cycle using the explainable 3-ratio rule.
        pump_on is used strictly to cut event phases, never as a model input.
        """
        # Step 1: Preprocess telemetry
        prep = preprocess_event_telemetry(
            time_sec=time_sec,
            water_level_cm=water_level_cm,
            pump_state=pump_on,
            sensor_distance_cm=sensor_distance_cm
        )
        
        if prep.get("is_bad_data", False):
            return {
                "rule_score": 0,
                "status": "Bad Data",
                "badge_color": "#9ca3af",
                "explanation": f"Rejected by preprocessor: {prep.get('discard_reason')}",
                "ratios": {"drawdown": 0.0, "duration": 0.0, "recovery": 0.0, "max_ratio": 0.0},
                "features": {},
                "ml_score": None,
                "is_bad_data": True
            }

        # Step 2: Extract features from preprocessed clean levels
        feats = self.extractor.extract_from_event(
            time_sec=prep["time_sec"],
            water_level_cm=prep["water_level_cm"],
            pump_state=prep["pump_state"]
        )

        dd = feats.get("max_drawdown_cm", 0.0)
        dur = feats.get("pump_duration_sec", 0.0)
        rec_pct = feats.get("recovery_pct", 100.0)

        # Step 3: Compute three explainable ratios
        r_dd = dd / max(0.1, self.typical_drawdown_cm)
        r_dur = dur / max(1.0, self.typical_duration_sec)
        
        # Recovery ratio: how much unrecovered fraction exceeds typical unrecovered fraction
        unrecovered = 100.0 - rec_pct
        typical_unrecovered = max(1.0, 100.0 - self.typical_recovery_pct)
        r_rec = unrecovered / typical_unrecovered

        # Max ratio dictates score
        max_ratio = max(r_dd, r_dur, r_rec)
        score = map_ratio_to_risk_score(max_ratio)
        status, badge_color = get_status_from_score(score)

        # Step 4: Plain English explanation of the trigger
        check_details = {
            "drawdown": (r_dd, f"Drawdown depth ({dd:.2f} cm vs typical {self.typical_drawdown_cm:.2f} cm, {r_dd:.2f}x)"),
            "duration": (r_dur, f"Pump duration ({dur:.1f} s vs typical {self.typical_duration_sec:.1f} s, {r_dur:.2f}x)"),
            "recovery": (r_rec, f"Incomplete/slow recovery ({rec_pct:.1f}% recovered vs typical {self.typical_recovery_pct:.1f}%, {r_rec:.2f}x deficit)")
        }
        primary_key = max(check_details.keys(), key=lambda k: check_details[k][0])
        primary_trigger_text = check_details[primary_key][1]

        if score < 50:
            explanation = f"Normal operation: Aquifer response within typical envelope. Primary variation: {primary_trigger_text}."
        elif score <= 70:
            explanation = f"Monitor recommended: Elevated stress flagged due to {primary_trigger_text}."
        else:
            explanation = f"Inspection recommended: Severe anomaly triggered by {primary_trigger_text}."

        # Optional second ML score
        ml_score = None
        if ml_pipeline is not None:
            try:
                ml_res = ml_pipeline.predict_event(
                    time_sec=prep["time_sec"],
                    water_level_cm=prep["water_level_cm"],
                    pump_state=prep["pump_state"]
                )
                ml_score = ml_res.get("extraction_risk_score")
            except Exception:
                ml_score = None

        return {
            "rule_score": score,
            "status": status,
            "badge_color": badge_color,
            "primary_check": primary_key,
            "explanation": explanation,
            "ratios": {
                "drawdown_ratio": round(r_dd, 3),
                "duration_ratio": round(r_dur, 3),
                "recovery_deficit_ratio": round(r_rec, 3),
                "max_ratio": round(max_ratio, 3)
            },
            "features": {
                "drawdown_cm": round(dd, 2),
                "duration_sec": round(dur, 1),
                "recovery_pct": round(rec_pct, 1),
                "typical_drawdown_cm": round(self.typical_drawdown_cm, 2),
                "typical_duration_sec": round(self.typical_duration_sec, 1),
                "typical_recovery_pct": round(self.typical_recovery_pct, 1)
            },
            "ml_score": ml_score,
            "is_bad_data": False
        }


def slice_continuous_events(df: pd.DataFrame, min_event_gap_sec: float = 15.0) -> List[pd.DataFrame]:
    """
    Slices continuous telemetry into individual pumping events based on pump transitions.
    Accepts columns: time/time_sec, level_cm/water_level_cm, pump_on/pump_state.
    pump_on is used ONLY to cut event boundaries, never as a feature.
    """
    # Normalize column names
    col_map = {}
    for c in df.columns:
        c_lower = c.lower().strip()
        if c_lower in ["time", "time_sec", "timestamp"]:
            col_map[c] = "time"
        elif c_lower in ["level_cm", "water_level_cm", "level", "water_level"]:
            col_map[c] = "level_cm"
        elif c_lower in ["pump_on", "pump_state", "pump"]:
            col_map[c] = "pump_on"

    df_clean = df.rename(columns=col_map).copy()
    if not {"time", "level_cm", "pump_on"}.issubset(df_clean.columns):
        raise ValueError(f"CSV must contain time, level_cm, and pump_on columns. Found: {list(df.columns)}")

    pump_arr = df_clean["pump_on"].values.astype(int)
    n = len(pump_arr)
    
    # Find rising edges (0 -> 1)
    rising_indices = np.where((pump_arr[:-1] == 0) & (pump_arr[1:] == 1))[0] + 1
    if pump_arr[0] == 1:
        rising_indices = np.insert(rising_indices, 0, 0)
        
    events = []
    for idx_start in rising_indices:
        # Pre-pump baseline margin (up to 15 seconds before start)
        t_start = max(0, idx_start - 15)
        
        # Find next pump stop
        post_start_pumps = np.where(pump_arr[idx_start:] == 0)[0]
        if len(post_start_pumps) == 0:
            idx_stop = n - 1
        else:
            idx_stop = idx_start + post_start_pumps[0]
            
        # Recovery margin (up to 90 seconds after pump stop or until next pump rising edge)
        next_risings = rising_indices[rising_indices > idx_stop]
        if len(next_risings) > 0:
            idx_end = min(idx_stop + 90, next_risings[0])
        else:
            idx_end = min(idx_stop + 90, n)
            
        ev_slice = df_clean.iloc[t_start:idx_end].copy().reset_index(drop=True)
        # Shift relative time to start at 0
        ev_slice["time"] = ev_slice["time"] - ev_slice["time"].iloc[0]
        events.append(ev_slice)
        
    return events
