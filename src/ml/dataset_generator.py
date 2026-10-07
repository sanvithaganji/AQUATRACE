"""
Dataset Generator for Groundwater Fingerprint
Generates realistic training and evaluation datasets representing:
1. Normal aquifer pumping behavior (Calibration baseline)
2. Excessive pumping (Long pump duration, high drawdown, slow recovery)
3. Unregistered / Unmetered pumping (Extraction outside permitted schedules)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from .hydro_physics import GroundwaterSimulator, AquiferConfig
from .feature_extractor import GroundwaterFeatureExtractor

class GroundwaterDatasetGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.config = AquiferConfig()
        self.simulator = GroundwaterSimulator(self.config)
        self.extractor = GroundwaterFeatureExtractor()

    def generate_dataset(
        self,
        n_normal: int = 800,
        n_excessive: int = 300,
        n_unregistered: int = 300
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Generate feature dataframe and raw time-series event records.
        """
        records = []
        raw_events = []

        total_samples = n_normal + n_excessive + n_unregistered
        print(f"Generating synthetic groundwater dataset ({total_samples} samples)...")

        # 1. Normal events
        for _ in range(n_normal):
            # Normal variations in baseline and pump run (10s to 30s)
            pump_dur = np.random.uniform(12.0, 28.0)
            start_sec = np.random.uniform(10.0, 20.0)
            
            event = self.simulator.generate_event(
                event_type="normal",
                duration_sec=120,
                dt=1.0,
                pump_start_sec=start_sec,
                pump_duration_sec=pump_dur,
                is_registered=True
            )
            
            features = self.extractor.extract_from_event(
                time_sec=event["time_sec"],
                water_level_cm=event["measured_water_level"],
                pump_state=event["pump_state"],
                is_registered=1
            )
            features["scenario"] = "normal"
            features["is_anomaly"] = 0
            features["risk_category"] = "LOW"
            # Target calibrated risk ~ 10 - 30
            features["target_risk_score"] = float(np.clip(15 + np.random.normal(0, 5), 5, 32))
            
            records.append(features)
            raw_events.append(event)

        # 2. Excessive events
        for _ in range(n_excessive):
            # Excessive pump duration (45s to 90s)
            pump_dur = np.random.uniform(45.0, 85.0)
            start_sec = np.random.uniform(10.0, 15.0)
            
            event = self.simulator.generate_event(
                event_type="excessive",
                duration_sec=180,
                dt=1.0,
                pump_start_sec=start_sec,
                pump_duration_sec=pump_dur,
                is_registered=True
            )
            
            features = self.extractor.extract_from_event(
                time_sec=event["time_sec"],
                water_level_cm=event["measured_water_level"],
                pump_state=event["pump_state"],
                is_registered=1
            )
            features["scenario"] = "excessive"
            features["is_anomaly"] = 1
            features["risk_category"] = "HIGH"
            # Target calibrated risk ~ 75 - 95
            features["target_risk_score"] = float(np.clip(85 + np.random.normal(0, 5), 70, 98))
            
            records.append(features)
            raw_events.append(event)

        # 3. Unregistered events
        for _ in range(n_unregistered):
            # Pumping outside registered schedule
            pump_dur = np.random.uniform(20.0, 60.0)
            start_sec = np.random.uniform(10.0, 20.0)
            
            event = self.simulator.generate_event(
                event_type="unregistered",
                duration_sec=140,
                dt=1.0,
                pump_start_sec=start_sec,
                pump_duration_sec=pump_dur,
                is_registered=False
            )
            
            features = self.extractor.extract_from_event(
                time_sec=event["time_sec"],
                water_level_cm=event["measured_water_level"],
                pump_state=event["pump_state"],
                is_registered=0
            )
            features["scenario"] = "unregistered"
            features["is_anomaly"] = 1
            features["risk_category"] = "CRITICAL"
            # Target calibrated risk ~ 88 - 99
            features["target_risk_score"] = float(np.clip(94 + np.random.normal(0, 3), 85, 100))
            
            records.append(features)
            raw_events.append(event)

        df = pd.DataFrame(records)
        return df, raw_events

if __name__ == "__main__":
    gen = GroundwaterDatasetGenerator()
    df, _ = gen.generate_dataset(n_normal=100, n_excessive=50, n_unregistered=50)
    print("Dataset generated successfully!")
    print(df.groupby("scenario")[["max_drawdown_cm", "recovery_duration_sec", "drawdown_recovery_ratio", "target_risk_score"]].mean())
