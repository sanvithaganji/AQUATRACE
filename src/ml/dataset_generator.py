"""
Dataset Generator for Groundwater Fingerprint
=============================================
Generates realistic training and evaluation datasets representing:
1. Normal aquifer pumping behavior (Calibration baseline) with sensor noise (0.1 to 0.4 cm),
   spikes, and missing readings, processed using the unified preprocessing pipeline.
2. Excessive pumping (Long pump duration, high drawdown, slow recovery)
3. Outside expected pattern (Unmetered / Unregistered extraction)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from .hydro_physics import GroundwaterSimulator, AquiferConfig
from .preprocessing import preprocess_event_telemetry
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
        Generate feature dataframe and raw time-series event records with realistic sensor
        noise (0.1 to 0.4 cm), spikes, and missing readings.
        """
        records = []
        raw_events = []

        total_samples = n_normal + n_excessive + n_unregistered
        print(f"Generating realistic groundwater dataset ({total_samples} samples) with 0.1-0.4 cm noise, spikes & gaps...")

        # 1. Normal events with sensor noise (0.1 to 0.4 cm), spikes, and missing readings
        while len([r for r in records if r["scenario"] == "normal"]) < n_normal:
            pump_dur = float(np.random.uniform(12.0, 28.0))
            start_sec = float(np.random.uniform(10.0, 20.0))
            noise_std = float(np.random.uniform(0.10, 0.40))
            
            event = self.simulator.generate_event(
                event_type="normal",
                duration_sec=120,
                dt=1.0,
                pump_start_sec=start_sec,
                pump_duration_sec=pump_dur,
                is_registered=True
            )
            
            raw_dist = event["sensor_distance_cm"].copy()
            n_pts = len(raw_dist)
            
            # Inject sensor noise (0.10 to 0.40 cm)
            raw_dist += np.random.normal(0, noise_std, size=n_pts)
            
            # Inject spikes on ~2.5% of readings
            spike_mask = np.random.rand(n_pts) < 0.025
            raw_dist[spike_mask] += np.random.choice([-1, 1], size=np.sum(spike_mask)) * np.random.uniform(3.0, 7.0, size=np.sum(spike_mask))
            
            # Inject missing readings (~3% NaNs, avoiding large gaps)
            nan_mask = np.random.rand(n_pts) < 0.03
            raw_dist[nan_mask] = np.nan
            
            prep = preprocess_event_telemetry(
                time_sec=event["time_sec"],
                sensor_distance_cm=raw_dist,
                pump_state=event["pump_state"]
            )
            
            if prep["is_bad_data"]:
                continue
                
            event["measured_water_level"] = prep["water_level_cm"]
            event["sensor_distance_cm"] = raw_dist
            
            features = self.extractor.extract_from_event(
                time_sec=prep["time_sec"],
                water_level_cm=prep["water_level_cm"],
                pump_state=prep["pump_state"],
                is_registered=1
            )
            features["scenario"] = "normal"
            features["is_anomaly"] = 0
            features["risk_category"] = "LOW"
            features["target_risk_score"] = float(np.clip(15 + np.random.normal(0, 5), 5, 32))
            
            records.append(features)
            raw_events.append(event)

        # 2. Excessive events
        while len([r for r in records if r["scenario"] == "excessive"]) < n_excessive:
            pump_dur = float(np.random.uniform(45.0, 85.0))
            start_sec = float(np.random.uniform(10.0, 15.0))
            noise_std = float(np.random.uniform(0.10, 0.40))
            
            event = self.simulator.generate_event(
                event_type="excessive",
                duration_sec=180,
                dt=1.0,
                pump_start_sec=start_sec,
                pump_duration_sec=pump_dur,
                is_registered=True
            )
            
            raw_dist = event["sensor_distance_cm"].copy()
            n_pts = len(raw_dist)
            raw_dist += np.random.normal(0, noise_std, size=n_pts)
            spike_mask = np.random.rand(n_pts) < 0.02
            raw_dist[spike_mask] += np.random.choice([-1, 1], size=np.sum(spike_mask)) * np.random.uniform(3.0, 6.0, size=np.sum(spike_mask))
            nan_mask = np.random.rand(n_pts) < 0.03
            raw_dist[nan_mask] = np.nan

            prep = preprocess_event_telemetry(event["time_sec"], sensor_distance_cm=raw_dist, pump_state=event["pump_state"])
            if prep["is_bad_data"]:
                continue

            event["measured_water_level"] = prep["water_level_cm"]
            event["sensor_distance_cm"] = raw_dist

            features = self.extractor.extract_from_event(
                time_sec=prep["time_sec"],
                water_level_cm=prep["water_level_cm"],
                pump_state=prep["pump_state"],
                is_registered=1
            )
            features["scenario"] = "excessive"
            features["is_anomaly"] = 1
            features["risk_category"] = "HIGH"
            features["target_risk_score"] = float(np.clip(85 + np.random.normal(0, 5), 70, 98))
            
            records.append(features)
            raw_events.append(event)

        # 3. Outside expected pattern (unregistered)
        while len([r for r in records if r["scenario"] == "outside_pattern"]) < n_unregistered:
            pump_dur = float(np.random.uniform(20.0, 60.0))
            start_sec = float(np.random.uniform(10.0, 20.0))
            noise_std = float(np.random.uniform(0.10, 0.40))
            
            event = self.simulator.generate_event(
                event_type="unregistered",
                duration_sec=140,
                dt=1.0,
                pump_start_sec=start_sec,
                pump_duration_sec=pump_dur,
                is_registered=False
            )
            
            raw_dist = event["sensor_distance_cm"].copy()
            n_pts = len(raw_dist)
            raw_dist += np.random.normal(0, noise_std, size=n_pts)
            spike_mask = np.random.rand(n_pts) < 0.02
            raw_dist[spike_mask] += np.random.choice([-1, 1], size=np.sum(spike_mask)) * np.random.uniform(3.0, 6.0, size=np.sum(spike_mask))
            nan_mask = np.random.rand(n_pts) < 0.03
            raw_dist[nan_mask] = np.nan

            prep = preprocess_event_telemetry(event["time_sec"], sensor_distance_cm=raw_dist, pump_state=event["pump_state"])
            if prep["is_bad_data"]:
                continue

            event["measured_water_level"] = prep["water_level_cm"]
            event["sensor_distance_cm"] = raw_dist

            features = self.extractor.extract_from_event(
                time_sec=prep["time_sec"],
                water_level_cm=prep["water_level_cm"],
                pump_state=prep["pump_state"],
                is_registered=0
            )
            features["scenario"] = "outside_pattern"
            features["is_anomaly"] = 1
            features["risk_category"] = "CRITICAL"
            features["target_risk_score"] = float(np.clip(94 + np.random.normal(0, 3), 85, 100))
            
            records.append(features)
            raw_events.append(event)

        df = pd.DataFrame(records)
        return df, raw_events

if __name__ == "__main__":
    gen = GroundwaterDatasetGenerator()
    df, _ = gen.generate_dataset(n_normal=50, n_excessive=20, n_unregistered=20)
    print("Dataset generated successfully!")
    print(df.groupby("scenario")[["max_drawdown_cm", "recovery_duration_sec", "drawdown_recovery_ratio", "target_risk_score"]].mean())
