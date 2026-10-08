"""
Groundwater Calibration Tool (calibrate.py)
===========================================
Reads a CSV of normal operations (columns: time, level_cm, pump_on),
applies the unified preprocessing pipeline, cuts individual events, and computes
'typical' hydrogeological values:
  - typical_drawdown_cm : 95th percentile of normal drawdown
  - typical_duration_sec : 95th percentile of normal pumping duration
  - typical_recovery_pct : 5th percentile of normal recovery %

Saves the calibrated parameters to 'calibration.json'.
Warns if fewer than 30 events are present.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ml.preprocessing import preprocess_event_telemetry
from src.ml.feature_extractor import GroundwaterFeatureExtractor
from src.ml.explainable_detector import slice_continuous_events, ExplainableDetector
from src.ml.hydro_physics import GroundwaterSimulator, AquiferConfig


def generate_synthetic_normal_calibration_csv(filepath: str, n_events: int = 50):
    """
    Generates a realistic continuous calibration CSV of normal pumping cycles
    with ultrasonic noise (0.1 to 0.4 cm), occasional spikes, and sensor gaps.
    """
    print(f"Generating realistic normal calibration CSV ({n_events} cycles) -> {filepath}")
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    rng = np.random.RandomState(42)
    sim = GroundwaterSimulator(AquiferConfig(sensor_noise_std_cm=0.20))
    
    rows = []
    current_time = 0.0
    
    for _ in range(n_events):
        # Idle pre-pump window (15-25 seconds)
        idle_len = int(rng.uniform(15, 25))
        for _ in range(idle_len):
            current_time += 1.0
            dist = 12.0 + rng.normal(0, rng.uniform(0.1, 0.3))
            level = 38.0 - dist
            rows.append({"time": round(current_time, 2), "level_cm": round(level, 2), "pump_on": 0})
            
        # Pumping event (14 - 26 seconds)
        dur = float(rng.uniform(14.0, 26.0))
        ev = sim.generate_event("normal", pump_duration_sec=dur, duration_sec=int(dur + 65))
        
        # Inject noise and occasional spikes
        raw_dist = ev["sensor_distance_cm"].copy()
        n_pts = len(raw_dist)
        raw_dist += rng.normal(0, rng.uniform(0.1, 0.35), size=n_pts)
        spike_mask = rng.rand(n_pts) < 0.02
        raw_dist[spike_mask] += rng.choice([-1, 1], size=np.sum(spike_mask)) * rng.uniform(3.0, 6.0, size=np.sum(spike_mask))
        
        for d, p in zip(raw_dist, ev["pump_state"]):
            current_time += 1.0
            lvl = 38.0 - d
            rows.append({"time": round(current_time, 2), "level_cm": round(lvl, 2), "pump_on": int(p)})
            
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved calibration dataset: {len(df)} readings ({n_events} pumping cycles)")
    return df


def calibrate_from_csv(csv_path: str, output_json: str = "calibration.json"):
    print("=" * 70)
    print("GROUNDWATER DETECTOR CALIBRATION (calibrate.py)")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found.")
        print("Generating a representative normal calibration CSV with 50 pumping cycles...")
        generate_synthetic_normal_calibration_csv(csv_path, n_events=50)

    print(f"Reading normal calibration dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total time-series rows: {len(df)}")

    # Slice continuous telemetry into individual pumping cycles using pump_on
    raw_event_slices = slice_continuous_events(df)
    total_raw_events = len(raw_event_slices)
    print(f"Detected pumping cycles: {total_raw_events}")

    extractor = GroundwaterFeatureExtractor()
    drawdowns = []
    durations = []
    recoveries = []
    discarded_count = 0

    for ev_df in raw_event_slices:
        prep = preprocess_event_telemetry(
            time_sec=ev_df["time"].values,
            water_level_cm=ev_df["level_cm"].values,
            pump_state=ev_df["pump_on"].values
        )
        if prep["is_bad_data"]:
            discarded_count += 1
            continue

        feats = extractor.extract_from_event(
            time_sec=prep["time_sec"],
            water_level_cm=prep["water_level_cm"],
            pump_state=prep["pump_state"]
        )

        drawdowns.append(feats["max_drawdown_cm"])
        durations.append(feats["pump_duration_sec"])
        recoveries.append(feats["recovery_pct"])

    valid_events_count = len(drawdowns)
    print(f"Valid events used for calibration: {valid_events_count} (Discarded: {discarded_count})")

    # Check for warning threshold
    if valid_events_count < 30:
        print("\n" + "!" * 70)
        print(f"⚠️  WARNING: Only {valid_events_count} events used for calibration.")
        print("    At least 30 normal pumping cycles are recommended for stable 95th percentiles.")
        print("!" * 70 + "\n")

    if valid_events_count == 0:
        raise ValueError("No valid pumping events found in calibration CSV.")

    # Calculate typical values: 99th percentile for drawdown/duration, 1st percentile for recovery %, plus 5% margin
    p99_dd = float(np.percentile(drawdowns, 99.0))
    p99_dur = float(np.percentile(durations, 99.0))
    p01_rec = float(np.percentile(recoveries, 1.0))

    typical_dd = p99_dd * 1.05
    typical_dur = p99_dur * 1.05
    typical_rec = max(0.0, p01_rec * 0.95)

    cal_results = {
        "typical_drawdown_cm": round(typical_dd, 3),
        "typical_duration_sec": round(typical_dur, 2),
        "typical_recovery_pct": round(typical_rec, 2),
        "events_used": valid_events_count,
        "raw_events_found": total_raw_events,
        "discarded_events": discarded_count,
        "statistics": {
            "drawdown": {
                "median": round(float(np.median(drawdowns)), 2),
                "p95": round(float(np.percentile(drawdowns, 95.0)), 2),
                "p99": round(p99_dd, 2),
                "typical_with_margin": round(typical_dd, 2),
                "max": round(float(np.max(drawdowns)), 2)
            },
            "duration": {
                "median": round(float(np.median(durations)), 1),
                "p95": round(float(np.percentile(durations, 95.0)), 1),
                "p99": round(p99_dur, 1),
                "typical_with_margin": round(typical_dur, 1),
                "max": round(float(np.max(durations)), 1)
            },
            "recovery_pct": {
                "median": round(float(np.median(recoveries)), 1),
                "p05": round(float(np.percentile(recoveries, 5.0)), 1),
                "p01": round(p01_rec, 1),
                "typical_with_margin": round(typical_rec, 1),
                "min": round(float(np.min(recoveries)), 1)
            }
        }
    }

    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(cal_results, f, indent=2)

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE — TYPICAL ENVELOPE (99th/1st PCT + 5% MARGIN):")
    print("=" * 70)
    print(f"  • Typical Drawdown (99th pct + 5% margin):  {typical_dd:.2f} cm (p99={p99_dd:.2f})")
    print(f"  • Typical Duration (99th pct + 5% margin):  {typical_dur:.1f} s  (p99={p99_dur:.1f})")
    print(f"  • Typical Recovery % (1st pct - 5% margin): {typical_rec:.1f} %  (p01={p01_rec:.1f})")
    print(f"  • Parameters saved to:                      {output_json}")
    print("=" * 70)

    return cal_results


def main():
    parser = argparse.ArgumentParser(description="Calibrate explainable groundwater detector on normal runs CSV.")
    parser.add_argument("--csv", default="data/normal_calibration.csv", help="Path to input CSV of normal runs.")
    parser.add_argument("--output", default="calibration.json", help="Path to output calibration JSON.")
    args = parser.parse_args()

    calibrate_from_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
