"""
Groundwater Detection Engine (detect.py)
========================================
Scores new pumping events from continuous CSV telemetry (columns: time, level_cm, pump_on)
and live ESP32 IoT sensor packets using 'calibration.json'.

Features:
  1. Uses pump_on strictly to cut event boundaries, never as a feature.
  2. Applies the unified 8-step preprocessing pipeline.
  3. Computes the explainable 3-ratio score:
       - ratio_drawdown = drawdown / typical_drawdown
       - ratio_duration = duration / typical_duration
       - ratio_recovery = (100 - recovery_pct) / (100 - typical_recovery_pct)
       - ratio = max of the three
       - mapped smoothly to 0-100 score (1.0 = 50, 1.5 = 85)
  4. Status:
       - Normal  : Below 50
       - Monitor : 50 - 70
       - Inspect : Above 70
  5. Plain-English explanation of the primary trigger.
  6. Optional secondary ML score shown beside the rule score (not used for status).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ml.explainable_detector import ExplainableDetector, slice_continuous_events
from src.ml.pipeline import GroundwaterPipeline


def load_ml_pipeline_optional(model_dir: str = "models") -> Optional[GroundwaterPipeline]:
    """Loads trained ML autoencoder model if available, otherwise returns None."""
    try:
        pipeline = GroundwaterPipeline(model_dir=model_dir)
        pipeline.load()
        return pipeline
    except Exception:
        return None


def detect_from_csv(csv_path: str, calibration_path: str = "calibration.json", enable_ml: bool = True) -> List[Dict[str, Any]]:
    print("=" * 78)
    print("GROUNDWATER ANOMALY DETECTION (detect.py)")
    print("=" * 78)

    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"Calibration file '{calibration_path}' not found. Run calibrate.py first.")

    detector = ExplainableDetector(calibration_path)
    print(f"Loaded calibration parameters from: {calibration_path}")
    print(f"  • Typical Drawdown:   {detector.typical_drawdown_cm:.2f} cm")
    print(f"  • Typical Duration:   {detector.typical_duration_sec:.1f} s")
    print(f"  • Typical Recovery %: {detector.typical_recovery_pct:.1f} %")

    ml_pipe = load_ml_pipeline_optional() if enable_ml else None
    if ml_pipe:
        print("✓ Optional ML secondary diagnostic scorer: ENABLED (shown beside rule score)")
    else:
        print("• Optional ML secondary diagnostic scorer: DISABLED/NOT FOUND")

    print(f"\nProcessing telemetry file: {csv_path}")
    df = pd.read_csv(csv_path)
    event_slices = slice_continuous_events(df)
    print(f"Identified {len(event_slices)} pumping cycles to evaluate.\n")

    results = []
    print(f"{'Event':<7} {'Rule Score':<12} {'Status':<10} {'ML Score':<10} {'Primary Trigger':<36}")
    print("-" * 78)

    for i, ev_df in enumerate(event_slices, 1):
        res = detector.score_event(
            time_sec=ev_df["time"].values,
            water_level_cm=ev_df["level_cm"].values,
            pump_on=ev_df["pump_on"].values,
            ml_pipeline=ml_pipe
        )
        results.append(res)

        score_str = f"{res['rule_score']}/100"
        ml_str = f"{res['ml_score']}/100" if res["ml_score"] is not None else "-"
        status_str = res["status"]
        
        # Short primary check snippet
        ratios = res.get("ratios", {})
        p_check = res.get("primary_check", "none")
        if p_check == "drawdown":
            p_desc = f"Drawdown ({ratios.get('drawdown_ratio', 0):.2f}x)"
        elif p_check == "duration":
            p_desc = f"Duration ({ratios.get('duration_ratio', 0):.2f}x)"
        elif p_check == "recovery":
            p_desc = f"Slow recovery ({ratios.get('recovery_deficit_ratio', 0):.2f}x)"
        else:
            p_desc = "Normal envelope"

        print(f"#{i:<6} {score_str:<12} {status_str:<10} {ml_str:<10} {p_desc:<36}")

    print("-" * 78)

    # Detailed audit card for flagged events
    flagged = [r for r in results if r["status"] in ["Monitor", "Inspect"]]
    if len(flagged) > 0:
        print(f"\n[FLAGGED EVENTS AUDIT DETAILS ({len(flagged)} events require attention)]")
        print("=" * 78)
        for idx, r in enumerate(flagged, 1):
            f = r.get("features", {})
            print(f"Flagged #{idx} — Score: {r['rule_score']}/100 | Status: {r['status']} | ML Score: {r.get('ml_score', '-')}/100")
            print(f"  • {r['explanation']}")
            print(f"  • Measured: Drawdown={f.get('drawdown_cm')} cm, Duration={f.get('duration_sec')} s, Recovery={f.get('recovery_pct')}%")
            print("-" * 78)
    else:
        print("\n✓ All evaluated events are within the Normal operational envelope.")

    return results


class LiveESP32PacketDetector:
    """
    Streaming detector for live ESP32 telemetry packets.
    Buffer telemetry, tracks pump transitions, and evaluates complete cycles.
    """
    def __init__(self, calibration_path: str = "calibration.json"):
        self.detector = ExplainableDetector(calibration_path)
        self.ml_pipeline = load_ml_pipeline_optional()
        self.buffer = []
        self.in_pumping_event = False
        self.post_pump_counter = 0

    def ingest_packet(self, timestamp: float, level_cm: float, pump_on: int) -> Optional[Dict[str, Any]]:
        """
        Ingest single live reading. Returns event result upon cycle completion.
        """
        self.buffer.append({"time": timestamp, "level_cm": level_cm, "pump_on": pump_on})
        
        # Keep buffer bounded (last 150 points)
        if len(self.buffer) > 150:
            self.buffer.pop(0)

        # Detect pump transition
        if pump_on == 1 and not self.in_pumping_event:
            self.in_pumping_event = True
            self.post_pump_counter = 0

        elif pump_on == 0 and self.in_pumping_event:
            self.post_pump_counter += 1
            # After 45 seconds of recovery or buffer full, evaluate completed event
            if self.post_pump_counter >= 45:
                df_ev = pd.DataFrame(self.buffer)
                res = self.detector.score_event(
                    time_sec=df_ev["time"].values,
                    water_level_cm=df_ev["level_cm"].values,
                    pump_on=df_ev["pump_on"].values,
                    ml_pipeline=self.ml_pipeline
                )
                self.in_pumping_event = False
                self.post_pump_counter = 0
                return res

        return None


def main():
    parser = argparse.ArgumentParser(description="Score groundwater events using explainable rule and calibration.json.")
    parser.add_argument("--csv", default="data/normal_calibration.csv", help="Path to input CSV.")
    parser.add_argument("--calibration", default="calibration.json", help="Path to calibration JSON.")
    parser.add_argument("--no-ml", action="store_true", help="Disable secondary ML score.")
    args = parser.parse_args()

    detect_from_csv(args.csv, args.calibration, enable_ml=(not args.no_ml))


if __name__ == "__main__":
    main()
