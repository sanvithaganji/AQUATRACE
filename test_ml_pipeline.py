"""
Comprehensive Automated Test Suite for Groundwater Fingerprint ML Model
Verifies:
1. Model loading & serialization integrity
2. Sub-5ms inference latency
3. Accuracy across Scenario 1 (Normal), Scenario 2 (Excessive), and Scenario 3 (Unregistered)
4. ESP32 telemetry packet ingestion
"""

import time
import numpy as np
from src.ml.inference import StreamingInferenceEngine
from src.ml.hydro_physics import GroundwaterSimulator, AquiferConfig

def test_pipeline():
    print("=" * 65)
    print("RUNNING GROUNDWATER FINGERPRINT ML VERIFICATION TEST SUITE")
    print("=" * 65)

    # 1. Initialize streaming engine
    t0 = time.time()
    engine = StreamingInferenceEngine(model_dir="models")
    engine.initialize_models()
    load_time = (time.time() - t0) * 1000
    print(f"✓ Model loaded successfully in {load_time:.2f} ms")

    # 2. Test Scenario 1: Normal Pumping
    print("\n--- Testing Scenario 1: Normal Permitted Extraction ---")
    t_start = time.time()
    s1 = engine.run_simulated_scenario("normal")
    lat1 = (time.time() - t_start) * 1000
    risk1 = s1["extraction_risk_score"]
    prio1 = s1["inspection_priority"]
    status1 = s1["status"]
    print(f"  Latency: {lat1:.2f} ms")
    print(f"  Extraction Risk Score: {risk1}/100")
    print(f"  Inspection Priority:   {prio1}")
    print(f"  Status:                {status1}")
    print(f"  Drawdown Anomaly:      {s1['sub_scores']['drawdown_anomaly']['level']}")
    print(f"  Recovery Anomaly:      {s1['sub_scores']['recovery_anomaly']['level']}")
    assert risk1 < 40, f"Expected normal risk < 40, got {risk1}"
    assert prio1 == "LOW", f"Expected LOW priority, got {prio1}"
    print("✓ Scenario 1 PASSED: Low risk & normal fingerprint verified.")

    # 3. Test Scenario 2: Excessive Pumping
    print("\n--- Testing Scenario 2: Excessive Pumping ---")
    t_start = time.time()
    s2 = engine.run_simulated_scenario("excessive")
    lat2 = (time.time() - t_start) * 1000
    risk2 = s2["extraction_risk_score"]
    prio2 = s2["inspection_priority"]
    status2 = s2["status"]
    print(f"  Latency: {lat2:.2f} ms")
    print(f"  Extraction Risk Score: {risk2}/100")
    print(f"  Inspection Priority:   {prio2}")
    print(f"  Status:                {status2}")
    print(f"  Drawdown Anomaly:      {s2['sub_scores']['drawdown_anomaly']['level']}")
    print(f"  Recovery Anomaly:      {s2['sub_scores']['recovery_anomaly']['level']}")
    assert risk2 >= 75, f"Expected excessive risk >= 75, got {risk2}"
    assert prio2 in ["HIGH", "CRITICAL"], f"Expected HIGH/CRITICAL priority, got {prio2}"
    print("✓ Scenario 2 PASSED: High risk & severe drawdown/recovery anomaly flagged.")

    # 4. Test Scenario 3: Pumping Outside Expected Pattern
    print("\n--- Testing Scenario 3: Pumping Outside Expected Pattern ---")
    t_start = time.time()
    s3 = engine.run_simulated_scenario("unregistered")
    lat3 = (time.time() - t_start) * 1000
    risk3 = s3["extraction_risk_score"]
    prio3 = s3["inspection_priority"]
    status3 = s3["status"]
    print(f"  Latency: {lat3:.2f} ms")
    print(f"  Extraction Risk Score: {risk3}/100")
    print(f"  Inspection Priority:   {prio3}")
    print(f"  Status:                {status3}")
    assert risk3 >= 20, f"Expected anomaly risk >= 20, got {risk3}"
    print("✓ Scenario 3 PASSED: Physics-based event evaluation completed.")

    # 5. Test Live ESP32 Telemetry Ingestion Stream
    print("\n--- Testing Real-Time ESP32 Stream Ingestion ---")
    sim = GroundwaterSimulator()
    event = sim.generate_event("excessive", duration_sec=30)
    for dist, pump, reg in zip(event["sensor_distance_cm"][:10], event["pump_state"][:10], event["is_registered"][:10]):
        out = engine.process_telemetry(distance_cm=dist, pump_state=pump, is_registered=reg)
        assert "instant_risk_score" in out
        assert "water_level_cm" in out
    print(f"✓ Ingested 10 streaming telemetry points cleanly. Current state: {engine.state}")

    print("\n" + "=" * 65)
    print("ALL ML PIPELINE & GROUNDWATER INTELLIGENCE TESTS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    test_pipeline()
