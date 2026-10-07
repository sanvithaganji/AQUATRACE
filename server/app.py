"""
Groundwater Fingerprint — Decision Support & Ingestion Server
Hosts REST API for ESP32-S3 telemetry, real-time ML inference,
scenario simulation, and the interactive web dashboard.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, render_template, send_from_directory
from src.ml.inference import StreamingInferenceEngine

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
)

# Global streaming engine instance
engine = StreamingInferenceEngine(model_dir="models")

# Live node metadata
node_meta = {
    "node_id": "ESP32S3-GW-01",
    "last_seen": None,
    "ip_address": "Waiting...",
    "pump_command": "NONE"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/telemetry", methods=["POST"])
def receive_telemetry():
    """
    Ingest live telemetry packet from ESP32-S3 or simulator.
    Payload: { "node_id": str, "distance_cm": float, "pump_state": int, "is_registered": int }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    distance_cm = float(data.get("distance_cm", 15.0))
    pump_state = int(data.get("pump_state", 0))
    is_registered = int(data.get("is_registered", 1))

    node_meta["last_seen"] = time.time()
    node_meta["node_id"] = data.get("node_id", "ESP32S3-GW-01")
    node_meta["ip_address"] = request.remote_addr

    processed = engine.process_telemetry(
        distance_cm=distance_cm,
        pump_state=pump_state,
        is_registered=is_registered
    )

    response = {
        "status": "ok",
        "processed": processed,
        "command": node_meta["pump_command"]
    }
    # Reset transient command
    node_meta["pump_command"] = "NONE"
    return jsonify(response)

@app.route("/api/status", methods=["GET"])
def get_status():
    """
    Get current real-time state, risk scores, and recent telemetry buffer.
    """
    recent_count = min(60, len(engine.water_levels))
    recent_levels = list(engine.water_levels)[-recent_count:] if recent_count > 0 else []
    recent_times = list(engine.timestamps)[-recent_count:] if recent_count > 0 else []
    recent_pumps = list(engine.pump_states)[-recent_count:] if recent_count > 0 else []

    is_online = (node_meta["last_seen"] is not None) and (time.time() - node_meta["last_seen"] < 5.0)

    # If an event completed recently, use its detailed breakdown
    last_res = engine.last_event_result
    
    return jsonify({
        "node_info": {
            "node_id": node_meta["node_id"],
            "online": is_online,
            "ip_address": node_meta["ip_address"],
            "last_seen_sec_ago": round(time.time() - node_meta["last_seen"], 1) if node_meta["last_seen"] else None
        },
        "system_state": engine.state,
        "current_water_level_cm": round(engine.water_levels[-1], 2) if engine.water_levels else engine.config.baseline_water_level_cm,
        "baseline_level_cm": round(engine.current_baseline_cm, 2),
        "recent_time_series": {
            "timestamps": recent_times,
            "water_levels": recent_levels,
            "pump_states": recent_pumps
        },
        "last_event_result": last_res
    })

@app.route("/api/simulate_scenario", methods=["POST"])
def simulate_scenario():
    """
    Execute instant simulation of competition demo scenarios:
    'normal', 'excessive', or 'unregistered'.
    """
    scenario = request.args.get("scenario", "normal")
    if scenario not in ["normal", "excessive", "unregistered"]:
        return jsonify({"error": "Invalid scenario name"}), 400

    result = engine.run_simulated_scenario(scenario)
    return jsonify(result)

@app.route("/api/pump_control", methods=["POST"])
def pump_control():
    """
    Send remote ON/OFF command to the ESP32 relay.
    """
    data = request.get_json(force=True) or {}
    state = data.get("state", "OFF").upper()
    node_meta["pump_command"] = f"PUMP_{state}"
    return jsonify({"command_dispatched": node_meta["pump_command"]})

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Return training and ROC-AUC evaluation metrics.
    """
    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"status": "no metrics file found"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\nStarting Groundwater Fingerprint ML Server on http://0.0.0.0:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
