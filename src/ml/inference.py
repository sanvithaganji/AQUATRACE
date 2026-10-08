"""
Real-Time Streaming Inference Engine for Groundwater Fingerprint
Manages continuous sensor streaming from ESP32-S3, noise filtering,
state machine (IDLE -> DRAWDOWN -> RECOVERY), and real-time risk scoring.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List
from .pipeline import GroundwaterPipeline
from .hydro_physics import AquiferConfig, GroundwaterSimulator

class StreamingInferenceEngine:
    """
    Stateful streaming engine for incoming ESP32-S3 telemetry.
    Filters ultrasonic jitter, tracks baseline level, detects pumping events,
    and computes the Extraction Risk Score.
    """
    def __init__(self, model_dir: str = "models", config: Optional[AquiferConfig] = None, buffer_size: int = 150):
        self.config = config or AquiferConfig()
        self.pipeline = GroundwaterPipeline(model_dir=model_dir)
        self.buffer_size = buffer_size
        
        # Ring buffers for real-time telemetry
        self.timestamps = deque(maxlen=buffer_size)
        self.raw_distances = deque(maxlen=buffer_size)
        self.filtered_distances = deque(maxlen=buffer_size)
        self.water_levels = deque(maxlen=buffer_size)
        self.pump_states = deque(maxlen=buffer_size)
        self.registered_flags = deque(maxlen=buffer_size)

        # Baseline tracker
        self.current_baseline_cm = self.config.baseline_water_level_cm
        
        # State machine: 'IDLE', 'DRAWDOWN', 'RECOVERY'
        self.state = "IDLE"
        self.event_start_time = 0.0
        self.last_event_result: Optional[Dict[str, Any]] = None

        # Cumulative extraction metrics
        self.total_pump_duration_today_sec = 0.0
        self.events_count = 0

    def initialize_models(self):
        if not self.pipeline.is_loaded:
            self.pipeline.load()

    def process_telemetry(
        self,
        distance_cm: float,
        pump_state: int,
        is_registered: int = 1,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single reading from ESP32-S3:
        1. Ultrasonic outlier filtering
        2. Water level conversion
        3. Event detection state machine
        4. Real-time risk estimation
        """
        self.initialize_models()
        now = timestamp if timestamp is not None else time.time()
        
        # Steps a & b: Blind zone check (<2 cm) and tank bounds check
        if distance_cm < 2.0 or distance_cm > (self.config.tank_height_cm + self.config.sensor_offset_cm):
            # Dropped impossible reading (blind zone or out of range)
            filtered_dist = float(self.filtered_distances[-1]) if len(self.filtered_distances) > 0 else 12.0
        else:
            self.raw_distances.append(distance_cm)
            recent_dists = list(self.raw_distances)[-3:]
            filtered_dist = float(np.median(recent_dists))
        self.filtered_distances.append(filtered_dist)

        # Water level column height (calibrated tank height + sensor offset - sensor distance)
        water_level = (self.config.tank_height_cm + self.config.sensor_offset_cm) - filtered_dist
        water_level = float(np.clip(water_level, 0.0, self.config.tank_height_cm))
        self.water_levels.append(water_level)
        self.timestamps.append(now)
        self.pump_states.append(pump_state)
        self.registered_flags.append(is_registered)

        # 2. State Machine: Detect Pumping & Recovery Cycles
        if self.state == "IDLE":
            if pump_state == 1:
                self.state = "DRAWDOWN"
                self.event_start_time = now
            else:
                # Update baseline slowly during steady state
                if len(self.water_levels) >= 10:
                    self.current_baseline_cm = float(np.median(list(self.water_levels)[-10:]))
        
        elif self.state == "DRAWDOWN":
            if pump_state == 0:
                # Pump just shut off -> entered recovery phase
                self.state = "RECOVERY"

        elif self.state == "RECOVERY":
            # Recovered when water level returns within 0.3 cm of baseline or 45s elapsed
            if abs(water_level - self.current_baseline_cm) < 0.3:
                self.state = "IDLE"
                self.events_count += 1
                # Trigger complete event evaluation
                self.last_event_result = self._evaluate_current_window()
            elif (now - self.event_start_time) > 90.0:
                # Timeout / incomplete recovery
                self.state = "IDLE"
                self.events_count += 1
                self.last_event_result = self._evaluate_current_window()

        # Real-time instantaneous risk assessment
        current_drawdown = max(0.0, self.current_baseline_cm - water_level)
        instant_risk = self._compute_instantaneous_risk(current_drawdown, pump_state, is_registered)

        return {
            "timestamp": now,
            "raw_distance_cm": round(distance_cm, 2),
            "filtered_distance_cm": round(filtered_dist, 2),
            "water_level_cm": round(water_level, 2),
            "baseline_level_cm": round(self.current_baseline_cm, 2),
            "current_drawdown_cm": round(current_drawdown, 2),
            "pump_state": pump_state,
            "is_registered": is_registered,
            "system_state": self.state,
            "instant_risk_score": instant_risk["risk_score"],
            "inspection_priority": instant_risk["priority"],
            "status": instant_risk["status"],
            "badge_color": instant_risk["badge_color"],
            "last_event_result": self.last_event_result
        }

    def _compute_instantaneous_risk(self, drawdown_cm: float, pump_state: int, is_registered: int) -> Dict[str, Any]:
        """
        Fast real-time heuristic scoring while pumping is in progress.
        """
        if pump_state == 1 and is_registered == 0:
            return {
                "risk_score": 94,
                "priority": "CRITICAL",
                "status": "SUSPECTED UNREGISTERED EXTRACTION",
                "badge_color": "#dc2626"
            }
        
        if drawdown_cm > 3.5:
            return {
                "risk_score": 87,
                "priority": "HIGH",
                "status": "SUSPECTED EXCESSIVE EXTRACTION",
                "badge_color": "#ef4444"
            }
        elif drawdown_cm > 2.0:
            return {
                "risk_score": 58,
                "priority": "MEDIUM",
                "status": "ELEVATED CONCERN",
                "badge_color": "#f59e0b"
            }
        elif pump_state == 1:
            return {
                "risk_score": 22,
                "priority": "LOW",
                "status": "NORMAL ACTIVE PUMPING",
                "badge_color": "#10b981"
            }
        else:
            return {
                "risk_score": 12,
                "priority": "LOW",
                "status": "NORMAL IDLE BASELINE",
                "badge_color": "#10b981"
            }

    def _evaluate_current_window(self) -> Dict[str, Any]:
        """
        Evaluate full window with ML pipeline when event completes.
        """
        if len(self.water_levels) < 15:
            return {}
        t_arr = np.array(list(self.timestamps))
        t_arr = t_arr - t_arr[0]
        w_arr = np.array(list(self.water_levels))
        p_arr = np.array(list(self.pump_states))
        is_reg = int(self.registered_flags[-1]) if len(self.registered_flags) > 0 else 1

        return self.pipeline.predict_event(
            time_sec=t_arr,
            water_level_cm=w_arr,
            pump_state=p_arr,
            is_registered=is_reg
        )

    def run_simulated_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Instantly run one of the 3 competition scenarios and return the full evaluation.
        scenario_name: 'normal', 'excessive', or 'unregistered'
        """
        self.initialize_models()
        sim = GroundwaterSimulator(self.config)
        event = sim.generate_event(event_type=scenario_name)
        
        eval_result = self.pipeline.predict_event(
            time_sec=event["time_sec"],
            water_level_cm=event["measured_water_level"],
            pump_state=event["pump_state"],
            is_registered=int(event["is_registered"][0])
        )
        
        # Attach time-series trajectories for web chart visualization
        eval_result["time_series"] = {
            "time_sec": event["time_sec"].tolist(),
            "water_level_cm": [round(float(x), 2) for x in event["measured_water_level"]],
            "sensor_distance_cm": [round(float(x), 2) for x in event["sensor_distance_cm"]],
            "pump_state": event["pump_state"].tolist(),
            "baseline_cm": round(float(self.config.baseline_water_level_cm), 2)
        }
        eval_result["scenario_name"] = scenario_name
        return eval_result
