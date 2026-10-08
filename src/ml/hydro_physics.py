"""
Hydrogeological & Well Simulator for Groundwater Fingerprint
Simulates physical tank well dynamics (HC-SR04 ultrasonic + 12V pump)
and real-world aquifer drawdown-recovery responses.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

@dataclass
class AquiferConfig:
    tank_height_cm: float = 35.0         # Height of physical simulation container
    sensor_offset_cm: float = 3.0        # Sensor mount distance above max water line
    baseline_water_level_cm: float = 26.0 # Normal standing water column height
    pump_drawdown_rate_cm_s: float = 0.08 # Rate of water level drop when pump is ON (~4.8 cm/min)
    normal_recharge_rate_k: float = 0.04  # Exponential recovery rate constant (1/s)
    depleted_recharge_rate_k: float = 0.009 # Stressed/slow recovery constant
    sensor_noise_std_cm: float = 0.15     # Ultrasonic HC-SR04 distance measurement noise

class GroundwaterSimulator:
    """
    Simulates water level time-series data for baseline, drawdown, and recovery events.
    Calculates distance measured by HC-SR04 ultrasonic sensor:
    distance_cm = tank_height_cm + sensor_offset_cm - water_level_cm
    """
    def __init__(self, config: AquiferConfig = None):
        self.config = config or AquiferConfig()

    def generate_event(
        self,
        event_type: str = "normal",
        duration_sec: int = 120,
        dt: float = 1.0,
        pump_start_sec: float = 15.0,
        pump_duration_sec: float = 25.0,
        is_registered: bool = True,
        start_level_cm: Optional[float] = None,
        custom_drawdown_rate: Optional[float] = None,
        custom_recharge_k: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simulate a complete pumping cycle:
        Pre-pumping baseline -> Pumping (Drawdown) -> Post-pumping (Recovery).
        
        event_type options:
          - 'normal': Permitted window, normal pump duration, swift recovery
          - 'excessive': Pump runs much longer / heavier draw, aquifer stress, slow recovery
          - 'unregistered': Drawdown occurs outside registered window (is_registered=False)
        """
        time_steps = np.arange(0, duration_sec, dt)
        n = len(time_steps)
        
        # Configure scenario parameters
        if event_type == "normal":
            p_dur = pump_duration_sec
            recharge_k = custom_recharge_k if custom_recharge_k is not None else self.config.normal_recharge_rate_k
            registered_flag = True
        elif event_type == "excessive":
            p_dur = pump_duration_sec * 2.5 # Long pump run (e.g. 62.5 sec)
            recharge_k = custom_recharge_k if custom_recharge_k is not None else self.config.depleted_recharge_rate_k # Aquifer cone of depression deep, slow recovery
            registered_flag = True
        elif event_type == "unregistered":
            p_dur = pump_duration_sec * 1.5
            recharge_k = custom_recharge_k if custom_recharge_k is not None else (self.config.normal_recharge_rate_k * 0.8)
            registered_flag = False # PUMPING WITHOUT PERMIT / OUTSIDE SCHEDULE
        else:
            raise ValueError(f"Unknown event type: {event_type}")

        pump_state = np.zeros(n, dtype=int)
        water_level = np.zeros(n, dtype=float)
        
        baseline = start_level_cm if start_level_cm is not None else self.config.baseline_water_level_cm
        current_level = baseline
        pump_end_sec = pump_start_sec + p_dur
        drawdown_rate = custom_drawdown_rate if custom_drawdown_rate is not None else self.config.pump_drawdown_rate_cm_s

        for i, t in enumerate(time_steps):
            is_pump_on = (t >= pump_start_sec) and (t < pump_end_sec)
            pump_state[i] = 1 if is_pump_on else 0
            
            if is_pump_on:
                # Water level drops due to pumping
                drawdown_step = drawdown_rate * dt
                # Slight non-linear drop as pump head changes
                current_level = max(2.0, current_level - drawdown_step)
            else:
                if current_level < baseline:
                    # Inflow / simulated aquifer recharge
                    recovery_step = recharge_k * (baseline - current_level) * dt
                    current_level = min(baseline, current_level + recovery_step)

            # Ambient fluctuation
            ambient_jitter = 0.02 * np.sin(2 * np.pi * t / 60.0)
            water_level[i] = current_level + ambient_jitter

        # Ultrasonic HC-SR04 sensor reading: distance to water surface
        # Distance increases as water level decreases (drawdown)
        sensor_distance = (self.config.tank_height_cm + self.config.sensor_offset_cm) - water_level
        # Add realistic ultrasonic acoustic jitter
        sensor_distance += np.random.normal(0, self.config.sensor_noise_std_cm, size=n)
        sensor_distance = np.clip(sensor_distance, 2.0, 400.0)

        # Re-derive measured water level from distance
        measured_water_level = (self.config.tank_height_cm + self.config.sensor_offset_cm) - sensor_distance

        return {
            "time_sec": time_steps,
            "water_level_true": water_level,
            "measured_water_level": measured_water_level,
            "sensor_distance_cm": sensor_distance,
            "pump_state": pump_state,
            "is_registered": np.full(n, 1 if registered_flag else 0, dtype=int),
            "event_type": event_type,
            "pump_start_sec": pump_start_sec,
            "pump_duration_sec": p_dur,
            "recharge_rate_k": recharge_k
        }
