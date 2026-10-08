"""
Unified Data Preprocessing Pipeline for Groundwater Fingerprint Telemetry
========================================================================
Used identically across:
  - Training dataset generation
  - Offline robustness evaluation & stress testing
  - Live streaming ESP32-S3 IoT sensor packets

Preprocessing Order:
  a. Convert sensor distance to water level using calibrated tank height
  b. Drop impossible readings: outside tank range, blind zone (<2 cm), or jump > max_rate_cm_s
  c. Remove spikes with Hampel filter (median & MAD based), not a rolling median
  d. Fill short gaps (<= 3 samples) via linear interpolation; mark > 3 samples as bad data
  e. Resample to a fixed time step
  f. Estimate baseline from median of pre-pump window, and final level from median of last 10 points
  g. Log repaired readings; discard events where > 20% of readings were repaired
  h. Fit scalers/thresholds on training data only
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List


def hampel_filter_1d(
    values: np.ndarray,
    k: int = 3,
    n_sigmas: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hampel filter (Median Absolute Deviation outlier detector).
    Preserves true slope while eliminating isolated ultrasonic ping spikes.
    
    Returns:
        filtered_values: Array with detected spikes replaced by local median
        spike_mask: Boolean array indicating which points were identified as spikes
    """
    n = len(values)
    filtered = values.copy()
    spike_mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        # Window boundaries
        start = max(0, i - k)
        end = min(n, i + k + 1)
        window = values[start:end]
        
        # Valid non-nan window points
        valid_win = window[~np.isnan(window)]
        if len(valid_win) == 0:
            continue
            
        med = float(np.median(valid_win))
        # Median Absolute Deviation (scaled for normal distribution consistency)
        mad = float(1.4826 * np.median(np.abs(valid_win - med)))
        threshold = n_sigmas * max(mad, 1e-4)
        
        val = values[i]
        if not np.isnan(val) and abs(val - med) > threshold:
            spike_mask[i] = True
            filtered[i] = med
            
    return filtered, spike_mask


def preprocess_event_telemetry(
    time_sec: np.ndarray,
    sensor_distance_cm: Optional[np.ndarray] = None,
    water_level_cm: Optional[np.ndarray] = None,
    pump_state: Optional[np.ndarray] = None,
    calibrated_tank_height_cm: float = 35.0,
    sensor_offset_cm: float = 3.0,
    blind_zone_cm: float = 2.0,
    max_jump_rate_cm_s: float = 3.0,
    hampel_k: int = 3,
    hampel_n_sigmas: float = 3.0,
    max_gap_samples: int = 3,
    target_dt: float = 1.0,
    max_repair_pct_limit: float = 20.0
) -> Dict[str, Any]:
    """
    Executes the strict 8-step preprocessing sequence on time-series telemetry.
    """
    time_arr = np.asarray(time_sec, dtype=float)
    total_samples = len(time_arr)
    
    if pump_state is None:
        pump_state = np.zeros(total_samples, dtype=int)
    else:
        pump_state = np.asarray(pump_state, dtype=int)
        
    total_height = calibrated_tank_height_cm + sensor_offset_cm
    
    # -----------------------------------------------------------------
    # Step a: Convert sensor distance to water level using calibrated tank height
    # -----------------------------------------------------------------
    if sensor_distance_cm is not None:
        raw_dist = np.asarray(sensor_distance_cm, dtype=float).copy()
        raw_level = total_height - raw_dist
    else:
        raw_level = np.asarray(water_level_cm, dtype=float).copy()
        raw_dist = total_height - raw_level
        
    working_level = raw_level.copy()
    repaired_mask = np.zeros(total_samples, dtype=bool)
    
    # -----------------------------------------------------------------
    # Step b: Drop impossible readings
    #   - Outside tank range (0 to calibrated_tank_height_cm)
    #   - Closer than blind_zone_cm (2.0 cm)
    #   - Jumping more than max_jump_rate_cm_s
    # -----------------------------------------------------------------
    impossible_mask = np.zeros(total_samples, dtype=bool)
    
    for i in range(total_samples):
        d_val = raw_dist[i]
        l_val = working_level[i]
        
        # Check if already NaN
        if np.isnan(d_val) or np.isnan(l_val):
            impossible_mask[i] = True
            working_level[i] = np.nan
            continue
            
        # 1. Blind zone check
        if d_val < blind_zone_cm:
            impossible_mask[i] = True
            working_level[i] = np.nan
            continue
            
        # 2. Outside tank range
        if l_val < 0.0 or l_val > calibrated_tank_height_cm:
            impossible_mask[i] = True
            working_level[i] = np.nan
            continue
            
    # Jump rate check on valid points
    valid_indices = np.where(~np.isnan(working_level))[0]
    for idx_prev, idx_curr in zip(valid_indices[:-1], valid_indices[1:]):
        dt = max(1e-3, time_arr[idx_curr] - time_arr[idx_prev])
        rate = abs(working_level[idx_curr] - working_level[idx_prev]) / dt
        if rate > max_jump_rate_cm_s:
            # Drop the jumping point
            impossible_mask[idx_curr] = True
            working_level[idx_curr] = np.nan
            
    repaired_mask |= impossible_mask

    # -----------------------------------------------------------------
    # Step c: Remove spikes with a Hampel filter (median-based)
    # -----------------------------------------------------------------
    level_hampel, spike_mask = hampel_filter_1d(working_level, k=hampel_k, n_sigmas=hampel_n_sigmas)
    repaired_mask |= spike_mask
    working_level = level_hampel

    # -----------------------------------------------------------------
    # Step d: Check gap lengths and fill short gaps (<= 3 samples)
    # -----------------------------------------------------------------
    nan_mask = np.isnan(working_level)
    max_gap_found = 0
    current_gap = 0
    
    for is_nan in nan_mask:
        if is_nan:
            current_gap += 1
            max_gap_found = max(max_gap_found, current_gap)
        else:
            current_gap = 0
            
    if max_gap_found > max_gap_samples:
        return {
            "status": "bad_data",
            "is_bad_data": True,
            "discard_reason": f"Sensor gap of {max_gap_found} consecutive samples exceeds limit ({max_gap_samples})",
            "readings_fixed": int(np.sum(repaired_mask)),
            "total_readings": total_samples,
            "repair_pct": float(np.sum(repaired_mask) / max(1, total_samples) * 100.0),
            "max_gap": max_gap_found
        }
        
    # Linear interpolation for short gaps
    series = pd.Series(working_level)
    interpolated = series.interpolate(method="linear").bfill().ffill().to_numpy()
    # Any point that was NaN and got filled counts as repaired
    repaired_mask |= nan_mask
    working_level = interpolated

    # -----------------------------------------------------------------
    # Step e: Resample to a fixed time step
    # -----------------------------------------------------------------
    t_start = time_arr[0]
    t_end = time_arr[-1]
    resampled_times = np.arange(t_start, t_end + 1e-6, target_dt)
    
    if len(resampled_times) > 1 and (len(resampled_times) != total_samples or np.max(np.abs(np.diff(time_arr) - target_dt)) > 1e-4):
        resampled_level = np.interp(resampled_times, time_arr, working_level)
        resampled_pump = np.interp(resampled_times, time_arr, pump_state).round().astype(int)
    else:
        resampled_times = time_arr
        resampled_level = working_level
        resampled_pump = pump_state

    # -----------------------------------------------------------------
    # Step f: Baseline median of pre-pump window & final level from last 10 points
    # -----------------------------------------------------------------
    pump_indices = np.where(resampled_pump > 0)[0]
    if len(pump_indices) > 0:
        pump_start_idx = pump_indices[0]
        pre_window = resampled_level[:pump_start_idx]
        baseline_level = float(np.median(pre_window)) if len(pre_window) > 0 else float(resampled_level[0])
    else:
        baseline_level = float(np.median(resampled_level[:min(15, len(resampled_level))]))
        
    last_10_pts = resampled_level[-10:] if len(resampled_level) >= 10 else resampled_level
    final_level = float(np.median(last_10_pts))

    # -----------------------------------------------------------------
    # Step g: Log fixed readings and discard if > 20% repaired
    # -----------------------------------------------------------------
    readings_fixed = int(np.sum(repaired_mask))
    repair_pct = float((readings_fixed / max(1, total_samples)) * 100.0)
    
    is_discarded = repair_pct > max_repair_pct_limit
    discard_reason = f"Repaired {repair_pct:.1f}% of readings exceeds {max_repair_pct_limit}% limit" if is_discarded else None

    return {
        "status": "bad_data" if is_discarded else "ok",
        "is_bad_data": is_discarded,
        "discard_reason": discard_reason,
        "readings_fixed": readings_fixed,
        "total_readings": total_samples,
        "repair_pct": round(repair_pct, 2),
        "time_sec": resampled_times,
        "water_level_cm": resampled_level,
        "pump_state": resampled_pump,
        "baseline_level_cm": round(baseline_level, 3),
        "final_level_cm": round(final_level, 3),
        "max_gap": max_gap_found
    }
