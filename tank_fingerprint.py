"""
tank_fingerprint.py
===================
Audited Hardware Demo Tank Fingerprint & Pump Event Anomaly Detection:
  1. Harder Tank Simulator:
       - 100 normal events (70 train, 30 held-out test). Runs ~2 min (100-130s), normal recharge.
       - 30 harder abnormal events:
           a) Slightly longer runs (160-210s, i.e. 2.5-3.5 min, NOT 8-10 min).
           b) Normal-duration runs with slow recharge from throttled valve (k ~ 0.003/s).
           c) Abnormal runs at NORMAL daytime hours (time of day provides zero separation).
  2. Feature Ablation Experiment:
       - Run A: WITH start_hour feature.
       - Run B: WITHOUT start_hour feature (strictly hydrological: depth, drop rate, fall time, recovery %).
  3. Disclaimer:
       Simulated tank results demonstrate code execution and pipeline integrity only,
       NOT real-world hardware performance.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest

SEED = 42
np.random.seed(SEED)


def simulate_harder_tank_telemetry(n_normal=100, n_abnormal=30, sampling_sec=1):
    """
    Simulates harder, more realistic tank water level telemetry:
      - Normal events: 100 (70 train, 30 test)
      - Abnormal events: 30 (held-out test)
    """
    records = []
    event_metadata = []
    
    current_time = datetime(2026, 1, 1, 6, 0, 0)
    baseline_level = 100.0  # Full tank equilibrium = 100 cm
    
    # 70 train normal, followed by test pool (30 test normal + 30 test abnormal shuffled)
    normal_train = [("normal", False)] * 70
    test_pool = [("normal", False)] * 30
    
    # 30 harder abnormal events: 10 slight-overpump, 10 slow-valve, 10 normal-hour-draw
    abnormal_types = ["slight_excess_runtime"] * 10 + ["slow_valve_recovery"] * 10 + ["normal_hour_draw"] * 10
    test_pool.extend([(t, True) for t in abnormal_types])
    
    np.random.seed(SEED)
    np.random.shuffle(test_pool)
    
    all_events_plan = normal_train + test_pool
    current_level = baseline_level
    event_idx = 0
    
    for sub_type, is_abnormal in all_events_plan:
        event_idx += 1
        
        # Idle schedule interval
        if is_abnormal and sub_type == "slight_excess_runtime" and np.random.rand() > 0.5:
            # Advance to next day early morning
            current_time = (current_time + timedelta(days=1)).replace(
                hour=int(np.random.choice([1, 2, 3])),
                minute=int(np.random.randint(0, 59)),
                second=0
            )
        else:
            # Normal daytime schedule (morning 7-9 AM or evening 17-19 PM)
            idle_hours = float(np.random.uniform(4.0, 7.0))
            current_time += timedelta(hours=idle_hours)
            
        # Idle records leading to pump
        idle_duration = 30
        for _ in range(idle_duration):
            noise = np.random.normal(0, 0.04)
            records.append({'time': current_time, 'level_cm': baseline_level + noise, 'pump_on': 0})
            current_time += timedelta(seconds=sampling_sec)
            
        start_time = current_time
        start_level = baseline_level
        
        # Pumping physics parameters
        if not is_abnormal:
            # Normal: ~2 min (100 to 130s), k = 0.008/s
            pump_duration_sec = int(np.random.randint(100, 131))
            q_pump = float(np.random.uniform(0.18, 0.22))
            k_recharge = float(np.random.uniform(0.0075, 0.0085))
        else:
            if sub_type == "slight_excess_runtime":
                # Subtle over-pump: 160 to 210s (only 2.5 to 3.5 min, NOT 8-10 min)
                pump_duration_sec = int(np.random.randint(160, 211))
                q_pump = float(np.random.uniform(0.19, 0.22))
                k_recharge = float(np.random.uniform(0.0075, 0.0085))
            elif sub_type == "slow_valve_recovery":
                # Normal runtime (105-125s), but throttled recharge valve (k ~ 0.003/s)
                pump_duration_sec = int(np.random.randint(105, 126))
                q_pump = float(np.random.uniform(0.18, 0.22))
                k_recharge = float(np.random.uniform(0.0028, 0.0035))
            else:
                # Normal-hour draw with moderate increase (145-175s)
                pump_duration_sec = int(np.random.randint(145, 176))
                q_pump = float(np.random.uniform(0.20, 0.23))
                k_recharge = float(np.random.uniform(0.0055, 0.0065))
                
        # Pumping phase
        level = start_level
        for _ in range(pump_duration_sec):
            dh = -q_pump + k_recharge * (baseline_level - level) + np.random.normal(0, 0.06)
            level = max(level + dh, 10.0)
            records.append({'time': current_time, 'level_cm': round(level, 2), 'pump_on': 1})
            current_time += timedelta(seconds=sampling_sec)
            
        stop_time = current_time
        
        # Recovery phase (350 seconds)
        recovery_sec = 350
        for _ in range(recovery_sec):
            dh = k_recharge * (baseline_level - level) + np.random.normal(0, 0.04)
            level = min(level + dh, baseline_level)
            records.append({'time': current_time, 'level_cm': round(level, 2), 'pump_on': 0})
            current_time += timedelta(seconds=sampling_sec)
            
        event_metadata.append({
            'event_id': event_idx,
            'start_time': str(start_time),
            'stop_time': str(stop_time),
            'is_abnormal': int(is_abnormal),
            'anomaly_type': sub_type,
            'pump_duration_sec': pump_duration_sec
        })
        
    df_telemetry = pd.DataFrame(records)
    df_events = pd.DataFrame(event_metadata)
    return df_telemetry, df_events


def extract_pump_events(df_telemetry):
    """Slices events using pump_on transitions only."""
    df = df_telemetry.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    pump_diff = df['pump_on'].diff().fillna(0)
    starts = df.index[pump_diff == 1].tolist()
    stops = df.index[pump_diff == -1].tolist()
    
    events = []
    stop_idx_ptr = 0
    for s_idx in starts:
        while stop_idx_ptr < len(stops) and stops[stop_idx_ptr] <= s_idx:
            stop_idx_ptr += 1
        if stop_idx_ptr < len(stops):
            e_idx = stops[stop_idx_ptr]
            events.append((s_idx, e_idx))
            stop_idx_ptr += 1
            
    return df, events


def compute_event_features(df, events):
    """Extracts features strictly from water level time series."""
    feature_rows = []
    for i, (s_idx, e_idx) in enumerate(events, 1):
        start_time = df.loc[s_idx, 'time']
        stop_time = df.loc[e_idx, 'time']
        
        pumping_levels = df.loc[s_idx:e_idx, 'level_cm'].values
        pumping_times = df.loc[s_idx:e_idx, 'time']
        
        level_start = pumping_levels[0]
        min_level = np.min(pumping_levels)
        min_pos = np.argmin(pumping_levels)
        
        drawdown_depth = max(level_start - min_level, 0.1)
        fall_duration = max((pumping_times.iloc[min_pos] - start_time).total_seconds(), 1.0)
        drop_rate = drawdown_depth / fall_duration
        
        post_5min_time = stop_time + timedelta(seconds=300)
        post_slice = df[(df['time'] >= stop_time) & (df['time'] <= post_5min_time)]
        
        if len(post_slice) > 0:
            level_at_5min = post_slice['level_cm'].iloc[-1]
            recovered_cm = level_at_5min - min_level
            recovery_5min_pct = np.clip((recovered_cm / drawdown_depth) * 100.0, 0.0, 100.0)
        else:
            recovery_5min_pct = 0.0
            
        start_hour = start_time.hour
        
        feature_rows.append({
            'event_id': i,
            'start_time': str(start_time),
            'stop_time': str(stop_time),
            'drawdown_depth_cm': float(drawdown_depth),
            'drop_rate_cm_s': float(drop_rate),
            'fall_duration_s': float(fall_duration),
            'recovery_5min_pct': float(recovery_5min_pct),
            'start_hour': int(start_hour)
        })
        
    return pd.DataFrame(feature_rows)


def score_to_risk(decision_score, normal_scores):
    """Converts Isolation Forest score to 0-100 risk score."""
    p05 = np.percentile(normal_scores, 5)
    p50 = np.percentile(normal_scores, 50)
    scale = (p50 - p05) if (p50 - p05) > 1e-4 else 0.1
    z = (p05 - decision_score) / scale
    risk = 100.0 / (1.0 + np.exp(-1.8 * z))
    return np.clip(risk, 0.0, 100.0)


def evaluate_tank_experiment(features_df, feature_cols, experiment_name):
    """Trains on 70 normal events, evaluates on 30 test normal + 30 test abnormal."""
    train_normal = features_df[features_df['is_abnormal'] == 0].iloc[:70]
    test_normal = features_df[features_df['is_abnormal'] == 0].iloc[70:]
    test_abnormal = features_df[features_df['is_abnormal'] == 1]
    
    iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=SEED)
    iso.fit(train_normal[feature_cols])
    
    train_scores = iso.decision_function(train_normal[feature_cols])
    all_scores = iso.decision_function(features_df[feature_cols])
    
    risk_scores = np.array([score_to_risk(s, train_scores) for s in all_scores])
    features_df['risk_score'] = risk_scores
    flagged = (risk_scores >= 50.0).astype(int)
    features_df['flagged'] = flagged
    
    test_norm_flags = flagged[test_normal.index]
    test_abnorm_flags = flagged[test_abnormal.index]
    
    caught = test_abnorm_flags.sum()
    caught_pct = (caught / len(test_abnormal)) * 100.0
    
    fa = test_norm_flags.sum()
    fa_pct = (fa / len(test_normal)) * 100.0
    
    mean_norm_risk = risk_scores[features_df['is_abnormal'] == 0].mean()
    mean_abnorm_risk = risk_scores[features_df['is_abnormal'] == 1].mean()
    
    return {
        'experiment': experiment_name,
        'features_used': feature_cols,
        'train_normal_count': len(train_normal),
        'test_normal_count': len(test_normal),
        'abnormal_count': len(test_abnormal),
        'abnormal_caught_count': int(caught),
        'abnormal_caught_pct': float(caught_pct),
        'false_alarms_count': int(fa),
        'false_alarm_pct': float(fa_pct),
        'mean_normal_risk': float(mean_norm_risk),
        'mean_abnormal_risk': float(mean_abnorm_risk)
    }


def run_harder_tank_pipeline():
    """Executes the harder tank benchmark and ablation study."""
    print("=" * 80)
    print("RUNNING AUDITED HARDER TANK EXPERIMENT (100 Normal, 30 Harder Abnormal)")
    print("=" * 80)
    
    # Generate new harder dataset
    df_tel, df_gt = simulate_harder_tank_telemetry(n_normal=100, n_abnormal=30)
    df_clean, events = extract_pump_events(df_tel)
    features_df = compute_event_features(df_clean, events)
    features_df = pd.merge(features_df, df_gt[['event_id', 'is_abnormal', 'anomaly_type']], on='event_id', how='left')
    
    # Save simulated telemetry
    os.makedirs("data/processed", exist_ok=True)
    df_tel.to_csv("data/processed/tank_telemetry.csv", index=False)
    
    # 1. Run A: WITH start_hour
    cols_with_hour = ['drawdown_depth_cm', 'drop_rate_cm_s', 'fall_duration_s', 'recovery_5min_pct', 'start_hour']
    res_with_hour = evaluate_tank_experiment(features_df, cols_with_hour, "Run A (With start_hour)")
    
    # 2. Run B: WITHOUT start_hour (Level features only)
    cols_no_hour = ['drawdown_depth_cm', 'drop_rate_cm_s', 'fall_duration_s', 'recovery_5min_pct']
    res_no_hour = evaluate_tank_experiment(features_df, cols_no_hour, "Run B (WITHOUT start_hour - Level Only)")
    
    print(f"{'Experiment':<38} | {'Abnormal Caught':<17} | {'False Alarms':<14} | {'Norm Risk':<10} | {'Abnorm Risk':<10}")
    print("-" * 80)
    print(f"{res_with_hour['experiment']:<38} | {res_with_hour['abnormal_caught_count']}/{res_with_hour['abnormal_count']} ({res_with_hour['abnormal_caught_pct']:.1f}%) | "
          f"{res_with_hour['false_alarms_count']}/{res_with_hour['test_normal_count']} ({res_with_hour['false_alarm_pct']:.1f}%) | "
          f"{res_with_hour['mean_normal_risk']:<10.1f} | {res_with_hour['mean_abnormal_risk']:<10.1f}")
    print(f"{res_no_hour['experiment']:<38} | {res_no_hour['abnormal_caught_count']}/{res_no_hour['abnormal_count']} ({res_no_hour['abnormal_caught_pct']:.1f}%) | "
          f"{res_no_hour['false_alarms_count']}/{res_no_hour['test_normal_count']} ({res_no_hour['false_alarm_pct']:.1f}%) | "
          f"{res_no_hour['mean_normal_risk']:<10.1f} | {res_no_hour['mean_abnormal_risk']:<10.1f}")
    print("-" * 80)
    print("\n[IMPORTANT AUDIT NOTE]:")
    print(">> Note: Simulated tank results demonstrate code execution and mathematical pipeline")
    print("   integrity only, not real-world hardware performance.\n")
    
    # Save results
    metrics = {
        'with_start_hour': res_with_hour,
        'without_start_hour': res_no_hour,
        'audit_disclaimer': "Simulated tank results demonstrate code execution and pipeline integrity only, not real-world hardware performance."
    }
    with open("models/tank_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    features_df.to_csv("data/processed/tank_events_scored.csv", index=False)
    return metrics


if __name__ == "__main__":
    run_harder_tank_pipeline()
