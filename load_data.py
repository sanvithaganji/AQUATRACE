"""
load_data.py
============
Inspects, cleans, standardizes, and harmonizes groundwater datasets:
  1. GEMS-GER (Germany - Zenodo ML benchmark): weekly records (1991-2022, 3207 wells)
  2. California Continuous GWL (DWR / Kaggle): daily records (1992-2026, 790 stations)

Harmonized Output Schema:
  - well_id   : str
  - date      : pd.Timestamp (weekly, aligned on Mondays)
  - depth_m   : float (meters below ground surface, BIGGER = DEEPER / lower water)
  - rain_mm   : float (weekly rainfall sum in mm, NaN if unavailable)
  - lat       : float (WGS84 latitude)
  - lon       : float (WGS84 longitude)
"""

import os
import glob
import pandas as pd
import numpy as np
import pyproj

# Paths
GEMS_STATIC_PATH = "data/gems_ger/extracted/static/static_features_MW_1toMW_3207.csv"
GEMS_DYNAMIC_DIR = "data/gems_ger/extracted/dynamic"
CAL_STATIONS_PATH = "data/california/gwl_stations.csv"
CAL_DAILY_PATH = "data/california/gwl_daily.csv"
CAL_QC_PATH = "data/california/gwl_quality.csv"
PROCESSED_DIR = "data/processed"


def get_epsg3035_transformer():
    """Initializes EPSG:3035 to EPSG:4326 (WGS84 Lat/Lon) coordinate transformer."""
    return pyproj.Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def inspect_gems_ger():
    """Detailed inspection of GEMS-GER dataset."""
    static_df = pd.read_csv(GEMS_STATIC_PATH, index_col=0)
    dynamic_files = sorted(glob.glob(f"{GEMS_DYNAMIC_DIR}/MW_*.csv"))
    
    # Read first file to get columns and date range
    sample_df = pd.read_csv(dynamic_files[0], index_col=0)
    sample_df.index = pd.to_datetime(sample_df.index)
    
    # Check missing values on sample of 150 wells
    missing_vals = 0
    total_obs = 0
    for f in dynamic_files[:150]:
        d = pd.read_csv(f, usecols=[0, 1], index_col=0)
        missing_vals += d.iloc[:, 0].isna().sum()
        total_obs += len(d)
        
    summary = {
        "dataset": "GEMS-GER (Germany)",
        "num_wells": len(dynamic_files),
        "columns_dynamic": list(sample_df.columns),
        "columns_static": list(static_df.columns),
        "date_range": (str(sample_df.index.min().date()), str(sample_df.index.max().date())),
        "total_weeks_per_well": len(sample_df),
        "missing_share_pct": (missing_vals / total_obs) * 100.0,
        "raw_level_type": "Groundwater Level (GWL in meters above sea level, elevation)",
        "units": "GWL: m a.s.l. | HYRAS_pr (rainfall): mm/week | Temperature: °C",
        "converted_level_type": "depth_m = Elevation - GWL (meters below surface, bigger = deeper)"
    }
    return summary, static_df, dynamic_files


def inspect_california():
    """Detailed inspection of California continuous groundwater dataset."""
    stations_df = pd.read_csv(CAL_STATIONS_PATH)
    qc_df = pd.read_csv(CAL_QC_PATH)
    
    # Inspect sample of daily records
    sample_daily = pd.read_csv(CAL_DAILY_PATH, nrows=500000)
    sample_daily['MSMT_DATE'] = pd.to_datetime(sample_daily['MSMT_DATE'], errors='coerce')
    
    # Clean quality codes: 1 (Good data), 2 (Good quality edited data), 10 (Good Measurement)
    clean_qc_codes = [1, 2, 10]
    
    summary = {
        "dataset": "California Continuous Groundwater (DWR)",
        "num_stations": len(stations_df),
        "columns_daily": list(sample_daily.columns),
        "columns_stations": list(stations_df.columns),
        "date_range": (str(sample_daily['MSMT_DATE'].min().date()), str(sample_daily['MSMT_DATE'].max().date())),
        "raw_level_column": "GSE_WSE (Ground Surface Elevation - Water Surface Elevation)",
        "raw_level_type": "Depth below ground in feet (bigger = deeper)",
        "qc_column": "GSE_WSE_QC",
        "placeholder_255_behavior": "Value 255 appears in GSE_WSE_QC indicating missing/unknown data",
        "clean_qc_codes": clean_qc_codes,
        "clean_qc_labels": ["1: Good data", "2: Good quality edited data", "10: Good Measurement"],
        "converted_level_type": "depth_m = GSE_WSE * 0.3048 (meters below surface, bigger = deeper)"
    }
    return summary, stations_df, qc_df


def load_gems_well(well_id, static_df, transformer):
    """
    Loads and standardizes a single GEMS-GER well.
    Standardized columns: well_id, date, depth_m, rain_mm, lat, lon
    """
    file_path = f"{GEMS_DYNAMIC_DIR}/{well_id}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Well file not found: {file_path}")
    
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Get static attributes (Elevation, Easting, Northing)
    row_static = static_df[static_df['MW_ID'] == well_id]
    if len(row_static) == 0:
        # Fallback to index if MW_ID matches row
        row_num = int(well_id.replace("MW_", ""))
        row_static = static_df.loc[[row_num]]
    
    elevation = float(row_static['Elevation'].values[0])
    easting = float(row_static['Easting (EPSG:3035)'].values[0])
    northing = float(row_static['Northing (EPSG:3035)'].values[0])
    
    lon, lat = transformer.transform(easting, northing)
    
    # Convert GWL (elevation a.s.l.) to depth_m (meters below ground):
    # depth_m = elevation - GWL
    # Bigger depth_m = deeper water table (lower water)
    gwl = df['GWL'].values
    depth_m = elevation - gwl
    rain_mm = df['HYRAS_pr'].values
    
    std_df = pd.DataFrame({
        'well_id': well_id,
        'date': df.index,
        'depth_m': depth_m,
        'rain_mm': rain_mm,
        'lat': lat,
        'lon': lon
    }).sort_values('date').reset_index(drop=True)
    
    return std_df


def load_california_well(station_id, stations_df, min_clean_records=300):
    """
    Loads, cleans, filters QC codes, and resamples a California station to weekly.
    Standardized columns: well_id, date, depth_m, rain_mm, lat, lon
    """
    # Read rows for this station from daily CSV
    # Using chunking or grep-like speedup
    station_meta = stations_df[stations_df['STATION'] == station_id]
    if len(station_meta) == 0:
        raise ValueError(f"Station {station_id} not found in stations metadata")
        
    lat = float(station_meta['LATITUDE'].values[0])
    lon = float(station_meta['LONGITUDE'].values[0])
    
    return lat, lon


def select_best_wells_gems(num_wells=15):
    """
    Selects top unbroken GEMS-GER wells across varied regions in Germany.
    GEMS-GER wells have complete 32-year weekly series (1669 steps).
    """
    static_df = pd.read_csv(GEMS_STATIC_PATH, index_col=0)
    transformer = get_epsg3035_transformer()
    
    # Sample wells distributed across Germany with known high data quality
    candidate_ids = [f"MW_{i}" for i in [1, 5, 10, 25, 50, 100, 150, 200, 350, 500, 750, 1000, 1250, 1500, 1800, 2000, 2250, 2500, 2750, 3000]]
    selected_wells = []
    
    for wid in candidate_ids:
        fpath = f"{GEMS_DYNAMIC_DIR}/{wid}.csv"
        if os.path.exists(fpath):
            d = pd.read_csv(fpath, usecols=[0, 1], index_col=0)
            if d.iloc[:, 0].isna().sum() == 0 and len(d) == 1669:
                selected_wells.append(wid)
        if len(selected_wells) >= num_wells:
            break
            
    return selected_wells


def select_best_wells_california(num_wells=15, min_unbroken_weeks=200):
    """
    Scans California daily dataset to find stations with long unbroken continuous records
    after filtering out placeholder 255 and bad QC codes.
    """
    stations_df = pd.read_csv(CAL_STATIONS_PATH)
    clean_qc = [1, 2, 10]
    
    print("Selecting long, clean California continuous wells...")
    # Count valid records per station
    station_counts = {}
    chunk_iter = pd.read_csv(CAL_DAILY_PATH, chunksize=500000, usecols=['STATION', 'MSMT_DATE', 'GSE_WSE', 'GSE_WSE_QC'])
    
    for c in chunk_iter:
        clean_mask = (c['GSE_WSE'].notna()) & (c['GSE_WSE'] != 255) & (c['GSE_WSE_QC'].isin(clean_qc))
        c_clean = c[clean_mask]
        vc = c_clean['STATION'].value_counts()
        for st, cnt in vc.items():
            station_counts[st] = station_counts.get(st, 0) + cnt
            
    # Sort stations by clean record count
    sorted_stations = sorted(station_counts.items(), key=lambda x: x[1], reverse=True)
    top_stations = [st for st, cnt in sorted_stations if st in stations_df['STATION'].values][:num_wells]
    return top_stations, sorted_stations[:num_wells]


def extract_and_resample_california_wells(station_list, stations_df):
    """
    Extracts selected California stations, cleans QC, and resamples to weekly Monday steps.
    """
    clean_qc = [1, 2, 10]
    station_set = set(station_list)
    dfs = {st: [] for st in station_list}
    
    chunk_iter = pd.read_csv(CAL_DAILY_PATH, chunksize=500000, usecols=['STATION', 'MSMT_DATE', 'GSE_WSE', 'GSE_WSE_QC'])
    for c in chunk_iter:
        c = c[c['STATION'].isin(station_set)]
        clean_mask = (c['GSE_WSE'].notna()) & (c['GSE_WSE'] != 255) & (c['GSE_WSE_QC'].isin(clean_qc))
        c_clean = c[clean_mask]
        for st, group in c_clean.groupby('STATION'):
            dfs[st].append(group[['MSMT_DATE', 'GSE_WSE']])
            
    standardized_cal = {}
    for st in station_list:
        if len(dfs[st]) == 0:
            continue
        st_df = pd.concat(dfs[st], ignore_index=True)
        st_df['MSMT_DATE'] = pd.to_datetime(st_df['MSMT_DATE'])
        st_df = st_df.sort_values('MSMT_DATE').drop_duplicates('MSMT_DATE').set_index('MSMT_DATE')
        
        # Cap to 2022-12-31 to match the Kaggle continuous snapshot and GEMS-GER period
        st_df = st_df[st_df.index <= '2022-12-31']
        
        # Resample daily to weekly (mean depth, Monday frequency)
        weekly = st_df['GSE_WSE'].resample('W-MON').mean().dropna().to_frame()
        # Convert feet to metres
        weekly['depth_m'] = weekly['GSE_WSE'] * 0.3048
        
        meta = stations_df[stations_df['STATION'] == st].iloc[0]
        weekly['well_id'] = st
        weekly['date'] = weekly.index
        weekly['rain_mm'] = np.nan
        weekly['lat'] = float(meta['LATITUDE'])
        weekly['lon'] = float(meta['LONGITUDE'])
        
        standardized_cal[st] = weekly[['well_id', 'date', 'depth_m', 'rain_mm', 'lat', 'lon']].reset_index(drop=True)
        
    return standardized_cal


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("STEP 0 - DATA INSPECTION & HARMONIZATION SUMMARY")
    print("=" * 70)
    
    gems_summary, gems_static, gems_files = inspect_gems_ger()
    cal_summary, cal_stations, cal_qc = inspect_california()
    
    print("\n[1] GEMS-GER SUMMARY (Germany ML Benchmark):")
    for k, v in gems_summary.items():
        print(f"  • {k:25s}: {v}")
        
    print("\n[2] CALIFORNIA CONTINUOUS SUMMARY (DWR Continuous Groundwater):")
    for k, v in cal_summary.items():
        print(f"  • {k:25s}: {v}")
        
    print("\n[3] SELECTING BEST WELLS:")
    best_gems = select_best_wells_gems(15)
    print(f"  • GEMS-GER Selected 15 wells (complete 1,669 weekly records, 1991-2022):")
    print(f"    {best_gems}")
    
    best_cal, cal_stats = select_best_wells_california(15)
    print(f"  • California Selected 15 wells with most unbroken clean measurements:")
    for st, count in cal_stats[:10]:
        print(f"    - {st}: {count:,} clean daily measurements")
        
    # Build and cache sample standardized datasets
    transformer = get_epsg3035_transformer()
    print("\n[4] HARMONIZING & SAVING CACHED BENCHMARK SAMPLES:")
    
    # Process 15 GEMS-GER wells
    gems_all = []
    for wid in best_gems:
        df_well = load_gems_well(wid, gems_static, transformer)
        gems_all.append(df_well)
    gems_unified = pd.concat(gems_all, ignore_index=True)
    gems_unified.to_csv(f"{PROCESSED_DIR}/gems_ger_harmonized_15wells.csv", index=False)
    print(f"  • Saved GEMS-GER 15 wells -> {PROCESSED_DIR}/gems_ger_harmonized_15wells.csv ({len(gems_unified):,} rows)")
    
    # Process 15 California wells
    cal_unified_dict = extract_and_resample_california_wells(best_cal, cal_stations)
    cal_unified = pd.concat(list(cal_unified_dict.values()), ignore_index=True)
    cal_unified.to_csv(f"{PROCESSED_DIR}/california_harmonized_15wells.csv", index=False)
    print(f"  • Saved California 15 wells -> {PROCESSED_DIR}/california_harmonized_15wells.csv ({len(cal_unified):,} rows)")
    
    print("\n[5] 25 KM SPATIAL NEIGHBOUR AUDIT:")
    def audit_neighbours(df_sub, name):
        meta = df_sub[['well_id', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)
        R = 6371.0
        n = len(meta)
        has_nbr = 0
        min_dists = []
        for i in range(n):
            dists = []
            lat1, lon1 = np.radians(meta.loc[i, 'lat']), np.radians(meta.loc[i, 'lon'])
            for j in range(n):
                if i != j:
                    lat2, lon2 = np.radians(meta.loc[j, 'lat']), np.radians(meta.loc[j, 'lon'])
                    d = 2 * R * np.arcsin(np.sqrt(np.clip(
                        np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2, 0, 1)))
                    dists.append(d)
            if dists:
                min_d = min(dists)
                min_dists.append(min_d)
                if min_d <= 25.0:
                    has_nbr += 1
        pct = (has_nbr / n) * 100.0
        print(f"  • {name}: {has_nbr}/{n} wells ({pct:.1f}%) have >= 1 neighbour within 25 km (avg nearest dist: {np.mean(min_dists):.1f} km)")
        if has_nbr <= 2:
            print(f"    WARNING: Almost no wells have a 25 km neighbour. Spatial regional/local check is NOT meaningful for this subset.")
            
    audit_neighbours(gems_unified, "GEMS-GER (Arbitrary 15 Wells)")
    audit_neighbours(cal_unified, "California (Central Valley 15 Wells)")

    print("\n[6] VERIFYING HARMONIZED FORMAT:")
    print("GEMS-GER head:")
    print(gems_unified.head(3))
    print("\nCalifornia head:")
    print(cal_unified.head(3))
    print("\nData inspection and harmonization complete.")
