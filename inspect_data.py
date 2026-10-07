import os
import glob
import pandas as pd
import numpy as np
import pyproj

# =====================================================================
# 1. INSPECT GEMS-GER DATASET
# =====================================================================
print("=" * 70)
print("INSPECTING GEMS-GER DATASET")
print("=" * 70)

static_path = "data/gems_ger/extracted/static/static_features_MW_1toMW_3207.csv"
dynamic_dir = "data/gems_ger/extracted/dynamic"

static_df = pd.read_csv(static_path, index_col=0)
print(f"Static features shape: {static_df.shape}")
print(f"Static features columns (first 15): {list(static_df.columns[:15])}")
print(f"Elevation stats:\n{static_df['Elevation'].describe()}")

dynamic_files = sorted(glob.glob(f"{dynamic_dir}/MW_*.csv"))
print(f"Number of dynamic well files found: {len(dynamic_files)}")

# Check a sample file
sample_file = dynamic_files[0]
sample_df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
print(f"\nSample well file: {os.path.basename(sample_file)}")
print(f"Columns in dynamic file: {list(sample_df.columns)}")
print(f"Index name: {sample_df.index.name}, Date range: {sample_df.index.min()} to {sample_df.index.max()}")
print(f"Number of weekly rows per well: {len(sample_df)}")
print(f"Sample GWL head:\n{sample_df[['GWL', 'HYRAS_pr', 'GWL_flag']].head(3)}")

# Scan 50 random or first 50 wells to get aggregate missing value share across GEMS-GER
missing_counts = []
total_counts = []
min_dates = []
max_dates = []

for f in dynamic_files[:100]:
    df = pd.read_csv(f, usecols=[0, 1], index_col=0)
    df.index = pd.to_datetime(df.index)
    missing_counts.append(df['GWL'].isna().sum())
    total_counts.append(len(df))
    min_dates.append(df.index.min())
    max_dates.append(df.index.max())

print(f"\nAggregate over first 100 GEMS-GER wells:")
print(f"Total observations sampled: {sum(total_counts)}")
print(f"Total missing GWL: {sum(missing_counts)} ({sum(missing_counts)/sum(total_counts)*100:.3f}%)")
print(f"Overall date span: {min(min_dates)} to {max(max_dates)}")

# Check coordinate system and conversion to lat/lon
transformer = pyproj.Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
sample_easting = static_df.loc[1, 'Easting (EPSG:3035)']
sample_northing = static_df.loc[1, 'Northing (EPSG:3035)']
lon, lat = transformer.transform(sample_easting, sample_northing)
print(f"Coordinate transform check for MW_1: Easting={sample_easting}, Northing={sample_northing} -> Lon={lon:.4f}, Lat={lat:.4f}")

# =====================================================================
# 2. INSPECT CALIFORNIA DATASET
# =====================================================================
print("\n" + "=" * 70)
print("INSPECTING CALIFORNIA DATASET")
print("=" * 70)

cal_stations_path = "data/california/gwl_stations.csv"
cal_daily_path = "data/california/gwl_daily.csv"
cal_qc_path = "data/california/gwl_quality.csv"

stations_df = pd.read_csv(cal_stations_path)
print(f"California stations count: {len(stations_df)}")
print(f"Stations columns: {list(stations_df.columns)}")

qc_df = pd.read_csv(cal_qc_path)
print(f"\nQuality codes table (first 10):\n{qc_df.head(10)}")

# Inspect daily measurements chunkwise or with head / value_counts
print("\nReading sample of California daily measurements...")
chunk = pd.read_csv(cal_daily_path, nrows=100000)
print(f"Daily measurements columns: {list(chunk.columns)}")
print(f"GSE_WSE head:\n{chunk[['STATION', 'MSMT_DATE', 'GSE_WSE', 'GSE_WSE_QC', 'WSE']].head(5)}")

# Let's inspect unique stations and QC distribution across a 1M row sample
print("\nScanning QC codes and placeholder 255 in California data...")
chunk_iter = pd.read_csv(cal_daily_path, chunksize=250000, usecols=['STATION', 'MSMT_DATE', 'GSE_WSE', 'GSE_WSE_QC'])
total_rows = 0
qc_counts = {}
val_255_count = 0
nan_gse_count = 0
station_set = set()
min_cal_date = None
max_cal_date = None

for i, c in enumerate(chunk_iter):
    total_rows += len(c)
    station_set.update(c['STATION'].unique())
    c_dates = pd.to_datetime(c['MSMT_DATE'], errors='coerce')
    cur_min = c_dates.min()
    cur_max = c_dates.max()
    if min_cal_date is None or cur_min < min_cal_date:
        min_cal_date = cur_min
    if max_cal_date is None or cur_max > max_cal_date:
        max_cal_date = cur_max
    
    nan_gse_count += c['GSE_WSE'].isna().sum()
    val_255_count += (c['GSE_WSE'] == 255.0).sum()
    
    # QC counts
    vc = c['GSE_WSE_QC'].value_counts().to_dict()
    for k, v in vc.items():
        qc_counts[k] = qc_counts.get(k, 0) + v
    
    if i >= 7: # Read ~2 million rows to get representative statistics quickly
        break

print(f"Sampled rows: {total_rows}")
print(f"Unique stations in sample: {len(station_set)}")
print(f"Date range in sample: {min_cal_date} to {max_cal_date}")
print(f"GSE_WSE NaN count: {nan_gse_count} ({nan_gse_count/total_rows*100:.2f}%)")
print(f"GSE_WSE == 255.0 placeholder count: {val_255_count} ({val_255_count/total_rows*100:.2f}%)")
print(f"GSE_WSE_QC distribution (top 15):")
for k, v in sorted(qc_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
    desc = qc_df.loc[qc_df['QUALITY_CODE'] == k, 'DESCRIPTION'].values
    desc_str = desc[0] if len(desc) > 0 else "Unknown"
    print(f"  Code {k}: {v:>8d} ({v/total_rows*100:>5.2f}%) - {desc_str}")
