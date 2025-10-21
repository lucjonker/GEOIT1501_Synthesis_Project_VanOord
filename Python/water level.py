import geopandas as gpd
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

db_connection_url = "postgresql://postgres.miiedebavuhxxbzpndeq:SYbFFBRcyttS3XQy@aws-1-eu-west-3.pooler.supabase.com:5432/postgres"
con = create_engine(db_connection_url)
channels = gpd.read_postgis("select * from side_channels", con, geom_col="geom", index_col="scid")

channels["geom"] = channels.geometry.centroid
channels["X"] = channels.geometry.x
channels["Y"] = channels.geometry.y

stations = pd.read_csv(
    "input/stations.txt",
    sep="\t"
)

y_threshold = 434000


def inverse_distance_weighting(x, x1, x2, v1, v2):
    """Compute inverse distance weighted average between two points"""
    if x1 == x2:
        return (v1 + v2) / 2
    d1, d2 = abs(x - x1), abs(x - x2)
    w1, w2 = 1 / d1 if d1 != 0 else 1e9, 1 / d2 if d2 != 0 else 1e9
    return (v1 * w1 + v2 * w2) / (w1 + w2)


results = []

for scid, row in channels.iterrows():
    x_c, y_c = row["X"], row["Y"]

    # Select correct set of stations (UP = 1 for upper, 0 otherwise)
    up_flag = 1 if y_c > y_threshold else 0
    stations_subset = stations[stations["UP"] == up_flag]

    # Separate left and right stations
    left_stations = stations_subset[stations_subset["X"] < x_c]
    right_stations = stations_subset[stations_subset["X"] > x_c]

    # Get nearest left and right using .loc instead of .iloc
    left = None
    right = None

    if not left_stations.empty:
        left_idx = (x_c - left_stations["X"]).abs().idxmin()
        left = left_stations.loc[left_idx]

    if not right_stations.empty:
        right_idx = (x_c - right_stations["X"]).abs().idxmin()
        right = right_stations.loc[right_idx]

    # Compute interpolated height
    if left is not None and right is not None:
        height = inverse_distance_weighting(x_c, left["X"], right["X"], left["HEIGHT"], right["HEIGHT"])
    elif left is not None:
        height = left["HEIGHT"]
    elif right is not None:
        height = right["HEIGHT"]
    else:
        height = np.nan  # No available stations

    results.append({
        "scid": scid,
        "water_level": round(height,3)
    })

# Create DataFrame with results
df_results = pd.DataFrame(results)

# Save results to CSV (optional)
df_results.to_csv("output/water_level.csv", index=False)

print(df_results.head())