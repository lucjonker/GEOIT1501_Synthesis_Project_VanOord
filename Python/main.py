import geopandas
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

#Load data from PostGIS
db_connection_url = "postgresql://postgres.miiedebavuhxxbzpndeq:SYbFFBRcyttS3XQy@aws-1-eu-west-3.pooler.supabase.com:5432/postgres"
con = create_engine(db_connection_url)
sql = "SELECT * FROM side_channels"
df = geopandas.read_postgis(sql, con, index_col="scid")
df.sort_index(inplace=True)

#----------------- Data Normalization -----------------

# 1. Initialize the MinMaxScaler
scaler = MinMaxScaler()

# 2. Select the columns you want to normalize (must be passed as a DataFrame/2D array)
features_to_scale = df[['perimeter', 'area']]

# 3. Fit the scaler to the data and transform it
# The output is a NumPy array
normalized_data_array = scaler.fit_transform(features_to_scale)

# 4. Create new normalized columns in your GeoDataFrame
# We use the same column names but add '_norm' to them
df['perimeter_norm'] = normalized_data_array[:, 0]
df['area_norm'] = normalized_data_array[:, 1]

# Print the original and normalized values to check the result
print(df.head())
print("\n--- Original vs. Normalized Values (First 5 Rows) ---")
print(df[['perimeter', 'perimeter_norm', 'area', 'area_norm']].head())

# ----------------- Correlation Analysis -----------------

# 1. Calculate the Pearson correlation coefficient
correlation_matrix = df[['area', 'perimeter']].corr()

print("--- Correlation Matrix ---")
print(correlation_matrix)


# ----------------- Plotting -----------------

# --- Plot 1: Scatterplot of Normalized Data ---
fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(10, 6))

df.plot(
    kind='scatter',
    x='perimeter_norm',
    y='area_norm',
    ax=ax_scatter,
    color='red',
    marker='o',
    s=50,
    title='Scatterplot of Normalized Area vs. Normalized Perimeter'
)

ax_scatter.set_xlabel('Perimeter (Normalized)')
ax_scatter.set_ylabel('Area (Normalized)')
ax_scatter.grid(True, linestyle='--', alpha=0.6)

plt.show()

# --- Plot 2: Histogram of Perimeter (Original Data) ---
fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10, 6))

df['perimeter'].plot(
    kind='hist',
    ax=ax_hist,
    bins=30, # Use a good number of bins to see the shape
    edgecolor='black',
    alpha=0.7,
    color='darkblue',
    title='Distribution of Channel Perimeter (Original Data)'
)

# Add labels for the histogram
ax_hist.set_xlabel('Perimeter Value')
ax_hist.set_ylabel('Frequency (Count)')
ax_hist.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()