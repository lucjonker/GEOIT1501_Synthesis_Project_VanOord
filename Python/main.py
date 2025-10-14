import geopandas as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

def calculate_descriptive_stats(df):
    """
    Calculates key descriptive statistics (min, max, mean, std, skewness, kurtosis)
    for every numeric column in a pandas DataFrame.

    Non-numeric columns (like categorical or string types) are automatically excluded.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A new DataFrame where indices are the calculated statistics and
        columns are the original DataFrame's numeric features.
    """
    # Filter for only numeric columns before calculating statistics
    numeric_df = df.select_dtypes(include=np.number)

    # Define the list of statistics we want to calculate
    # Note: 'skew' and 'kurt' are built-in pandas methods for skewness and kurtosis.
    metrics = ['min', 'max', 'mean', 'std', 'skew', 'kurt']

    # Use the .agg() method to apply all metrics across the filtered DataFrame
    # The result is a DataFrame where rows are metrics and columns are features.
    stats_df = numeric_df.agg(metrics)

    return stats_df.T

def load_DB_data(query):
    #Load data from PostGIS
    db_connection_url = "postgresql://postgres.miiedebavuhxxbzpndeq:SYbFFBRcyttS3XQy@aws-1-eu-west-3.pooler.supabase.com:5432/postgres"
    con = create_engine(db_connection_url)
    df = pd.read_sql(query, con, index_col="tid")
    df.sort_index(inplace=True)

    return df

def min_max_normalize(df, features_to_scale):
    # 1. Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # 3. Fit the scaler to the data and transform it
    # The output is a NumPy array
    normalized_data_array = scaler.fit_transform(features_to_scale)

    # 4. Create new normalized columns in your GeoDataFrame
    for norm_feature in features_to_scale:
        df[f"{norm_feature}_norm"] = normalized_data_array[:, 0]

def plot_scatter(df, x, y):
    fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(10, 6))

    df.plot(
        kind='scatter',
        x=x,
        y=y,
        ax=ax_scatter,
        color='red',
        marker='o',
        s=50,
        title=f'Scatterplot of {x} vs. {y}'
    )

    ax_scatter.set_xlabel(x)
    ax_scatter.set_ylabel(y)
    ax_scatter.grid(True, linestyle='--', alpha=0.6)

    plt.show()

def plot_histogram(df, x):
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10, 6))

    df[f"{x}"].plot(
        kind='hist',
        ax=ax_hist,
        bins=30,  # Use a good number of bins to see the shape
        edgecolor='black',
        alpha=0.7,
        color='darkblue',
        title=f'Distribution of {x}'
    )

    # Add labels for the histogram
    ax_hist.set_xlabel(f'{x} Value')
    ax_hist.set_ylabel('Frequency (Count)')
    ax_hist.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()

# Set pandas options to display all rows and columns (in its entirety)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main():
    # ----------------- Load Data -----------------
    df = load_DB_data("SELECT * FROM tile_observations JOIN observations USING (oid) WHERE scid=5 AND year=2024;")
    statistics_result = calculate_descriptive_stats(df)

    print("--- Descriptive Statistics for Features ---")
    # Use .T to transpose the result for better readability (features as rows)
    print("Note: The table is transposed for easier reading (features as rows).")
    print(statistics_result)
    print("-" * 35)

    #----------------- Data Normalization -----------------
    # 2. Select the columns you want to normalize (must be passed as a DataFrame/2D array)
    features_to_scale = df[['slope', 'roughness', 'bed_level']]
    min_max_normalize(df, features_to_scale)
    # Print the original and normalized values to check the result
    print(df.head())
    print("\n--- Original vs. Normalized Values (First 5 Rows) ---")
    print(df[['bed_level', 'bed_level_norm', 'slope', 'slope_norm', 'roughness', 'roughness_norm']].head())

    # ----------------- Correlation Analysis -----------------
    # Calculate the Pearson correlation coefficient

    correction_matrix = df[['slope', 'roughness', 'bed_level']].corr(method='pearson')
    print(f"--- Correlation between features ---")
    print(correction_matrix)

    # ----------------- Plotting -----------------
    # Scatterplot
    for i, j in combinations(['slope', 'roughness', 'bed_level'],2):
        plot_scatter(df, i, j)

    # Histogram
    for feature in ['slope', 'roughness', 'bed_level']:
        plot_histogram(df, feature)

if __name__ == '__main__':
    main()