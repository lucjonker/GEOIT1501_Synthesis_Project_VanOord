import geopandas as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

DB_CONNECTION_URL = "postgresql://postgres.miiedebavuhxxbzpndeq:SYbFFBRcyttS3XQy@aws-1-eu-west-3.pooler.supabase.com:5432/postgres"

def calculate_difference(scid):
    df_2024 = load_db_data(
        "SELECT tid,bed_level FROM tile_observations JOIN observations USING (oid) WHERE scid=%(scid)s AND year=2024;",
        index_col='tid',
        params={"scid": scid}
    )

    df_2025 = load_db_data(
        "SELECT tid,bed_level FROM tile_observations JOIN observations USING (oid) WHERE scid=%(scid)s AND year=2025;",
        index_col='tid',
        params={"scid": scid}
    )

    df = pd.merge(df_2024, df_2025, on='tid', how='inner')
    df.dropna(inplace=True)
    df["change"] = df["bed_level_y"] - df["bed_level_x"]
    return df


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


def load_db_data(query, index_col, params=None):
    #Load data from PostGIS
    con = create_engine(DB_CONNECTION_URL)
    df = pd.read_sql(query, con, index_col=index_col, params=params)
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


def plot_scatter(df, x, y, scid):
    num_points= df.shape[0]
    fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(10, 6))

    df.plot(
        kind='scatter',
        x=x,
        y=y,
        ax=ax_scatter,
        color='red',
        marker='o',
        s=50,
        title=f'Scatterplot of {x} vs. {y} for SCID: {scid} (N={num_points} points)'
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


def get_usable_scids(year1, year2):
    """
    Queries the database to find side channel IDs (scid) that have
    corresponding tile observations (tid) in both the specified years.

    Args:
        year1 (int): The first year to check for data availability.
        year2 (int): The second year to check for data availability.

    Returns:
        list: A list of usable SCIDs (integers).
    """
    # SQL query to find scid that have bed_level values in both years.
    query = """
            SELECT scid
            FROM tile_observations
                     JOIN observations USING (oid)
            WHERE year IN (%(year1)s \
                , %(year2)s)
            GROUP BY scid
            HAVING COUNT (DISTINCT year) = 2; \
            """

    # Parameters dictionary to map placeholders to Python variables
    params = {
        'year1': year1,
        'year2': year2
    }

    # Database connection setup (using the connection URL defined in your existing code)
    con = create_engine(DB_CONNECTION_URL)

    # Read the data using the query and parameters
    df_scids = pd.read_sql(query, con, params=params)

    # Extract the 'scid' column and convert to a list of integers
    usable_scids = df_scids['scid'].astype(int).tolist()

    return usable_scids

def make_scatterplot_for_each_channel(feature):
    usable_scids = get_usable_scids(2024, 2025)
    for scid in usable_scids:
        print(f"Processing SCID: {scid}")

        df_diff = calculate_difference(scid=scid)

        # ----------------- Load Data -----------------
        df = load_db_data(
            "SELECT * FROM tile_observations JOIN observations USING (oid) WHERE scid=%(scid)s AND year=2024;",
            index_col= 'tid',
            params={"scid": scid}
            )
        df['change'] = df_diff['change']

        plot_scatter(df, feature, 'change', scid)

def make_plots_for_tiles_in_one_channel(scid, features):
    df_diff = calculate_difference(scid=scid)

    # ----------------- Load Data -----------------
    df = load_db_data(
        "SELECT * FROM tile_observations JOIN observations USING (oid) WHERE scid=%(scid)s AND year=2024;",
        index_col = 'tid',
        params={"scid": scid}
        )
    # df = load_db_data("SELECT * FROM side_channels;")
    df['change'] = df_diff['change']

    # export df to excel
    print(df.head(50))
    # df.to_excel(f"Output/scid_{scid}_data.xlsx")

    statistics_result = calculate_descriptive_stats(df)
    print("--- Descriptive Statistics for Features ---")
    print("Note: The table is transposed for easier reading (features as rows).")
    print(statistics_result)
    print("-" * 35)

    #----------------- Data Normalization -----------------
    # 2. Select the columns you want to normalize (must be passed as a DataFrame/2D array)
    features_to_scale = df[features]
    min_max_normalize(df, features_to_scale)
    # Print the original and normalized values to check the result
    print(df.head())
    print("\n--- Original vs. Normalized Values (First 5 Rows) ---")
    print(df[['bed_level', 'bed_level_norm', 'slope', 'slope_norm', 'roughness', 'roughness_norm']].head())

    # ----------------- Correlation Analysis -----------------
    # Calculate the Pearson correlation coefficient
    correction_matrix = df[features].corr(method='pearson')
    print(f"--- Correlation between features ---")
    print(correction_matrix)

    # ----------------- Plotting -----------------

    # Scatterplot
    for i, j in combinations(features,2):
        plot_scatter(df, i, j, scid)

    # Histogram
    for feature in features:
        plot_histogram(df, feature)

# Set pandas options to display all rows and columns (in its entirety)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main():
    # Define the features to analyze
    # features = ['area', 'perimeter', 'channel_length'] # global features
    features =  ['bed_level', 'slope', 'roughness', 'aspect'] # local features

    # make_plots_for_tiles_in_one_channel(5, features=features)

    for feature in features:
        make_scatterplot_for_each_channel(feature)

if __name__ == '__main__':
    main()