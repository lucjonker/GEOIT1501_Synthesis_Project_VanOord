import geopandas as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from itertools import combinations
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


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


def normalize_features(df, features, method='min_max'):
    """
    Applies MinMax, Standard, or Robust scaling to features and
    returns a new DataFrame with only the normalized columns.
    """

    # Select scaler based on method
    if method == 'min_max':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Invalid method: '{method}'")

    # Fit and transform the data
    data_to_scale = df[features].copy()
    scaled_array = scaler.fit_transform(data_to_scale)

    # Create normalized DataFrame
    df_normalized = pd.DataFrame(
        scaled_array,
        columns=features,
        index=df.index
    )

    return df_normalized


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


def transform_circular_aspect(df):
    """
    Transforms the circular 'aspect' column (in degrees) into two linear
    Cartesian components ('aspect_x' and 'aspect_y') using sine and cosine.
    The original 'aspect' column is dropped from the DataFrame.
    """
    if 'aspect' not in df.columns:
        return df

    # Convert degrees to radians (numpy trigonometric functions require radians)
    df['aspect_rad'] = np.deg2rad(df['aspect'])

    # Calculate the sine and cosine components
    df['aspect_x'] = np.cos(df['aspect_rad']) # cos(theta)
    df['aspect_y'] = np.sin(df['aspect_rad']) # sin(theta)

    # Drop the original circular column and the temporary radians column
    df.drop(columns=['aspect', 'aspect_rad'], inplace=True)

    return df


def transform_circular_flow(df):
    """
    Transforms the circular 'aspect' column (in degrees) into two linear
    Cartesian components ('aspect_x' and 'aspect_y') using sine and cosine.
    The original 'aspect' column is dropped from the DataFrame.
    """
    if 'flow_direction' not in df.columns:
        return df

    # Convert degrees to radians (numpy trigonometric functions require radians)
    df['flow_rad'] = np.deg2rad(df['flow_direction'])

    # Calculate the sine and cosine components
    df['flow_x'] = np.cos(df['flow_rad']) # cos(theta)
    df['flow_y'] = np.sin(df['flow_rad']) # sin(theta)

    # Drop the original circular column and the temporary radians column
    df.drop(columns=['flow_direction', 'flow_rad'], inplace=True)

    return df


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


def correlation_matrix(df, features):
    """Computes and displays the correlation matrix heatmap."""

    matrix = df[features].corr()

    fig_heatmap, ax_heatmap = plt.subplots(1, 1, figsize=(8, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap='coolwarm',
        linewidths=.5,
        linecolor='white',
        ax=ax_heatmap
    )

    ax_heatmap.set_title("Correlation Matrix", fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)

    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()


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

    #----------------- Display correlation matrix -----------------
    correlation_matrix(df,['bed_level', 'slope', 'roughness', 'aspect'])

    #----------------- Cosine-sine encode aspect feature -----------------
    print(f"Before transforming circular aspect:")
    print(df.head(10))
    df = transform_circular_aspect(df)
    print(f"After transforming circular aspect:")
    print(df.head(10))

    # export df to excel
    # df.to_excel(f"Output/scid_{scid}_data.xlsx")

    #----------------- Calculate Statistics -----------------
    statistics_result = calculate_descriptive_stats(df)
    print("--- Descriptive Statistics for Features ---")
    print(statistics_result)

    #----------------- Data Normalization -----------------
    # 2. Select the columns you want to normalize (must be passed as a DataFrame/2D array)
    df_norm = normalize_features(df, features, method='min_max') # method = min_max / standard / robust
    print("Normalization Result:")
    print(df_norm.head(10))

    # ----------------- Plotting -----------------

    # # Scatterplot
    # for i, j in combinations(features,2):
    #     plot_scatter(df, i, j, scid)
    #     # plot_scatter(df_norm, i, j, scid)
    #
    # # Histogram
    # for feature in features:
    #     plot_histogram(df, feature)
    #     # plot_histogram(df_norm, feature)

# Set pandas options to display all rows and columns (in its entirety)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def calculate_collinearity_stats(scid, features):
    """
    Calculates Variance Inflation Factor (VIF) and Tolerance
    for a given set of predictor variables in a DataFrame.
    """
    df_diff = calculate_difference(scid=scid)

    # ----------------- Load Data -----------------
    df = load_db_data(
        "SELECT * FROM tile_observations JOIN observations USING (oid) WHERE scid=%(scid)s AND year=2024;",
        index_col='tid',
        params={"scid": scid}
    )
    # df = load_db_data("SELECT * FROM side_channels;")
    df['change'] = df_diff['change']

    df = transform_circular_aspect(df)

    # 1. Create a new DataFrame 'X' containing only predictors
    X = df[features].dropna()

    # 2. Add a constant (intercept) term
    X_with_const = add_constant(X)

    # 3. Create a DataFrame to store the VIF results
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns

    # 4. Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                       for i in range(X_with_const.shape[1])]

    # 5. Calculate Tolerance (1 / VIF)
    vif_data["Tolerance"] = 1 / vif_data["VIF"].replace(0, float('inf'))

    # 6. Filter out the 'const' row before returning
    vif_data = vif_data[vif_data.Feature != 'const'].reset_index(drop=True)

    return vif_data

def main():
    # Define the features to analyze
    # features = ['area', 'perimeter', 'channel_length'] # global features
    features =  ['bed_level', 'slope', 'roughness', 'aspect_x', 'aspect_y'] # local features

    vif_data = calculate_collinearity_stats(5, features)
    print(vif_data)

    # make_plots_for_tiles_in_one_channel(5, features=features)

    # for feature in features:
    #     make_scatterplot_for_each_channel(feature)

if __name__ == '__main__':
    main()