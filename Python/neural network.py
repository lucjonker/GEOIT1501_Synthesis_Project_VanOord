from main import load_db_data, transform_circular_feature
import pandas as pd
import geopandas as gpd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
import numpy as np
import folium

def main():
    side_channels = "11, 17, 36, 40, 43, 49"
    df = prepare_data(side_channels,2024,2025)
    df = run_model(df,feature_reduction="pca",features_n=6,size=2000,runs=5)
    html_visualizer(df, side_channels)

def prepare_data(side_channels,year_1,year_2):
    """
    Prepares and consolidates all relevant data for the specified side channels and years.

    Parameters
    ----------
    side_channels : str
        Comma-separated list of side channel IDs to include.
    year_1 : int
        First observation year (used as baseline for change calculation).
    year_2 : int
        Second observation year (used to compute bed-level change).

    Returns
    -------
    pandas.DataFrame
    """

    # load all the database tables into pandas
    df_year_1 = load_db_data(f"SELECT tid,bed_level,slope,aspect,roughness,scid FROM tile_observations JOIN observations USING (oid) WHERE scid IN ({side_channels}) AND year={year_1};",
                           index_col='tid')
    transform_circular_feature(df_year_1, 'aspect')

    df_year_2 = load_db_data(f"SELECT tid,bed_level FROM tile_observations JOIN observations USING (oid) WHERE scid IN ({side_channels}) AND year={year_2};",
                           index_col='tid')

    df_tiles = load_db_data(f"SELECT tid,width,min_flow_threshold,flow_direction,distance_from_inlet,distance_to_bank FROM tiles WHERE scid IN ({side_channels});",
                            index_col='tid')
    try:
        transform_circular_feature(df_tiles, 'flow_direction')
    except:
        df_tiles.drop(columns=['flow_direction','distance_from_inlet'], inplace=True)

    df_side_channels = load_db_data(f"SELECT scid,area,perimeter,channel_length,relative_channel_length FROM side_channels WHERE scid IN ({side_channels});",
                            index_col='scid')

    df_observations = load_db_data(f"SELECT scid,vegetation_percentage FROM observations WHERE scid IN ({side_channels}) AND year={year_1};",
                            index_col='scid')

    # load the water_level data
    df_water_level = pd.read_csv("output/water_level.csv")

    # calculate the bed_level change
    df_change = pd.merge(df_tiles,pd.merge(df_year_1, df_year_2, on='tid', how='inner'),on='tid', how='inner')
    df_change["change"] = df_change["bed_level_y"] - df_change["bed_level_x"]

    # merge all data
    df_sc_all = pd.merge(df_observations,pd.merge(df_side_channels,df_water_level, on='scid', how='left'),on='scid', how='left')
    df = pd.merge(df_change, df_sc_all, on='scid', how='left', left_index = False)
    df["depth"] = df["water_level"] / 100 - df["bed_level_x"]
    df.index = df_change.index

    # drop useless columns
    df.drop(columns=['water_level','bed_level_x','bed_level_y'], inplace=True)

    return df

def run_model(df, features=None, feature_reduction=None, features_n=6, size=2000, runs=1):
    """
    Runs a neural network model on the prepared dataset with optional feature reduction and multiple runs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset
    features : list, optional
        List of feature column names to use for training.
    feature_reduction : str, optional
        Method for feature reduction ('pca' or 'forward_insertion').
    features_n : int, default=6
        Number of features or PCA components to retain.
    size : int, default=2000
        Number of samples per side channel used for training.
    runs : int, default=1
        Number of independent training runs to average results.

    Returns
    -------
    pandas.DataFrame
        DataFrame with model predictions and errors for each tile.
    """
    if feature_reduction == "pca":
        df = pca_data(df, n=features_n)
        if runs > 1:
            df_out = multiple_runs(df, [f"pca_{i + 1}" for i in range(features_n)], size=size, runs=runs)
        else:
            df_out = train_neural_network(df, [f"pca_{i + 1}" for i in range(features_n)], size=size)
    elif feature_reduction == "forward_insertion":
        insert = forward_insertion(df, features, size=200, amount=features_n)
        if runs > 1:
            df_out = multiple_runs(df, insert, size=size, runs=runs, scale=True)
        else:
            df_out = train_neural_network(df, insert, size=size, scale=True)
    else:
        if runs > 1:
            df_out = multiple_runs(df, features, size=size, runs=runs, scale=True)
        else:
            df_out = train_neural_network(df, features, size=size, scale=True)
    return df_out

def pca_data(df,exclude=None,n=4):
    # Separate features
    if exclude is not None:
        df.drop(columns=exclude,inplace=True)
    df.dropna(inplace=True)
    features = df.drop(columns=['scid', 'change'])


    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n)
    pca_features = pca.fit_transform(scaled_features)

    # Create PCA DataFrame
    pca_cols = [f"pca_{i+1}" for i in range(pca_features.shape[1])]
    df_pca = pd.DataFrame(pca_features, columns=pca_cols, index=df.index)

    # Combine results
    df_out = pd.concat([df[['scid', 'change']], df_pca], axis=1)

    # Print explained variance
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA completed: retained {pca_features.shape[1]} components explaining {explained:.2%} of variance")

    return df_out

def forward_insertion(df,features,size=2000,amount=3):
    features_to_keep = []
    n = 0
    while n < amount:
        print(f"feature {n+1}")
        min_loss = 1000
        feature_to_keep = ""
        for feature in features:
            print(feature)
            loss = train_neural_network(df,features_to_keep + [feature],size,test=True,graph=False)
            if loss < min_loss:
                min_loss = loss
                feature_to_keep = feature
        print(f"added: {feature_to_keep} - loss: {min_loss}")
        features_to_keep.append(feature_to_keep)
        features.remove(feature_to_keep)
        n += 1

    print(features_to_keep)
    return features_to_keep

def train_neural_network(df,features,size=2000,test=False,graph=True,hist=False,scale=False):

    y = df['change'].values.reshape(-1,1)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    df['change_norm'] = y_scaled


    df_samp = (
        df.groupby("scid", group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), size), random_state=23),include_groups=False)
    )

    features_change = features + ["change_norm"]
    df_sampled = df_samp[features_change].copy()
    df_sampled.dropna(inplace=True)

    X = df_sampled[features].values
    y = df_sampled['change_norm'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.add(layers.Dropout(0.1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse',
                  metrics=['mae', keras.metrics.R2Score()])

    if test:
        e = 5
        v = 0
    elif hist:
        e = 50
        v = 0
    else:
        e = 50
        v = "auto"

    history = model.fit(X_train, y_train, epochs=e, batch_size=4, validation_data=(X_test, y_test),verbose=v)

    loss, mae, r2 = model.evaluate(X_test, y_test)
    print(f"Test results — Loss: {loss:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
    if graph:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss (MSE)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    if test == True:
        if hist:
            return history, loss
        else:
            return loss

    X_all = df[features].values
    if scale:
        X_all = scaler.fit_transform(X_all)
    y_pred_norm = model.predict(X_all)
    df['predicted_target_norm'] = y_pred_norm
    y_pred = y_scaler.inverse_transform(y_pred_norm)
    df['predicted_target'] = y_pred
    df['error'] = df['predicted_target'] - df['change']

    df.to_csv("Output/result.csv")
    if hist:
        return history, df
    else:
        return df

def multiple_runs(df,features,size=2000,runs=5,scale=False):
    n = 0
    train_loss = None
    val_loss = None
    while n < runs:
        history, df_out = train_neural_network(df.copy(),features,size=size,graph=False,hist=True,scale=scale)
        if n == 0:
            df['predicted_target_norm'] = df_out['predicted_target_norm'] / runs
            df['predicted_target'] = df_out['predicted_target'] / runs
            df['change_norm'] = df_out['change_norm']
            train_loss = np.array(history.history['loss'])
            val_loss = np.array(history.history['val_loss'])
        else:
            df['predicted_target_norm'] += df_out['predicted_target_norm'] / runs
            df['predicted_target'] += df_out['predicted_target'] / runs
            train_loss = train_loss + np.array(history.history['loss'])
            val_loss = val_loss + np.array(history.history['val_loss'])
        n += 1

    df['error'] = df['predicted_target_norm'] - df['change_norm']
    df['sq_error'] = df['error'] ** 2
    MSE = df['sq_error'].mean()
    df.drop(columns=['sq_error'], inplace=True)
    print("MSE: ", MSE)
    print("Train loss: ", train_loss[-1] / runs)
    print("Test loss: ", val_loss[-1] / runs)

    df['error'] = df['predicted_target'] - df['change']
    total_tiles = df['error'].count()
    print("0 - 1 cm: ", round(df.query("abs(error) < 0.01")['error'].count() / total_tiles * 100, 2), "%")
    print("1 - 2 cm: ", round(df.query("abs(error) < 0.02 & abs(error) >= 0.01")['error'].count() / total_tiles * 100, 2), "%")
    print("2 - 5 cm: ", round(df.query("abs(error) < 0.05 & abs(error) >= 0.02")['error'].count() / total_tiles * 100, 2), "%")
    print(" 5 +  cm: ", round(df.query("abs(error) >= 0.05")['error'].count() / total_tiles * 100, 2), "%")
    print("right   : ", round(df.query("(change > 0 & predicted_target > 0) | (change < 0 & predicted_target < 0)")['error'].count() / total_tiles * 100, 2), "%")

    df.to_csv("Output/result.csv")

    plt.plot(train_loss / runs)
    plt.plot(val_loss / runs)
    plt.title('Model loss (MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return df

def html_visualizer(df,side_channels):
    db_connection_url = "postgresql://postgres.miiedebavuhxxbzpndeq:SYbFFBRcyttS3XQy@aws-1-eu-west-3.pooler.supabase.com:5432/postgres"
    con = create_engine(db_connection_url)
    tiles = gpd.read_postgis(f"SELECT tid,geom FROM tiles WHERE scid IN ({side_channels})", con, geom_col="geom", index_col="tid")

    # Merge prediction dataframe with tile geometries
    gdf = tiles.merge(df[["change", "predicted_target"]], left_index=True, right_index=True, how="inner")

    # Compute prediction error (difference)
    gdf["difference"] = gdf["predicted_target"] - gdf["change"]

    # Convert to WGS84 for Folium visualization (EPSG:4326)
    gdf = gdf.to_crs(epsg=4326)

    # Create Folium map centered on data
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=12)

    # Helper function for adding choropleth-style layers
    def add_layer(column_name, color_map, layer_name):
        folium.Choropleth(
            geo_data=gdf,
            name=layer_name,
            data=gdf,
            columns=[gdf.index, column_name],
            key_on="feature.id",
            fill_color=color_map,
            fill_opacity=1,
            line_opacity=0,
            legend_name=layer_name,
        ).add_to(m)

    # Add layers for change, predicted, and difference
    add_layer("change", "YlGnBu", "Observed Change")
    add_layer("predicted_target", "YlGnBu", "Predicted Change")
    add_layer("difference", "PuOr", "Prediction Error")

    # Add layer control and save to HTML
    folium.LayerControl().add_to(m)

    output_path = "Output/prediction_visualization.html"
    m.save(output_path)

    return 0

if __name__ == '__main__':
    main()