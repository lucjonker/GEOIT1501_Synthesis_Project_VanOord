from main import load_db_data
import pandas as pd
# import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df_2024 = load_db_data("SELECT tid,scid,slope,roughness,bed_level,aspect FROM tile_observations JOIN observations USING (oid) WHERE scid IN (1,5,6,19,24,42) AND year=2024;",
                       index_col='tid')

df_2025 = load_db_data("SELECT tid,bed_level FROM tile_observations JOIN observations USING (oid) WHERE scid IN (1,5,6,19,24,42) AND year=2025;",
                       index_col='tid')

df_width = load_db_data("SELECT tid,width,min_flow_threshold FROM tiles WHERE scid IN (1,5,6,19,24,42);",
                        index_col='tid')

df_water_height = pd.read_csv("output/water_level.csv")

df_change = pd.merge(df_2024, df_2025, on='tid', how='inner')
df_change["change"] = df_change["bed_level_y"] - df_change["bed_level_x"]

df_water = pd.merge(df_change, df_water_height, on='scid', how='left', left_index = False)
df_water["depth"] = df_water["bed_level_x"] - df_water["water_level"] / 100
df_water.index = df_change.index

print(df_water.head())
print(df_water.tail())

df = pd.merge(df_water, df_width, on='tid', how='inner')
df.dropna(inplace=True)

y = df['change'].values.reshape(-1,1)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)
df['change_norm'] = y_scaled

print(df.head())

X = df[['depth', 'slope', 'width', 'roughness', 'min_flow_threshold']].values
print("Look here")
print(df.head())
y = df['change_norm'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='tanh'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='tanh'),
    layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"mae: {acc:.3f}")

X_all = df[['depth', 'slope', 'width', 'roughness', 'min_flow_threshold']].values
X_all_scaled = scaler.fit_transform(X_all)
y_pred_norm = model.predict(X_all_scaled)
df['predicted_target_norm'] = y_pred_norm
y_pred = y_scaler.inverse_transform(y_pred_norm)
df['predicted_target'] = y_pred
print(df.head())

df.to_csv("Output/result.csv")