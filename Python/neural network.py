from main import load_DB_data
import pandas as pd
import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df_2024 = load_DB_data("SELECT tid,slope,roughness,bed_level,aspect FROM tile_observations JOIN observations USING (oid) WHERE scid=5 AND year=2024;")

df_2025 = load_DB_data("SELECT tid,bed_level FROM tile_observations JOIN observations USING (oid) WHERE scid=5 AND year=2025;")

df = pd.merge(df_2024, df_2025, on='tid', how='inner')
df.dropna(inplace=True)
df["change"] = df["bed_level_y"] - df["bed_level_x"]
y = df['change'].values.reshape(-1,1)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

df['change_norm'] = y_scaled

print(df.head())

X = df[['bed_level_x', 'slope', 'roughness', 'aspect']].values
y = df['change_norm'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"mae: {acc:.3f}")

X_all = df[['bed_level_x', 'slope', 'roughness', 'aspect']].values
X_all_scaled = scaler.fit_transform(X_all)
y_pred_norm = model.predict(X_all_scaled)
df['predicted_target_norm'] = y_pred_norm
y_pred = y_scaler.inverse_transform(y_pred_norm)
df['predicted_target'] = y_pred
print(df.head())

df.to_csv("Output/result.csv")