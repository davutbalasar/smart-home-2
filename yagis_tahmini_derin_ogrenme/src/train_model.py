import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "data/rain_sensor_data.csv"
MODEL_PATH = "models/rain_prediction_model.keras"
SCALER_PATH = "models/rain_scaler.pkl"

os.makedirs("models", exist_ok=True)

data = pd.read_csv(DATA_PATH)

X = data[["hour", "temperature", "humidity"]].values
y = data["rain"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = Sequential([
    Dense(32, activation="relu", input_shape=(3,)),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)
model.fit(X_scaled, y, epochs=200, batch_size=4, callbacks=[early_stop], verbose=1)

model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Model training completed.")
print(f"Model saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
