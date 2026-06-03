import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "light_usage_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "smart_light_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "light_scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)

features = ["hour", "minute", "day_of_week", "ambient_light", "lamp_status"]
target = "target_action"

X = data[features]
y = data[target]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)
model.fit(X_scaled, y, epochs=200, batch_size=8, callbacks=[early_stop], verbose=1)

model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Model training completed.")
print(f"Model saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
