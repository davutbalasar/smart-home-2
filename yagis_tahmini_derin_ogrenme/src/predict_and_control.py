import joblib
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/rain_prediction_model.keras"
SCALER_PATH = "models/rain_scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_rain(hour, temperature, humidity):
    input_data = np.array([[hour, temperature, humidity]])
    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled, verbose=0)[0][0]
    return probability

hour = 20
temperature = 17
humidity = 82

probability = predict_rain(hour, temperature, humidity)

print(f"Rain probability: %{probability * 100:.2f}")

if probability >= 0.5:
    print("Warning: Yağış ihtimali yüksek.")
else:
    print("Yağış ihtimali düşük.")
