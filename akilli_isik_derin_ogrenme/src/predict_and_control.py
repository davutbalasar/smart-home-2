import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "smart_light_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "light_scaler.pkl")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_light_action(hour, minute, day_of_week, ambient_light, lamp_status):
    input_data = np.array([[hour, minute, day_of_week, ambient_light, lamp_status]])
    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled, verbose=0)[0][0]
    return "Açık" if probability >= 0.5 else "Kapalı", probability

if __name__ == "__main__":
    state, prob = predict_light_action(19, 20, 1, 15, 0)
    print(f"Tahmini ışık durumu: {state}")
    print(f"Açık olma olasılığı: {prob:.2f}")
