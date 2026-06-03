import tkinter as tk
from datetime import datetime
import joblib
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/rain_prediction_model.keras"
SCALER_PATH = "models/rain_scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

current_time = datetime.now().replace(second=0, microsecond=0)

def predict_rain(hour, temperature, humidity):
    input_data = np.array([[hour, temperature, humidity]])
    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled, verbose=0)[0][0]
    return probability

def update_prediction():
    try:
        temperature = float(temp_entry.get())
        humidity = float(humidity_entry.get())
        hour = current_time.hour

        probability = predict_rain(hour, temperature, humidity)
        percent = probability * 100

        probability_label.config(text=f"Yağış İhtimali: %{percent:.1f}")

        if probability >= 0.5:
            result_label.config(text="Durum: Yağış Bekleniyor")
            warning_label.config(text="Uyarı: Şemsiye almanız önerilir.")
        else:
            result_label.config(text="Durum: Yağış Beklenmiyor")
            warning_label.config(text="Uyarı: Risk düşük.")
    except ValueError:
        warning_label.config(text="Sıcaklık ve nem değerlerini doğru giriniz.")

def refresh_clock():
    clock_label.config(text=current_time.strftime("%H:%M"))
    update_prediction()
    window.after(1000, refresh_clock)

def set_time():
    global current_time
    try:
        hour = int(hour_entry.get())
        minute = int(minute_entry.get())
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            current_time = current_time.replace(hour=hour, minute=minute)
            update_prediction()
        else:
            warning_label.config(text="Saat 0-23, dakika 0-59 aralığında olmalıdır.")
    except ValueError:
        warning_label.config(text="Saat ve dakika sayısal olmalıdır.")

window = tk.Tk()
window.title("Yağış Tahmini - Akıllı Ev")
window.geometry("470x430")
window.resizable(False, False)

main_title = tk.Label(window, text="Yağış Tahmini Sistemi", font=("Arial", 18, "bold"))
main_title.pack(pady=12)

clock_label = tk.Label(window, text="00:00", font=("Arial", 42, "bold"))
clock_label.pack(pady=5)

time_frame = tk.Frame(window)
time_frame.pack(pady=8)

tk.Label(time_frame, text="Saat:").pack(side="left")
hour_entry = tk.Entry(time_frame, width=5, justify="center")
hour_entry.insert(0, current_time.strftime("%H"))
hour_entry.pack(side="left", padx=4)

tk.Label(time_frame, text="Dakika:").pack(side="left")
minute_entry = tk.Entry(time_frame, width=5, justify="center")
minute_entry.insert(0, current_time.strftime("%M"))
minute_entry.pack(side="left", padx=4)

tk.Button(time_frame, text="Saati Ayarla", command=set_time).pack(side="left", padx=8)

input_frame = tk.Frame(window)
input_frame.pack(pady=12)

tk.Label(input_frame, text="Sıcaklık:").grid(row=0, column=0, padx=5, pady=5)
temp_entry = tk.Entry(input_frame, width=10, justify="center")
temp_entry.insert(0, "17")
temp_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Nem (%):").grid(row=1, column=0, padx=5, pady=5)
humidity_entry = tk.Entry(input_frame, width=10, justify="center")
humidity_entry.insert(0, "82")
humidity_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Button(window, text="Tahmini Güncelle", command=update_prediction, font=("Arial", 11)).pack(pady=8)

probability_label = tk.Label(window, text="Yağış İhtimali: -", font=("Arial", 14, "bold"))
probability_label.pack(pady=8)

result_label = tk.Label(window, text="Durum: -", font=("Arial", 15, "bold"))
result_label.pack(pady=8)

warning_label = tk.Label(window, text="Uyarı bekleniyor...", font=("Arial", 11))
warning_label.pack(pady=8)

refresh_clock()
window.mainloop()
