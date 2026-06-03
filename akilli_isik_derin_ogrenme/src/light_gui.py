import os
import tkinter as tk
from datetime import datetime, timedelta
import joblib
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "smart_light_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "light_scaler.pkl")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

current_time = datetime.now().replace(second=0, microsecond=0)
lamp_status = 0


def predict_state():
    global lamp_status
    hour = current_time.hour
    minute = current_time.minute
    day_of_week = current_time.weekday()

    try:
        ambient_light = int(light_entry.get())
    except ValueError:
        ambient_light = 20

    input_data = np.array([[hour, minute, day_of_week, ambient_light, lamp_status]])
    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled, verbose=0)[0][0]

    if probability >= 0.5:
        lamp_status = 1
        status_text.set("Işık: Açıldı")
    else:
        lamp_status = 0
        status_text.set("Işık: Kapandı")

    probability_text.set(f"Açık olma olasılığı: {probability:.2f}")


def refresh_screen():
    clock_text.set(current_time.strftime("%H:%M"))
    predict_state()


def set_time():
    global current_time
    try:
        h = int(hour_entry.get())
        m = int(minute_entry.get())
        current_time = current_time.replace(hour=h, minute=m)
        refresh_screen()
    except ValueError:
        status_text.set("Saat formatı hatalı")


def add_minutes(minutes):
    global current_time
    current_time = current_time + timedelta(minutes=minutes)
    refresh_screen()


def turn_on():
    global lamp_status
    lamp_status = 1
    status_text.set("Işık: Açıldı")


def turn_off():
    global lamp_status
    lamp_status = 0
    status_text.set("Işık: Kapandı")


window = tk.Tk()
window.title("Akıllı Işık - Derin Öğrenme")
window.geometry("460x420")
window.configure(bg="#20242a")

clock_text = tk.StringVar()
status_text = tk.StringVar(value="Işık: -")
probability_text = tk.StringVar(value="Açık olma olasılığı: -")

header = tk.Label(window, text="Akıllı Işık Kontrol Sistemi", font=("Arial", 18, "bold"), fg="white", bg="#20242a")
header.pack(pady=15)

clock_label = tk.Label(window, textvariable=clock_text, font=("Arial", 46, "bold"), fg="#00ffcc", bg="#20242a")
clock_label.pack(pady=10)

status_label = tk.Label(window, textvariable=status_text, font=("Arial", 22, "bold"), fg="white", bg="#20242a")
status_label.pack(pady=10)

probability_label = tk.Label(window, textvariable=probability_text, font=("Arial", 12), fg="white", bg="#20242a")
probability_label.pack(pady=5)

input_frame = tk.Frame(window, bg="#20242a")
input_frame.pack(pady=10)

tk.Label(input_frame, text="Saat", fg="white", bg="#20242a").grid(row=0, column=0, padx=5)
hour_entry = tk.Entry(input_frame, width=5, justify="center")
hour_entry.grid(row=1, column=0, padx=5)
hour_entry.insert(0, current_time.strftime("%H"))

tk.Label(input_frame, text="Dakika", fg="white", bg="#20242a").grid(row=0, column=1, padx=5)
minute_entry = tk.Entry(input_frame, width=5, justify="center")
minute_entry.grid(row=1, column=1, padx=5)
minute_entry.insert(0, current_time.strftime("%M"))

tk.Label(input_frame, text="Ortam Işığı", fg="white", bg="#20242a").grid(row=0, column=2, padx=5)
light_entry = tk.Entry(input_frame, width=8, justify="center")
light_entry.grid(row=1, column=2, padx=5)
light_entry.insert(0, "20")

tk.Button(window, text="Saati Ayarla ve Tahmin Et", command=set_time, width=25).pack(pady=8)

button_frame = tk.Frame(window, bg="#20242a")
button_frame.pack(pady=8)

tk.Button(button_frame, text="-10 dk", command=lambda: add_minutes(-10), width=8).grid(row=0, column=0, padx=5)
tk.Button(button_frame, text="+10 dk", command=lambda: add_minutes(10), width=8).grid(row=0, column=1, padx=5)
tk.Button(button_frame, text="Işığı Aç", command=turn_on, width=8).grid(row=0, column=2, padx=5)
tk.Button(button_frame, text="Işığı Kapat", command=turn_off, width=10).grid(row=0, column=3, padx=5)

refresh_screen()
window.mainloop()
