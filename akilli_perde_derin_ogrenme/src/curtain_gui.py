import os
import sys
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Proje klasör yapısını sisteme ekler (import işlemleri için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Konfigürasyon dosyasından gerekli path ve sabitleri alır
from src.config import DATA_PATH, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH, SEQUENCE_LENGTH, CONTROL_DELAY_MINUTES


def time_to_minutes(time_value):
    """Convert HH:MM time format to total minutes."""
    # HH:MM formatındaki zamanı dakika cinsine çevirir
    hour, minute = map(int, str(time_value).split(":"))
    return hour * 60 + minute


def minutes_to_time(minutes_value):
    """Convert total minutes to HH:MM time format."""
    # Dakika cinsinden zamanı tekrar HH:MM formatına çevirir
    minutes_value = int(round(minutes_value)) % (24 * 60)
    hour = minutes_value // 60
    minute = minutes_value % 60
    return f"{hour:02d}:{minute:02d}"


def predict_open_close_time():
    """Predict curtain opening and closing time by using the trained LSTM model."""
    
    # Model dosyası var mı kontrol eder
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file was not found. First run: python src/train_model.py")

    # Eğitim verisini yükler
    data = pd.read_csv(DATA_PATH)

    # Açılma ve kapanma saatlerini dakika cinsine çevirir
    data["open_minutes"] = data["open_time"].apply(time_to_minutes)
    data["close_minutes"] = data["close_time"].apply(time_to_minutes)

    # Model için kullanılacak feature'ları seçer
    features = data[["day_of_week", "open_minutes", "close_minutes"]].values

    # Daha önce kaydedilmiş scaler'ları yükler
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Eğitilmiş LSTM modelini yükler
    model = load_model(MODEL_PATH)

    # Son SEQUENCE_LENGTH kadar veriyi alır (LSTM input)
    last_sequence = features[-SEQUENCE_LENGTH:]

    # Veriyi normalize eder
    last_sequence_scaled = scaler_x.transform(last_sequence)
    last_sequence_scaled = np.array([last_sequence_scaled])

    # Tahmin yapar
    prediction_scaled = model.predict(last_sequence_scaled, verbose=0)

    # Tahmini tekrar gerçek değerlere çevirir
    prediction = scaler_y.inverse_transform(prediction_scaled)[0]

    # Açılma ve kapanma saatlerini yuvarlar
    predicted_open_minutes = int(round(prediction[0]))
    predicted_close_minutes = int(round(prediction[1]))

    return predicted_open_minutes, predicted_close_minutes


class CurtainClockApp:
    def __init__(self, window):
        # Ana pencere ayarları
        self.window = window
        self.window.title("Akıllı Perde - Dijital Saat")
        self.window.geometry("560x520")
        self.window.resizable(False, False)

        # Başlangıç değerleri
        self.current_time = datetime.now().replace(second=0, microsecond=0)
        self.curtain_status = "open"
        self.predicted_open_minutes = None
        self.predicted_close_minutes = None
        self.clock_running = False

        # Arayüzü oluşturur
        self.build_screen()

        # Model tahminlerini yükler
        self.load_prediction()

        # İlk ekran güncellemesi
        self.update_screen()

    def build_screen(self):
        # Başlık label'ı
        title = tk.Label(
            self.window,
            text="AKILLI PERDE KONTROL EKRANI",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=16)

        # Dijital saat göstergesi
        self.clock_label = tk.Label(
            self.window,
            text="00:00",
            font=("Digital-7", 54, "bold"),
            bg="black",
            fg="lime",
            width=8,
            relief="sunken"
        )
        self.clock_label.pack(pady=10)

        # Saat değiştirme butonları
        time_frame = tk.Frame(self.window)
        time_frame.pack(pady=6)

        tk.Button(time_frame, text="-10 dk", width=8, command=lambda: self.change_time(-10)).grid(row=0, column=0, padx=4)
        tk.Button(time_frame, text="-1 dk", width=8, command=lambda: self.change_time(-1)).grid(row=0, column=1, padx=4)
        tk.Button(time_frame, text="+1 dk", width=8, command=lambda: self.change_time(1)).grid(row=0, column=2, padx=4)
        tk.Button(time_frame, text="+10 dk", width=8, command=lambda: self.change_time(10)).grid(row=0, column=3, padx=4)

        # Manuel saat girişi alanı
        manual_frame = tk.Frame(self.window)
        manual_frame.pack(pady=8)

        tk.Label(manual_frame, text="Saat Gir (HH:MM):", font=("Arial", 11)).grid(row=0, column=0, padx=5)
        self.time_entry = tk.Entry(manual_frame, font=("Arial", 12), width=8, justify="center")
        self.time_entry.grid(row=0, column=1, padx=5)
        tk.Button(manual_frame, text="Saati Ayarla", command=self.set_manual_time).grid(row=0, column=2, padx=5)

        # Tahmin alanı
        prediction_frame = tk.LabelFrame(self.window, text="Yapay Zeka Tahmini", font=("Arial", 11, "bold"))
        prediction_frame.pack(pady=12, padx=20, fill="x")

        self.prediction_label = tk.Label(
            prediction_frame,
            text="Tahmin yükleniyor...",
            font=("Arial", 12)
        )
        self.prediction_label.pack(pady=10)

        # Perde durumu alanı
        status_frame = tk.LabelFrame(self.window, text="Perde Durumu", font=("Arial", 11, "bold"))
        status_frame.pack(pady=8, padx=20, fill="x")

        self.status_label = tk.Label(
            status_frame,
            text="AÇIK",
            font=("Arial", 24, "bold"),
            fg="green"
        )
        self.status_label.pack(pady=8)

        # İşlem bilgisi label'ı
        self.action_label = tk.Label(
            status_frame,
            text="Durum bekleniyor...",
            font=("Arial", 13, "bold")
        )
        self.action_label.pack(pady=6)

        # Manuel kontrol butonları
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Perdeyi Aç", width=14, command=self.open_curtain).grid(row=0, column=0, padx=6)
        tk.Button(button_frame, text="Perdeyi Kapat", width=14, command=self.close_curtain).grid(row=0, column=1, padx=6)
        tk.Button(button_frame, text="Otomatik Kontrol", width=16, command=self.automatic_control).grid(row=0, column=2, padx=6)

        # Saat başlat/durdur butonları
        run_frame = tk.Frame(self.window)
        run_frame.pack(pady=5)

        tk.Button(run_frame, text="Saati Başlat", width=14, command=self.start_clock).grid(row=0, column=0, padx=6)
        tk.Button(run_frame, text="Saati Durdur", width=14, command=self.stop_clock).grid(row=0, column=1, padx=6)

    def load_prediction(self):
        # Model tahminini yükler ve ekrana yazar
        try:
            self.predicted_open_minutes, self.predicted_close_minutes = predict_open_close_time()

            open_time = minutes_to_time(self.predicted_open_minutes)
            close_time = minutes_to_time(self.predicted_close_minutes)

            self.prediction_label.config(
                text=f"Tahmini Açılma: {open_time}   |   Tahmini Kapanma: {close_time}\n"
                     f"Kontrol Gecikmesi: {CONTROL_DELAY_MINUTES} dakika"
            )

        except Exception as error:
            # Hata durumunda kullanıcıya mesaj gösterir
            self.prediction_label.config(text="Tahmin yüklenemedi. Önce modeli eğitin.")
            messagebox.showerror("Model Hatası", str(error))

    def update_screen(self):
        # Saat ve input alanını günceller
        self.clock_label.config(text=self.current_time.strftime("%H:%M"))
        self.time_entry.delete(0, tk.END)
        self.time_entry.insert(0, self.current_time.strftime("%H:%M"))

        # Durum yazısını günceller
        self.update_status_text()

    def update_status_text(self):
        # Perde durumunu ekrana yansıtır
        if self.curtain_status == "open":
            self.status_label.config(text="AÇIK", fg="green")
        else:
            self.status_label.config(text="KAPALI", fg="red")

    def current_minutes(self):
        # Şu anki zamanı dakika cinsine çevirir
        return self.current_time.hour * 60 + self.current_time.minute

    def change_time(self, minute_count):
        # Saati ileri/geri alır
        self.current_time += timedelta(minutes=minute_count)
        self.update_screen()
        self.automatic_control()

    def set_manual_time(self):
        # Kullanıcının girdiği saate göre zamanı ayarlar
        try:
            selected_time = datetime.strptime(self.time_entry.get(), "%H:%M")
            self.current_time = self.current_time.replace(hour=selected_time.hour, minute=selected_time.minute)

            self.update_screen()
            self.automatic_control()

        except ValueError:
            # Hatalı format girilirse uyarı verir
            messagebox.showwarning("Saat Hatası", "Lütfen saati HH:MM formatında girin. Örnek: 20:40")

    def open_curtain(self):
        # Perdeyi açar
        self.curtain_status = "open"
        self.update_status_text()
        self.action_label.config(text="Perde Açıldı")

    def close_curtain(self):
        # Perdeyi kapatır
        self.curtain_status = "closed"
        self.update_status_text()
        self.action_label.config(text="Perde Kapandı")

    def automatic_control(self):
        # Yapay zeka tahminine göre otomatik kontrol yapar
        if self.predicted_open_minutes is None or self.predicted_close_minutes is None:
            self.action_label.config(text="Model tahmini yok. Önce modeli eğitin.")
            return

        now_minutes = self.current_minutes()

        # Gecikmeli kontrol zamanları
        open_control_time = self.predicted_open_minutes + CONTROL_DELAY_MINUTES
        close_control_time = self.predicted_close_minutes + CONTROL_DELAY_MINUTES

        # Açma kontrolü
        if now_minutes >= open_control_time and now_minutes < self.predicted_close_minutes and self.curtain_status == "closed":
            self.open_curtain()
            self.action_label.config(text="Yapay zeka tahminine göre perde otomatik açıldı")

        # Kapatma kontrolü
        elif now_minutes >= close_control_time and self.curtain_status == "open":
            self.close_curtain()
            self.action_label.config(text="Yapay zeka tahminine göre perde otomatik kapandı")

        else:
            # Hiçbir işlem gerekmezse
            self.action_label.config(text="Şu anda otomatik işlem gerekmiyor")

    def start_clock(self):
        # Saati başlatır
        self.clock_running = True
        self.run_clock()

    def stop_clock(self):
        # Saati durdurur
        self.clock_running = False

    def run_clock(self):
        # Saat her saniye 1 dakika ilerler (simülasyon)
        if self.clock_running:
            self.current_time += timedelta(minutes=1)
            self.update_screen()
            self.automatic_control()

            # 1 saniye sonra tekrar çalışır
            self.window.after(1000, self.run_clock)


# Programın başlangıç noktası
if __name__ == "__main__":
    root = tk.Tk()
    app = CurtainClockApp(root)
    root.mainloop()