import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Proje klasör yolunu sisteme ekler (import işlemlerinin çalışması için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Konfigürasyon dosyasından gerekli sabit ve dosya yollarını alır
from src.config import DATA_PATH, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH, SEQUENCE_LENGTH, CONTROL_DELAY_MINUTES

# Perdeyi açma ve kapama fonksiyonlarını import eder
from src.curtain_controller import open_curtain, close_curtain


def time_to_minutes(time_value):
    """Convert HH:MM time format to total minutes."""
    # HH:MM formatındaki zamanı dakika cinsine çevirir
    hour, minute = map(int, str(time_value).split(":"))
    return hour * 60 + minute


def minutes_to_time(minutes_value):
    """Convert total minutes to HH:MM time format."""
    # Dakika cinsindeki zamanı HH:MM formatına çevirir
    minutes_value = int(round(minutes_value))
    hour = minutes_value // 60
    minute = minutes_value % 60
    return f"{hour:02d}:{minute:02d}"


def now_to_minutes():
    """Convert current system time to total minutes."""
    # Sistemdeki mevcut zamanı dakika cinsine çevirir
    now = datetime.now()
    return now.hour * 60 + now.minute


def predict_next_open_close_time():
    """Predict next opening and closing times by using the trained LSTM model."""
    
    # Veri setini yükler
    data = pd.read_csv(DATA_PATH)

    # Açılma ve kapanma saatlerini dakika cinsine çevirir
    data["open_minutes"] = data["open_time"].apply(time_to_minutes)
    data["close_minutes"] = data["close_time"].apply(time_to_minutes)

    # Modelde kullanılacak feature'ları seçer
    features = data[["day_of_week", "open_minutes", "close_minutes"]].values

    # Daha önce kaydedilmiş scaler'ları yükler
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Eğitilmiş LSTM modelini yükler
    model = load_model(MODEL_PATH)

    # Son SEQUENCE_LENGTH kadar veriyi alır (LSTM giriş verisi)
    last_sequence = features[-SEQUENCE_LENGTH:]

    # Veriyi normalize eder
    last_sequence_scaled = scaler_x.transform(last_sequence)
    last_sequence_scaled = np.array([last_sequence_scaled])

    # Model ile tahmin yapar
    prediction_scaled = model.predict(last_sequence_scaled, verbose=0)

    # Tahmini tekrar gerçek değerlere çevirir
    prediction = scaler_y.inverse_transform(prediction_scaled)[0]

    # Açılma ve kapanma zamanlarını alır
    predicted_open_minutes = prediction[0]
    predicted_close_minutes = prediction[1]

    return predicted_open_minutes, predicted_close_minutes


def control_curtain(current_curtain_status):
    """
    Decide whether the curtain should be opened or closed.
    current_curtain_status must be either 'open' or 'closed'.
    """
    
    # Modelden tahmini açılma ve kapanma saatlerini alır
    predicted_open, predicted_close = predict_next_open_close_time()

    # Mevcut zamanı dakika cinsine çevirir
    current_minutes = now_to_minutes()

    # Kontrol için gecikme eklenmiş limitler
    open_limit = predicted_open + CONTROL_DELAY_MINUTES
    close_limit = predicted_close + CONTROL_DELAY_MINUTES

    # Bilgi amaçlı çıktılar
    print(f"Predicted opening time: {minutes_to_time(predicted_open)}")
    print(f"Predicted closing time: {minutes_to_time(predicted_close)}")
    print(f"Current curtain status: {current_curtain_status}")

    # Eğer açılma zamanı geçtiyse ve perde hala kapalıysa → aç
    if current_minutes >= open_limit and current_curtain_status == "closed" and current_minutes < predicted_close:
        open_curtain()
        return "Curtain opened automatically."

    # Eğer kapanma zamanı geçtiyse ve perde hala açıksa → kapat
    if current_minutes >= close_limit and current_curtain_status == "open":
        close_curtain()
        return "Curtain closed automatically."

    # Hiçbir işlem gerekmezse
    return "No automatic action is needed right now."


if __name__ == "__main__":
    # Bu değer ileride gerçek sensör veya IoT sisteminden alınacaktır
    current_curtain_status = "open"

    # Perde kontrol fonksiyonu çalıştırılır
    result = control_curtain(current_curtain_status)

    # Sonuç ekrana yazdırılır
    print(result)