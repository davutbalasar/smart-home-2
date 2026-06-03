import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Proje klasör yolunu sisteme ekler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Gerekli dosya yollarını ve sabit değerleri config dosyasından alır
from src.config import DATA_PATH, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH, SEQUENCE_LENGTH


def time_to_minutes(time_value):
    """Convert HH:MM time format to total minutes."""
    # HH:MM formatındaki saati toplam dakika değerine çevirir
    hour, minute = map(int, str(time_value).split(":"))
    return hour * 60 + minute


def create_sequences(features, targets, sequence_length):
    """Create time-series sequences for the LSTM model."""
    # LSTM modeli için giriş ve hedef verilerini tutacak listeler
    x_values = []
    y_values = []

    # Verileri belirlenen sequence uzunluğuna göre sıralı parçalara böler
    for index in range(len(features) - sequence_length):
        x_values.append(features[index:index + sequence_length])
        y_values.append(targets[index + sequence_length])

    # Listeleri numpy array formatına çevirir
    return np.array(x_values), np.array(y_values)


def main():
    # CSV veri dosyasını okur
    data = pd.read_csv(DATA_PATH)

    # Convert opening and closing hours to numerical minute values.
    # Açılma ve kapanma saatlerini dakika cinsinden sayısal değerlere çevirir
    data["open_minutes"] = data["open_time"].apply(time_to_minutes)
    data["close_minutes"] = data["close_time"].apply(time_to_minutes)

    # The model uses day of week and previous curtain usage times as input features.
    # Model, haftanın günü ve önceki perde kullanım saatlerini giriş verisi olarak kullanır
    features = data[["day_of_week", "open_minutes", "close_minutes"]].values

    # The model predicts both opening and closing minutes for the next day.
    # Model, sonraki gün için açılma ve kapanma dakikalarını tahmin eder
    targets = data[["open_minutes", "close_minutes"]].values

    # Giriş ve çıkış verileri için ölçekleyiciler oluşturur
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Verileri 0-1 aralığına ölçekler
    features_scaled = scaler_x.fit_transform(features)
    targets_scaled = scaler_y.fit_transform(targets)

    # LSTM için eğitim dizilerini oluşturur
    x_train, y_train = create_sequences(features_scaled, targets_scaled, SEQUENCE_LENGTH)

    # Yeterli veri yoksa hata verir
    if len(x_train) == 0:
        raise ValueError("Not enough data. Add more rows to the CSV file.")

    # LSTM tabanlı yapay sinir ağı modelini oluşturur
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, x_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(2)
    ])

    # Modeli derler
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Eğitim sırasında iyileşme olmazsa erken durdurma uygular
    early_stop = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)

    # Modeli eğitir
    model.fit(
        x_train,
        y_train,
        epochs=200,
        batch_size=4,
        callbacks=[early_stop],
        verbose=1
    )

    # Model klasörü yoksa oluşturur
    os.makedirs("models", exist_ok=True)

    # Eğitilmiş modeli ve scaler dosyalarını kaydeder
    model.save(MODEL_PATH)
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    # Eğitim tamamlandığında bilgi mesajı yazdırır
    print("Model training completed.")
    print(f"Model saved to: {MODEL_PATH}")


# Dosya doğrudan çalıştırıldığında main fonksiyonunu başlatır
if __name__ == "__main__":
    main()