# Project configuration values
# Proje ayar değerleri

DATA_PATH = "data/curtain_usage_sample.csv"
MODEL_PATH = "models/curtain_lstm_model.keras"
SCALER_X_PATH = "models/scaler_x.pkl"
SCALER_Y_PATH = "models/scaler_y.pkl"

# If the curtain is still in the wrong state after this delay, automation starts.
# Tahmin edilen saatten sonra bu kadar dakika geçerse ve perde yanlış durumdaysa otomasyon başlar.
CONTROL_DELAY_MINUTES = 10

# Number of past days used by the LSTM model
# LSTM modelinin kullanacağı geçmiş gün sayısı
SEQUENCE_LENGTH = 7
