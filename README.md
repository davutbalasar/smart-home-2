# Akıllı Ev — Derin Öğrenme Projeleri

Bu repo, gerçek sensör verilerine dayalı olarak ev otomasyonunu öğrenen üç bağımsız derin öğrenme projesini içermektedir. Her proje kendi modelini eğitir, tahmin üretir ve bir tkinter arayüzüyle simüle edilebilir.

---

## Projeler

| # | Proje | Model | Görev |
|---|---|---|---|
| 1 | [Akıllı Işık](#isik) | Dense NN | Işığın açık/kapalı olması gerektiğini tahmin eder |
| 2 | [Akıllı Perde](#perde) | LSTM | Perdenin açılma ve kapanma saatini tahmin eder |
| 3 | [Yağış Tahmini](#️yagis) | Dense NN | Saat, sıcaklık ve neme göre yağış ihtimalini tahmin eder |

---

<a name="isik"></a>
## Akıllı Işık Kontrol Sistemi

Geçmiş kullanım verilerinden öğrenerek ışığın o an açık mı kapalı mı olması gerektiğini tahmin eden bir sinir ağı sistemi.

### Özellikler
- Saat, gün, ortam ışığı ve önceki lamba durumuna göre tahmin
- Gerçek zamanlı tkinter GUI — olasılık göstergesi, saat simülasyonu, manuel kontrol
- CLI ile hızlı tahmin

### Model Bilgisi

| Özellik | Değer |
|---|---|
| Mimari | Fully Connected Neural Network |
| Girdi | `hour`, `minute`, `day_of_week`, `ambient_light`, `lamp_status` |
| Çıktı | Işığın açık olma ihtimali (0.0 – 1.0) |
| Eşik | ≥ 0.5 → Işık açık |
| Kayıp Fonksiyonu | Binary Crossentropy |
| Düzenleme | Dropout (0.2), Early Stopping (patience=20) |

### Veri Formatı (`light_usage_data.csv`)

| Kolon | Açıklama |
|---|---|
| `hour` | Saat (0–23) |
| `minute` | Dakika (0–59) |
| `day_of_week` | Haftanın günü (0=Pazartesi) |
| `ambient_light` | Ortam ışık değeri (lux / sensör) |
| `lamp_status` | Önceki lamba durumu (0/1) |
| `target_action` | Hedef etiket (0=kapalı, 1=açık) |

### Kullanım

```bash
# Model eğitimi
python src/train_model.py

# Arayüz
python src/light_gui.py

# CLI tahmini
python src/predict_and_control.py
```

---

<a name="perde"></a>
## Akıllı Perde Kontrol Sistemi

Haftalık perde kullanım alışkanlıklarını öğrenerek ertesi gün için açılma ve kapanma saatini tahmin eden LSTM tabanlı sistem.

### Özellikler
- 7 günlük geçmiş veriye dayalı LSTM tahmini
- Tahmin edilen saatten sonra perde hâlâ yanlış durumdaysa otomatik devreye giriş
- Gerçek zamanlı tkinter GUI — dijital saat, tahmin paneli, saat simülasyonu
- Donanım entegrasyonuna hazır motor kontrol arayüzü (`curtain_controller.py`)
- Tüm ayarlar tek dosyada yönetilir (`config.py`)

### Model Bilgisi

| Özellik | Değer |
|---|---|
| Mimari | Stacked LSTM (2 katman) |
| Girdi | `day_of_week`, `open_minutes`, `close_minutes` (son 7 gün) |
| Çıktı | Yarın için `open_minutes` ve `close_minutes` tahmini |
| Sequence Uzunluğu | 7 gün (ayarlanabilir) |
| Kayıp Fonksiyonu | Mean Squared Error |
| Düzenleme | Dropout (0.2), Early Stopping (patience=20) |

### Konfigürasyon (`src/config.py`)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `SEQUENCE_LENGTH` | 7 | LSTM için kullanılan geçmiş gün sayısı |
| `CONTROL_DELAY_MINUTES` | 10 | Tahmin sonrası otomasyon için bekleme süresi (dk) |
| `DATA_PATH` | `data/curtain_usage_sample.csv` | Eğitim verisi yolu |
| `MODEL_PATH` | `models/curtain_lstm_model.keras` | Model kayıt/yükleme yolu |

### Veri Formatı (`curtain_usage_sample.csv`)

| Kolon | Açıklama |
|---|---|
| `day_of_week` | Haftanın günü (0=Pazartesi) |
| `open_time` | Açılma saati (`HH:MM` formatında) |
| `close_time` | Kapanma saati (`HH:MM` formatında) |

Minimum satır sayısı: `SEQUENCE_LENGTH + 1` (varsayılan: 8 satır)

### Kullanım

```bash
# Model eğitimi
python src/train_model.py

# Arayüz
python src/curtain_gui.py

# CLI otomasyonu
python src/predict_and_control.py
```

### Donanım Entegrasyonu

Gerçek bir perde motoru bağlamak için `src/curtain_controller.py` dosyasını düzenleyin:

```python
def open_curtain():
    # GPIO.output(RELAY_PIN, HIGH)
    # veya: requests.post("http://esp32/open")
    pass

def close_curtain():
    # GPIO.output(RELAY_PIN, LOW)
    pass
```

---

<a name="yagis"></a>
## Yağış Tahmini Sistemi

Saat, sıcaklık ve nem değerlerine bakarak yağış ihtimalini tahmin eden sinir ağı tabanlı sistem.

### Özellikler
- Saat, sıcaklık ve neme göre yağış olasılığı tahmini
- Gerçek zamanlı tkinter GUI — canlı saat, sensör girişleri, uyarı göstergesi
- Yağış ihtimali %50 üzerindeyse şemsiye uyarısı
- CLI ile anlık tahmin

### Model Bilgisi

| Özellik | Değer |
|---|---|
| Mimari | Fully Connected Neural Network |
| Girdi | `hour`, `temperature`, `humidity` |
| Çıktı | Yağış ihtimali (0.0 – 1.0) |
| Eşik | ≥ 0.5 → Yağış bekleniyor |
| Kayıp Fonksiyonu | Binary Crossentropy |
| Düzenleme | Dropout (0.2), Early Stopping (patience=20) |

### Veri Formatı (`rain_sensor_data.csv`)

| Kolon | Açıklama |
|---|---|
| `hour` | Saat (0–23) |
| `temperature` | Sıcaklık (°C) |
| `humidity` | Bağıl nem (0–100) |
| `rain` | Hedef etiket (0=yağmur yok, 1=yağmur var) |

### Kullanım

```bash
# Model eğitimi
python src/train_model.py

# Arayüz
python src/rain_gui.py

# CLI tahmini
python src/predict_and_control.py
```

**CLI Örnek Çıktı:**
```
Rain probability: %73.45
Warning: Yağış ihtimali yüksek.
```

---

## Genel Kurulum

Tüm projeler aynı bağımlılıkları paylaşır:

```bash
pip install tensorflow scikit-learn joblib numpy pandas
```

> Python 3.8+ gereklidir. `tkinter` standart kütüphane ile gelir, ek kurulum gerekmez.

---

## Genel Çalışma Mantığı

Her proje aynı üç aşamalı akışı takip eder:

```
Veri Toplama  →  Model Eğitimi  →  Tahmin & Kontrol
   (CSV)        (train_model.py)   (GUI veya CLI)
```

1. Geçmiş kullanım verileri CSV formatında toplanır
2. `train_model.py` veriyi normalize eder, modeli eğitir, `models/` klasörüne kaydeder
3. GUI veya CLI, kaydedilen modeli yükleyerek gerçek zamanlı tahmin üretir
