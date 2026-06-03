# Yağış Tahmini - Derin Öğrenme Projesi

Bu proje sıcaklık, nem ve saat verilerini kullanarak yağış ihtimalini tahmin eder.
Model, geçmiş sensör verilerinden kullanıcıya erken uyarı verecek şekilde tasarlanmıştır.

## Klasör Yapısı

- data/rain_sensor_data.csv: Eğitim verisi
- src/train_model.py: Model eğitimi
- src/predict_and_control.py: Konsol tahmin dosyası
- src/rain_gui.py: Görsel arayüz
- models/: Eğitilen model ve scaler dosyaları

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Modeli Eğitme

```bash
python src/train_model.py
```

## Konsolda Tahmin

```bash
python src/predict_and_control.py
```

## Ekranı Açma

```bash
python src/rain_gui.py
```

## Veri Formatı

CSV dosyasında şu kolonlar bulunmalıdır:

```text
hour,temperature,humidity,rain
```

- hour: Saat bilgisi, örnek 14 veya 20
- temperature: Sıcaklık değeri
- humidity: Nem oranı
- rain: Yağış durumu, 1 = yağış var, 0 = yağış yok
