# Akıllı Işık Derin Öğrenme Projesi

Bu proje, kullanıcının günlük ışık açma/kapatma alışkanlıklarını öğrenerek belirli saatlerde ışığın açık mı kapalı mı olması gerektiğini tahmin eder.

## Çalışma Mantığı

Model şu verileri kullanır:

- Saat
- Dakika
- Haftanın günü
- Ortam ışık seviyesi
- Mevcut ışık durumu

Model çıktısı:

- 1 = Işık açık olmalı
- 0 = Işık kapalı olmalı

## Kurulum

```bash
pip install -r requirements.txt
```

## Model Eğitimi

```bash
python src/train_model.py
```

## Tahmin Testi

```bash
python src/predict_and_control.py
```

## Dijital Saatli Ekran

```bash
python src/light_gui.py
```

## Kendi Verini Ekleme

Kendi verilerini `data/light_usage_data.csv` dosyasına aynı kolon yapısıyla ekleyebilirsin.
