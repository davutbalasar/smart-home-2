# Akıllı Perde Kontrolü - Derin Öğrenme Projesi

Bu proje, akıllı evdeki perdelerin kullanıcının geçmiş kullanım alışkanlıklarına göre tahmini açılma ve kapanma saatlerini öğrenmesi için hazırlanmıştır.

## Çalışma Mantığı

Sistem geçmiş perde kullanım verilerini inceler. Örneğin kullanıcı çoğunlukla perdeleri 20:00 - 20:30 arasında kapatıyorsa model bu alışkanlığı öğrenir. Tahmin edilen kapanma saatinden 10 dakika sonra perde hâlâ açıksa sistem otomatik olarak kapatma komutu üretir.

Aynı mantık açılma işlemi için de geçerlidir. Kullanıcı genellikle sabah belirli bir saat aralığında perdeleri açıyorsa model bu saati tahmin eder. Tahmin edilen açılma saatinden 10 dakika sonra perde hâlâ kapalıysa sistem otomatik olarak açma komutu üretir.

## Dosya Yapısı

- `data/curtain_usage_sample.csv`: Örnek veri seti
- `src/train_model.py`: Derin öğrenme modelini eğitir
- `src/predict_and_control.py`: Tahmin yapar ve otomatik kontrol kararını verir
- `src/config.py`: Ayarlar
- `src/curtain_controller.py`: Perde açma/kapatma komutları için örnek kontrol dosyası
- `requirements.txt`: Gerekli kütüphaneler

## Veri Seti Formatı

Kendi verinizi aşağıdaki formatta hazırlayabilirsiniz:

```csv
date,day_of_week,open_time,close_time
2026-04-01,2,07:35,20:15
2026-04-02,3,07:40,20:20
```

Açıklama:

- `date`: Tarih
- `day_of_week`: Haftanın günü. Pazartesi = 0, Pazar = 6
- `open_time`: Perdenin kullanıcı tarafından açıldığı saat
- `close_time`: Perdenin kullanıcı tarafından kapatıldığı saat

## Kurulum

```bash
pip install -r requirements.txt
```

## Modeli Eğitme

```bash
python src/train_model.py
```

## Tahmin ve Kontrol Çalıştırma

```bash
python src/predict_and_control.py
```

## Not

Bu proje gerçek perde motoruna doğrudan bağlanmaz. `curtain_controller.py` dosyasında bulunan fonksiyonlar örnek olarak hazırlanmıştır. Daha sonra Raspberry Pi, Arduino, ESP32 veya akıllı ev API entegrasyonu bu dosyaya eklenebilir.
