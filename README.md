## NASDAQ Trader Transcript Analyzer (TR)

YouTube’daki hisse/finans videolarının sesini indirir, Türkçe’ye transkribe eder, videoda GERÇEKTEN söylenenlerden yola çıkarak trade fikirlerini çıkarır ve hem okunabilir rapor hem de JSON üretir. Raporda video üreticisinin adı da yer alır. Proje, “halüsinasyon yok” prensibiyle yalnızca video metnine dayanır.

### Kimler için?
- Kod bilmeyen yatırımcılar ve trader’lar
- Türkçe finans içeriklerinden hızlı, güvenilir ve “kaynağa dayalı” özet/trade fikri çıkarmak isteyenler

### Başlıca Özellikler
- YouTube → Ses (MP3) indirip önbellekler
- Whisper ile Türkçe transkript ve zaman segmentleri (timestamp) üretir
- Gemini ile yapılandırılmış trade fikirleri (JSON) çıkarır; yalnız transkriptte geçen bilgiler kullanılır
- Video üreticisi (kanal/uploader) rapor başında yer alır
- Tekrar çalıştırmalarda aynı transkript için aynı sonuçlar (deterministik önbellekleme)

---

## Gereksinimler
- Python 3.10+
- FFmpeg (ses işleme için)
- Google Gemini API anahtarı (GOOGLE_API_KEY)

## Kurulum (Adım Adım)

1) Python’u kurun
- `https://www.python.org/downloads/` adresinden indirin ve kurun.

2) FFmpeg’i kurun (Windows)
- `https://www.gyan.dev/ffmpeg/builds/` → “release full” sürümünü indirin → zip’i açın → `bin` klasörünü PATH’e ekleyin.
- Doğrulama (Yeni terminal):
```
ffmpeg -version
```
Sürüm çıktısı görmelisiniz.

3) Projeyi indirin
- GitHub’da yeşil “Code” → “Download ZIP” → zip’i açın.

4) API anahtarınızı tanımlayın
- `.env.example` dosyasını kopyalayıp `.env` olarak yeniden adlandırın.
- `.env` içine şunu yazın:
```
GOOGLE_API_KEY=buraya_anahtarınızı_yazın
```

5) Bağımlılıkları yükleyin
Terminalde proje klasörüne geçip:
```
pip install -r requirements.txt
```
Not: CPU ile kullanacaksanız PyTorch GPU kurulumunu atlayabilirsiniz. FFmpeg’in kurulu olması şarttır.

6) Video linklerini ekleyin
- `video_list.txt` dosyasına YouTube linklerinizi (her satıra bir link) girin.

7) Jupyter ile çalıştırma (Önerilir)
- `Nasdaq_Trader.ipynb` dosyasını açın.
- “1) Full Pipeline Code” hücresini çalıştırın (tanımlar yüklenir).
- “2) Run the Pipeline” hücresini çalıştırın (işlem başlar).
- Çıktılar `summary/` klasöründe oluşur.

8) Komut satırından çalıştırma (Opsiyonel)
```
python nasdaq_trader.py
```

---

## Üretilen Çıktılar
- `summary/summary_<VIDEOID>_<ZAMAN>.txt`: Okunabilir rapor
- `summary/summary_<VIDEOID>_<ZAMAN>.json`: Makine okunabilir sonuçlar
- `summary/run.log`: Günlük (log) dosyası

## “Halüsinasyon Yok” Garantisi Nasıl Sağlanır?
- Modelin yönergeleri yalnızca transkriptte yer alan bilgilere dayanmayı zorunlu kılar.
- Zaman damgaları (timestamp) Whisper segmentlerinden seçilir.
- Transkriptte geçmeyen alanlar boş/0.0 bırakılır; tahmin edilmez.
- Aynı transkript ve ayarlar için sonuçlar önbellekten deterministik olarak alınır.

## Sık Karşılaşılan Sorunlar
**FFmpeg bulunamadı**
- PATH ayarınızı kontrol edin, terminalde `ffmpeg -version` çıkıyor olmalı.

**yt-dlp/Whisper/Google import hatası**
- `pip install -r requirements.txt` komutunu doğru ortamda (aynı Python) çalıştırdığınızdan emin olun.

**503 veya kota hataları (Gemini)**
- Bir süre sonra tekrar deneyin; model yoğun olabilir. Tekrar denemede önbellek sayesinde önceki sonuçlar korunur.

**Çıktılar neden farklı olabilir?**
- Son sürüm deterministik önbellekleme kullanır. Aynı transkript için tekrar işlem yapsanız bile aynı sonuçlar gelir (ayarlar değişmediği sürece).

## Gizlilik ve Sınırlamalar
- Yalnızca videoda geçen bilgiler raporlanır; dış kaynak eklenmez.
- Transkript kalitesi videonun ses kalitesine bağlıdır.

## Katkı
- Hata/öneri için GitHub “Issues” bölümünden bildirebilirsiniz.


