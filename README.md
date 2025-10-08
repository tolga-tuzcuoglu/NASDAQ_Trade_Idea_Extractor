# 🚀 NASDAQ Trader - AI Destekli YouTube Finans Video Analizörü

**Otomatik YouTube Finans Video Analiz Pipeline'ı** - ses indirir, Türkçe transkript oluşturur, videoda GERÇEKTEN söylenenlere dayanarak trade fikirlerini çıkarır ve kapsamlı raporlar/JSON üretir. **Sıfır halüsinasyon** prensipleriyle inşa edilmiştir - raporlar sadece video içeriğine dayanır.

## 🎯 **Kimler İçin**
- **Kod bilmeyenler**: Yatırımcılar ve trader'lar için otomatik analiz
- **Finans profesyonelleri**: Hızlı, güvenilir, kaynak tabanlı trade fikri çıkarma
- **İçerik üreticileri**: Kendi finans videolarını analiz etme
- **Araştırmacılar**: Finans video içeriğinin sistematik analizi

## ✨ **Temel Özellikler**

### 🔒 **Sıfır Halüsinasyon Garantisi**
- ✅ **AI halüsinasyonu yok** - raporlar sadece video içeriğine dayanır
- ✅ **Kaynak doğrulama** - sadece videoda gerçekten bahsedilen bilgiler
- ✅ **Her seferinde taze AI analizi** (AI önbelleği yok)
- ✅ **Deterministik önbellekleme** - aynı transkript = aynı sonuçlar

### 🎬 **Video İşleme**
- ✅ **YouTube ses indirme** gelişmiş kimlik doğrulama ile
- ✅ **Türkçe transkripsiyon** OpenAI Whisper kullanarak
- ✅ **Otomatik zaman damgası çıkarma** video segmentlerinden
- ✅ **Video üretici bilgisi** raporlarda yer alır
- ✅ **İlerleme takibi** ETA ve gerçek zamanlı güncellemeler ile

### 📊 **Trade Analizi**
- ✅ **Tüm varlık türleri**: Hisse senetleri, ETF'ler, Kripto, Emtialar
- ✅ **Dinamik ticker çözümleme** yfinance API kullanarak
- ✅ **Akıllı tekrar önleme** - aynı raporda gereksiz ticker'lar yok
- ✅ **Kapsamlı raporlar** Türkçe tez ve risk değerlendirmesi ile
- ✅ **JSON çıktı** programatik erişim için

### 🛡️ **Gelişmiş Kimlik Doğrulama**
- ✅ **Çoklu strateji YouTube kimlik doğrulama** (tarayıcı çerezleri, manuel çerezler, başlıklar)
- ✅ **Windows uyumluluğu** çerez veritabanı erişim düzeltmeleri ile
- ✅ **Korumalı videolar için yedek mekanizmalar**
- ✅ **Zorlu videolar için manuel çerez çıkarma rehberi**

## 📋 **Gereksinimler**
- **Python 3.10+**
- **FFmpeg** (ses işleme için)
- **Google Gemini API anahtarı** (ücretsiz katman mevcut)
- **yt-dlp** (YouTube indirmeleri için)

## 🚀 **Hızlı Başlangıç**

### **1. Kurulum**
```bash
# Depoyu klonlayın
git clone https://github.com/yourusername/nasdaq-trader-pipeline.git
cd nasdaq-trader-pipeline

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### **2. Yapılandırma**
```bash
# Yapılandırma şablonunu kopyalayın
cp config.yaml config_local.yaml

# Ortam dosyası oluşturun
echo "GEMINI_API_KEY=api_anahtarınız_buraya" > .env
```

### **3. Videolarınızı Ekleyin**
```bash
# video_list.txt dosyasını YouTube URL'lerinizle düzenleyin
echo "https://youtube.com/watch?v=VIDEO_ID" >> video_list.txt
```

### **4. Analizi Çalıştırın**
```bash
# Seçenek 1: Jupyter Notebook (Önerilen)
jupyter notebook Nasdaq_Trader.ipynb

# Seçenek 2: Komut Satırı
python nasdaq_trader.py
```

## 📁 **Proje Yapısı**
```
nasdaq-trader-pipeline/
├── nasdaq_trader.py              # Ana Python scripti
├── Nasdaq_Trader.ipynb          # Jupyter notebook
├── config.yaml                  # Yapılandırma şablonu
├── requirements.txt             # Python bağımlılıkları
├── cookies_template.txt         # Manuel çerez çıkarma rehberi
├── .gitignore                  # Git ignore dosyası
├── README.md                   # Bu dosya
└── .env                        # API anahtarlarınız (bunu oluşturun)
```

## ⚙️ **Yapılandırma Seçenekleri**

### **Model Performansı**
```yaml
MODEL_PERFORMANCE:
  WHISPER_MODEL: "medium"        # tiny|small|medium|large|large-v2|large-v3
  GEMINI_MODEL: "gemini-2.5-flash" # gemini-2.5-flash|gemini-1.5-pro
  CHUNK_SIZE: 12000              # Uzun transkriptler için karakter sayısı
  ENABLE_MODEL_CACHING: true     # Whisper modelini bellekte tut
```

### **İşleme Ayarları**
```yaml
PROCESSING:
  ENABLE_PROGRESS_TRACKING: true  # İlerleme çubukları ve ETA göster
  ENABLE_DETAILED_LOGGING: true   # Daha ayrıntılı günlük
  MAX_VIDEO_LENGTH_MINUTES: 120  # Bu süreden uzun videoları atla
  MIN_VIDEO_LENGTH_MINUTES: 1    # Bu süreden kısa videoları atla
```

### **YouTube Kimlik Doğrulama**
```yaml
YOUTUBE_AUTHENTICATION:
  ENABLE_BROWSER_COOKIES: true    # Kimlik doğrulama için tarayıcı çerezleri kullan
  PREFERRED_BROWSERS: ["chrome", "firefox", "edge", "safari"]
  MAX_RETRIES_PER_BROWSER: 3      # Tarayıcı başına deneme sayısı
  FALLBACK_TO_NO_AUTH: true       # Son çare olarak kimlik doğrulama olmadan dene
```

## 📊 **Üretilen Çıktılar**

### **Rapor Dosyaları**
- `summary/summary_<VIDEO_ID>_<ZAMAN>.txt` - İnsan tarafından okunabilir rapor
- `summary/summary_<VIDEO_ID>_<ZAMAN>.json` - Makine tarafından okunabilir JSON
- `summary/run.log` - İşleme günlükleri

### **Rapor Yapısı**
```
--- YÖNETİCİ TRADE ÖZET RAPORU ---
Oluşturulma Tarihi: 2025-01-08 22:30:00
Kaynak Video: https://youtube.com/watch?v=VIDEO_ID
Video Başlığı: Finans Video Başlığı
Video Üreticisi: Kanal Adı
Rapor Dosyası: summary_VIDEO_ID_ZAMAN.txt
-------------------------------------

=== TRADE FİKRİ 1: Hisse Senedi | AAPL (00:02:15'te başlar) ===
  Aksiyon:        Long
  Sentiment:      BULLISH
  Ufuk:          Uzun Vadeli
  Risk:          Orta
  Piyasa:        NASDAQ
  Destek:        $150.00
  Direnç:        $200.00
  Hedef Fiyat:   $180.00
  Stop Loss:     $140.00
-------------------------------------
  Ana Tez (TR Orijinal) - AAPL (Hisse Senedi):
  Apple'ın yeni ürün lansmanları ve güçlü finansal performansı...
-------------------------------------
```

## 🔧 **Gelişmiş Özellikler**

### **Gelişmiş Kimlik Doğrulama Sistemi**
Sistem otomatik olarak birden fazla kimlik doğrulama stratejisi dener:

1. **Manuel Çerezler** (en güvenilir)
2. **Tarayıcı Çerezleri** (Firefox, Chrome, Edge)
3. **Gelişmiş Başlıklar** (gerçek tarayıcıyı taklit eder)
4. **Minimal Seçenekler** (genel videolar için yedek)

### **Dinamik Ticker Çözümleme**
- Gerçek zamanlı ticker çözümleme için `yfinance` API kullanır
- Hardcoded şirket eşleştirmeleri yok
- Çözümleme başarısız olursa orijinal isme geri döner
- Tüm varlık türlerini destekler: Hisse senetleri, ETF'ler, Kripto, Emtialar

### **Akıllı Önbellekleme Sistemi**
- **Ses önbelleği**: İndirilen MP3 dosyaları
- **Transkript önbelleği**: Oluşturulan transkriptler
- **AI önbelleği**: Taze analiz için otomatik silinir
- **Model önbelleği**: Whisper modeli bellekte tutulur

## 🛠️ **Sorun Giderme**

### **Yaygın Sorunlar**

**"FFmpeg bulunamadı"**
```bash
# Windows: https://www.gyan.dev/ffmpeg/builds/ adresinden indirin
# PATH'e ekleyin, sonra doğrulayın:
ffmpeg -version
```

**"yt-dlp/Whisper/Google import hatası"**
```bash
# Doğru ortamda olduğunuzdan emin olun:
pip install -r requirements.txt
```

**"Bot olmadığınızı doğrulamak için giriş yapın"**
- Sistem otomatik olarak birden fazla kimlik doğrulama yöntemi deneyecek
- Hepsi başarısız olursa, `cookies_template.txt` dosyasındaki manuel çerez çıkarma rehberini takip edin

**"Geçersiz veya desteklenmeyen YouTube URL"**
- Desteklenen formatlar: `youtube.com/watch?v=`, `youtu.be/`, `/shorts/`, `/live/`
- URL formatınızı kontrol edin

### **Performans Optimizasyonu**
- Daha hızlı işleme için `WHISPER_MODEL: "small"` kullanın
- Birden fazla video için `ENABLE_MODEL_CACHING: true` etkinleştirin
- Transkript uzunluğunuza göre `CHUNK_SIZE` ayarlayın

## 🔒 **Güvenlik ve Gizlilik**

### **GitHub'a Yüklenmesi Güvenli Olanlar**
- ✅ Kaynak kod dosyaları (`.py`, `.ipynb`)
- ✅ Yapılandırma şablonları (`config.yaml`)
- ✅ Dokümantasyon dosyaları (`README.md`)
- ✅ Şablon dosyaları (`cookies_template.txt`)

### **Yüklenmesi Güvenli OLMAYANlar**
- ❌ `cookies.txt` (kişisel kimlik doğrulamanız)
- ❌ `.env` (API anahtarlarınız)
- ❌ `video_cache/` (indirilen ses dosyaları)
- ❌ `summary/` (analiz raporlarınız)
- ❌ `logs/` (ayrıntılı günlükler)

## 📈 **Performans Metrikleri**

### **Tipik İşleme Süreleri**
- **Ses İndirme**: 30-60 saniye
- **Transkripsiyon**: 2-5 dakika (video uzunluğuna bağlı)
- **AI Analizi**: 1-3 dakika
- **Rapor Oluşturma**: 10-30 saniye

### **Kaynak Kullanımı**
- **RAM**: 2-4 GB (model önbellekleme ile)
- **Depolama**: Video başına 50-200 MB (ses + önbellek)
- **API Çağrıları**: Video başına 1-3 Gemini API çağrısı

## 🤝 **Katkıda Bulunma**

### **Nasıl Katkıda Bulunulur**
1. Depoyu fork edin
2. Özellik dalı oluşturun
3. Değişikliklerinizi yapın
4. Kapsamlı test edin
5. Pull request gönderin

### **Sorun Bildirme**
- Hata raporları için GitHub Issues kullanın
- Hata günlükleri ve sistem bilgilerini dahil edin
- Sorunu yeniden üretme adımlarını sağlayın

## 📄 **Lisans**

Bu proje açık kaynaklıdır ve MIT Lisansı altında kullanılabilir.

## 🙏 **Teşekkürler**

- **OpenAI Whisper** - konuşmadan metne transkripsiyon için
- **Google Gemini** - AI destekli analiz için
- **yt-dlp** - YouTube video indirme için
- **yfinance** - dinamik ticker çözümleme için

---

**🚀 Finans videolarınızı analiz etmeye hazır mısınız? Yukarıdaki Hızlı Başlangıç rehberiyle başlayın!**