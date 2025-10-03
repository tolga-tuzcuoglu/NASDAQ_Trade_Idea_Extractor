# NASDAQ Borsa İşlem Fikirleri Edinme (Gemini + Whisper) 🤖📈

Bu araç, YouTube borsa analiz videolarını (Türkçe) otomatik olarak analiz eder, konuşmaları metne dönüştürür (Whisper), yatırım fikirlerini ayıklar (Gemini) ve yapılandırılmış bir özet raporu oluşturur.

Bu aracı kullanmak için hiçbir kodlama bilgisine veya kod satırına dokunmanıza gerek yoktur, ancak aşağıdaki 5 adımı sırayla ve dikkatle tamamlamanız gerekir.

-----

## 🛠️ Adım 1: Gerekli Programları Kurma (En Baştan)

Bu programın çalışması için bilgisayarınızda üç temel araç olmalıdır. Lütfen bunları kurduğunuzdan emin olun:

### 1\. Python 🐍 (Programın Çalışma Ortamı)

  * Python İndirme: Google'da "Python indir" diye aratın veya resmi siteye gidin.
  * En son sürümü (örneğin Python 3.12) indirin.
  * Kurulum dosyasını çalıştırın. KURULUM EKRANINDA ÇOK ÖNEMLİ: Ekranda göreceğiniz "Add python.exe to PATH" (veya benzer bir ifade) kutucuğunu kesinlikle İŞARETLEYİN. Bu kutucuk işaretli olmazsa program çalışmaz\!
  * Kurulumu bitirin.

### 2\. yt-dlp ve FFmpeg 🎬 (Video İşleme Araçları)

  * Bu iki araç, programın YouTube videosunun sesini indirip hazırlaması için gereklidir.
  * Arama Yapın: Google'da ayrı ayrı "yt-dlp kurulumu" ve "FFmpeg kurulumu" diye aratarak kendi işletim sisteminize (Windows, Mac) uygun basit kurulum rehberlerini bulun ve kurun.
  * Kurulum sırasında, bu programların da bilgisayarınızın "PATH" kısmına eklendiğinden emin olun. Bu, komut ekranında çalışabilmeleri için gereklidir.

-----

## 🔑 Adım 2: Google Gemini API Anahtarını Hazırlama

Bu program, yapay zeka analizi için Google'ın Gemini 2.5 Flash modelini kullanır. Bu hizmeti kullanmak için size özel bir anahtar almalısınız.

1.  Anahtarınızı Alın: [Google AI Studio](https://aistudio.google.com/app/apikey) adresine gidin ve ücretsiz bir API anahtarı oluşturun.
2.  Projeyi İndirin: Bu GitHub sayfasındaki \< \> Code (Kod) düğmesine tıklayıp "Download ZIP" (ZIP İndir) seçeneğiyle projeyi bilgisayarınıza indirin ve bir klasöre açın.
3.  `.env` Dosyasını Oluşturun:
      * İndirdiğiniz bu klasörde, `env__example.txt` adında bir şablon dosyası göreceksiniz.
      * Bu dosyanın adını değiştirip tam olarak sadece `.env` yapın (Dosya adında ön başta nokta olmalı).
4.  Anahtarınızı Ekleyin: Yeni ismini verdiğiniz `.env` dosyasını Not Defteri veya herhangi bir metin düzenleyiciyle açın. İçindeki `"[YOUR_GEMINI_API_KEY_GOES_HERE]"` metnini silerek, kendi gerçek API anahtarınızla değiştirin.
      * Örnek görünüm: `GEMINI_API_KEY="AIzaSy...sizin_gercek_gizli_anahtarınız...xyz123"`

-----

## 👨‍💻 Adım 3: Gerekli Program Eklentilerini Kurma

Bu adımda, Python'un programı çalıştırması için gerekli olan eklentileri (kütüphaneleri) kuracağız.

1.  Komut Ekranını Açın: Proje klasörünüze (ZIP'ten çıkardığınız klasör) girin. Klasörün içinde, üstteki adres çubuğuna tıklayın.
2.  Adres çubuğundaki her şeyi silin ve sadece `cmd` yazıp Enter tuşuna basın. (Bu, tam olarak bu klasörde bir komut penceresi açar.)
3.  Açılan siyah ekrana aşağıdaki komutu yazın ve Enter tuşuna basın:
    ```bash
    pip install -r requirements.txt
    ```
    *(Bu komut, programın ihtiyacı olan tüm küçük yardımcı yazılımları otomatik olarak kurar. Birkaç dakika sürebilir.)*

-----

## Adım 4: Analiz Edilecek Videoları Belirleme (Çok Önemli\!)

Programın hangi videoları analiz edeceğini bu dosyaya yazıyorsunuz.

1.  Ana çalışma klasörünüzdeki `video_list.txt` dosyasını Not Defteri ile açın.
2.  Analiz edilmesini istediğiniz her bir YouTube videosunun URL'sini yeni bir satıra ekleyin.
      * Her satıra sadece bir tane YouTube linki koyun.
      * Örnek `video_list.txt` içeriği:
        ```
        # Analiz etmek istediğiniz YouTube video linklerini buraya ekleyin.
        https://www.youtube.com/watch?v=dmmD9xvtjE4
        https://www.youtube.com/watch?v=bir_baska_video
        ```

-----

## Adım 5: Analizi Başlatma

1.  Hâlâ açık olan siyah komut ekranına (veya kapattıysanız, yeniden açtığınız komut ekranına) aşağıdaki iki komutu sırayla yazın ve her birinden sonra Enter tuşuna basın:
    ```bash
    jupyter nbconvert --to script NASDAQ_Trade_Idea_Extractor_GitHub.ipynb
    python NASDAQ_Trade_Idea_Extractor_GitHub.py
    ```
    *(İlk komut, ana program dosyasını çalıştırılabilir bir Python betiğine dönüştürür. İkinci komut ise programı başlatır.)*
2.  Program çalışırken ekranda indirme ve analiz adımlarını göreceksiniz. Lütfen bitmesini bekleyin.

### Raporu İnceleme

1.  Program "BATCH PROCESSING COMPLETE" (Toplu İşlem Tamamlandı) yazdıktan sonra, ana çalışma klasörünüzdeki `summary` adında yeni bir klasöre girin.
2.  Bu klasörün içinde, video ID'si ve tarih/saat bilgisi içeren isimli Türkçe Özet Rapor dosyalarını bulacaksınız. Bu `.txt` dosyalarını açarak analiz sonuçlarını okuyabilirsiniz.
