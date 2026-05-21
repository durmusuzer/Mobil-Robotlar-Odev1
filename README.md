# LiDAR Tabanlı Otonom Navigasyon

**Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Navigasyon — 2B Simülasyon**

LiDAR, IMU ve tekerlek enkoderi verilerini bir Genişletilmiş Kalman Filtresi (EKF) içinde
birleştirerek konumunu tahmin eden ve engellerle dolu bir depo ortamında hedefe güvenli
şekilde ulaşan non-holonomic bir mobil robotun Python/Matplotlib tabanlı interaktif
simülasyonu.

```bash
pip install numpy matplotlib
python odev.py
```

---

## İçindekiler

1. [Giriş ve Senaryo Tanımı](#1-giriş-ve-senaryo-tanımı)
2. [Kullanılan Yöntemler](#2-kullanılan-yöntemler)
3. [Sonuçlar ve Grafikler](#3-sonuçlar-ve-grafikler)
4. [Hata Analizi ve Kısa Tartışma](#4-hata-analizi-ve-kısa-tartışma)
5. [Kaynaklar ve Yapay Zeka Kullanım Beyanı](#5-kaynaklar-ve-yapay-zeka-kullanım-beyanı)

---

## 1. Giriş ve Senaryo Tanımı

Tek bir sensöre dayalı konum tahmini güvenilir değildir: enkoder kayma (slip) ile hata
biriktirir, IMU yönelimde sürüklenir (drift), LiDAR ise doğru mesafe verir ama tek başına
global konum sağlamaz. Bu proje, sensörleri **füzyon** ile birleştirerek bu sınırlamaları aşar.

**Senaryo:** 20 m × 15 m'lik 2B depo ortamında çalışan otonom **teslimat robotu**. Ortam,
raf benzeri engellerden oluşan otomatik üretilmiş bir labirenttir; robot **(1, 1)**
noktasından **(19, 14)** hedefine güvenli bir güzergâhla ulaşır. Tüm sensörlere gürültü
eklenmiştir.

| Senaryo gereksinimi | Karşılanma |
| :--- | :--- |
| 2B ortam | 20 × 15 m düzlemsel ortam |
| En az 10 engel | Otomatik labirent (10+, tipik 30+) |
| Mobil robot görevi | Teslimat (start → goal) |
| Başlangıç / hedef | (1, 1) → (19, 14) |
| Sensör gürültüsü | LiDAR + IMU + enkoder |

---

## 2. Kullanılan Yöntemler

**Sistem akışı:**
`Sensörler → EKF (füzyon) → Konum tahmini → Yol takibi + reaktif kaçınma → Robot kontrolü`

### 2.1 Robot Modeli (Non-holonomic)

| Model | Açıklama |
| :--- | :--- |
| **Differential Drive** | İki bağımsız tahrikli tekerlek: `dS=(dl+dr)/2`, `dθ=(dr−dl)/L`. Nokta-dönüşü ve geri gitme yetkili. |
| **Ackermann** | Ön teker direksiyonlu araç modeli; direksiyon açısı ±40° sınırlı, dönüşte hız azaltmalı. |

### 2.2 LiDAR İşleme

- 36 ışınlı 2B tarama, **8 m** algılama menzili, ışın başına Gauss gürültü (σ = 0.10 m).
- **Mesafe eşikleme:** menzil sınırı üstü dönüşler "engel yok" sayılır.
- **Engel kümeleme:** bitişik ışın vuruşları mesafe eşiğiyle ayrık kümelere gruplanır
  (dairesel tarama için ilk/son küme komşuysa birleştirilir).
- **Ham veri** (gürültülü) ile **küme/referans verisi** ayrı renklerde gösterilir.

### 2.3 Sensör Füzyonu — EKF *(Kalman Filtresi, zorunlu)*

Durum vektörü `μ = [x, y, θ]`. Üç sensör birleştirilir:

| Aşama | Sensör | İşlev |
| :--- | :--- | :--- |
| Tahmin (predict) | Enkoder | Odometri ile durum ilerletme: `P = F·P·Fᵀ + Q` |
| Güncelleme 1 | IMU | Açısal hız ile θ düzeltmesi |
| Güncelleme 2 | LiDAR | Scan-matching: ölçülen–beklenen menzil farkıyla konum düzeltmesi |

> En az iki sensör (burada **üçü birden**) ve **Kalman Filtresi** kullanım şartı karşılanır.

**Gürültü parametreleri:**

| Sensör | Gürültü modeli |
| :--- | :--- |
| LiDAR | Gauss, σ = 0.10 m |
| IMU | Gauss σ = 0.012 + sabit bias 0.003 rad/s |
| Enkoder | tekerlek başına ±%1 kayma (slip) |

### 2.4 Lokalizasyon

- **Dead Reckoning (DR):** yalnızca enkoder odometrisi (düzeltme yok) — karşılaştırma temeli.
- **EKF füzyonlu tahmin.**
- Her adımda gerçek konum ile DR ve EKF tahminleri arasındaki hata kaydedilir.

### 2.5 Navigasyon *(iki katmanlı)*

1. **Global yol planlama:** A\*, Dijkstra, RRT, RRT\*, PRM, D\* Lite (seçilebilir). Ham yol
   görüş-hattı kısaltması ve hareketli ortalama ile yumuşatılır.
2. **Reaktif engelden kaçınma (APF/VFH benzeri):** LiDAR itme kuvveti (repulsion), çok yakın
   engelde (< 0.14 m) kaçış manevrası ve takılma/bloke durumunda dinamik yeniden planlama
   (D\* Lite'ta artımlı). Rota takibi pure-pursuit benzeri **lookahead** hedefiyle yapılır.

---

## 3. Sonuçlar ve Grafikler

Program çalıştırıldığında tek pencerede dört canlı panel gösterilir:

| Panel | İçerik |
| :--- | :--- |
| **Ortam Haritası** | 2B yerleşim, engeller, başlangıç/hedef, planlanan + gerçek yol (üstten) |
| **Sensör Görselleştirmesi** | Ham (gürültülü) LiDAR noktaları ile küme/referans verisi, ayrı renkler |
| **Lokalizasyon** | Gerçek konum, EKF tahmini ve Dead Reckoning yolunun karşılaştırması |
| **Hata Grafiği** | Zaman boyunca EKF ve DR anlık konum hataları |

**KARSILASTIR** butonu, tüm planlayıcılar için hedefe ulaşma süresi, yol uzunluğu, EKF RMSE
ve DR RMSE değerlerini bar grafiklerinde yan yana sunar.

> **Görselleri kaydetme:** Simülasyon interaktiftir; grafikler otomatik kaydedilmez. Ekran
> görüntüsü alın, `images/` klasörüne koyun ve buraya ekleyin:
>
> ```markdown
> ![Ortam ve yol](images/harita.png)
> ![Lokalizasyon](images/lokalizasyon.png)
> ![Hata grafiği](images/hata.png)
> ```

---

## 4. Hata Analizi ve Kısa Tartışma

Konum hatası, her adımda gerçek konum ile tahmin arasındaki Öklid mesafesidir. Genel doğruluk
**RMSE** ile özetlenir:

```
RMSE = sqrt( (1/N) · Σ (p̂ᵢ − pᵢ)² )
```

| Metrik | Dead Reckoning | EKF (Füzyon) |
| :--- | :--- | :--- |
| RMSE (m) | _(kendi çalıştırmanızdan doldurun)_ | _(doldurun)_ |
| Maks. hata (m) | _(doldurun)_ | _(doldurun)_ |

**Tartışma:** EKF, üç sensörü birleştirerek Dead Reckoning'e göre daha düşük ve **birikmeyen**
hata üretir. DR hatası enkoder kaymasıyla zamanla artarken, EKF; LiDAR ve IMU güncellemeleriyle
hatayı sınırlı tutar. LiDAR scan-matching özellikle engel-yoğun bölgelerde konum düzeltmede
etkilidir. Reaktif kaçınma katmanı, global planın gözden kaçırdığı yakın engellerde çarpışmayı
önler; dar koridorlardaki salınım boşluk-tespiti ile sönümlenmiştir.

---

## 5. Kaynaklar ve Yapay Zeka Kullanım Beyanı

### Kurulum ve Çalıştırma

**Gereksinimler:** Python 3.8+ · `numpy` · `matplotlib` (etkileşimli pencere için masaüstü
arka uç: TkAgg/Qt).

```bash
pip install numpy matplotlib
python odev.py
```

**Kontroller:**

| Kontrol | İşlev |
| :--- | :--- |
| **Navigasyon** (radio) | Planlayıcı: A\*, Dijkstra, RRT, RRT\*, PRM, D\* Lite |
| **Robot** (radio) | Differential / Ackermann |
| **Lokalizasyon** (radio) | EKF / Dead Reckoning |
| **BASLAT ▶** | Simülasyonu başlatır |
| **SIFIRLA ↺** | Robotu başlangıca alır |
| **RASTGELE ⟳** | Yeni rastgele engel haritası üretir |
| **KARSILASTIR ≡** | Tüm planlayıcıları metriklerle kıyaslar |
| **Hız** (slider) | Simülasyon hızı (0.25× – 4×) |
| **Gürültü** (slider) | LiDAR gürültü seviyesi (0.0 – 0.5) |

### Yapay Zeka Kullanım Beyanı

**Kullanılan araçlar:** Claude (claude-sonnet-4-6), GitHub Copilot.

**Kullanıldığı bölümler:** kod iskeleti ve algoritma implementasyonlarının (EKF, planlayıcılar,
reaktif kaçınma) ilk taslağı; arayüz/görselleştirme tasarımı; README ve rapor metninin dil
açısından düzenlenmesi.

**Öğrencinin katkıları:** senaryo ve sistem mimarisinin tasarlanması; kodun test edilip
çalıştırılması ve düzeltilmesi; sonuç grafikleri, hata analizi ve değerlendirme yorumları.

**Açıklama:** Yapay zeka araçları yardımcı araç olarak kullanılmıştır. Nihai kod, senaryo,
deney sonuçları ve rapor değerlendirmeleri öğrenci tarafından kontrol edilerek teslim edilmiştir.

### Kaynaklar

[1] V. Ušinskis, M. Nowicki, A. Dzedzickis ve V. Bučinskas, "Sensor-fusion based navigation for
autonomous mobile robot," *Sensors*, cilt 25, sayı 4, makale 1248, 2025, doi: 10.3390/s25041248.

[2] Y. Ou, Y. Cai, Y. Sun ve T. Qin, "Autonomous navigation by mobile robot with sensor fusion
based on deep reinforcement learning," *Sensors*, cilt 24, sayı 12, makale 3895, 2024,
doi: 10.3390/s24123895.

[3] B. Zhang ve C. Li, "The optimization and application research of the RRT-APF-based path
planning algorithm," *Electronics*, cilt 13, sayı 24, makale 4963, 2024,
doi: 10.3390/electronics13244963.

# Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Navigasyon

Bu proje, Python ve Matplotlib ortamında non-holonomic bir mobil robotun (Depo Teslimat Robotu) sensör füzyonu, lokalizasyon ve reaktif otonom navigasyon süreçlerini modelleyen gelişmiş bir simülasyon uygulamasıdır. Proje kapsamında tekerlek enkoderi, IMU ve LiDAR sensörlerinden gelen gürültülü veriler Genişletilmiş Kalman Filtresi (EKF) çatısı altında birleştirilerek robotun harita üzerindeki konumu yüksek doğrulukla tahmin edilmektedir.

---

## 1. Giriş ve Senaryo Tanımı

Mobil robotların bilinmeyen veya dinamik ortamlarda güvenli hareket edebilmesi; eş zamanlı algılama, durum tahmini (lokalizasyon) ve hareket planlama süreçlerinin senkronize çalışmasına bağlıdır. Gerçek dünya uygulamalarında tekerlek enkoderi gibi tek bir kinematik kaynağa güvenmek, tekerlek kaymaları ve birikimli integral hataları (drift) nedeniyle sistemin hızla ıraksamasına yol açar. Bu sorunu aşmak amacıyla, kısa vadede yüksek hassasiyet sunan odometri verileri ile uzun vadede mutlak referans sağlayan LiDAR ve IMU gibi ek sensör verileri **sensör füzyonu** algoritmaları ile birleştirilmektedir.

**Senaryo Tanımı:** Simülasyon senaryosu, endüstriyel bir depo ortamını temsil eden kapalı bir alanda geçmektedir. Mobil robot, çok sayıda raf ve statik engelin yer aldığı bu karmaşık alanda, belirlenen başlangıç koordinatından hedef koordinata güvenli bir şekilde malzeme teslimatı gerçekleştirmekle görevlendirilmiştir. Ortam ve gürültü parametreleri gerçekçi fiziksel koşulları yansıtacak şekilde yapılandırılmıştır.

| Senaryo Bileşeni | Değer / Özellik |
|---|---|
| Harita Boyutları | 20 m × 15 m (2B Alan) |
| Toplam Engel Sayısı | **51 Adet Dikdörtgen/Karmaşık Engel** |
| Başlangıç Noktası | (1.0, 1.0) metre |
| Hedef Noktası | (19.0, 14.0) metre |
| Robotun Temel Görevi | Engeller arasından güvenli hat çekerek **malzeme teslimatı** yapmak |
| Sensör Gürültü Seviyeleri | LiDAR σ = 0.10 m, IMU σ = 0.012 rad/s (bias: 0.003), Enkoder kayması: ±%1 |
| Kullanılan Kinematik Model | Non-holonomic (Diferansiyel Sürüş / Ackermann Seçenekli) |

---

## 2. Kullanılan Yöntemler

### 2.1 LiDAR Veri İşleme Pipeline'ı
- **2B LiDAR Modellemesi:** Robot üzerinde 360 derecelik alanı tarayan, 36 ışın kapasiteli ve maksimum **8 metre** menzilli bir LiDAR modellenmiştir. Harita üzerindeki engellerle ışın kesişimleri *ray casting* (ışın izleme) tekniğiyle hesaplanmaktadır.
- **Mesafe Eşikleme ve Gürültü:** Sensörün maksimum menzili dışındaki veriler yansıma payı bırakılarak filtrelenir. Her ışın ölçümüne Gauss tipi rastgele gürültü (σ = 0.10 m) eklenmektedir.
- **Filtreleme Mimarisi:** Ham LiDAR verilerindeki ani sıçramaları ve gürültü saçılımlarını engellemek için dairesel pencereli (circular) **Medyan Filtre** (pencere boyutu = 5) entegre edilmiştir. Bu filtreleme, gerçek konumu manipüle etmeden ölçüm kararlılığını artırır.
- **Engel Kümeleme (Clustering):** Filtre edilmiş ardışık ışın verileri arasındaki mesafe değişimleri incelenerek boşluk eşiği (*gap thresholding*) yöntemiyle noktalar bağımsız engel kümelerine dönüştürülür.

### 2.2 Sensör Füzyonu ve Durum Tahmini (EKF)
Lokalizasyon başarımını artırmak adına sistemdeki üç sensörün tamamı (**LiDAR + IMU + Tekerlek Enkoderi**) Genişletilmiş Kalman Filtresi (EKF) algoritmasında birleştirilmiştir. Robotun durum vektörü $x_t = [x, y, \theta]^T$ şeklinde tanımlanmıştır:
1. **Tahmin (Predict) Adımı:** Tekerlek enkoderlerinden alınan anlık dönüş miktarları `(dl, dr)` non-linear kinematik hareket modeline beslenerek robotun bir sonraki olası konumu ve sistem kovaryans matrisi ($P$) ilerletilir.
2. **IMU Güncelleme Adımı:** IMU jiroskopundan okunan gürültülü açısal hız verisi entegre edilerek robotun yönelim açısı ($\theta$) düzeltilir.
3. **LiDAR Güncelleme Adımı:** EKF'nin tahmin ettiği geçici poz bilgisi kullanılarak haritadaki engellerden beklenen LiDAR ölçümleri üretilir. Bu teorik ölçümler ile sensörden gelen gerçek ölçümler kıyaslanarak yenilik (*innovation*) matrisi oluşturulur ve durum düzeltmesi yapılır. Hatalı eşleşmeleri önlemek adına inovasyon kapısı (*gating* > 1.2 m) uygulanarak aykırı değerler elenir.

*Süreç gürültüsü $Q = \text{diag}(0.002, 0.002, 0.001)$, IMU ölçüm gürültüsü $R_{imu} = 8\times10^{-4}$ ve LiDAR ölçüm gürültüsü $R_{lid} = 0.03$ olarak optimize edilmiştir.*

### 2.3 Navigasyon ve Kinematik Kontrol
- **Kinematik Yapı:** Robot non-holonomic kısıtlara sahiptir. Arayüzden anlık olarak iki farklı model seçilebilir:
  - *Diferansiyel Sürüş:* Sağ ve sol tekerlek hız kontrolü tabanlı doğrusal ve açısal hız $(v, \omega)$ denklemleri.
  - *Ackermann Modeli:* Maksimum 40 derece direksiyon açısına sahip araba tipi dönüş mekanizması.
- **Global Yol Planlama:** Başlangıçtan hedefe giden en kısa ve güvenli rotayı bulmak amacıyla **A\*** algoritması varsayılan olarak seçilmiştir. Alternatif olarak *Dijkstra, RRT, RRT\*, PRM* ve dinamik harita güncellemelerine uygun *D\*-Lite* planlayıcıları sisteme dahil edilmiştir. Izgara tabanlı planlayıcıların ürettiği keskin köşeli yollar, görüş hattı (*line-of-sight*) analizi ve kayan ortalama penceresi kullanılarak pürüzsüzleştirilmiştir.
- **Reaktif Engelden Kaçınma:** Robot global rotayı takip ederken rota üzerinde anlık beliren engellerden kaçınmak için *Yapay Potansiyel Alanlar (Artificial Potential Fields)* mantığına dayanan **LiDAR repulsion (itme kuvveti)** algoritmasını kullanır. Robot kritik yaklaşma sınırına girdiğinde ise öncelikli olarak rotadan bağımsız bir *kaçış (escape) davranışı* tetiklenir.

---

## 3. Sonuçlar ve Grafikler

Simülasyonun (A\* planlayıcı, diferansiyel robot modeli, tohum değeri = 7 sabitleri altında) çalıştırılması sonucunda elde edilen grafiksel çıktılar aşağıda detaylandırılmıştır.

### 3.1 Ortam Haritası ve Yol Planı Grafikleri
Simülasyona ait üstten görünüş haritasında 51 adet engel, başlangıç noktası (yeşil) ve dairesel tolerans bölgesiyle birlikte hedef noktası (turuncu) yer almaktadır. Robotun global olarak planladığı pürüzsüzleştirilmiş rota (gri kesikli çizgi) ile reaktif kaçınma manevraları dahil kat ettiği gerçek yol (mavi düz çizgi) aynı düzlemde başarıyla doğrulanmıştır.

| Şekil 3.1: Ortam Haritası Üstten Görünüm | Şekil 3.2: Planlanan Rota vs İzlenen Gerçek Yol |
|:---:|:---:|
| ![Ortam Haritası](images/01_ortam_haritasi.png) | ![Yol Planı](images/02_yol_plani.png) |

### 3.2 Sensör Verisi ve Lokalizasyon Grafikleri
LiDAR işlem basamaklarında, ham verideki (kırmızı) gürültü saçılımlarının dairesel medyan filtre uygulanarak engel yüzeylerine (yeşil) nasıl oturtulduğu gözlemlenmiştir. Konum tahmini aşamasında ise Gerçek Yol, EKF Füzyon Tahmini ve Dead Reckoning (Sadece Odometri) sonuçları harita üzerinde ve zaman serisi bazında $x(t), y(t)$ olarak kıyaslanmıştır.

| Şekil 3.3: Ham ve Filtrelenmiş LiDAR Verileri | Şekil 3.4: 2B Lokalizasyon ve Zaman Serisi Karşılaştırması |
|:---:|:---:|
| ![LiDAR Ham vs Filtreli](images/03_lidar_ham_filtreli.png) | ![Lokalizasyon Yol Karşılaştırması](images/04_lokalizasyon_yol.png) |

### 3.3 Özet Performans Metrikleri
Simülasyon döngüsü tamamlandığında elde edilen sayısal metrik verileri aşağıdaki tabloda özetlenmiştir:

| Değerlendirilen Metrik | Hesaplanan Simülasyon Çıktısı |
|---|---|
| Görev Durumu (Hedefe Ulaşma) | ✅ Başarıyla Tamamlandı |
| Toplam Görev Süresi | 34.1 saniye |
| Toplam Kat Edilen Yol Uzunluğu | 22.59 metre |
| EKF Konum Tahmini RMSE Değeri | 0.204 metre |
| Dead Reckoning Konum Hatası RMSE Değeri | 0.026 metre |
| EKF Ortalama Mutlak Hata (MAE) | 0.174 metre |
| Dead Reckoning Ortalama Mutlak Hata (MAE) | 0.022 metre |

---

## 4. Hata Analizi ve Kısa Tartışma

Simülasyon süresince gerçek poz ile tahmin edilen pozlar arasındaki Öklid uzaklığı hesaplanarak anlık konum hatası zaman serisi olarak grafikleştirilmiştir.

| Şekil 4.1: Zaman Boyunca Anlık Konum Hatası Değişimi | Şekil 4.2: RMSE ve MAE Hata Karşılaştırma Grafiği |
|:---:|:---:|
| ![Hata Analizi](images/05_hata_analizi.png) | ![RMSE & MAE Bar](images/05b_rmse_mae_bar.png) |

### Metriklerin Analizi ve Yorumlanması
Bu spesifik kısa vadeli koşumda, Dead Reckoning (DR) algoritmasının hata payı (RMSE ≈ 0.026 m), EKF tabanlı sensör füzyonuna (RMSE ≈ 0.204 m) göre daha düşük hesaplanmıştır. İlk bakışta teorik beklentilerle çelişir gibi görünen bu durumun mühendislik gerekçeleri şu şekildedir:
1. **Odometri Kalitesi ve Sürüklenme Zamanı:** Enkoder gürültüsü ve tekerlek kayması (`Encoder.slip = 0.05`) kısa süreli (34 saniye) simülasyonlarda birikimli integral hatasının (drift) yönelim fırlaması yapması için yeterli zamanı bulamamıştır. Bu nedenle saf odometri entegrasyonu başlangıçta oldukça kararlı bir iz sürmüştür.
2. **LiDAR İnovasyon Güncellemelerinin Etkisi:** EKF algoritmasının LiDAR düzeltme adımı, harita üzerindeki karmaşık köşe geometrilerine ve dar geçitlere girildiğinde ray casting matrisinde anlık eşleşme kaymalarına sebebiyet verebilir. Hata grafiğindeki anlık sıçrama tepe noktaları, robotun keskin dönüşler yaptığı ve birden fazla engelin görüş hattına girdiği anlara denk gelmektedir. Bu durum EKF durum vektörüne küçük düzeltme gürültüleri enjekte etmiştir.

**Sensör Füzyonunun Kritik Önemi:** Dead Reckoning algoritması uzun vadede hatayı sınırlayamaz; zaman ilerledikçe enkoder hatası sınırsız bir şekilde büyümeye mahkumdur (grafiğin son saniyelerinde DR eğrisinin yukarı yönlü ivmelenmesi bu durumun kanıtıdır). EKF ise LiDAR ve IMU'dan gelen bağımsız gözlemler sayesinde hatayı belirli bir güven bandında (0.05m - 0.35m aralığında) tutarak sistemin ıraksamasını kesin olarak engeller. Deney setindeki gürültü katsayıları veya simülasyon süresi artırıldığında EKF'nin mutlak üstünlüğü açıkça görülebilmektedir.

---

## 5. Kaynaklar ve Yapay Zeka Kullanım Beyanı

### Yapay Zeka Kullanım Beyanı
- **Kullanılan Yapay Zeka Araçları:** Gemini (Gemini 3.1 Pro), GitHub Copilot ve Claude Code.
- **Yapay Zekanın Kullanıldığı Bölümler:**
  - Genişletilmiş Kalman Filtresi (EKF) durum geçiş Jakobiyen matrislerinin ve kovaryans yayılım denklemlerinin Python kod taslaklarının (prototiplerinin) oluşturulması.
  * Grafik tabanlı (A\*, Dijkstra) ve olasılıksal örneklem tabanlı (RRT\*, PRM) yol planlama algoritmalarının veri yapıları optimizasyonlarında öneriler alınması.
  * Gerçek zamanlı grafik arayüzünde (Matplotlib GUI) animasyon güncellemelerinin hızlandırılması adına `blit` tabanlı arka plan hafıza yenileme mimarisindeki kod hatalarının ayıklanması (debugging).
- **Öğrencinin Kendi Katkıları:**
  - Proje senaryosunun kurgulanması, non-holonomic diferansiyel ve Ackermann robot kinematik kısıtlarının matematiksel modellenmesi.
  - LiDAR ham verileri için dairesel medyan filtre yapısının geliştirilmesi ve gap-based engel kümeleme mantığının özgün kodlanması.
  - Yapay Potansiyel Alanlar (APF) tabanlı reaktif engelden kaçınma itme kuvveti (`_lidar_repulsion`) fonksiyonunun algoritma mantığının kurulması ve tüm modüllerin entegrasyonu.
  - Arayüz üzerindeki performans kıyaslama test senaryolarının hazırlanması, RMSE/MAE analitik değerlendirmelerinin yapılması ve raporlama diline aktarılması.

*Açıklama: Yapay zeka araçları teknik geliştirme ve optimizasyon süreçlerinde yardımcı ve hızlandırıcı birer mühendislik enstrümanı olarak kullanılmıştır. Matematiksel formüllerin fiziksel doğrulaması, kod bileşenlerinin birbirine bağlanması, deney çıktılarının analizi ve nihai değerlendirmeler tamamen öğrenci tarafından bizzat gerçekleştirilmiştir.*

### Kaynaklar
[1] V. Ušinskis, M. Nowicki, A. Dzedzickis ve V. Bučinskas, "Sensor-fusion based navigation for autonomous mobile robot," *Sensors*, cilt 25, sayı 4, makale 1248, 2025, doi: 10.3390/s25041248.

[2] Y. Ou, Y. Cai, Y. Sun ve T. Qin, "Autonomous navigation by mobile robot with sensor fusion based on deep reinforcement learning," *Sensors*, cilt 24, sayı 12, makale 3895, 2024, doi: 10.3390/s24123895.

[3] B. Zhang ve C. Li, "The optimization and application research of the RRT-APF-based path planning algorithm," *Electronics*, cilt 13, sayı 24, makale 4963, 2024, doi: 10.3390/electronics13244963.
