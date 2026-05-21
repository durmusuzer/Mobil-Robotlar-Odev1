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

---

## 4. Hata Analizi ve Kısa Tartışma

Konum hatası, her adımda gerçek konum ile tahmin arasındaki Öklid mesafesidir. Genel doğruluk
**RMSE** ile özetlenir:

```
RMSE = sqrt( (1/N) · Σ (p̂ᵢ − pᵢ)² )
```

| Metrik | Dead Reckoning | EKF (Füzyon) |
| :--- | :--- | :--- |
| RMSE (m) | _(2.172m)_ | _(0.116)_ |

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
