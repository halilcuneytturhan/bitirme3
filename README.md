# Sürücü Uyku ve Yorgunluk Tespiti

Bu proje, kamera görüntüsü üzerinden sürücünün uyku ve yorgunluk durumunu tespit etmeyi amaçlayan bir bitirme projesidir. Projede görüntü tabanlı sınıflandırma için MLP ve CNN modelleri karşılaştırılmış, elde edilen sonuçlara göre CNN modeli final model olarak seçilmiştir.

Sistem; göz açık/kapalı durumu ve esneme davranışı üzerinden sürücünün yorgunluk seviyesini analiz etmeyi hedefler. Model eğitimi tamamlandıktan sonra sistemin canlı kamera görüntüsü üzerinde test edilmesi ve daha sonra Jetson Nano üzerinde çalıştırılması planlanmaktadır.

---

## Proje Amacı

Trafik kazalarının önemli nedenlerinden biri sürücü yorgunluğu ve uyku halidir. Bu proje ile sürücünün yüz görüntüsünden göz kapanması ve esneme gibi belirtiler analiz edilerek gerçek zamanlı uyarı sistemi geliştirilmesi amaçlanmıştır.

Projenin temel hedefleri:

- Sürücü uyku ve yorgunluk durumunu görüntü işleme ile tespit etmek
- MLP ve CNN modellerini karşılaştırmak
- En başarılı modeli seçerek canlı kamera sistemine entegre etmek
- OpenCV ile gerçek zamanlı tahmin yapmak
- Jetson Nano üzerinde çalışabilecek bir yapı oluşturmak

---

## Kullanılan Yöntemler

Projede iki farklı model yaklaşımı değerlendirilmiştir:

### 1. MLP Modeli

MLP modeli, görüntüleri tek boyutlu vektör haline getirerek sınıflandırma yapar. Ancak görüntü içerisindeki pikseller arasındaki konumsal ilişkileri koruyamadığı için görüntü sınıflandırma problemlerinde CNN’e göre daha düşük başarı göstermiştir.

### 2. CNN Modeli

CNN modeli, convolution ve pooling katmanları sayesinde görüntü içerisindeki yerel özellikleri daha başarılı şekilde öğrenir. Göz açıklığı, ağız açıklığı, kenar ve şekil bilgileri gibi görsel özellikleri daha iyi yakalayabildiği için bu projede final model olarak CNN tercih edilmiştir.

---

## Model Seçim Gerekçesi

Yapılan deneysel çalışmalarda CNN modeli, MLP modeline göre daha yüksek doğruluk değeri elde etmiştir.

MLP modeli görüntüleri düzleştirerek işlediği için görüntüdeki uzamsal ilişkileri yeterince koruyamamaktadır. CNN modeli ise evrişim katmanları sayesinde göz ve ağız bölgesindeki önemli özellikleri daha etkili şekilde öğrenmektedir.

Bu nedenle nihai sistemde CNN tabanlı modelin kullanılmasına karar verilmiştir.

---

## Veri Seti Yapısı

Projede kullanılan veri seti sınıflandırma formatındadır. Veri seti aşağıdaki sınıflardan oluşmaktadır:

- closed_eye
- open_eye
- yawn
- no_yawn

Örnek veri seti klasör yapısı:

dataset/
├── train/
│   ├── closed_eye/
│   ├── open_eye/
│   ├── yawn/
│   └── no_yawn/
├── val/
│   ├── closed_eye/
│   ├── open_eye/
│   ├── yawn/
│   └── no_yawn/
└── test/
    ├── closed_eye/
    ├── open_eye/
    ├── yawn/
    └── no_yawn/

---

## Proje Klasör Yapısı

driver-drowsiness-detection/
│
├── data/
│   └── dataset dosyaları
│
├── models/
│   └── drowsiness_cnn.h5
│
├── src/
│   ├── train_cnn.py
│   ├── train_mlp.py
│   ├── evaluate.py
│   └── realtime_demo.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

## Kullanılan Teknolojiler

Projede kullanılan temel teknolojiler:

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- CNN
- MLP
- Jetson Nano

---

## Kurulum

Öncelikle projeyi bilgisayarınıza klonlayın:

git clone https://github.com/kullaniciadi/driver-drowsiness-detection.git
cd driver-drowsiness-detection

Gerekli kütüphaneleri yükleyin:

pip install -r requirements.txt

Eğer requirements.txt dosyası yoksa aşağıdaki kütüphaneler kurulabilir:

pip install tensorflow opencv-python numpy matplotlib scikit-learn

---

## Model Eğitimi

CNN modelini eğitmek için:

python src/train_cnn.py

MLP modelini eğitmek için:

python src/train_mlp.py

Eğitim tamamlandıktan sonra CNN modeli aşağıdaki konuma kaydedilir:

models/drowsiness_cnn.h5

---

## Model Değerlendirme

Model başarısını değerlendirmek için:

python src/evaluate.py

Değerlendirme sonucunda aşağıdaki metrikler incelenir:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Loss grafiği
- Accuracy grafiği

Bu projede yalnızca accuracy değerine bakılmamış, özellikle uyku durumunu doğru yakalamak için recall ve F1-score değerleri de dikkate alınmıştır.

---

## Canlı Kamera Testi

Eğitilen CNN modelini canlı kamera üzerinde test etmek için:

python src/realtime_demo.py

Bu aşamada OpenCV kullanılarak kameradan görüntü alınır. Görüntü CNN modelinin giriş boyutuna uygun hale getirilir ve model tahmini ekrana yazdırılır.

Sistem aşağıdaki durumları algılamayı hedefler:

open_eye    → Göz açık
closed_eye  → Göz kapalı
yawn        → Esneme var
no_yawn     → Esneme yok

---

## Karar Mekanizması

Tek bir karede göz kapalı tespit edilmesi doğrudan uyku olarak değerlendirilmez. Çünkü bu durum normal göz kırpma hareketinden kaynaklanabilir.

Bu nedenle zamana bağlı karar mekanizması kullanılmıştır:

Göz kapalı durumu belirli bir süreden uzun devam ederse:
    Uykulu sürücü uyarısı verilir.

Esneme algılanırsa:
    Yorgunluk uyarısı verilir.

Örnek karar:

closed_eye durumu 2 saniyeden uzun sürerse:
UYARI: Sürücü uykulu!

---

## Jetson Nano Planı

Sistem öncelikle geliştirme bilgisayarı üzerinde canlı kamera ile test edilecektir. Modelin doğru çalıştığı doğrulandıktan sonra proje Jetson Nano üzerine taşınacaktır.

Jetson Nano üzerinde hedeflenen yapı:

Kamera
↓
OpenCV ile görüntü alma
↓
CNN modeli ile tahmin
↓
Uyku/yorgunluk karar mekanizması
↓
Görsel veya sesli uyarı

Jetson Nano üzerinde performans yetersiz kalırsa modelin daha hafif hale getirilmesi veya TFLite/TensorRT gibi optimizasyon yöntemlerinin kullanılması planlanmaktadır.

---

## YOLO Neden Kullanılmadı?

Proje sürecinde YOLO tabanlı nesne tespiti yöntemi de değerlendirilmiştir. Ancak kullanılan veri seti sınıflandırma formatındadır. YOLO detection modeli için her görüntüye ait bounding box etiketlerinin bulunması gerekmektedir.

Mevcut veri setinde bu etiketler bulunmadığı için YOLO detection yöntemi yerine CNN tabanlı sınıflandırma yaklaşımı tercih edilmiştir.

YOLO, ilerleyen çalışmalarda alternatif ve geliştirilebilir bir yöntem olarak değerlendirilebilir.

---

## Genel Sistem Akışı

Kamera görüntüsü
        ↓
Görüntü ön işleme
        ↓
CNN modeli
        ↓
Sınıf tahmini
        ↓
Zamana bağlı karar mekanizması
        ↓
Uyku / yorgunluk uyarısı

---

## Beklenen Çıktılar

Proje sonunda sistemin aşağıdaki çıktıları vermesi beklenmektedir:

- Canlı kamera görüntüsü üzerinde tahmin sonucu
- Göz açık / kapalı sınıflandırması
- Esneme algılama
- Uykulu sürücü uyarısı
- Yorgunluk uyarısı
- Model başarı metrikleri
- MLP ve CNN karşılaştırması

---

## Sonuç

Bu projede sürücü uyku ve yorgunluk tespiti için görüntü tabanlı bir yaklaşım geliştirilmiştir. MLP ve CNN modelleri karşılaştırılmış, CNN modelinin görüntü sınıflandırma probleminde daha başarılı olduğu gözlemlenmiştir.

Bu nedenle final sistemde CNN modeli kullanılmıştır. Sistem OpenCV ile canlı kamera görüntüsü üzerinde test edilecek ve daha sonra Jetson Nano üzerinde çalıştırılması hedeflenecektir.

---

## Geliştirilebilir Yönler

İlerleyen aşamalarda proje şu şekilde geliştirilebilir:

- MediaPipe ile göz ve ağız bölgelerinin daha doğru tespit edilmesi
- YOLO detection için veri setinin bounding box ile yeniden etiketlenmesi
- Jetson Nano üzerinde TensorRT optimizasyonu
- Sesli alarm sistemi eklenmesi
- Gece/gündüz ışık koşullarına göre modelin iyileştirilmesi
- Daha büyük ve dengeli veri setleriyle yeniden eğitim yapılması
