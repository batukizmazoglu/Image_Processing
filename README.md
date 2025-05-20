# Görüntü İşleme ile Konuşan Kişiyi Belirleme ve Duygu Durumu Tahmini

Bu proje, görüntü işleme teknikleri kullanarak:
1. Bilinen kişileri tanıma
2. Konuşan kişiyi belirleme
3. Duygu durumu tahmini
yapmak için geliştirilmiştir.

## Kurulum

Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install face_recognition opencv-python numpy tqdm
```

**Not:** `face_recognition` kütüphanesi için `dlib` kütüphanesinin kurulması gereklidir. Windows'ta kurulum için şu adımları izleyebilirsiniz:
1. [Visual Studio Build Tools](https://visualstudio.microsoft.com/tr/visual-cpp-build-tools/) yükleyin
2. [CMake](https://cmake.org/download/) yükleyin
3. `pip install dlib` komutunu çalıştırın
4. `pip install face_recognition` komutunu çalıştırın

## Kullanım

### 1. Kişi Veritabanı Oluşturma

İlk olarak, tanımak istediğiniz kişilerin fotoğraflarını şu klasör yapısında düzenleyin:

```
dataset/
    Ahmet/
        ahmet1.jpg
        ahmet2.png
    Ayse/
        ayse1.jpg
        ...
```

Sonra, fotoğraflardan yüz veritabanı oluşturmak için aşağıdaki komutu çalıştırın:

```bash
python face_encoder.py
```

Bu komut, her fotoğraftaki yüzleri tespit edecek, yüz özelliklerini çıkaracak ve `known_faces.dat` dosyasına kaydedecektir.

Farklı parametrelerle çalıştırmak için:

```bash
python face_encoder.py -d "dataset_klasoru" -m "cnn" -o "veritabani.dat"
```

Parametreler:
- `-d, --dataset`: Yüz verilerinin bulunduğu klasör
- `-m, --model`: Yüz tespiti modeli (hog: hızlı ama daha az hassas, cnn: GPU ile daha hassas)
- `-o, --output`: Oluşturulacak veritabanı dosyasının adı

### 2. Yüz Tanıma Kullanımı

Veritabanını oluşturduktan sonra, `face_recognizer.py` ile tanıma yapmak için:

```bash
# Bir resim dosyasındaki yüzleri tanıma
python face_recognizer.py -i "resim.jpg"

# Sonucu bir dosyaya kaydetme
python face_recognizer.py -i "resim.jpg" -o "sonuc.jpg"

# Kamera ile canlı yüz tanıma
python face_recognizer.py -c

# Farklı parametrelerle çalıştırma
python face_recognizer.py -c -m "cnn" -t 0.5 -e "known_faces.dat"
```

Parametreler:
- `-i, --image`: Tanınacak resim dosyası
- `-c, --camera`: Kamera kullanarak canlı tanıma yapmak için
- `-m, --model`: Yüz tespiti modeli (hog/cnn)
- `-t, --tolerance`: Eşleştirme toleransı (0.6 varsayılan)
- `-e, --encodings`: Yüz veritabanı dosyası
- `-o, --output`: Sonuç resmi kaydetme yolu
- `--camera-id`: Birden fazla kamera varsa ID numarası (varsayılan: 0)

### 3. Test Etme

Sistemi hızlıca test etmek için:

```bash
# Test ve örnek yapılandırma
python test.py

# Özel bir resim ile test etme
python test.py --test-image "test.jpg"
```

### 4. Sonraki Adımlar

- Konuşan kişiyi belirleme
- Duygu durumu tahmini

Bu bölümler geliştirme aşamasındadır.

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır. "# Image_Processing" 
