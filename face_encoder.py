import os
import face_recognition_models
import face_recognition
import pickle
from tqdm import tqdm
import numpy as np

def encode_known_faces(dataset_path="dataset", model="hog", encodings_path="known_faces.dat"):
    """
    dataset/<kisi_ismi>/<foto.jpg> yapısındaki klasörden yüzleri tespit eder,
    embedding'lerini çıkarır ve bir dosyada saklar.
    
    Parametreler:
        dataset_path: Yüz verilerinin bulunduğu klasör yolu
        model: Yüz tespiti için kullanılacak model ('hog' veya 'cnn')
        encodings_path: Kaydedilecek dosyanın yolu
    """
    # Veri saklama yapıları
    known_encodings = []
    known_names = []
    
    # dataset klasöründeki her bir kişi klasörünü tara
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        # Eğer bu bir klasör değilse, atla
        if not os.path.isdir(person_dir):
            continue
        
        # Bu kişiye ait tüm fotoğrafları bul
        image_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        # İlerleme çubuğu ile her bir fotoğrafı işle
        print(f"{person_name} için {len(image_paths)} fotoğraf işleniyor...")
        for image_path in tqdm(image_paths, desc=person_name):
            try:
                # Resmi yükle ve RGB formatına dönüştür
                image = face_recognition.load_image_file(image_path)
                
                # Yüzleri bul
                face_locations = face_recognition.face_locations(image, model=model)
                
                # Eğer hiç yüz bulunamadıysa, uyarı ver ve atla
                if not face_locations:
                    print(f"Uyarı: {image_path} dosyasında yüz bulunamadı.")
                    continue
                
                # Her bir yüz için encoding'leri hesapla
                encodings = face_recognition.face_encodings(image, face_locations)
                
                # Her encoding'i ve ilgili kişi adını kaydet
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person_name)
                    
            except Exception as e:
                print(f"Hata: {image_path} işlenirken sorun oluştu: {e}")
    
    # Tüm kişi ve encoding'leri içeren sözlük
    data = {
        "encodings": known_encodings,
        "names": known_names
    }
    
    # Veriyi pickle dosyasına kaydet
    with open(encodings_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"İşlem tamamlandı. {len(known_names)} yüz encoding'i kaydedildi.")
    return data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Yüz tanıma sistemi için veri tabanı oluşturucu")
    parser.add_argument("-d", "--dataset", type=str, default="dataset",
                        help="Yüz verilerini içeren dataset klasörünün yolu")
    parser.add_argument("-m", "--model", type=str, default="hog", choices=["hog", "cnn"],
                        help="Yüz tespiti modeli (hog: hızlı ama daha az hassas, cnn: yavaş ama daha hassas)")
    parser.add_argument("-o", "--output", type=str, default="known_faces.dat",
                        help="Kaydedilecek encoding dosyasının adı")
    
    args = parser.parse_args()
    
    print(f"Dataset klasörü: {args.dataset}")
    print(f"Kullanılan model: {args.model}")
    print(f"Çıktı dosyası: {args.output}")
    
    # Encoding işlemini başlat
    encode_known_faces(args.dataset, args.model, args.output) 