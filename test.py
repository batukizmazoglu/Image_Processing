import os
import shutil
import face_encoder
import face_recognizer
import argparse

def create_test_dataset():
    """Test için örnek bir dataset oluşturur"""
    if os.path.exists("test_dataset"):
        print("test_dataset klasörü zaten var, siliniyor...")
        shutil.rmtree("test_dataset")
    
    # Test klasör yapısını oluştur
    os.makedirs("test_dataset/Test_Kisi1", exist_ok=True)
    os.makedirs("test_dataset/Test_Kisi2", exist_ok=True)
    
    print("Test için örnek resim dosyaları oluşturuluyor...")
    
    # Bilgi mesajı
    print("""
UYARI: Bu test betiği gerçek resim dosyalarını oluşturmaz.
Kendi resimlerinizle test etmek için:
1. 'test_dataset/<kisi_ismi>/' klasörüne JPEG veya PNG resimler yerleştirin
2. Yüz içeren resimlerin yüklü olduğundan emin olun
    """)

def main():
    parser = argparse.ArgumentParser(description="Yüz tanıma sistemi testi")
    parser.add_argument("--create", action="store_true",
                      help="Test için örnek bir dataset yapısı oluşturur")
    parser.add_argument("--encode", action="store_true",
                      help="Test dataset'ini kodlar ve kaydeder")
    parser.add_argument("--test-image", type=str,
                      help="Test edilecek görüntü dosyası")
    parser.add_argument("--test-camera", action="store_true",
                      help="Kamera ile canlı test gerçekleştirir")
    
    args = parser.parse_args()
    
    # Hiçbir argüman verilmezse, tüm test adımlarını çalıştır
    if not (args.create or args.encode or args.test_image or args.test_camera):
        args.create = True
        args.encode = True
        args.test_camera = True
    
    # Test dataset'i oluştur
    if args.create:
        create_test_dataset()
    
    # Encoding işlemi
    if args.encode:
        print("\nTest dataset'i kodlanıyor...")
        face_encoder.encode_known_faces(
            dataset_path="test_dataset",
            model="hog",
            encodings_path="test_faces.dat"
        )
    
    # Resim ile test
    if args.test_image:
        print(f"\n{args.test_image} üzerinde yüz tanıma yapılıyor...")
        data = face_recognizer.load_known_faces("test_faces.dat")
        if data:
            output_image = face_recognizer.recognize_faces_in_image(
                args.test_image, data, model="hog", tolerance=0.6
            )
            
            import cv2
            cv2.imshow("Test Sonucu", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Kamera ile test
    if args.test_camera:
        print("\nKamera ile canlı yüz tanıma testi yapılıyor...")
        data = face_recognizer.load_known_faces("test_faces.dat")
        if data:
            face_recognizer.recognize_faces_from_camera(data, model="hog", tolerance=0.6)

if __name__ == "__main__":
    main() 