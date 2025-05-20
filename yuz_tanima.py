import cv2
import face_recognition
import pickle
import numpy as np
import os
import tempfile
from urllib.parse import urlparse
import argparse
import traceback

# DeepFace kütüphanesini import et
try:
    from deepface import DeepFace
    deepface_available = True
except ImportError:
    deepface_available = False
    print("DeepFace yüklü değil. Duygu analizi yapılamayacak.")

# YouTube video indirmek için yt-dlp kütüphanesini import et
try:
    import yt_dlp
    youtube_dl_available = True
except ImportError:
    youtube_dl_available = False
    print("yt-dlp yüklü değil. YouTube videoları işlenemeyecek.")

def load_known_faces(file_path="known_faces.dat"):
    """Bilinen yüzlerin embedding'lerini ve isimlerini yükle"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"{len(data['encodings'])} bilinen yüz yüklendi.")
        return data
    except Exception as e:
        print(f"Hata: {file_path} dosyası yüklenemedi: {e}")
        return {"encodings": [], "names": []}

def is_youtube_url(url):
    """Verilen URL'nin YouTube URL'si olup olmadığını kontrol et"""
    parsed_url = urlparse(url)
    return any(domain in parsed_url.netloc for domain in ['youtube.com', 'youtu.be'])

def download_youtube_video(url):
    """YouTube videosunu indir ve geçici dosya yolunu döndür"""
    if not youtube_dl_available:
        print("YouTube videoları indirmek için yt-dlp gereklidir.")
        return None
    
    try:
        print("YouTube videosu indiriliyor...")
        
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, "downloaded_video.mp4")
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_filename,
            'quiet': False,
            'no_warnings': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        print(f"Video indirildi: {temp_filename}")
        return temp_filename
    except yt_dlp.utils.DownloadError as de:
        print(f"yt-dlp indirme hatası: {de}")
        print("Lütfen yt-dlp kütüphanesinin güncel olduğundan emin olun: pip install --upgrade yt-dlp")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Video indirme sırasında genel bir hata oluştu: {e}")
        traceback.print_exc()
        return None

def analyze_emotion(face_image):
    """DeepFace ile yüz görüntüsünün duygu durumunu analiz eder"""
    if not deepface_available:
        print("DeepFace kütüphanesi yüklü değil!")
        return "Bilinmeyen"
    
    try:
        # Görüntünün geçerli olup olmadığını kontrol et
        if face_image.size == 0 or face_image is None:
            print("Geçersiz yüz görüntüsü!")
            return "Bilinmeyen"
            
        # Görüntünün minimum boyutunu kontrol et
        if face_image.shape[0] < 48 or face_image.shape[1] < 48:
            # Yüz çok küçükse yeniden boyutlandır
            face_image = cv2.resize(face_image, (96, 96), interpolation=cv2.INTER_AREA)
            
        # DeepFace ile duygu analizi yap
        result = DeepFace.analyze(img_path=face_image, 
                                 actions=['emotion'], 
                                 enforce_detection=False,
                                 silent=True)
        
        # Sonucu kontrol et
        if result is None or len(result) == 0:
            print("DeepFace sonuç döndürmedi!")
            return "Bilinmeyen"
            
        # En baskın duyguyu al
        if isinstance(result, list):
            result = result[0]  # İlk sonucu al
        
        # Sonuç yapısını yazdır (debug için)
        print(f"DeepFace sonucu: {result.keys()}")
        
        if 'dominant_emotion' in result:
            dominant_emotion = result['dominant_emotion']
        elif 'emotion' in result and isinstance(result['emotion'], dict):
            # En yüksek değerli duyguyu bul
            emotions = result['emotion']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        else:
            print("Duygu verisi bulunamadı!")
            return "Bilinmeyen"
        
        print(f"Tespit edilen duygu: {dominant_emotion}")
        
        # Duygu isimlerini Türkçe'ye çevir
        emotion_tr = {
            'angry': 'Kızgın',
            'disgust': 'İğrenmiş',
            'fear': 'Korkmuş',
            'happy': 'Mutlu',
            'sad': 'Üzgün',
            'surprise': 'Şaşkın',
            'neutral': 'Nötr'
        }
        
        return emotion_tr.get(dominant_emotion, dominant_emotion)
    except Exception as e:
        print(f"Duygu analizi hatası: {e}")
        traceback.print_exc()
        return "Bilinmeyen"

def process_video(video_path, known_faces_data):
    """Video işle ve yüzleri tanı"""
    # Veri çıkar
    known_face_encodings = known_faces_data["encodings"]
    known_face_names = known_faces_data["names"]
    
    # Video yakalayıcıyı aç
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Hata: Video açılamadı - {video_path}")
        return
    
    frame_count = 0
    
    # DeepFace'in ilk kullanımda çok yavaş olduğunu belirt
    if deepface_available:
        print("DeepFace ilk kullanımda model dosyalarını indiriyor. Lütfen bekleyin...")
        print("İlk birkaç kare yavaş işlenebilir, endişelenmeyin.")
    
    while True:
        # Videodan bir kare oku
        ret, frame = video_capture.read()
        
        # Video bittiyse döngüden çık
        if not ret:
            break
        
        # Her kareyi işlememek için (performans iyileştirmesi)
        frame_count += 1
        if frame_count % 3 != 0:  # Her 3 kareden birini işle
            continue
        
        # BGR'den RGB'ye dönüştür (face_recognition RGB bekler)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # İşlem hızlandırmak için boyutu küçült
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        
        # Yüz lokasyonlarını bul
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        # Her bir tespit edilen yüz için
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Ölçeği tekrar ayarla
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Bilinen yüzlerle karşılaştır
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Bilinmeyen"
            emotion_text = "" # Duygu metni için değişken
            
            # Eğer eşleşme varsa, en yakın eşleşmeyi bul
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            
            # Yüzü kırp (orijinal BGR kare üzerinden)
            # face_recognition'dan gelen top, right, bottom, left koordinatları zaten orijinal frame'e göre ölçeklenmiş durumda.
            face_image_bgr = frame[top:bottom, left:right]

            # Kırpılan yüzün geçerli olup olmadığını kontrol et (çok küçük veya boş olabilir)
            if face_image_bgr.size > 0:
                 emotion_text = analyze_emotion(face_image_bgr)
            
            # Yüzün etrafına kutu çiz
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Yüzün altına isim ve duygu etiketi yaz
            label = f"{name} ({emotion_text})" if emotion_text and emotion_text != "Bilinmeyen" else name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Sonucu göster
        cv2.imshow('Video', frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Temizle
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Yüz tanıma sistemi')
    parser.add_argument('--video', type=str, help='Video dosyası yolu veya YouTube URL')
    parser.add_argument('--faces', type=str, default='known_faces.dat', help='Bilinen yüzler veri dosyası')
    
    args = parser.parse_args()
    
    # DeepFace kütüphanesini kontrol et
    if not deepface_available:
        print("UYARI: DeepFace kütüphanesi bulunamadı. Lütfen 'pip install deepface' komutunu çalıştırın.")
        print("Duygu analizi olmadan devam etmek istiyor musunuz? (E/H)")
        response = input().lower()
        if response != 'e':
            print("Program sonlandırılıyor. Lütfen DeepFace kütüphanesini yükleyip tekrar deneyin.")
            return
    else:
        print("DeepFace kütüphanesi yüklü. Duygu analizi aktif.")
        try:
            # DeepFace modellerinin yüklü olup olmadığını kontrol et - boş bir resimle test
            import numpy as np
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            # Bu çağrı, modelleri yükleyecek ve ilk başlatma gecikmesini burada yaşatacak
            print("DeepFace model dosyaları kontrol ediliyor/indiriliyor...")
            try:
                DeepFace.analyze(img_path=test_img, actions=['emotion'], enforce_detection=False, silent=False)
                print("DeepFace modelleri hazır!")
            except Exception as e:
                print(f"DeepFace modelleri yükleniyor... İlk çalıştırmada biraz zaman alabilir: {e}")
        except Exception as e:
            print(f"DeepFace test hatası: {e}")
    
    # Eğer komut satırından argüman verilmediyse kullanıcıdan al
    video_path = args.video
    if not video_path:
        video_path = input("Video dosya yolu veya YouTube URL'si girin: ")
    else:
        video_path = video_path.strip('"')
    
    # Bilinen yüzleri yükle
    known_faces_data = load_known_faces(args.faces)
    
    # YouTube linki ise indir
    if is_youtube_url(video_path):
        video_path = download_youtube_video(video_path)
        if not video_path:
            print("Video indirilemedi veya işlenemedi.")
            return
    
    # Videoyu işle
    process_video(video_path, known_faces_data)

if __name__ == "__main__":
    main()