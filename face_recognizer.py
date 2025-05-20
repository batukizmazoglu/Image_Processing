import face_recognition
import cv2
import pickle
import os
import argparse
import numpy as np
import time # Zaman takibi için eklendi
from deepface import DeepFace # Duygu analizi için eklendi

def load_known_faces(encodings_path="known_faces.dat"):
    """Önceden kaydedilmiş yüz encoding'lerini yükler"""
    if not os.path.exists(encodings_path):
        print(f"Hata: {encodings_path} dosyası bulunamadı. Önce face_encoder.py ile veritabanı oluşturun.")
        return None
    
    print(f"{encodings_path} dosyasından bilinen yüzler yükleniyor...")
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Toplam {len(data['encodings'])} adet yüz encoding'i yüklendi.")
    return data

def recognize_faces_in_image(image_path, data, model="hog", tolerance=0.6):
    """Bir görüntüdeki yüzleri tanır ve etiketlerle görselleştirir"""
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Yüzleri bul
    print("Yüzler tespit ediliyor...")
    face_locations = face_recognition.face_locations(rgb_image, model=model)
    print(f"{len(face_locations)} yüz tespit edildi.")
    
    if len(face_locations) == 0:
        return image
    
    # Yüzleri kodla
    print("Yüzler kodlanıyor...")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # Yüzleri tanı
    print("Yüzler tanınıyor...")
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Bilinen yüzlerle karşılaştır
        matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=tolerance)
        name = "Bilinmeyen"
        
        # En iyi eşleşmeyi bul
        if True in matches:
            matched_indices = [i for i, match in enumerate(matches) if match]
            counts = {}
            
            for i in matched_indices:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            
            name = max(counts, key=counts.get)
        
        # Yüz etrafına dikdörtgen çiz
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # İsmi yaz
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    
    return image

def recognize_faces_from_camera(data, model="hog", tolerance=0.6, camera_id=0):
    """Kameradan gelen görüntülerde yüzleri tanır, duygu analizi yapar ve süreleri takip eder"""
    # Kamerayı aç
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Hata: {camera_id} ID'li kamera açılamadı.")
        return
    
    print("Kamera açıldı. Çıkış için 'q' tuşuna basın.")
    
    person_timers = {}  # {name: total_time_seconds}
    person_last_emotion = {} # {name: last_emotion_detected}
    
    prev_time = time.time()
    fps_display_time = time.time()
    fps_frame_counter = 0
    displayed_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera okuma hatası.")
            break

        current_time = time.time()
        time_elapsed_this_frame = current_time - prev_time
        prev_time = current_time

        # FPS hesaplama (her saniye güncelle)
        fps_frame_counter += 1
        if (current_time - fps_display_time) >= 1.0:
            displayed_fps = fps_frame_counter / (current_time - fps_display_time)
            fps_frame_counter = 0
            fps_display_time = current_time
        
        # Görüntüyü BGR'dan RGB'ye çevir (face_recognition için)
        # Performans için kare boyutunu küçültebilirsiniz:
        # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) 
        # rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Yüzleri tespit et
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        
        active_persons_in_frame = set()

        if face_locations:
            # Yüzleri kodla
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Her yüz için
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Ölçeklendirme yapıldıysa koordinatları geri ölçekle
                # top *= 2; right *= 2; bottom *= 2; left *= 2 # Eğer small_frame kullanılıyorsa
                
                # Bilinen yüzlerle karşılaştır
                matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=tolerance)
                name = "Bilinmeyen"
                face_img_for_emotion = frame[top:bottom, left:right]

                emotion = person_last_emotion.get(name, "N/A") # Varsayılan duygu
                
                if True in matches:
                    matched_indices = [i for i, match in enumerate(matches) if match]
                    counts = {}
                    for i in matched_indices:
                        name_candidate = data["names"][i]
                        counts[name_candidate] = counts.get(name_candidate, 0) + 1
                    name = max(counts, key=counts.get)
                
                active_persons_in_frame.add(name)

                if name != "Bilinmeyen":
                    # Duygu analizi (DeepFace)
                    try:
                        if face_img_for_emotion.size > 0: # Yüz görüntüsü boş değilse
                            # Not: enforce_detection=False, yüz bulamazsa hata vermesini engeller,
                            # ama yine de emotion kısmı olmayabilir.
                            analysis = DeepFace.analyze(face_img_for_emotion, actions=['emotion'], enforce_detection=False, silent=True)
                            # DeepFace bazen liste dönebilir, ilk sonucu alalım
                            if isinstance(analysis, list) and len(analysis) > 0:
                                emotion = analysis[0]['dominant_emotion']
                            elif isinstance(analysis, dict): # Tek bir sonuç döndüğünde
                                emotion = analysis['dominant_emotion']
                            else: # Beklenmedik format veya duygu bulunamadı
                                emotion = person_last_emotion.get(name, "N/A") # Önceki duyguyu kullan veya N/A
                            person_last_emotion[name] = emotion
                        else:
                             emotion = person_last_emotion.get(name, "N/A")
                    except Exception as e:
                        # print(f"{name} için duygu analizi hatası: {e}")
                        emotion = person_last_emotion.get(name, "N/A") # Hata durumunda önceki duyguyu kullan

                    # Süre takibi
                    person_timers[name] = person_timers.get(name, 0) + time_elapsed_this_frame
                else: # Bilinmeyen kişi için
                    emotion = "N/A"
                
                # Yüz etrafına dikdörtgen çiz
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # İsmi, duyguyu ve süreyi yaz
                display_text = f"{name} ({emotion}) - {int(person_timers.get(name, 0))} sn"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, display_text, (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # "Kim ne kadar süre konuştu:" başlığı altında bilgileri göster
        y_offset = 30
        cv2.putText(frame, "Kim ne kadar sure konustu:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_offset += 25
        for person_name, total_time in sorted(person_timers.items()):
            if person_name == "Bilinmeyen": continue # Bilinmeyenleri listeleme
            
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            time_str = ""
            if minutes > 0:
                time_str = f"{minutes} dk {seconds} sn"
            else:
                time_str = f"{seconds} sn"
                
            summary_text = f"{person_name}: {time_str}"
            cv2.putText(frame, summary_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 20

        # FPS bilgisini ekranda göster
        cv2.putText(frame, f"FPS: {displayed_fps:.2f}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Sonucu göster
        cv2.imshow("Yuz Tanima ve Duygu Analizi", frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Temizle
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Komut satırı argümanlarını ayarla
    parser = argparse.ArgumentParser(description="Yüz tanıma sistemi")
    parser.add_argument("-e", "--encodings", type=str, default="known_faces.dat",
                      help="Bilinen yüz kodlarını içeren dosya yolu")
    parser.add_argument("-i", "--image", type=str,
                      help="Yüz tanıma için kullanılacak görüntü dosyası yolu")
    parser.add_argument("-c", "--camera", action="store_true",
                      help="Kamera kullanarak canlı yüz tanıma yapmak için bu bayrağı kullanın")
    parser.add_argument("-m", "--model", type=str, default="hog", choices=["hog", "cnn"],
                      help="Yüz tespiti modeli (hog: hızlı ama daha az hassas, cnn: yavaş ama daha hassas)")
    parser.add_argument("-t", "--tolerance", type=float, default=0.6,
                      help="Yüz eşleştirme toleransı (daha düşük: daha kesin, daha yüksek: daha esnek)")
    parser.add_argument("-o", "--output", type=str,
                      help="Tanımlanan yüzleri içeren çıktı görüntüsünün kaydedileceği dosya yolu")
    parser.add_argument("--camera-id", type=int, default=0,
                      help="Kullanılacak kamera ID'si (varsayılan: 0)")
    
    args = parser.parse_args()
    
    # Bilinen yüzleri yükle
    data = load_known_faces(args.encodings)
    if data is None:
        return
    
    # Ya kamera ya da resim modunda çalış
    if args.camera:
        recognize_faces_from_camera(data, args.model, args.tolerance, args.camera_id)
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Hata: {args.image} dosyası bulunamadı.")
            return
        
        output_image = recognize_faces_in_image(args.image, data, args.model, args.tolerance)
        
        # Sonucu göster
        cv2.imshow("Yüz Tanıma", output_image)
        cv2.waitKey(0)
        
        # İsteğe bağlı olarak sonucu kaydet
        if args.output:
            cv2.imwrite(args.output, output_image)
            print(f"Sonuç {args.output} dosyasına kaydedildi.")
    else:
        print("Hata: Ya bir görüntü dosyası belirtin (-i) ya da kamera modunu etkinleştirin (-c)")

if __name__ == "__main__":
    main() 