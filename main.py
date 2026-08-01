import cv2#kameradan görüntü almak için
import mediapipe as mp#elimizdeki 21 eklem  noktaını milisaniyeler içinde tespşt eder
import math#parmak  uçları  arası meafe için 
import numpy as np#şeffaf renk harmanlama için

# 1. MediaPipe ve Kamera Kurulumu
mp_hands = mp.solutions.hands# el tanıma modülü
mp_draw = mp.solutions.drawing_utils#bulunan 21 noktayı ekrana renkli çizgilerle çzidirmek için yardımcı araç

hands = mp_hands.Hands(#asıl yapay zeka modülü bu
    static_image_mode=False,#vvideo işlediğimiz için hızlı takip modülünü aktfileştirir
    max_num_hands=1,#sadece bir ele odaklanmasını ister
    min_detection_confidence=0.7,#%70 emin olmadan bu eldir deme diyorum
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)#bilgisyarın ana kamerası

# 2. Parmakların Açık/Kapalı Durumunu Analiz Eden Fonksiyon
def get_finger_states(hand_landmarks):
    # [İşaret, Orta, Yüzük, Serçe]
    tips = [8, 12, 16, 20]
    states = []
    
    for tip_id in tips:
        # Parmağın ucu boğumundan daha yukarıda mı?
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            states.append(True)  # Açık
        else:
            states.append(False) # Kapalı
            
    # Başparmak Kontrolü
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_open = math.hypot(thumb_tip.x - hand_landmarks.landmark[17].x, 
                            thumb_tip.y - hand_landmarks.landmark[17].y) > 0.25
    
    return states, thumb_open

# 3. İtalyan El İşareti (🤌) Hassas Kontrolü
def is_italian_gesture(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    coords = [(hand_landmarks.landmark[t].x, hand_landmarks.landmark[t].y) for t in tips]
    
    avg_x = sum([c[0] for c in coords]) / 5
    avg_y = sum([c[1] for c in coords]) / 5
    
    max_dist = max([math.hypot(c[0] - avg_x, c[1] - avg_y) for c in coords])
    
    # Sadece 5 parmak ucu birbirine ÇOK yakınsa (0.045) İtalya olsun (Yanlış tetiklenmeyi önler)
    return max_dist < 0.045

# 4. Kore Parmak Kalbi (🫰 Finger Heart) Kontrolü
def is_korea_finger_heart(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    
    dist_to_tip = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    dist_to_pip = math.hypot(thumb_tip.x - index_pip.x, thumb_tip.y - index_pip.y)
    
    touching = (dist_to_tip < 0.08) or (dist_to_pip < 0.08)
    
    middle_closed = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_closed = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_closed = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    
    return touching and middle_closed and ring_closed and pinky_closed

# 5. Ekrana Bilgi Kartı Çizen Fonksiyon
def draw_info_card(frame, title, gesture_name, info_list):
    h, w, _ = frame.shape
    
    card_w, card_h = 340, 180
    x1, y1 = 20, 20
    x2, y2 = x1 + card_w, y1 + card_h

    # Yarı saydam arka plan
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2) # Sarı Çerçeve
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Başlık ve Detaylar
    cv2.putText(frame, title, (x1 + 15, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, f"Jest: {gesture_name}", (x1 + 15, y1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    cv2.line(frame, (x1 + 15, y1 + 75), (x2 - 15, y1 + 75), (100, 100, 100), 1)

    y_offset = y1 + 105
    for info in info_list:
        cv2.putText(frame, f"- {info}", (x1 + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

print("Gelişmiş Ülke ve Jest Bilgi Sistemi Başlatıldı...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            (index_open, middle_open, ring_open, pinky_open), thumb_open = get_finger_states(hand_landmarks)

            # --- JEST VE ÜLKE KONTROLLERİ ---

            # 🇹🇷 TÜRKİYE: Bozkurt
            if index_open and pinky_open and not middle_open and not ring_open:
                draw_info_card(
                    frame, 
                    title="TURKIYE [TR]", 
                    gesture_name="Bozkurt / Wolf Sign", 
                    info_list=["Baskent: Ankara", "Meshur: Kebap, Cay, Doner", "Simge: Kiz Kulesi, Peri Bacalari"]
                )

            # 🇰🇷 GÜNEY KORE: Parmak Kalbi (Finger Heart)
            elif is_korea_finger_heart(hand_landmarks):
                draw_info_card(
                    frame, 
                    title="GUNEY KORE [KR]", 
                    gesture_name="Finger Heart (Parmak Kalbi)", 
                    info_list=["Baskent: Seul", "Meshur: K-Pop, Kimchi, Teknoloji", "Kültür: Kalp Hareketi & Dizi"]
                )

            # 🇮🇹 İTALYA: Parmak Birleştirme
            elif is_italian_gesture(hand_landmarks):
                draw_info_card(
                    frame, 
                    title="ITALYA [IT]", 
                    gesture_name="Che Vuoi? (Parmak Birlestirme)", 
                    info_list=["Baskent: Roma", "Meshur: Pizza, Pasta, Gelato", "Simge: Kolezyum, Pisa Kulesi"]
                )

            # 🇯🇵 JAPONYA: V / Peace İşareti
            elif index_open and middle_open and not ring_open and not pinky_open:
                draw_info_card(
                    frame, 
                    title="JAPONYA [JP]", 
                    gesture_name="Peace / V Sign", 
                    info_list=["Baskent: Tokyo", "Meshur: Susi, Anime, Sakura", "Kültür: Teknoloji & Gelenek"]
                )

            # 🌺 HAWAII (ABD): Shaka İşareti
           # 🌺 HAWAII (ABD): Shaka İşareti (Serçe AÇIK, İşaret-Orta-Yüzük KAPALI, Başparmak Yana Açık)
            elif pinky_open and not index_open and not middle_open and not ring_open:
                 draw_info_card(
                     frame, 
                     title="HAWAII [US]", 
                    gesture_name="Shaka / Hang Loose", 
                    info_list=["Baskent: Honolulu", "Meshur: Sorf, Volkanlar", "Kultur: Aloha Ruhu"]
    )

            # 🇬🇧 İNGİLTERE / GLOBAL: Thumbs Up
            elif thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
                draw_info_card(
                    frame, 
                    title="INGILTERE [UK]", 
                    gesture_name="Thumbs Up (Begenme)", 
                    info_list=["Baskent: Londra", "Meshur: Big Ben, Bes Cayi", "Kültür: Pop Muzik & Futbol"]
                )

    cv2.imshow("El Jesti ile Ulke Bilgi Kartlari", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#İki Durumu KıyaslayalımDurum A: Parmağın HAVADA (Açık)Parmağını yukarı kaldırdığında tırnağın (uç point), boğumunun üstünde kalır.Parmağın Ucu ($Y$): Ekranın yukarısında olduğu için küçük bir sayıdır. (Örn: $Y = 200$)Boğumu ($Y$): Daha aşağıda olduğu için büyük bir sayıdır. (Örn: $Y = 350$)Kodun yaptığı kıyaslama:Uç (200) < Boğum (350) -> DOĞRU (True)!Bilgisayar der ki: "Harika, uç noktası eklemden daha yukarıda! Demek ki parmak açık!"Durum B: Parmağın KAPALI (Kıvrık/Yumruk)Parmağını avucunun içine büktüğünde veya kapattığında tırnağın aşağıya düşer ve boğumunun altında kalır.Parmağın Ucu ($Y$): Aşağı düştüğü için sayısı büyür. (Örn: $Y = 400$)Boğumu ($Y$): Yukarıda kaldığı için sayısı küçüktür. (Örn: $Y = 350$)Kodun yaptığı kıyaslama:Uç (400) < Boğum (350) -> YANLIŞ (False)!Bilgisayar der ki: "Uç noktası boğumun altında kalmış. Demek ki parmak kapalı!"📝 Özetle:landmark[8].y < landmark[6].y derken Python'a aslında Türkçe olarak şunu soruyoruz:"8 numaralı tırnak ucumun $Y$ yüksekliği, 6 numaralı parmak boğumumun $Y$ yüksekliğinden daha mı küçük (yani ekranda daha mı yukarıda)?"Eğer cevap "Evet" ise parmak havada demektir ve sayacı 1 artırırız!
#uç küçük<boğum ise uç boğumdan küçükse parmak açık