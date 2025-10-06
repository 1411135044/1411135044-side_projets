from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import math
import time
import threading
import webbrowser

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands_solution = mp.solutions.hands # Renamed for clarity

# 全局變量來存儲手勢歷史和確認時間
GESTURE_CONFIRMATION_DURATION = 2  # 秒，統一手勢確認時間

gesture_history = {
    'current_gesture': '',
    'start_time': 0,
    'confirmed': False
}

# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 根據手指角度的串列內容，返回對應的手勢名稱 (1-8)
# 您可以根據實際測試效果微調這些角度閾值
def hand_pos(finger_angle):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮 (這是一個基本參考值，可能需要調整)
    # 確保手勢定義清晰且不易混淆

    if f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50: # 食指伸直
        return '1'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50: # 食指、中指伸直
        return '2'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 > 50: # 食指、中指、無名指伸直
        return '3'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50: # 四指伸直 (不含大拇指)
        return '4'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50: # 五指全伸
        return '5'
    # 手勢 6, 7, 8 的定義需要更仔細，以下為範例，您需要根據實際情況調整
    elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50: # 大拇指、小拇指伸直 (類似 '6')
        return '6'
    elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50: # 大拇指、食指伸直 (類似 'OK' 但不完全是，或是 L 手勢)
        return '7'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50: # 大拇指、食指、中指伸直
        return '8'
    # 可以添加一個 '0' 或其他非遊戲手勢的返回值
    # elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50: # 全都握拳
    #     return '0'
    else:
        return '' # 未定義或無法識別的手勢

def update_gesture_state(new_gesture):
    global gesture_history
    current_time = time.time()

    if new_gesture != gesture_history['current_gesture']:
        gesture_history.update({
            'current_gesture': new_gesture,
            'start_time': current_time,
            'confirmed': False
        })
        # print(f"[偵測] 手勢改變: {new_gesture}, 重置計時")
        return False, GESTURE_CONFIRMATION_DURATION # 返回未確認和總需時間

    duration = current_time - gesture_history['start_time']
    remaining = max(GESTURE_CONFIRMATION_DURATION - duration, 0)

    if duration >= GESTURE_CONFIRMATION_DURATION and not gesture_history['confirmed']:
        if gesture_history['current_gesture'] != '': # 只有在當前手勢非空時才確認
            gesture_history['confirmed'] = True
            # print(f"[偵測] 手勢確認: {gesture_history['current_gesture']} (持續 {duration:.2f} 秒)")
            return True, 0 # 返回已確認，剩餘0秒
        else: # 如果是空手勢持續了足夠時間，不標記為confirmed
            # print(f"[偵測] 空手勢持續，不確認")
            return False, remaining


    elif gesture_history['confirmed']:
         # 如果已經確認過，並且手勢仍然是同一個，則保持確認狀態，但重置以便下次手勢改變時能重新確認
        # 或者，如果希望確認一次後，只要手勢不變就一直發送 confirmed: True，則不需要重置 confirmed
        # 目前的邏輯是，一旦手勢改變，confirmed 會變 False，然後重新計時確認
        return True, 0

    # print(f"[偵測] 手勢: {gesture_history['current_gesture']}, 持續: {duration:.2f}s, 剩餘: {remaining:.2f}s, 確認中...")
    return False, remaining

# 手勢辨識執行緒
def gesture_detection():
    with mp_hands_solution.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=2  # 修改為允許偵測最多兩隻手
    ) as hands:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("錯誤：無法開啟攝像頭")
            return

        print("攝像頭已成功開啟，開始手勢偵測...")
        w_cam, h_cam = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_cam)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_cam)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("忽略空的攝像頭畫面。")
                continue

            # 定義中央區域（60% 寬高）
            center_region = (int(w_cam*0.2), int(h_cam*0.2), int(w_cam*0.8), int(h_cam*0.8))
            cv2.rectangle(image, (center_region[0], center_region[1]), 
                         (center_region[2], center_region[3]), (0,255,0), 2)

            # 翻轉與顏色轉換（必須在處理手勢前完成）
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            current_gesture_detected = ''
            if results.multi_hand_landmarks:
                # --- 新增邏輯：篩選最近且在中央區域的手 ---
                hands_data = []
                for hand_landmarks in results.multi_hand_landmarks:
                    # 提取手部邊界框座標
                    x_coords = [lm.x * w_cam for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h_cam for lm in hand_landmarks.landmark]
                    min_x, max_x = int(min(x_coords)), int(max(x_coords))
                    min_y, max_y = int(min(y_coords)), int(max(y_coords))
                    area = (max_x - min_x) * (max_y - min_y)
                    hand_center_x = (min_x + max_x) // 2
                    hand_center_y = (min_y + max_y) // 2
                    
                    # 檢查是否在中央區域
                    if (center_region[0] < hand_center_x < center_region[2] and
                        center_region[1] < hand_center_y < center_region[3]):
                        hands_data.append((area, hand_landmarks))

                # 選擇面積最大的手（最近的手）
                if hands_data:
                    hands_data.sort(reverse=True, key=lambda x: x[0])  # 按面積降序
                    selected_hand = hands_data[0][1]  # 取最大面積的手

                    # 繪製選中手的標註（可選）
                    mp_drawing.draw_landmarks(
                        image,
                        selected_hand,
                        mp_hands_solution.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 提取關節點座標並辨識手勢
                    landmark_list = []
                    for lm in selected_hand.landmark:
                        cx, cy = int(lm.x * w_cam), int(lm.y * h_cam)
                        landmark_list.append((cx, cy))
                    
                    if landmark_list:
                        finger_angles = hand_angle(landmark_list)
                        current_gesture_detected = hand_pos(finger_angles)

            # --- 後續原有邏輯保持不變 ---
            confirmed, remaining_time = update_gesture_state(current_gesture_detected)
            socketio.emit('gesture', {
                'gesture': gesture_history['current_gesture'],
                'remaining': round(remaining_time, 2),
                'confirmed': confirmed,
                'timestamp': time.time()
            })

            # 顯示畫面資訊（原有邏輯）
            cv2.putText(image, f"Gesture: {gesture_history['current_gesture']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Countdown: {remaining_time:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if not confirmed else (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Confirmed: {confirmed}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if not confirmed else (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('手勢辨識預覽 (按 Q 離開)', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            time.sleep(0.02)
        
        cap.release()
        cv2.destroyAllWindows()
        print("攝像頭已關閉。")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # 確保 static 資料夾存在
    import os
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    if not os.path.exists('static/sounds'):
        os.makedirs('static/sounds')
    if not os.path.exists('static/css'): # 雖然您提供了css，但確保路徑
        os.makedirs('static/css')
    if not os.path.exists('static/js'): # 確保js路徑
        os.makedirs('static/js')


    print("啟動手勢偵測執行緒...")
    detection_thread = threading.Thread(target=gesture_detection, daemon=True)
    detection_thread.start()

    # 延遲一點時間確保 Flask 伺服器先啟動
    # time.sleep(1)
    # webbrowser.open("http://127.0.0.1:5000") # 通常本地是 127.0.0.1

    print("Flask伺服器運行在 http://127.0.0.1:5000")
    # use_reloader=False 很重要，因為我們有背景執行緒
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)