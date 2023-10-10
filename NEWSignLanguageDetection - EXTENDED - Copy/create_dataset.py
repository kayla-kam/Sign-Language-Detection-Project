import mediapipe as mp
import pickle
import cv2
import os
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence = 0.3)

data = []
labels = []

for dir_ in os.listdir('./data'):
    for img_path in os.listdir(os.path.join('./data', dir_)):
        data_aux = [] 
        img = cv2.imread(os.path.join('./data', dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x)
                    data_aux.append(hand_landmarks.landmark[i].y)

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

        