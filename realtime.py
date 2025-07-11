import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# 🔥 Cargar modelo
model = tf.keras.models.load_model("ResNet152V2-ASL.h5")

# 📋 Detectar número de clases desde el modelo
output_shape = model.output_shape[-1]  # e.g., (None, 29)
print(f"📦 Modelo tiene {output_shape} clases de salida")

# Crear lista de clases según cantidad de salidas
if output_shape == 26:
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
elif output_shape == 29:
    # Ejemplo: ASL incluye 'space', 'del', 'nothing'
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['space', 'del', 'nothing']
else:
    raise ValueError(f"⚠️ No reconozco {output_shape} clases. Ajusta class_names manualmente.")

print(f"🔤 Clases detectadas: {class_names}")

# 🖐️ Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 🎥 Webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara."); exit()
print("🎥 Webcam iniciada. Presiona 'q' para salir.")

# 🔄 Loop
while True:
    ret, frame = cap.read()
    if not ret: break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Bounding Box
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            margin = 30
            x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
            x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0: continue

            # 🏁 Preprocesamiento
            roi_resized = cv2.resize(roi, (256, 256))  # ⚡ Para ResNet152V2
            roi_normalized = roi_resized / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=0)

            # 🔮 Predicción
            pred = model.predict(roi_expanded, verbose=0)[0]
            pred_idx = int(np.argmax(pred))
            if pred_idx < len(class_names):
                pred_label = class_names[pred_idx]
            else:
                pred_label = "Unknown"
            confidence = pred[pred_idx]

            # Texto en recuadro
            text = f'{pred_label}: {confidence:.2%}'
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    else:
        cv2.putText(frame, "No se detecta mano", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("ASL Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
