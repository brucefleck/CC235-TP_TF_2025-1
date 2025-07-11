import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

model = load_model("asl_mobilenet_model.h5")

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
          'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'space', 'del', 'nothing']

cap = cv2.VideoCapture(0)
predicted_text = ""
history = deque(maxlen=20)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and draw rectangle
    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (224, 224))
    roi_normalized = roi_resized / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Predict
    preds = model.predict(roi_expanded, verbose=0)
    pred_label = labels[np.argmax(preds)]

    # Debounce and accumulate
    history.append(pred_label)
    most_common = max(set(history), key=history.count)

    if most_common != "nothing":
        if most_common == "space":
            predicted_text += " "
        elif most_common == "del":
            predicted_text = predicted_text[:-1]
        else:
            predicted_text += most_common

        history.clear()

    # Show on screen
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Letter: {pred_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(frame, f"Text: {predicted_text}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Reconocimiento", frame)

    # Exit on Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

