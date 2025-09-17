import sys
import os
import cv2
import numpy as np

# Добавляем корень проекта в sys.path, чтобы Python видел пакеты
sys.path.append(os.path.join(os.path.dirname(__file__), "TROSS-recognition"))
sys.path.append(os.path.join(os.path.dirname(__file__), "TROSS-distance-counter"))

from recognizer import HaarFaceRecognizer
from orchestrator import PyOrchestrator

def main():
    # Инициализация Orchestrator
    orchestrator = PyOrchestrator()

    # Инициализация распознавателя лиц
    recognizer = HaarFaceRecognizer(
        "TROSS-recognition/haarcascades/haarcascade_frontalface_default.xml"
    )

    # Callback для демонстрации
    def detection_callback(result_ptr):
        print("Detection received!")

    orchestrator.set_detection_callback(detection_callback)
    orchestrator.start()

    # Камера
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Отправка кадра в Orchestrator
        orchestrator.push_frame(frame)

        # Отображение локально через Haarcascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = recognizer.detect_faces(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    orchestrator.stop()

if __name__ == "__main__":
    main()
