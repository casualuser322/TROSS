import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "TROSS-recognition"))
sys.path.append(os.path.join(os.path.dirname(__file__), "TROSS-distance-counter"))

from recognizer import HaarFaceRecognizer
from orchestrator import PyOrchestrator

def main():
    orchestrator = PyOrchestrator()

    recognizer = HaarFaceRecognizer(
        "TROSS-recognition/haarcascades/haarcascade_frontalface_default.xml"
    )

    def detection_callback(result_ptr):
        print("Detection received!")

    orchestrator.set_detection_callback(detection_callback)
    orchestrator.start()

    cap = cv2.VideoCapture(0)

    REAL_FACE_WIDTH = 0.16  
    FOCAL_LENGTH = 600.0 

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        orchestrator.push_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = recognizer.detect_faces(gray)
        for (x, y, w, h) in faces:
            distance = (REAL_FACE_WIDTH * FOCAL_LENGTH) / w
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #using this in the test because most of people 
            #dont have 2 cameras included
            cv2.putText(
                frame,
                f"{distance:.2f} m",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Detection + Distance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    orchestrator.stop()


if __name__ == "__main__":
    main()
