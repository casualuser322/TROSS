import cv2
import numpy as np
import time
from pathlib import Path

from orchestrator import PyOrchestrator
from types_ import DetectionResult, DistanceResult

import sys
sys.path.append('TROSS-recognition')
sys.path.append('TROSS-distance-counter')

from TROSS_recognition.recognizer import HaarFaceRecognizer
from TROSS_distance_counter.stereo_vision import StereoVision
from TROSS_distance_counter.calibrator import StereoCalibrator
from TROSS_distance_counter.dataset_collector import DistanceDatasetCollector

HAAR_PATH = "TROSS-recognition/haarcascades/\
            haarcascade_frontalface_default.xml"

class TROSSIntegratedSystem:
    def __init__(self):
        self.orchestrator = PyOrchestrator()
        self.face_recognizer = HaarFaceRecognizer(HAAR_PATH)
        self.stereo_vision = None
        self.dataset_collector = DistanceDatasetCollector("distance_dataset")
        
        self.orchestrator.set_detection_callback(self.on_detection)
        self.orchestrator.set_distance_callback(self.on_distance)
        
    def on_detection(self, det: DetectionResult):
        print(f"Detected: {det.object_class} at ({det.x:.1f}, {det.y:.1f}) "
              f"size: {det.width:.1f}x{det.height:.1f} \
                conf: {det.confidence:.2f}")
    
    def on_distance(self, dist: DistanceResult):
        print(f"Distance to \
              {dist.object_class}: {dist.distance:.2f}m\
                at ({dist.x:.1f}, {dist.y:.1f})")
    
    def initialize_face_detection(self):
        return self.orchestrator.initialize_haar_cascade(HAAR_PATH)
    
    def initialize_stereo_vision(self, calibration_file="calibration.json"):
        success = self.orchestrator.initialize_stereo_vision(calibration_file)
        if success:
            self.orchestrator.enable_stereo_mode(True)
        return success
    
    def initialize_mono_depth(self):
        self.orchestrator.enable_mono_depth(True)
    
    def process_webcam(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        self.orchestrator.start()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.orchestrator.push_frame(frame)
                
                cv2.imshow('TROSS System', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.03)
                
        finally:
            self.orchestrator.stop()
            cap.release()
            cv2.destroyAllWindows()
    
    def process_stereo_cameras(self, left_cam_id=0, right_cam_id=1):
        left_cap = cv2.VideoCapture(left_cam_id)
        right_cap = cv2.VideoCapture(right_cam_id)
        
        if not left_cap.isOpened() or not right_cap.isOpened():
            print("Error: Cannot open stereo cameras")
            return
        
        left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.orchestrator.start()
        
        try:
            while True:
                ret_left, left_frame = left_cap.read()
                ret_right, right_frame = right_cap.read()
                
                if not ret_left or not ret_right:
                    break
                
                self.orchestrator.push_stereo_frame(left_frame, right_frame)
                
                mosaic = np.hstack([left_frame, right_frame])
                cv2.imshow('Stereo Cameras', mosaic)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.03)
                
        finally:
            self.orchestrator.stop()
            left_cap.release()
            right_cap.release()
            cv2.destroyAllWindows()

def main():
    system = TROSSIntegratedSystem()
    
    print("TROSS Integrated System")
    print("1. Face detection only")
    print("2. Face detection with mono depth")
    print("3. Stereo vision")
    
    choice = input("Select mode (1-3): ").strip()
    
    if choice == "1":
        if system.initialize_face_detection():
            print("Face detection initialized")
            system.process_webcam()
        else:
            print("Failed to initialize face detection")
    
    elif choice == "2":
        if system.initialize_face_detection():
            system.initialize_mono_depth()
            print("Face detection with mono depth initialized")
            system.process_webcam()
        else:
            print("Failed to initialize face detection")
    
    elif choice == "3":
        if system.initialize_stereo_vision():
            print("Stereo vision initialized")
            system.process_stereo_cameras()
        else:
            print("Failed to initialize stereo vision")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()