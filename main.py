import cv2
from orchestrator import PyOrchestrator


if __name__ == '__main__':
    orch = PyOrchestrator()
    orch.start()

    frame = cv2.imread("test.jpg")
    orch.push_frame(frame)

    orch.stop()
