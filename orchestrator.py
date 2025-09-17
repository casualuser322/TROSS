import os
import ctypes
import numpy as np

from typing import Callable, Any


class PyOrchestrator:
    def __init__(self):
        # uploading c++ lib
        self.lib = ctypes.CDLL(
            os.path.join(
                os.path.dirname(__file__), 
                "build", 
                "liborchestrator.so"
            )
        )

        self.lib.create_orchestrator.argtypes = []
        self.lib.create_orchestrator.restype = ctypes.c_void_p

        self.lib.start.argtypes = [ctypes.c_void_p]
        self.lib.stop.argtypes = [ctypes.c_void_p]

        self.lib.push_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        #creating orchestrator
        self.orchestrator_ptr = self.lib.create_orchestrator()
    
    def start(self):
        self.lib.start(self.orchestrator_ptr)
    
    def stop(self):
        self.lib.stop(self.orchestrator_ptr)
    
    def push_frame(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1

        self.lib.push_frame(
            self.orchestrator_ptr,
            frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            width, 
            height, 
            channels
        )
    
    def set_detection_callback(self, callback: Callable[[Any], None]):
        #callbacks
        pass

    def __del__(self):
        if hasattr(self, "orchestrator_ptr") and self.orchestrator_ptr:
            self.stop()