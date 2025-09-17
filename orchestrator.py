import os
import ctypes
import numpy as np
from typing import Callable, Any

class PyOrchestrator:
    def __init__(self):
        # Загружаем C++ библиотеку
        lib_path = os.path.join(os.path.dirname(__file__), "build", "liborchestrator.so")
        self.lib = ctypes.CDLL(lib_path)

        self.lib.create_orchestrator.restype = ctypes.c_void_p
        self.lib.destroy_orchestrator.argtypes = [ctypes.c_void_p]

        self.lib.start.argtypes = [ctypes.c_void_p]
        self.lib.stop.argtypes = [ctypes.c_void_p]

        self.lib.push_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.orchestrator_ptr = self.lib.create_orchestrator()
        self._detection_cb = None

    def start(self):
        self.lib.start(self.orchestrator_ptr)

    def stop(self):
        if self.orchestrator_ptr:
            self.lib.stop(self.orchestrator_ptr)
            self.lib.destroy_orchestrator(self.orchestrator_ptr)
            self.orchestrator_ptr = None

    def push_frame(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        channels = frame.shape[2] if frame.ndim == 3 else 1

        self.lib.push_frame(
            self.orchestrator_ptr,
            frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            width,
            height,
            channels
        )

    def set_detection_callback(self, callback: Callable[[Any], None]):
        # Определяем C-тип callback
        CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        self._detection_cb = CALLBACK_TYPE(lambda result_ptr: callback(result_ptr))
        
        # Передаем callback в C++ (если есть функция в bindings.cpp)
        if hasattr(self.lib, "set_detection_callback"):
            self.lib.set_detection_callback.argtypes = [ctypes.c_void_p, CALLBACK_TYPE]
            self.lib.set_detection_callback(self.orchestrator_ptr, self._detection_cb)

    def __del__(self):
        self.stop()
