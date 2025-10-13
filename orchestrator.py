import os
import ctypes
import numpy as np
from typing import Callable, Any, List

from types_ import DistanceResult, DetectionResult

class PyOrchestrator:
    def __init__(self):
        lib_path = os.path.join(
            os.path.dirname(__file__), "build", "liborchestrator.so"
        )
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

        self.lib.push_stereo_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte), 
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte), 
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.enable_stereo_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.enable_mono_depth.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.lib.initialize_onnx_model.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_float
        ]
        self.lib.initialize_onnx_model.restype = ctypes.c_int

        self.lib.initialize_haar_cascade.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p
        ]
        self.lib.initialize_haar_cascade.restype = ctypes.c_int

        self.lib.initialize_stereo_vision.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_float,
            ctypes.c_float
        ]
        self.lib.initialize_stereo_vision.restype = ctypes.c_int

        self.orchestrator_ptr = self.lib.create_orchestrator()
        self._detection_cb = None
        self._distance_cb = None

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

    def push_stereo_frame(self, left_frame: np.ndarray, 
                          right_frame: np.ndarray):
        left_height, left_width = left_frame.shape[:2]
        left_channels = left_frame.shape[2] if left_frame.ndim == 3 else 1
        
        right_height, right_width = right_frame.shape[:2]
        right_channels = right_frame.shape[2] if right_frame.ndim == 3 else 1

        self.lib.push_stereo_frame(
            self.orchestrator_ptr,
            left_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            left_width, left_height, left_channels,
            right_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            right_width, right_height, right_channels
        )

    def enable_stereo_mode(self, enable: bool):
        self.lib.enable_stereo_mode(self.orchestrator_ptr, 1 if enable else 0)

    def enable_mono_depth(self, enable: bool):
        self.lib.enable_mono_depth(self.orchestrator_ptr, 1 if enable else 0)

    def initialize_onnx_model(self, model_path: str, class_names: List[str], 
                              confidence_threshold: float = 0.5):
        model_path_bytes = model_path.encode('utf-8')
        
        class_names_array = (ctypes.c_char_p * len(class_names))()
        for i, name in enumerate(class_names):
            class_names_array[i] = name.encode('utf-8')
        
        success = self.lib.initialize_onnx_model(
            self.orchestrator_ptr,
            model_path_bytes,
            class_names_array,
            len(class_names),
            ctypes.c_float(confidence_threshold)
        )
        
        return bool(success)

    def initialize_haar_cascade(self, cascade_path: str):
        cascade_path_bytes = cascade_path.encode('utf-8')
        success = self.lib.initialize_haar_cascade(
            self.orchestrator_ptr,
            cascade_path_bytes
        )
        return bool(success)

    def initialize_stereo_vision(self, calibration_file: str, 
                                 baseline: float = 0.12, 
                                 focal_length: float = 1250.0):
        calibration_file_bytes = calibration_file.encode('utf-8')
        success = self.lib.initialize_stereo_vision(
            self.orchestrator_ptr,
            calibration_file_bytes,
            ctypes.c_float(baseline),
            ctypes.c_float(focal_length)
        )
        return bool(success)

    def set_detection_callback(self, 
                               callback: Callable[[DetectionResult], None]):
        CALLBACK_TYPE = ctypes.CFUNCTYPE(
            None, 
            ctypes.POINTER(DetectionResult)
        )
        self._detection_cb = CALLBACK_TYPE(
            lambda result_ptr: callback(result_ptr.contents)
        )
        
        self.lib.set_detection_callback.argtypes = [
            ctypes.c_void_p, 
            CALLBACK_TYPE
        ]
        self.lib.set_detection_callback(
            self.orchestrator_ptr, 
            self._detection_cb
        )
    
    def set_distance_callback(self, 
                              callback: Callable[[DistanceResult], None]):
        CALLBACK_TYPE = ctypes.CFUNCTYPE(
            None, 
            ctypes.POINTER(DistanceResult)
        )
        self._distance_cb = CALLBACK_TYPE(
            lambda result_ptr: callback(result_ptr.contents)
        )

        self.lib.set_distance_callback.argtypes = [
            ctypes.c_void_p,
            CALLBACK_TYPE
        ]
        self.lib.set_distance_callback(
            self.orchestrator_ptr,
            self._distance_cb
        )

    def __del__(self):
        self.stop()
