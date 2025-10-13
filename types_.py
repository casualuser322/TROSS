import ctypes
from typing import TypedDict

class DetectionResult(ctypes.Structure):
    _fields_ = [
        ("object_class", ctypes.c_char_p),
        ("confidence", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float)
    ]

class DistanceResult(ctypes.Structure):
    _fields_ = [
        ("object_class", ctypes.c_char_p),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("distance", ctypes.c_float)
    ]

class SpeechResult(ctypes.Structure):
    _fields_ = [
        ("command", ctypes.c_char_p),
        ("object_name", ctypes.c_char_p),
        ("confidence", ctypes.c_float)
    ]

class PyDetectionResult(TypedDict):
    object_class: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

class PyDistanceResult(TypedDict):
    object_class: str
    x: float
    y: float
    distance: float