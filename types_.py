import ctypes

class DetectionResult(ctypes.Structure):
    _fields_ = [
        ("object_class", ctypes.c_char_p),
        ("confidence", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
    ]

class DistanceResult(ctypes.Structure):
    _fields_ = [
        ("object_class", ctypes.c_char_p),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("distance", ctypes.c_float),
    ]
