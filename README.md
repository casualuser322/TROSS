# TROSS

**TROSS** — a modular framework for autonomous antropomorphic robot, that unites a C++ real-time orchestrator with Python mobules for stereo vision, monocular depth model training and simple recognition. This README documents the **entire TROSS system** as inspected in the uploaded repositories (`TROSS`, `TROSS-distance-counter`, `TROSS-recognition-main`).

## Project summary

* **C++ Orchestrator** — runs capture, processing, detection and distance estimation in separate threads with internal queues; written for low-latency, real-time usage;
* **Python modules** for data collection and model development:

  * Face recognition - recognising face using haar cascade
  * Stereo vision - calculating distance to recognised objects, using 2 cameras
  * Monocular depth - estimating distance to recognised objects, using model trained on stereo vision's work



## Key features

* Multi-threaded orchestrator managing queues for recognition tasks.
* C-style bindings to instantiate and control the orchestrator from Python.
* Stereo vision for distance and image depth calculation.
* Cameras' calibration utilities for camera intrinsic/extrinsic estimation.
* Monocular depth training model to estimate image depth using single-frame image.

---

## Repository layout

Top-level modules (three uploaded packages):

```
TROSS/              
  ├── CMakeLists.txt
  ├── requirements.txt
  ├── include
    └── orchestrator.h            
  ├──src/
    ├─ orchestrator.cpp
    └─ bindings.cpp
  ├── orchestrator.py       

TROSS-distance-counter
  ├── calibrator.py
  ├── dataset_collector.py
  ├── README.md
  ├── requirements.txt
  ├── stereo_vision.py
  └── train_mono_depth.py

TROSS-recognition-main/  
  ├─ README.md
  ├─ recognizer.py
  └─ haarcascades/
     └─ haarcascade_frontalface_default.xml
```

## Requirements & dependencies

### Build / runtime (C++)

* CMake >= **3.16**
* C++ compiler with `C++17` support
* **OpenCV** with `dnn`, `calib3d`, `imgproc` modules
* Python3::Development `3.11+`

### Python (for distance-counter & recognition)

* Use a virtual environment (.venv / conda).
* Typical packages (`TROSS-distance-counter/requirements.txt`):

  * `opencv-python`
  * `numpy`
  * `torch`
  * `tqdm`

---

## Build & install

From `TROSS-main`:

```bash
mkdir -p TROSS-main/build
cd TROSS-main/build
cmake ..
make -j16 (nproc)
```

It will create shared library `liborchestrator.so` linked against OpenCV and Python development libs.

> If CMake cannot find OpenCV or Python headers, install system packages `libopencv-dev`, `python3-dev` or point CMake to custom locations.

---

## Using the orchestrator from Python (bindings)

The C++ library exposes C-style bindings. During inspection, at least `create_orchestrator()` was present. Typical usage pattern (template — verify symbols in `src/bindings.cpp`):

```python
import ctypes
from ctypes import c_void_p

lib = ctypes.CDLL("/path/to/liborchestrator.so")
lib.create_orchestrator.restype = c_void_p

orc = lib.create_orchestrator()

lib.start.argtypes = [c_void_p]
lib.start.restype = None
lib.start(orc)

lib.destroy_orchestrator(orc)
```

## Testing & debugging tips

* Start with small test video or webcam to validate pipeline.
* Build in `Debug` mode if you need symbols or to use sanitizers:

  ```bash
  cmake -DCMAKE_BUILD_TYPE=Debug ..
  cmake --build .
  ```
* If threads do not exit cleanly, confirm orchestrator pushes sentinel/empty objects to queues at shutdown — the codebase uses that pattern to unblock waiting threads.
* Use logging in both C++ and Python layers to trace flow across the boundary.

## License

A `LICENSE` file exists in the main repo. Confirm the license type there and ensure all submodules and third-party assets (e.g., Haarcascade) are compatible.

---

## TODO

1. **Add sync scripts for camera**
2. **Docker dev image** with CMake, OpenCV and Python dev libs for reproducible builds.
3. **Add unit tests and lints** for Python scripts and C++ components.

