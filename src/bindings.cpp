#include "orchestrator.h"

extern "C" {

Orchestrator* create_orchestrator() {
    return new Orchestrator();
}

void destroy_orchestrator(Orchestrator* o) {
    delete o;
}

void start(Orchestrator* o) {
    o->start();
}

void stop(Orchestrator* o) {
    o->stop();
}

void push_frame(Orchestrator* o, unsigned char* data, int width, int height, int channels) {
    cv::Mat frame(height, width, (channels == 3 ? CV_8UC3 : CV_8UC1), data);
    o->push_frame(frame);
}

struct CDetectionResult {
    const char* object_class;
    float confidence;
    float x, y, width, height;
};

using DetectionCallback = void(*)(CDetectionResult);

void set_detection_callback(Orchestrator* o, DetectionCallback cb) {
    o->set_detection_callback([cb](const DetectionResult& det) {
        CDetectionResult cdet;
        cdet.object_class = det.object_class.c_str();
        cdet.confidence = det.confidence;
        cdet.x = det.x;
        cdet.y = det.y;
        cdet.width = det.width;
        cdet.height = det.height;
        cb(cdet);
    });
}

}