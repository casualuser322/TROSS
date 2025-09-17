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

}