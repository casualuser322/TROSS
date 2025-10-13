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

void push_stereo_frame(Orchestrator* o, 
                      unsigned char* left_data, int left_width, 
                      int left_height, int left_channels,
                      unsigned char* right_data, int right_width, 
                      int right_height, int right_channels) {
    cv::Mat left_frame(
                        left_height, left_width, 
                        (left_channels == 3 ? CV_8UC3 : CV_8UC1), 
                        left_data
                    );
    cv::Mat right_frame(
                        right_height, right_width, 
                        (right_channels == 3 ? CV_8UC3 : CV_8UC1), 
                        right_data
                    );
    o->push_stereo_frame(left_frame, right_frame);
}

void enable_stereo_mode(Orchestrator* o, int enable) {
    o->enable_stereo_mode(enable != 0);
}

void enable_mono_depth(Orchestrator* o, int enable) {
    o->enable_mono_depth(enable != 0);
}

int initialize_onnx_model(
                            Orchestrator* o, 
                            const char* model_path, 
                            const char** class_names, 
                            int num_classes, 
                            float confidence_threshold) {
    ModelConfig config;
    config.model_path = model_path;
    config.model_type = "onnx";
    config.confidence_threshold = confidence_threshold;
    config.input_size = cv::Size(416, 416);
    
    for (int i = 0; i < num_classes; i++) {
        config.class_names.push_back(class_names[i]);
    }
    
    return o->initialize_cv_model(config) ? 1 : 0;
}

int initialize_haar_cascade(Orchestrator* o, const char* cascade_path) {
    ModelConfig config;
    config.model_path  = cascade_path;
    config.model_type  = "haar";
    config.class_names = {"face"};
    
    return o->initialize_cv_model(config) ? 1 : 0;
}

int initialize_stereo_vision(Orchestrator* o, 
                             const char* calibration_file, 
                             float baseline, float focal_length) {
    ModelConfig config;
    config.model_type       = "stereo";
    config.calibration_file = calibration_file;
    config.baseline         = baseline;
    config.focal_length     = focal_length;
    
    return o->initialize_stereo_model(config) ? 1 : 0;
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
        cdet.confidence   = det.confidence;
        cdet.x            = det.x;
        cdet.y            = det.y;
        cdet.width        = det.width;
        cdet.height       = det.height;
        cb(cdet);
    });
}

struct CDistanceResult {
    const char* object_class;
    float x, y, distance;
};

using DistanceCallback = void(*)(CDistanceResult);

void set_distance_callback(Orchestrator* o, DistanceCallback cb) {
    o->set_distance_callback([cb](const DistanceResult& dist) {
        CDistanceResult cdist;
        cdist.object_class = dist.object_class.c_str();
        cdist.x            = dist.x;
        cdist.y            = dist.y;
        cdist.distance     = dist.distance;
        cb(cdist);
    });
}
}