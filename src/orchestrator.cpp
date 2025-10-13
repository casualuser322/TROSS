#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <memory>

#include "orchestrator.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>


class HaarCascadeEngine : public ICVEngine {
private:
    cv::CascadeClassifier cascade;
    ModelConfig config;

public:
    bool initialize(const ModelConfig& cfg) override {
        config = cfg;
        if (!cascade.load(config.model_path)) {
            std::cerr << "Failed to load Haar cascade: " << config.model_path << std::endl;
            return false;
        }
        std::cout << "Haar cascade loaded successfully: " << config.model_path << std::endl;
        return true;
    }

    std::vector<DetectionResult> process_frame(const cv::Mat& frame) override {
        std::vector<DetectionResult> detections;

        if (frame.empty()) return detections;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> objects;
        cascade.detectMultiScale(gray, objects, 1.1, 3, 0, cv::Size(30, 30));

        for (auto& obj : objects) {
            DetectionResult det;
            det.object_class = "face";
            det.confidence = 1.0f;
            det.x      = obj.x;
            det.y      = obj.y;
            det.width  = obj.width;
            det.height = obj.height;
            detections.push_back(det);
        }

        return detections;
    }
};

class ONNXEngine : public ICVEngine {
private:
    cv::dnn::Net net;
    ModelConfig config;
    std::vector<std::string> output_names;

public:
    bool initialize(const ModelConfig& cfg) override {
        config = cfg;
        try {
            net = cv::dnn::readNetFromONNX(config.model_path);
            
            #ifdef WITH_CUDA
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            #else
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            #endif

            output_names = net.getUnconnectedOutLayersNames();
            std::cout << "ONNX model loaded successfully: " << config.model_path << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<DetectionResult> process_frame(const cv::Mat& frame) override {
        std::vector<DetectionResult> detections;
        if (frame.empty()) return detections;

        try {
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, config.scale_factor, 
                                 config.input_size, config.mean, config.swap_rb, false);
            net.setInput(blob);

            std::vector<cv::Mat> outputs;
            net.forward(outputs, output_names);

            for (const auto& output : outputs) {
                for (int i = 0; i < output.rows; i++) {
                    const float* data = output.ptr<float>(i);
                    
                    float confidence = data[4];
                    if (confidence < config.confidence_threshold) continue;

                    int class_id = 0;
                    float max_class_conf = 0;
                    for (int j = 5; j < output.cols; j++) {
                        if (data[j] > max_class_conf) {
                            max_class_conf = data[j];
                            class_id = j - 5;
                        }
                    }

                    float final_confidence = confidence * max_class_conf;
                    if (final_confidence < config.confidence_threshold) continue;

                    float center_x = data[0] * frame.cols;
                    float center_y = data[1] * frame.rows;
                    float width    = data[2] * frame.cols;
                    float height   = data[3] * frame.rows;

                    DetectionResult det;
                    det.object_class = (class_id < config.class_names.size()) ? 
                                      config.class_names[class_id] : "unknown";
                    det.confidence = final_confidence;
                    det.x      = center_x - width / 2;
                    det.y      = center_y - height / 2;
                    det.width  = width;
                    det.height = height;
                    
                    detections.push_back(det);
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "ONNX processing error: " << e.what() << std::endl;
        }

        return detections;
    }
};

class StereoVisionEngine : public IStereoEngine {
private:
    struct CalibrationData {
        cv::Mat left_map_x, left_map_y;
        cv::Mat right_map_x, right_map_y;
        float baseline = 0.12f;
        float focal_length = 1250.0f;
    };

    CalibrationData calib_data;
    ModelConfig config;
    cv::Ptr<cv::StereoBM> stereo_bm;
    bool calibrated = false;

public:
    bool initialize(const ModelConfig& cfg) override {
        config = cfg;
        
        calibrated = load_calibration(config.calibration_file);
        
        stereo_bm = cv::StereoBM::create(160, 21);
        
        std::cout << "Stereo vision engine initialized" << std::endl;
        return calibrated;
    }

    std::vector<DistanceResult> process_stereo_frame(
                                                const cv::Mat& left, 
                                                const cv::Mat& right) override {
        std::vector<DistanceResult> results;
        
        if (!calibrated || left.empty() || right.empty()) return results;

        try {
            cv::Mat left_rect, right_rect;
            cv::remap(
                left, left_rect, 
                calib_data.left_map_x, 
                calib_data.left_map_y, 
                cv::INTER_LINEAR
            );
            cv::remap(
                right, right_rect, 
                calib_data.right_map_x, 
                calib_data.right_map_y, 
                cv::INTER_LINEAR
            );

            cv::Mat left_gray, right_gray;
            cv::cvtColor(left_rect, left_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right_rect, right_gray, cv::COLOR_BGR2GRAY);

            cv::Mat disparity;
            stereo_bm->compute(left_gray, right_gray, disparity);
            disparity.convertTo(disparity, CV_32F, 1.0/16.0);

            results = detect_objects_from_disparity(disparity);

        } catch (const std::exception& e) {
            std::cerr << "Stereo processing error: " << e.what() << std::endl;
        }

        return results;
    }

    float calculate_distance(const DetectionResult& det, const cv::Mat& left_frame) override {
        if (!calibrated) return -1.0f;

        float focal_length = calib_data.focal_length;
        float real_size = 0.0f;

        if (det.object_class == "face") {
            real_size = 0.16f;
            return (real_size * focal_length) / det.width;
        } else if (det.object_class == "person") {
            real_size = 0.5f;
            return (real_size * focal_length) / det.width;
        } else {
            real_size = 0.3f;
            return (real_size * focal_length) / ((det.width + det.height) / 2);
        }
    }

private:
    bool load_calibration(const std::string& calibration_file) {
        calib_data.baseline = config.baseline;
        calib_data.focal_length = config.focal_length;
        
        cv::Size image_size(640, 480);
        calib_data.left_map_x  = cv::Mat::zeros(image_size, CV_32F);
        calib_data.left_map_y  = cv::Mat::zeros(image_size, CV_32F);
        calib_data.right_map_x = cv::Mat::zeros(image_size, CV_32F);
        calib_data.right_map_y = cv::Mat::zeros(image_size, CV_32F);
        
        for (int i = 0; i < image_size.height; i++) {
            for (int j = 0; j < image_size.width; j++) {
                calib_data.left_map_x.at<float>(i, j) = j;
                calib_data.left_map_y.at<float>(i, j) = i;
                calib_data.right_map_x.at<float>(i, j) = j;
                calib_data.right_map_y.at<float>(i, j) = i;
            }
        }
        
        return true;
    }

    std::vector<DistanceResult> detect_objects_from_disparity(const cv::Mat& disparity) {
        std::vector<DistanceResult> results;
        
        cv::Mat valid_disparity = disparity > 0;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(valid_disparity, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            if (contour.size() < 5) continue;
            
            cv::Rect bbox = cv::boundingRect(contour);
            if (bbox.area() < 100) continue;
            
            cv::Mat roi = disparity(bbox);
            cv::Scalar mean_val = cv::mean(roi, roi > 0);
            float mean_disparity = mean_val[0];
            
            if (mean_disparity <= 0) continue;
            
            float distance = (calib_data.baseline * calib_data.focal_length) / mean_disparity;
            
            DistanceResult result;
            result.object_class = "object";
            result.x = bbox.x + bbox.width / 2;
            result.y = bbox.y + bbox.height / 2;
            result.distance = distance;
            
            results.push_back(result);
        }
        
        return results;
    }
};

struct SpeechEngine : ISpeechEngine {
    std::vector<SpeechResult> process_audio(const std::vector<float>&) override { 
        return {}; 
    }
};

Orchestrator::Orchestrator() {}
Orchestrator::~Orchestrator() { stop(); }

bool Orchestrator::initialize_cv_model(const ModelConfig& config) {
    cv_model_config = config;
    
    if (config.model_type == "onnx") {
        cv_engine = std::make_unique<ONNXEngine>();
    } else if (config.model_type == "haar") {
        cv_engine = std::make_unique<HaarCascadeEngine>();
    } else {
        std::cerr << "Unknown CV model type: " << config.model_type << std::endl;
        return false;
    }
    
    return cv_engine->initialize(config);
}

bool Orchestrator::initialize_stereo_model(const ModelConfig& config) {
    stereo_model_config = config;
    stereo_engine = std::make_unique<StereoVisionEngine>();
    return stereo_engine->initialize(config);
}

bool Orchestrator::initialize_speech_model(const std::string& model_path) {
    speech_engine = std::make_unique<SpeechEngine>();
    return true;
}

void Orchestrator::start() {
    if (!cv_engine && !stereo_engine) {
        std::cerr << 
        "No engines initialized. Call initialize_cv_model() or initialize_stereo_model() first." 
        << std::endl;
        return;
    }

    running = true;

    if (cv_engine) {
        cv_thread = std::thread(&Orchestrator::cv_thread_func, this);
        distance_thread = std::thread(&Orchestrator::distance_thread_func, this);
    }
    
    if (stereo_engine && stereo_mode) {
        stereo_thread = std::thread(&Orchestrator::stereo_thread_func, this);
    }
    
    speech_thread = std::thread(&Orchestrator::speech_thread_func, this);
    decision_thread = std::thread(&Orchestrator::decision_thread_func, this);

    std::cout << "Orchestrator started - CV: " << (cv_engine ? "yes" : "no")
              << ", Stereo: " << (stereo_engine && stereo_mode ? "yes" : "no")
              << ", MonoDepth: " << (mono_depth_mode ? "yes" : "no") << std::endl;
}

void Orchestrator::stop() {
    running = false;

    frame_queue.push(cv::Mat());
    stereo_queue.push(StereoFrame());
    audio_queue.push({});
    detection_queue.push({});
    distance_queue.push({});
    speech_queue.push({});

    if (cv_thread.joinable()) cv_thread.join();
    if (stereo_thread.joinable()) stereo_thread.join();
    if (distance_thread.joinable()) distance_thread.join();
    if (speech_thread.joinable()) speech_thread.join();
    if (decision_thread.joinable()) decision_thread.join();

    std::cout << "Orchestrator stopped\n";
}

void Orchestrator::push_frame(const cv::Mat& frame) { 
    frame_queue.push(frame.clone()); 
}

void Orchestrator::push_stereo_frame(const cv::Mat& left, const cv::Mat& right) {
    StereoFrame stereo_frame;
    stereo_frame.left = left.clone();
    stereo_frame.right = right.clone();
    stereo_frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stereo_queue.push(stereo_frame);
}

void Orchestrator::push_audio(const std::vector<float>& audio_data) { 
    audio_queue.push(audio_data); 
}

void Orchestrator::cv_thread_func() {
    while (running) {
        cv::Mat frame = frame_queue.wait_and_pop();
        if (!running || frame.empty()) continue;
        
        auto results = cv_engine->process_frame(frame);
        for (auto& r : results) {
            detection_queue.push(r);
            if (detection_callback) detection_callback(r);
            
            if (mono_depth_mode && stereo_engine) {
                float distance = stereo_engine->calculate_distance(r, frame);
                DistanceResult dist;
                dist.object_class = r.object_class;
                dist.x = r.x + r.width / 2;
                dist.y = r.y + r.height / 2;
                dist.distance = distance;
                
                distance_queue.push(dist);
                if (distance_callback) distance_callback(dist);
            }
        }
    }
}

void Orchestrator::stereo_thread_func() {
    while (running) {
        StereoFrame stereo_frame = stereo_queue.wait_and_pop();
        if (!running || stereo_frame.left.empty() || stereo_frame.right.empty()) continue;
        
        auto results = stereo_engine->process_stereo_frame(stereo_frame.left, stereo_frame.right);
        for (auto& r : results) {
            distance_queue.push(r);
            if (distance_callback) distance_callback(r);
        }
    }
}

void Orchestrator::distance_thread_func() {
    while (running) {
        DetectionResult det = detection_queue.wait_and_pop();
        if (!running) break;
        
        if (!stereo_mode && !mono_depth_mode) {
            float focal_length = 600.0f;
            float real_size = 0.0f;
            
            if (det.object_class == "face") {
                real_size = 0.16f;
            } else if (det.object_class == "person") {
                real_size = 0.5f;
            } else {
                real_size = 0.3f;
            }
            
            float distance = (real_size * focal_length) / det.width;
            
            DistanceResult dist;
            dist.object_class = det.object_class;
            dist.x = det.x + det.width / 2;
            dist.y = det.y + det.height / 2;
            dist.distance = distance;
            
            distance_queue.push(dist);
            if (distance_callback) distance_callback(dist);
        }
    }
}

void Orchestrator::speech_thread_func() {
    if (!speech_engine) {
        speech_engine = std::make_unique<SpeechEngine>();
    }

    while (running) {
        std::vector<float> audio = audio_queue.wait_and_pop();
        if (!running) break;
        
        auto results = speech_engine->process_audio(audio);
        for (auto& r : results) {
            speech_queue.push(r);
            if (speech_callback) speech_callback(r);
        }
    }
}

void Orchestrator::decision_thread_func() {
    while (running) {
        DetectionResult det;
        DistanceResult dist;
        SpeechResult speech;
        
        if (detection_queue.try_pop(det)) {}
        
        if (distance_queue.try_pop(dist)) {}
        
        if (speech_queue.try_pop(speech)) {}
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void Orchestrator::set_detection_callback(std::function<void(const DetectionResult&)> cb) { 
    detection_callback = cb; 
}

void Orchestrator::set_distance_callback(std::function<void(const DistanceResult&)> cb) { 
    distance_callback = cb; 
}

void Orchestrator::set_speech_callback(std::function<void(const SpeechResult&)> cb) { 
    speech_callback = cb; 
}