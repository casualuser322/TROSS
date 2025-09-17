#include <iostream>
#include <thread>
#include <chrono>

#include "orchestrator.h"
#include <opencv2/imgproc.hpp>


struct CVEngine : ICVEngine {
    cv::CascadeClassifier face_cascade;

    CVEngine(const std::string& model_path) {
        face_cascade.load(model_path);
    }

    std::vector<DetectionResult> process_frame(const cv::Mat& frame) override {
        std::vector<DetectionResult> detections;

        if (frame.empty()) return detections;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

        for (auto& f : faces) {
            DetectionResult det;
            det.object_class = "face";
            det.confidence = 1.0f;
            det.x = f.x;
            det.y = f.y;
            det.width = f.width;
            det.height = f.height;
            detections.push_back(det);
        }

        return detections;
    }
};

struct SpeechEngine : ISpeechEngine {
    std::vector<SpeechResult> process_audio(const std::vector<float>&) override { return {}; }
};

inline float calculate_distance(const DetectionResult& det) {
    float focal_length = 600.0f;
    float real_face_width = 0.16f;
    return (real_face_width * focal_length) / det.width;
}

Orchestrator::Orchestrator() {}
Orchestrator::~Orchestrator() { stop(); }

void Orchestrator::start() {
    running = true;

    cv_thread = std::thread(&Orchestrator::cv_thread_func, this);
    distance_thread = std::thread(&Orchestrator::distance_thread_func, this);
    speech_thread = std::thread(&Orchestrator::speech_thread_func, this);
    detection_thread = std::thread(&Orchestrator::decision_thread_func, this);

    std::cout << "Orchestrator started\n";
}

void Orchestrator::stop() {
    running = false;

    // Push dummy to unblock threads
    frame_queue.push(cv::Mat());
    audio_queue.push({});
    detection_queue.push({});
    distance_queue.push({});
    speech_queue.push({});

    if (cv_thread.joinable()) cv_thread.join();
    if (detection_thread.joinable()) detection_thread.join();
    if (distance_thread.joinable()) distance_thread.join();
    if (speech_thread.joinable()) speech_thread.join();

    std::cout << "Orchestrator stopped\n";
}

void Orchestrator::push_frame(const cv::Mat& frame) { frame_queue.push(frame); }
void Orchestrator::push_audio(const std::vector<float>& audio_data) { audio_queue.push(audio_data); }

void Orchestrator::cv_thread_func() {
    CVEngine engine("model.onnx");
    while (running) {
        cv::Mat frame = frame_queue.wait_and_pop();
        if (!running) break;
        auto results = engine.process_frame(frame);
        for (auto& r : results) {
            detection_queue.push(r);
            if (detection_callback) detection_callback(r);
        }
    }
}

void Orchestrator::distance_thread_func() {
    while (running) {
        DetectionResult det = detection_queue.wait_and_pop();
        if (!running) break;
        DistanceResult dist;
        dist.object_class = det.object_class;
        dist.distance = calculate_distance(det);
        if (distance_callback) distance_callback(dist);
    }
}

void Orchestrator::speech_thread_func() {
    SpeechEngine engine;
    while (running) {
        std::vector<float> audio = audio_queue.wait_and_pop();
        if (!running) break;
        auto results = engine.process_audio(audio);
        for (auto& r : results)
            if (speech_callback) speech_callback(r);
    }
}

void Orchestrator::decision_thread_func() {
    while (running) {
        // ...
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
