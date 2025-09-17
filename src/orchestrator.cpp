#include <iostream>
#include <thread>
#include <chrono>

#include "orchestrator.h"
#include <opencv2/imgproc.hpp>


// Mock engines
struct CVEngine : ICVEngine {
    CVEngine(const std::string&) {}
    std::vector<DetectionResult> process_frame(const cv::Mat&) override { return {}; }
};

struct SpeechEngine : ISpeechEngine {
    std::vector<SpeechResult> process_audio(const std::vector<float>&) override { return {}; }
};

inline float calculate_distance(const DetectionResult&) { return 0.0f; }

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
