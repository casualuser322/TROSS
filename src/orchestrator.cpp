#include "orchestrator.h"

struct CVEngine {
    CVEngine(const std::string&) {}
    std::vector<DetectionResult> process_frame(const cv::Mat&) {
        return {};
    }
};

struct SpeechEngine {
    std::vector<SpeechResult> process_audio(const std::vector<float>&) {
        return {};
    }
};

inline float calculate_distance(const DetectionResult&) {
    return 0.0f; // просто 0
}


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

Orchestrator::Orchestrator() {
    //module initialisation
}

Orchestrator::~Orchestrator() {
    stop();
}

void Orchestrator::start() {
    running = true;

    //threads managing
    cv_thread        = std::thread(&Orchestrator::cv_thread_func, this);
    detection_thread = std::thread(&Orchestrator::decision_thread_func, this);
    distance_thread  = std::thread(&Orchestrator::distance_thread_func, this);
    speech_thread    = std::thread(&Orchestrator::speech_thread_func, this);

    std::cout << "Orchestrator started" << std::endl;
}

void Orchestrator::stop() {
    running = false;

    //notifing all threads
    frame_cova.notify_all();
    audio_cova.notify_all();
    detection_cova.notify_all();
    distance_cova.notify_all();
    speech_cova.notify_all();

    //waiting threads to stop
    if (cv_thread.joinable()) cv_thread.join();
    if (detection_thread.joinable()) detection_thread.join();
    if (distance_thread.joinable()) distance_thread.join();
    if (speech_thread.joinable()) speech_thread.join();

    std::cout << "Orchestrator stoped" << std::endl;
}

void Orchestrator::push_frame(const cv::Mat& frame) {
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        frame_queue.push(frame.clone());
    }
    frame_cova.notify_one();
}

void Orchestrator::push_audio(const std::vector<float>& audio_data) {
    {
        std::lock_guard<std::mutex> lock(audio_mutex);
        audio_queue.push(audio_data);
    }
    audio_cova.notify_one();
}


void Orchestrator::cv_thread_func() {
    CVEngine cv_engine("model.onnx");

    while (running) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            frame_cova.wait(lock, [this]() {return !frame_queue.empty() || !running; });

            if (!running) break;

            frame = frame_queue.front();
            frame_queue.pop();
        }

        auto results = cv_engine.process_frame(frame);

        //sending result
        for (const auto& result : results) {
            if (detection_callback) {
                detection_callback(result);
            }

            //putting distance counter thread
            {
                std::lock_guard<std::mutex> lock(detection_mutex);
                detection_queue.push(result);
            }
            detection_cova.notify_one();
        }
    }
}

void Orchestrator::distance_thread_func() {
    while (running) {
        DetectionResult detection;
        {
            std::unique_lock<std::mutex> lock(detection_mutex);
            detection_cova.wait(lock, [this]() {return !detection_queue.empty() || !running;});
        
            if (!running) break;

            detection = detection_queue.front();
            detection_queue.pop();
        }

        //counting distance
        DistanceResult distance_result;
        distance_result.object_class = detection.object_class;
        distance_result.distance = calculate_distance(detection);

        //sending result
        if (distance_callback){
            distance_callback(distance_result);
        }
        
    }
}

void Orchestrator::speech_thread_func() {
    SpeechEngine speech_engine;

    while (running) {
        std::vector<float> audio_data;
        {
            std::unique_lock<std::mutex> lock(audio_mutex);
            audio_cova.wait(lock, [this]() { return !audio_queue.empty() || !running; });

            if (!running) break;

            audio_data = audio_queue.front();
            audio_queue.pop();
        }

        auto results = speech_engine.process_audio(audio_data);
        
        //sending results
        for (const auto& result : results) {
            if (speech_callback) {
                speech_callback(result);
            }
        }

    }
}

void Orchestrator::decision_thread_func() {
    //like a main function
    while (running){
        // ...

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
}

//callbacks
void Orchestrator::set_detection_callback(std::function<void(const DetectionResult&)> callback) {
    detection_callback = callback;
}

void Orchestrator::set_distance_callback(std::function<void(const DistanceResult&)> callback) {
    distance_callback = callback;
}

void Orchestrator::set_speech_callback(std::function<void(const SpeechResult&)> callback) {
    speech_callback = callback;
}