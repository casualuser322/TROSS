#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


//modules for intermodule interaction
struct DetectionResult {
    std::string object_class;
    float confidence;
    float x, y, width, height;
};

struct DistanceResult {
    std::string object_class;
    float x, y, distance;
};

struct SpeechResult {
    std::string command;
    std::string object_name;
    float confidence;
};


//Orchestrator
class Orchestrator {
public:
    Orchestrator();
    ~Orchestrator();

    void start();
    void stop();

    //module interaction
    void push_frame(const cv::Mat& frame);
    void push_audio(const std::vector<float>& audio_data);

    //Callbacks for results
    void set_detection_callback(std::function<void(const DetectionResult&)> callback);
    void set_distance_callback(std::function<void(const DistanceResult&)> callback);
    void set_speech_callback(std::function<void(const SpeechResult&)> callback);

private:
    void cv_thread_func();
    void distance_thread_func();
    void decision_thread_func(); //everything is correct here
    void speech_thread_func();

    //Q
    std::queue<cv::Mat> frame_queue;
    std::queue<std::vector<float>> audio_queue;
    std::queue<DetectionResult>    detection_queue;
    std::queue<DistanceResult>     distance_result;
    std::queue<SpeechResult>       speech_queue;

    //Sync
    std::mutex frame_mutex, audio_mutex, detection_mutex, distance_mutex, speech_mutex;
    std::condition_variable frame_cova, audio_cova, detection_cova, distance_cova, speech_cova;

    //Threads
    std::thread cv_thread;
    std::thread detection_thread;
    std::thread distance_thread;
    std::thread speech_thread;

    //Flags
    std::atomic<bool> running{false};
    
    //Callbacks
    std::function<void(const DetectionResult&)> detection_callback;
    std::function<void(const DistanceResult&)>  distance_callback;
    std::function<void(const SpeechResult&)>    speech_callback;
};
