#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <memory>
#include <queue>
#include <thread>
#include <vector>
#include <string>

#include <opencv2/core.hpp>

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

template<typename T>
class ThreadSafeQueue {
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    T wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

struct IEngine {
    virtual ~IEngine() = default;
};

struct ICVEngine : IEngine {
    virtual std::vector<DetectionResult> process_frame(const cv::Mat&) = 0;
};

struct ISpeechEngine : IEngine {
    virtual std::vector<SpeechResult> process_audio(const std::vector<float>&) = 0;
};

class Orchestrator {
public:
    Orchestrator();
    ~Orchestrator();

    void start();
    void stop();

    void push_frame(const cv::Mat& frame);
    void push_audio(const std::vector<float>& audio_data);

    void set_detection_callback(std::function<void(const DetectionResult&)> callback);
    void set_distance_callback(std::function<void(const DistanceResult&)> callback);
    void set_speech_callback(std::function<void(const SpeechResult&)> callback);

private:
    void cv_thread_func();
    void distance_thread_func();
    void decision_thread_func();
    void speech_thread_func();

    ThreadSafeQueue<cv::Mat> frame_queue;
    ThreadSafeQueue<std::vector<float>> audio_queue;
    ThreadSafeQueue<DetectionResult> detection_queue;
    ThreadSafeQueue<DistanceResult> distance_queue;
    ThreadSafeQueue<SpeechResult> speech_queue;

    std::thread cv_thread;
    std::thread detection_thread;
    std::thread distance_thread;
    std::thread speech_thread;

    std::atomic<bool> running{false};

    std::function<void(const DetectionResult&)> detection_callback;
    std::function<void(const DistanceResult&)> distance_callback;
    std::function<void(const SpeechResult&)> speech_callback;
};