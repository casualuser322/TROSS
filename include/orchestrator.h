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
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>


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

struct StereoFrame {
    cv::Mat left;
    cv::Mat right;
    uint64_t timestamp;
};

struct ModelConfig {
    std::string model_path;
    std::string model_type; 
    std::vector<std::string> class_names;
    float confidence_threshold = 0.5f;
    cv::Size input_size        = cv::Size(416, 416);
    float scale_factor         = 1.0 / 255.0;
    cv::Scalar mean            = cv::Scalar(0, 0, 0);
    bool swap_rb               = true;
    
    std::string calibration_file = "calibration.json";
    float baseline     = 0.12f;  
    float focal_length = 1250.0f;
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

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
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
    virtual bool initialize(const ModelConfig& config) = 0;
};

struct IStereoEngine : IEngine {
    virtual std::vector<DistanceResult> process_stereo_frame(
        const cv::Mat& left, const cv::Mat& right) = 0;
    virtual bool initialize(const ModelConfig& config) = 0;
    virtual float calculate_distance(const DetectionResult& det, const cv::Mat& left_frame) = 0;
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
    void push_stereo_frame(const cv::Mat& left, const cv::Mat& right);
    void push_audio(const std::vector<float>& audio_data);

    void set_detection_callback(std::function<void(const DetectionResult&)> callback);
    void set_distance_callback(std::function<void(const DistanceResult&)> callback);
    void set_speech_callback(std::function<void(const SpeechResult&)> callback);

    bool initialize_cv_model(const ModelConfig& config);
    bool initialize_stereo_model(const ModelConfig& config);
    bool initialize_speech_model(const std::string& model_path);

    void enable_stereo_mode(bool enable) { stereo_mode = enable; }
    void enable_mono_depth(bool enable) { mono_depth_mode = enable; }

private:
    void cv_thread_func();
    void stereo_thread_func();
    void distance_thread_func();
    void decision_thread_func();
    void speech_thread_func();
    
    void process_mono_depth(const cv::Mat& frame);
    void process_stereo_depth(const cv::Mat& left, const cv::Mat& right);

    ThreadSafeQueue<cv::Mat>            frame_queue;
    ThreadSafeQueue<StereoFrame>        stereo_queue;
    ThreadSafeQueue<std::vector<float>> audio_queue;
    ThreadSafeQueue<DetectionResult>    detection_queue;
    ThreadSafeQueue<DistanceResult>     distance_queue;
    ThreadSafeQueue<SpeechResult>       speech_queue;

    std::unique_ptr<ICVEngine>     cv_engine;
    std::unique_ptr<IStereoEngine> stereo_engine;
    std::unique_ptr<ISpeechEngine> speech_engine;

    std::thread cv_thread;
    std::thread stereo_thread;
    std::thread distance_thread;
    std::thread speech_thread;
    std::thread decision_thread;

    std::atomic<bool> running{false};
    std::atomic<bool> stereo_mode{false};
    std::atomic<bool> mono_depth_mode{false};

    std::function<void(const DetectionResult&)> detection_callback;
    std::function<void(const DistanceResult&)>  distance_callback;
    std::function<void(const SpeechResult&)>    speech_callback;

    ModelConfig cv_model_config;
    ModelConfig stereo_model_config;
};