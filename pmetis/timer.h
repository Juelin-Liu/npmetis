#pragma once
#include <chrono>

namespace pmetis {

class Timer {
public:
    // Starts the timer
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    // Stops the timer
    void end() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    // Calculates and returns the duration between start and end in nanoseconds
    double nanosec() const {
        auto elapsed = end_time - start_time;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    }

    // double sec() const {
    //     auto elapsed = end_time - start_time;
    //     return std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    // }

    // Resets the timer (sets start and end times to the current time)
    void reset() {
        start_time = end_time = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};

} // namespace pmetis