#pragma once
// Running a program (Phase 4): ffmpeg and ffprobe, as the Python runs them
// with subprocess. The arguments are passed as a vector, never through a
// shell. Two shapes: run() to completion with stdout and stderr captured,
// and Process to stream a large stdout (decoded frames) while stderr goes to
// a file, as the Python's Popen(stdout=PIPE, stderr=log) does.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

// shutil.which: the program on PATH (with .exe on Windows), or nullopt.
std::optional<std::filesystem::path> find_executable(const std::string& name);

struct ProcessOutput {
    int exit_code = 0;
    std::string out, err;
};

// Runs argv[0] with the rest as its arguments and waits. An error only when
// it could not be started; a non-zero exit is in exit_code.
Result<ProcessOutput> run_process(const std::vector<std::string>& argv);

// video.executable: shutil.which or "<name> is required on PATH".
Result<std::filesystem::path> require_executable(const std::string& name);
// video.run: stdout, or "<program> failed: <last 4000 bytes of stderr>".
Result<std::string> run_tool(const std::vector<std::string>& argv);

// A running program whose stdout is read as it comes.
class Process {
public:
    ~Process();
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    // stderr goes to `stderr_file` (truncated), or is discarded when empty.
    static Result<std::unique_ptr<Process>> start(const std::vector<std::string>& argv,
                                                  const std::filesystem::path& stderr_file = {});
    // Reads up to out.size() bytes, looping until the buffer is full or the
    // stream ends; returns how many were read (less only at the end).
    std::size_t read(std::span<std::uint8_t> out);
    // Waits for the program to end; its exit code.
    int wait();
    // Ends it now (the Python's kill on an error path).
    void kill();
    bool running() const;

private:
    Process() = default;
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace rudra
