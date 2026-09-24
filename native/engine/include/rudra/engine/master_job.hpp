#pragma once
// The master as a background job (Phase 3 step 9): the page's master()
// loop, off the UI thread. For each target of the render plan, in order: the
// frame is prepared (decoded and through the network, by the caller's
// function, which shares the app's backend), written and published beside
// its sidecar (deliver/master.hpp publish_master), and reported. Cancel stops
// it after the frame in hand; a failure stops it and says after how many.

#include <atomic>
#include <filesystem>
#include <optional>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct MasterFrame {
    SdrImage sdr;
    int source_bits = 8;
    Fields fields;
    FrameScalars scalars;
};
// Frame i of the render: decode it and run the network.
using PrepareMasterFrame = std::function<Result<MasterFrame>(std::size_t index)>;

struct MasterProgress {
    std::size_t done = 0, total = 0;   // frames finished, frames in the render
    MasterResult last;                 // the one just saved
};

struct MasterOutcome {
    std::size_t completed = 0, total = 0;
    bool cancelled = false;
    std::optional<Error> error;        // what stopped it, if anything did
};

class MasterJob {
public:
    MasterJob(ModelConstants model, MasterRequest request, std::vector<std::filesystem::path> targets,
              PrepareMasterFrame prepare);
    ~MasterJob();   // cancels and waits
    MasterJob(const MasterJob&) = delete;
    MasterJob& operator=(const MasterJob&) = delete;

    // Callbacks run on the job's thread: hand them to the UI thread yourself.
    void start(std::function<void(std::size_t index)> on_begin, std::function<void(const MasterProgress&)> on_frame,
               std::function<void(const MasterOutcome&)> on_done);
    void cancel() { cancel_ = true; }
    bool running() const { return running_; }
    const std::vector<std::filesystem::path>& targets() const { return targets_; }

private:
    ModelConstants model_;
    MasterRequest request_;
    std::vector<std::filesystem::path> targets_;
    PrepareMasterFrame prepare_;
    std::atomic<bool> cancel_{false}, running_{false};
    std::thread thread_;
};

}  // namespace rudra
