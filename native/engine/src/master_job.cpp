#include "rudra/engine/master_job.hpp"

namespace rudra {

MasterJob::MasterJob(ModelConstants model, MasterRequest request, std::vector<std::filesystem::path> targets,
                     PrepareMasterFrame prepare)
    : model_(model), request_(std::move(request)), targets_(std::move(targets)), prepare_(std::move(prepare)) {}

MasterJob::~MasterJob() {
    cancel_ = true;
    if (thread_.joinable()) thread_.join();
}

void MasterJob::start(std::function<void(std::size_t)> on_begin, std::function<void(const MasterProgress&)> on_frame,
                      std::function<void(const MasterOutcome&)> on_done) {
    if (running_ || thread_.joinable()) return;
    running_ = true;
    thread_ = std::thread([this, on_begin = std::move(on_begin), on_frame = std::move(on_frame),
                           on_done = std::move(on_done)] {
        MasterOutcome out;
        out.total = targets_.size();
        for (std::size_t i = 0; i < targets_.size(); ++i) {
            if (cancel_) {
                out.cancelled = true;
                break;
            }
            if (on_begin) on_begin(i);
            auto frame = prepare_(i);
            if (!frame) {
                out.error = frame.error();
                break;
            }
            auto r = publish_master(frame->sdr, frame->source_bits, frame->fields, frame->scalars, model_, request_,
                                    targets_[i]);
            if (!r) {
                out.error = r.error();
                break;
            }
            ++out.completed;
            if (on_frame) on_frame(MasterProgress{out.completed, out.total, *r});
        }
        running_ = false;
        if (on_done) on_done(out);
    });
}

}  // namespace rudra
