#include "rudra/deliver/queue.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <set>

#include <nlohmann/json.hpp>

#include "rudra/platform/hash.hpp"

#if defined(_WIN32)
#include <io.h>
#include <sys/locking.h>
#include <fcntl.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace rudra {
namespace fs = std::filesystem;
using pyjson::Dict;
using pyjson::List;
using pyjson::Value;

namespace {

// rudra/video.py add_arguments, the options a queue job may carry.
enum class Kind { Path, Str, Float, Int };
struct Opt {
    const char* name;
    Kind kind;
    std::vector<const char*> choices;
};
const std::vector<Opt>& video_options() {
    static const std::vector<Opt> o{
        {"output", Kind::Path, {}},
        {"checkpoint", Kind::Path, {}},
        {"format", Kind::Str, {"hdr10", "hlg", "prores422", "prores422hq", "prores4444"}},
        {"alpha-mode", Kind::Str, {"straight"}},
        {"device", Kind::Str, {}},
        {"input-transfer", Kind::Str, {"auto", "srgb", "rec709", "gamma22", "gamma24"}},
        {"input-primaries", Kind::Str, {"auto", "rec709", "rec2020"}},
        {"input-matrix", Kind::Str, {"auto", "bt709", "bt2020nc", "gbr"}},
        {"input-range", Kind::Str, {"auto", "full", "limited"}},
        {"audio", Kind::Str, {"copy", "aac", "none"}},
        {"peak-nits", Kind::Float, {}},
        {"min-nits", Kind::Float, {}},
        {"knee-nits", Kind::Float, {}},
        {"crf", Kind::Int, {}},
        {"preset", Kind::Str, {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}},
        {"tile-size", Kind::Int, {}},
        {"tile-overlap", Kind::Int, {}},
        {"shadow-smoothing", Kind::Float, {}},
        {"cut-threshold", Kind::Float, {}},
        {"work-dir", Kind::Path, {}},
    };
    return o;
}

// str(value) as Python prints a JSON value.
std::string py_str(const nlohmann::ordered_json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_null()) return "None";
    if (v.is_number_integer()) return std::to_string(v.get<std::int64_t>());
    if (v.is_number_unsigned()) return std::to_string(v.get<std::uint64_t>());
    if (v.is_number_float()) return pyjson::repr(v.get<double>());
    return v.dump();
}

bool parses_float(const std::string& s) {
    try {
        std::size_t used = 0;
        (void)std::stod(s, &used);
        return used == s.size();
    } catch (...) {
        return false;
    }
}
bool parses_int(const std::string& s) {
    if (s.empty()) return false;
    std::size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    if (i == s.size()) return false;
    return std::all_of(s.begin() + std::ptrdiff_t(i), s.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// argparse's matching: the exact name, else a unique prefix.
const Opt* match(const std::string& key) {
    const Opt* hit = nullptr;
    int prefix = 0;
    for (const auto& o : video_options()) {
        if (key == o.name) return &o;
        if (std::string(o.name).rfind(key, 0) == 0) { hit = &o; ++prefix; }
    }
    return prefix == 1 ? hit : nullptr;
}

Value to_py(const nlohmann::ordered_json& j) {
    if (j.is_null()) return nullptr;
    if (j.is_boolean()) return j.get<bool>();
    if (j.is_number_integer()) return j.get<std::int64_t>();
    if (j.is_number_unsigned()) return static_cast<std::int64_t>(j.get<std::uint64_t>());
    if (j.is_number_float()) return j.get<double>();
    if (j.is_string()) return j.get<std::string>();
    if (j.is_array()) {
        List l;
        for (const auto& e : j) l.push_back(to_py(e));
        return l;
    }
    Dict d;
    for (const auto& [k, v] : j.items()) d.emplace_back(k, to_py(v));
    return d;
}

Dict& as_dict(Value& v) { return *std::get<std::shared_ptr<Dict>>(v.v); }
List& as_list(Value& v) { return *std::get<std::shared_ptr<List>>(v.v); }
std::string as_str(const Value* v) {
    if (!v) return {};
    if (auto s = std::get_if<std::string>(&v->v)) return *s;
    return {};
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

fs::path with_suffix_json(const fs::path& p) { return fs::path(p.string() + ".json"); }
fs::path state_path_of(const fs::path& q) { return fs::path(q.string() + ".state.json"); }
fs::path lock_path_of(const fs::path& q) { return fs::path(q.string() + ".lock"); }

// save(): write a temporary file beside the target, flush it to disk, replace.
Result<void> save(const fs::path& path, const Value& data) {
    const std::string text = pyjson::dumps(data, 2);
    const fs::path tmp = path.parent_path() / (path.filename().string() + ".rudra-native.tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return make_error(ErrorCode::IoError, "The queue state could not be written.", tmp.string());
        f.write(text.data(), std::streamsize(text.size()));
        f.flush();
        if (!f) return make_error(ErrorCode::IoError, "The queue state could not be written.", tmp.string());
    }
#if !defined(_WIN32)
    if (int fd = ::open(tmp.c_str(), O_RDONLY); fd >= 0) { ::fsync(fd); ::close(fd); }
#endif
    std::error_code ec;
    fs::rename(tmp, path, ec);   // os.replace
    if (ec) {
        fs::remove(tmp, ec);
        return make_error(ErrorCode::IoError, "The queue state could not be replaced.", path.string());
    }
    return {};
}

Result<std::string> digest(const fs::path& p) { return sha256_file(p); }

// The lock: one byte of <queue>.lock, non-blocking, released on scope exit.
class QueueLock {
public:
    static Result<QueueLock> take(const fs::path& p) {
        QueueLock l;
#if defined(_WIN32)
        l.fd_ = _wopen(p.c_str(), _O_RDWR | _O_CREAT | _O_BINARY | _O_APPEND, 0666);
        if (l.fd_ < 0) return make_error(ErrorCode::IoError, "The queue lock could not be opened.", p.string());
        if (_lseek(l.fd_, 0, SEEK_END) == 0) (void)_write(l.fd_, "0", 1);
        _lseek(l.fd_, 0, SEEK_SET);
        if (_locking(l.fd_, _LK_NBLCK, 1) != 0) return make_error(ErrorCode::Busy, "This queue is already running");
#else
        l.fd_ = ::open(p.c_str(), O_RDWR | O_CREAT | O_APPEND, 0666);
        if (l.fd_ < 0) return make_error(ErrorCode::IoError, "The queue lock could not be opened.", p.string());
        if (::lseek(l.fd_, 0, SEEK_END) == 0) (void)!::write(l.fd_, "0", 1);
        if (::flock(l.fd_, LOCK_EX | LOCK_NB) != 0) return make_error(ErrorCode::Busy, "This queue is already running");
#endif
        l.held_ = true;
        return l;
    }
    QueueLock() = default;
    QueueLock(QueueLock&& o) noexcept : fd_(o.fd_), held_(o.held_) { o.fd_ = -1; o.held_ = false; }
    QueueLock& operator=(QueueLock&&) = delete;
    ~QueueLock() {
        if (fd_ < 0) return;
#if defined(_WIN32)
        if (held_) { _lseek(fd_, 0, SEEK_SET); _locking(fd_, _LK_UNLCK, 1); }
        _close(fd_);
#else
        if (held_) ::flock(fd_, LOCK_UN);
        ::close(fd_);
#endif
    }

private:
    int fd_ = -1;
    bool held_ = false;
};

}  // namespace

fs::path QueueJob::sidecar() const { return with_suffix_json(output); }

Result<std::vector<QueueJob>> load_queue(const fs::path& queue) {
    nlohmann::ordered_json spec;
    try {
        spec = nlohmann::ordered_json::parse(read_file(queue));
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The queue is not valid JSON.", e.what());
    }
    if (!spec.is_object() || !spec.contains("version") || spec["version"] != 1 || !spec.contains("jobs") ||
        !spec["jobs"].is_array() || spec["jobs"].empty())
        return make_error(ErrorCode::InvalidArgument, "Queue requires version 1 and a nonempty jobs list");
    const fs::path base = queue.parent_path();
    auto resolve = [&](const std::string& v) { return fs::weakly_canonical(fs::absolute(base / v)); };
    std::vector<QueueJob> jobs;
    for (const auto& job : spec["jobs"]) {
        nlohmann::ordered_json options = spec.contains("defaults") ? spec["defaults"] : nlohmann::ordered_json::object();
        if (job.contains("options"))
            for (const auto& [k, v] : job["options"].items()) options[k] = v;
        if (options.contains("input") || options.contains("output"))
            return make_error(ErrorCode::InvalidArgument, "Set input/output on the job, not in options");
        if (!job.contains("input") || !job.contains("output"))
            return make_error(ErrorCode::InvalidArgument, "Invalid video options in queue", "a job needs input and output");
        QueueJob q;
        q.input = resolve(py_str(job["input"]));
        q.output = resolve(py_str(job["output"]));
        bool have_checkpoint = false;
        for (const auto& [k, v] : options.items()) {
            std::string key = k;
            std::replace(key.begin(), key.end(), '_', '-');
            const Opt* o = match(key);
            const std::string value = py_str(v);
            const bool ok = o && (o->kind != Kind::Float || parses_float(value)) &&
                            (o->kind != Kind::Int || parses_int(value)) &&
                            (o->choices.empty() || std::find_if(o->choices.begin(), o->choices.end(),
                                                                [&](const char* c) { return value == c; }) != o->choices.end());
            if (!ok) return make_error(ErrorCode::InvalidArgument, "Invalid video options in queue", key);
            const std::string name = o->name;
            if (name == "output") q.output = resolve(value);
            else if (name == "checkpoint") { q.checkpoint = resolve(value); have_checkpoint = true; }
            else if (name == "work-dir") q.work_dir = resolve(value);
            q.options.emplace_back(name, value);
        }
        if (!have_checkpoint) return make_error(ErrorCode::InvalidArgument, "Invalid video options in queue", "checkpoint is required");
        jobs.push_back(std::move(q));
    }
    const fs::path absq = fs::weakly_canonical(fs::absolute(queue));
    std::set<fs::path> reserved{absq, state_path_of(absq), lock_path_of(absq)}, sources, targets;
    for (const auto& j : jobs) { sources.insert(j.input); sources.insert(j.checkpoint); }
    for (const auto& j : jobs)
        for (const auto& p : {j.output, j.sidecar()}) {
            if (reserved.count(p) || sources.count(p) || targets.count(p))
                return make_error(ErrorCode::InvalidArgument, "Queue path collision: " + p.string());
            targets.insert(p);
        }
    return jobs;
}

Result<int> run_queue(const fs::path& queue_in, bool retry_failed, const QueueRunner& runner, const QueueLog& log) {
    const fs::path queue = fs::weakly_canonical(fs::absolute(queue_in));
    const fs::path state_path = state_path_of(queue);
    auto lock = QueueLock::take(lock_path_of(queue));
    if (!lock) return lock.error();
    auto jobs = load_queue(queue);
    if (!jobs) return jobs.error();
    auto signature = digest(queue);
    if (!signature) return signature.error();

    Value state;
    if (fs::exists(state_path)) {
        try {
            state = to_py(nlohmann::ordered_json::parse(read_file(state_path)));
        } catch (const std::exception& e) {
            return make_error(ErrorCode::ParseError, "The queue state is not valid JSON.", e.what());
        }
        auto& sd = as_dict(state);
        if (as_str(pyjson::get(sd, "queue_sha256")) != *signature)
            return make_error(ErrorCode::InvalidArgument, "Queue changed; use a new queue filename for changed jobs");
        const Value* js = pyjson::get(sd, "jobs");
        if (!js || std::get<std::shared_ptr<List>>(js->v)->size() != jobs->size())
            return make_error(ErrorCode::InvalidArgument, "Queue state has an invalid job count");
    } else {
        List records;
        for (std::size_t i = 0; i < jobs->size(); ++i) records.push_back(Dict{{"status", "pending"}});
        state = Dict{{"version", 1}, {"queue_sha256", *signature}, {"jobs", std::move(records)}};
    }
    if (auto s = save(state_path, state); !s) return s.error();
    auto say = [&](const std::string& m) { if (log) log(m); };

    auto& records = as_list(*const_cast<Value*>(pyjson::get(as_dict(state), "jobs")));
    const std::size_t total = jobs->size();
    for (std::size_t index = 0; index < total; ++index) {
        const QueueJob& job = (*jobs)[index];
        Dict& record = as_dict(records[index]);
        if (as_str(pyjson::get(record, "status")) == "failed" && !retry_failed) continue;
        // Everything a job can refuse or fail with lands in the record, as the
        // Python's except clause does, and the queue moves on.
        auto attempt = [&]() -> Result<bool> {
            auto src = digest(job.input);
            if (!src) return make_error(ErrorCode::NotFound, "[Errno 2] No such file or directory: '" + job.input.string() + "'");
            auto ck = digest(job.checkpoint);
            if (!ck) return make_error(ErrorCode::NotFound, "[Errno 2] No such file or directory: '" + job.checkpoint.string() + "'");
            const Value identity = Dict{{"source", *src}, {"checkpoint", *ck}};
            if (const Value* prev = pyjson::get(record, "identity"); prev && pyjson::dumps(*prev) != pyjson::dumps(identity))
                return make_error(ErrorCode::InvalidArgument, "Source or checkpoint changed since this job started");
            pyjson::set(record, "identity", identity);
            if (as_str(pyjson::get(record, "status")) == "complete") {
                auto a = digest(job.output), b = digest(job.sidecar());
                const Value* art = pyjson::get(record, "artifacts");
                const Value now = List{a ? Value(*a) : Value(nullptr), b ? Value(*b) : Value(nullptr)};
                if (!art || !a || !b || pyjson::dumps(*art) != pyjson::dumps(now))
                    return make_error(ErrorCode::InvalidArgument, "Completed export changed; refusing to skip or overwrite it");
                say("Job " + std::to_string(index + 1) + "/" + std::to_string(total) + ": verified, skipped");
                return false;
            }
            if (fs::exists(job.output) || fs::exists(job.sidecar()))
                return make_error(ErrorCode::InvalidArgument,
                                  "Existing output needs review; refusing overwrite (including interrupted publication)");
            pyjson::set(record, "status", "running");
            pyjson::set(record, "progress", Dict{{"phase", "starting"}});
            pyjson::erase(record, "error");
            if (auto s = save(state_path, state); !s) return s.error();
            say("Job " + std::to_string(index + 1) + "/" + std::to_string(total) + ": " + job.input.filename().string());
            auto ran = runner(job, [&](Value v) {
                pyjson::set(record, "progress", std::move(v));
                (void)save(state_path, state);
            });
            if (!ran) return ran.error();
            bool passed = false;
            try {
                const auto rep = nlohmann::json::parse(read_file(job.sidecar()));
                passed = rep.contains("qc") && rep["qc"].is_object() && rep["qc"].contains("passed") &&
                         rep["qc"]["passed"].is_boolean() && rep["qc"]["passed"].get<bool>();
            } catch (...) {
                passed = false;
            }
            if (!passed) return make_error(ErrorCode::ParityError, "Export report does not confirm passing QC");
            auto src2 = digest(job.input), ck2 = digest(job.checkpoint);
            if (!src2 || !ck2 || *src2 != *src || *ck2 != *ck)
                return make_error(ErrorCode::ParityError, "Source or checkpoint changed during export");
            auto a = digest(job.output), b = digest(job.sidecar());
            if (!a || !b) return make_error(ErrorCode::IoError, "The export's files are missing after it finished");
            pyjson::set(record, "status", "complete");
            pyjson::set(record, "artifacts", List{*a, *b});
            pyjson::set(record, "progress", Dict{{"phase", "complete"}});
            return true;
        };
        auto r = attempt();
        if (!r) {
            pyjson::set(record, "status", "failed");
            pyjson::set(record, "error", r.error().message);
            say("Job " + std::to_string(index + 1) + " failed: " + r.error().message);
        } else if (!*r) {
            continue;   // verified and skipped: the Python `continue`s before saving
        }
        if (auto s = save(state_path, state); !s) return s.error();
    }
    for (auto& rec : records)
        if (as_str(pyjson::get(as_dict(rec), "status")) != "complete") return 1;
    return 0;
}

std::string queue_status(const fs::path& queue) {
    const fs::path p = state_path_of(fs::weakly_canonical(fs::absolute(queue)));
    return fs::exists(p) ? read_file(p) : "Queue has not run yet.";
}

}  // namespace rudra
