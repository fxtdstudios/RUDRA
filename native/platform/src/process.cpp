#include "rudra/platform/process.hpp"

#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace rudra {
namespace fs = std::filesystem;

std::optional<fs::path> find_executable(const std::string& name) {
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos) {
        std::error_code ec;
        if (fs::is_regular_file(name, ec)) return fs::path(name);
        return std::nullopt;
    }
    const char* path = std::getenv("PATH");
    if (!path) return std::nullopt;
#ifdef _WIN32
    constexpr char sep = ';';
    const std::vector<std::string> exts = {".exe", ".com", ".bat", ""};
#else
    constexpr char sep = ':';
    const std::vector<std::string> exts = {""};
#endif
    std::string p(path);
    std::size_t at = 0;
    while (at <= p.size()) {
        const std::size_t end = std::min(p.find(sep, at), p.size());
        const std::string dir = p.substr(at, end - at);
        at = end + 1;
        if (dir.empty()) continue;
        for (const auto& ext : exts) {
            const fs::path cand = fs::path(dir) / (name + ext);
            std::error_code ec;
            if (!fs::is_regular_file(cand, ec)) continue;
#ifndef _WIN32
            if (::access(cand.c_str(), X_OK) != 0) continue;
#endif
            return cand;
        }
    }
    return std::nullopt;
}

#ifdef _WIN32

namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(std::size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}

// One argument quoted as CommandLineToArgvW / the MSVC runtime parse it.
void append_quoted(std::wstring& cmd, const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        cmd += arg;
        return;
    }
    cmd += L'"';
    for (auto it = arg.begin();; ++it) {
        std::size_t backslashes = 0;
        while (it != arg.end() && *it == L'\\') {
            ++it;
            ++backslashes;
        }
        if (it == arg.end()) {
            cmd.append(backslashes * 2, L'\\');
            break;
        }
        if (*it == L'"') {
            cmd.append(backslashes * 2 + 1, L'\\');
            cmd += L'"';
        } else {
            cmd.append(backslashes, L'\\');
            cmd += *it;
        }
    }
    cmd += L'"';
}

std::wstring command_line(const std::vector<std::string>& argv) {
    std::wstring cmd;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i) cmd += L' ';
        append_quoted(cmd, widen(argv[i]));
    }
    return cmd;
}

std::string drain(HANDLE h) {
    std::string out;
    char buf[65536];
    DWORD n = 0;
    while (ReadFile(h, buf, sizeof buf, &n, nullptr) && n > 0) out.append(buf, n);
    return out;
}

}  // namespace

struct Process::Impl {
    PROCESS_INFORMATION pi{};
    HANDLE out = nullptr;
    bool done = false;
    int code = -1;
};

Result<std::unique_ptr<Process>> Process::start(const std::vector<std::string>& argv, const fs::path& stderr_file) {
    if (argv.empty()) return make_error(ErrorCode::InvalidArgument, "No program to run.");
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 20)) return make_error(ErrorCode::IoError, "Could not create a pipe.");
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE err = INVALID_HANDLE_VALUE;
    if (!stderr_file.empty())
        err = CreateFileW(stderr_file.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    else
        err = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    HANDLE in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in;
    si.hStdOutput = wr;
    si.hStdError = err;
    std::wstring cmd = command_line(argv);
    auto impl = std::make_unique<Impl>();
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si,
                                   &impl->pi);
    CloseHandle(wr);
    if (err != INVALID_HANDLE_VALUE) CloseHandle(err);
    if (in != INVALID_HANDLE_VALUE) CloseHandle(in);
    if (!ok) {
        CloseHandle(rd);
        return make_error(ErrorCode::NotFound, "Could not start the program.", argv[0]);
    }
    impl->out = rd;
    std::unique_ptr<Process> p(new Process());
    p->d_ = std::move(impl);
    return p;
}

std::size_t Process::read(std::span<std::uint8_t> out) {
    std::size_t got = 0;
    while (got < out.size()) {
        DWORD n = 0;
        if (!ReadFile(d_->out, out.data() + got, DWORD(std::min<std::size_t>(out.size() - got, 1u << 30)), &n, nullptr) ||
            n == 0)
            break;
        got += n;
    }
    return got;
}

int Process::wait() {
    if (!d_->done) {
        WaitForSingleObject(d_->pi.hProcess, INFINITE);
        DWORD code = 0;
        GetExitCodeProcess(d_->pi.hProcess, &code);
        d_->code = int(code);
        d_->done = true;
    }
    return d_->code;
}

void Process::kill() {
    if (running()) TerminateProcess(d_->pi.hProcess, 1);
}

bool Process::running() const {
    return !d_->done && WaitForSingleObject(d_->pi.hProcess, 0) == WAIT_TIMEOUT;
}

Process::~Process() {
    if (!d_) return;
    if (running()) kill();
    wait();
    if (d_->out) CloseHandle(d_->out);
    CloseHandle(d_->pi.hThread);
    CloseHandle(d_->pi.hProcess);
}

Result<ProcessOutput> run_process(const std::vector<std::string>& argv) {
    if (argv.empty()) return make_error(ErrorCode::InvalidArgument, "No program to run.");
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE ord = nullptr, owr = nullptr, erd = nullptr, ewr = nullptr;
    if (!CreatePipe(&ord, &owr, &sa, 0) || !CreatePipe(&erd, &ewr, &sa, 0))
        return make_error(ErrorCode::IoError, "Could not create a pipe.");
    SetHandleInformation(ord, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(erd, HANDLE_FLAG_INHERIT, 0);
    HANDLE in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in;
    si.hStdOutput = owr;
    si.hStdError = ewr;
    std::wstring cmd = command_line(argv);
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(owr);
    CloseHandle(ewr);
    if (in != INVALID_HANDLE_VALUE) CloseHandle(in);
    if (!ok) {
        CloseHandle(ord);
        CloseHandle(erd);
        return make_error(ErrorCode::NotFound, "Could not start the program.", argv[0]);
    }
    ProcessOutput r;
    std::thread t([&] { r.err = drain(erd); });
    r.out = drain(ord);
    t.join();
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    r.exit_code = int(code);
    CloseHandle(ord);
    CloseHandle(erd);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return r;
}

#else  // POSIX

namespace {

std::vector<char*> c_argv(const std::vector<std::string>& argv) {
    std::vector<char*> v;
    for (const auto& a : argv) v.push_back(const_cast<char*>(a.c_str()));
    v.push_back(nullptr);
    return v;
}

std::string drain(int fd) {
    std::string out;
    char buf[65536];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof buf);
        if (n > 0) out.append(buf, std::size_t(n));
        else if (n < 0 && errno == EINTR) continue;
        else break;
    }
    return out;
}

int exit_code_of(int status) {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

}  // namespace

struct Process::Impl {
    pid_t pid = -1;
    int out = -1;
    bool done = false;
    int code = -1;
};

Result<std::unique_ptr<Process>> Process::start(const std::vector<std::string>& argv, const fs::path& stderr_file) {
    if (argv.empty()) return make_error(ErrorCode::InvalidArgument, "No program to run.");
    int fds[2];
    if (::pipe(fds) != 0) return make_error(ErrorCode::IoError, "Could not create a pipe.");
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    posix_spawn_file_actions_addclose(&fa, fds[1]);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    const std::string err = stderr_file.empty() ? std::string("/dev/null") : stderr_file.string();
    posix_spawn_file_actions_addopen(&fa, 2, err.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    auto args = c_argv(argv);
    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, argv[0].c_str(), &fa, nullptr, args.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(fds[1]);
    if (rc != 0) {
        ::close(fds[0]);
        return make_error(ErrorCode::NotFound, "Could not start the program.", argv[0] + ": " + std::strerror(rc));
    }
    std::unique_ptr<Process> p(new Process());
    p->d_ = std::make_unique<Impl>();
    p->d_->pid = pid;
    p->d_->out = fds[0];
    return p;
}

std::size_t Process::read(std::span<std::uint8_t> out) {
    std::size_t got = 0;
    while (got < out.size()) {
        const ssize_t n = ::read(d_->out, out.data() + got, out.size() - got);
        if (n > 0) got += std::size_t(n);
        else if (n < 0 && errno == EINTR) continue;
        else break;
    }
    return got;
}

int Process::wait() {
    if (!d_->done) {
        int status = 0;
        while (::waitpid(d_->pid, &status, 0) < 0 && errno == EINTR) {
        }
        d_->code = exit_code_of(status);
        d_->done = true;
    }
    return d_->code;
}

void Process::kill() {
    if (!d_->done) ::kill(d_->pid, SIGKILL);
}

bool Process::running() const {
    if (d_->done) return false;
    int status = 0;
    const pid_t r = ::waitpid(d_->pid, &status, WNOHANG);
    if (r == d_->pid) {
        d_->code = exit_code_of(status);
        d_->done = true;
        return false;
    }
    return true;
}

Process::~Process() {
    if (!d_) return;
    if (d_->out >= 0) ::close(d_->out);   // a writer blocked on a full pipe gets SIGPIPE
    if (!d_->done) {
        kill();
        wait();
    }
}

Result<ProcessOutput> run_process(const std::vector<std::string>& argv) {
    if (argv.empty()) return make_error(ErrorCode::InvalidArgument, "No program to run.");
    int o[2], e[2];
    if (::pipe(o) != 0) return make_error(ErrorCode::IoError, "Could not create a pipe.");
    if (::pipe(e) != 0) {
        ::close(o[0]);
        ::close(o[1]);
        return make_error(ErrorCode::IoError, "Could not create a pipe.");
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addclose(&fa, o[0]);
    posix_spawn_file_actions_addclose(&fa, e[0]);
    posix_spawn_file_actions_adddup2(&fa, o[1], 1);
    posix_spawn_file_actions_adddup2(&fa, e[1], 2);
    posix_spawn_file_actions_addclose(&fa, o[1]);
    posix_spawn_file_actions_addclose(&fa, e[1]);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    auto args = c_argv(argv);
    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, argv[0].c_str(), &fa, nullptr, args.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(o[1]);
    ::close(e[1]);
    if (rc != 0) {
        ::close(o[0]);
        ::close(e[0]);
        return make_error(ErrorCode::NotFound, "Could not start the program.", argv[0] + ": " + std::strerror(rc));
    }
    ProcessOutput r;
    std::thread t([&] { r.err = drain(e[0]); });
    r.out = drain(o[0]);
    t.join();
    ::close(o[0]);
    ::close(e[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    r.exit_code = exit_code_of(status);
    return r;
}

#endif

Result<std::filesystem::path> require_executable(const std::string& name) {
    if (auto found = find_executable(name)) return *found;
    return make_error(ErrorCode::NotFound, name + " is required on PATH");
}

Result<std::string> run_tool(const std::vector<std::string>& argv) {
    auto r = run_process(argv);
    if (!r) return r.error();
    if (r->exit_code != 0) {
        const std::string& err = r->err;
        const std::string tail = err.size() > 4000 ? err.substr(err.size() - 4000) : err;
        return make_error(ErrorCode::IoError, std::filesystem::path(argv.front()).filename().string() + " failed: " + tail);
    }
    return std::move(r->out);
}

}  // namespace rudra
