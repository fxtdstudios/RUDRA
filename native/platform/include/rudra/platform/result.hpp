#pragma once
// Result<T>: a value or a typed, user-presentable error. No exceptions cross a
// module boundary (NATIVE_ARCHITECTURE.md 5.8). C++20 has no std::expected, so
// this is the small subset the codebase needs.

#include <cassert>
#include <string>
#include <utility>
#include <variant>

namespace rudra {

enum class ErrorCode {
    InvalidArgument,
    NotFound,
    IoError,
    ParseError,
    Unsupported,        // refused on purpose: the input is outside what RUDRA handles
    ContractMismatch,   // a model package or file format this build does not understand
    IntegrityError,     // a hash did not match
    BackendError,       // an inference runtime failed
    ParityError,        // a result disagreed with its golden beyond tolerance
};

const char* to_string(ErrorCode code) noexcept;

struct Error {
    ErrorCode code;
    std::string message;   // one sentence a user can read
    std::string detail;    // what an engineer needs
};

inline Error make_error(ErrorCode code, std::string message, std::string detail = {}) {
    return Error{code, std::move(message), std::move(detail)};
}

template <class T>
class [[nodiscard]] Result {
public:
    Result(T value) : v_(std::move(value)) {}          // NOLINT(google-explicit-constructor)
    Result(Error error) : v_(std::move(error)) {}      // NOLINT(google-explicit-constructor)

    bool ok() const noexcept { return v_.index() == 0; }
    explicit operator bool() const noexcept { return ok(); }

    T& value() & { assert(ok()); return std::get<0>(v_); }
    const T& value() const& { assert(ok()); return std::get<0>(v_); }
    T&& value() && { assert(ok()); return std::get<0>(std::move(v_)); }
    const Error& error() const { assert(!ok()); return std::get<1>(v_); }

    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }
    T& operator*() & { return value(); }
    const T& operator*() const& { return value(); }

private:
    std::variant<T, Error> v_;
};

template <>
class [[nodiscard]] Result<void> {
public:
    Result() = default;
    Result(Error error) : error_(std::move(error)), ok_(false) {}   // NOLINT(google-explicit-constructor)
    bool ok() const noexcept { return ok_; }
    explicit operator bool() const noexcept { return ok_; }
    const Error& error() const { assert(!ok_); return error_; }

private:
    Error error_{};
    bool ok_ = true;
};

}  // namespace rudra
