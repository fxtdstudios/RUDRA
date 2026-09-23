#pragma once
// JSON exactly as Python's json.dumps writes it, so a sidecar the native app
// writes is byte-identical to the one the Python writes (ADR-008): insertion
// order kept, floats as repr() (shortest round trip, ".0" on integral values,
// exponent from 1e16 and below 1e-4), NaN and Infinity as Python emits them.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace rudra::pyjson {

struct Value;
using List = std::vector<Value>;
using Dict = std::vector<std::pair<std::string, Value>>;   // insertion order, like a Python dict

struct Value {
    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string,
                 std::shared_ptr<List>, std::shared_ptr<Dict>> v;
    Value() : v(nullptr) {}
    Value(std::nullptr_t) : v(nullptr) {}                                          // NOLINT
    Value(bool b) : v(b) {}                                                        // NOLINT
    Value(int i) : v(std::int64_t(i)) {}                                           // NOLINT
    Value(std::int64_t i) : v(i) {}                                                // NOLINT
    Value(double d) : v(d) {}                                                      // NOLINT
    Value(const char* s) : v(std::string(s)) {}                                    // NOLINT
    Value(std::string s) : v(std::move(s)) {}                                      // NOLINT
    Value(List l) : v(std::make_shared<List>(std::move(l))) {}                     // NOLINT
    Value(Dict d) : v(std::make_shared<Dict>(std::move(d))) {}                     // NOLINT
};

// repr(float) as Python 3 prints it.
std::string repr(double d);

// json.dumps(value, indent=indent, sort_keys=sort_keys). indent < 0 means
// None: one line with ", " and ": " separators.
std::string dumps(const Value& value, int indent = -1, bool sort_keys = false);

}  // namespace rudra::pyjson
