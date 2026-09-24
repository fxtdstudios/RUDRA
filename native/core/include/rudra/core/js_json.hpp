#pragma once
// JSON.stringify(value, null, 2), for the texts the page copies (Phase 3
// step 11): keys in insertion order, numbers as String(x) (NaN and the
// infinities as null), strings escaped as JSON.stringify escapes them, two
// spaces a level, an empty array or object as [] and {}.

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rudra {

class JsValue {
public:
    enum class Kind { Null, Bool, Number, String, Array, Object };
    JsValue() = default;                                   // null
    JsValue(std::nullptr_t) {}
    JsValue(bool b) : kind_(Kind::Bool), b_(b) {}
    JsValue(double d) : kind_(Kind::Number), d_(d) {}
    JsValue(int i) : kind_(Kind::Number), d_(double(i)) {}
    JsValue(const char* s) : kind_(Kind::String), s_(s) {}
    JsValue(std::string s) : kind_(Kind::String), s_(std::move(s)) {}
    template <class T>
    JsValue(const std::optional<T>& o) {
        if (o) *this = JsValue(*o);
    }
    static JsValue array(std::vector<JsValue> items = {});
    static JsValue object();
    JsValue& add(std::string key, JsValue v);             // object: appended in order
    JsValue& push(JsValue v);                              // array
    Kind kind() const { return kind_; }
    std::string stringify(int indent = 2) const;

private:
    void write(std::string& out, int indent, int depth) const;
    Kind kind_ = Kind::Null;
    bool b_ = false;
    double d_ = 0.0;
    std::string s_;
    std::vector<JsValue> items_;
    std::vector<std::string> keys_;
};

// A string as JSON.stringify quotes it.
std::string js_quote(const std::string& utf8);

}  // namespace rudra
