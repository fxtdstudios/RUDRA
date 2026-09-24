#include "rudra/core/js_json.hpp"

#include <cmath>
#include <cstdio>

#include "rudra/core/js_format.hpp"

namespace rudra {

JsValue JsValue::array(std::vector<JsValue> items) {
    JsValue v;
    v.kind_ = Kind::Array;
    v.items_ = std::move(items);
    return v;
}

JsValue JsValue::object() {
    JsValue v;
    v.kind_ = Kind::Object;
    return v;
}

JsValue& JsValue::add(std::string key, JsValue v) {
    keys_.push_back(std::move(key));
    items_.push_back(std::move(v));
    return *this;
}

JsValue& JsValue::push(JsValue v) {
    items_.push_back(std::move(v));
    return *this;
}

std::string js_quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(c));
                    out += buf;
                } else {
                    out += char(c);
                }
        }
    }
    return out + "\"";
}

void JsValue::write(std::string& out, int indent, int depth) const {
    const std::string pad(std::size_t(indent * (depth + 1)), ' '), end(std::size_t(indent * depth), ' ');
    switch (kind_) {
        case Kind::Null: out += "null"; return;
        case Kind::Bool: out += b_ ? "true" : "false"; return;
        case Kind::Number: out += std::isfinite(d_) ? js_number(d_) : "null"; return;
        case Kind::String: out += js_quote(s_); return;
        case Kind::Array:
        case Kind::Object: {
            const bool obj = kind_ == Kind::Object;
            if (items_.empty()) {
                out += obj ? "{}" : "[]";
                return;
            }
            out += obj ? '{' : '[';
            for (std::size_t i = 0; i < items_.size(); ++i) {
                out += i ? ",\n" : "\n";
                out += pad;
                if (obj) out += js_quote(keys_[i]) + ": ";
                items_[i].write(out, indent, depth + 1);
            }
            out += "\n" + end + (obj ? '}' : ']');
            return;
        }
    }
}

std::string JsValue::stringify(int indent) const {
    std::string out;
    write(out, indent, 0);
    return out;
}

}  // namespace rudra
