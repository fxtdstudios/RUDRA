#include "rudra/platform/pyjson.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace rudra::pyjson {
namespace {

void escape(std::string& out, const std::string& s) {
    out += '"';
    auto hex = [&](unsigned cp) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "\\u%04x", cp);
        out += buf;
    };
    for (std::size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            default: break;
        }
        if (c < 0x20) { hex(c); continue; }
        if (c < 0x80) { out += char(c); continue; }
        // ensure_ascii: decode UTF-8 to a code point, \uXXXX (surrogate pair above the BMP).
        unsigned cp = 0;
        int extra = 0;
        if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
        else { hex(0xFFFD); continue; }
        for (int k = 0; k < extra && i + 1 < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[++i]) & 0x3F);
        if (cp >= 0x10000) {
            cp -= 0x10000;
            hex(0xD800 + (cp >> 10));
            hex(0xDC00 + (cp & 0x3FF));
        } else {
            hex(cp);
        }
    }
    out += '"';
}

void write(std::string& out, const Value& v, int indent, bool sort_keys, int level);

void newline(std::string& out, int indent, int level) {
    out += '\n';
    out.append(std::size_t(indent) * std::size_t(level), ' ');
}

void write(std::string& out, const Value& v, int indent, bool sort_keys, int level) {
    if (std::holds_alternative<std::nullptr_t>(v.v)) { out += "null"; return; }
    if (auto b = std::get_if<bool>(&v.v)) { out += *b ? "true" : "false"; return; }
    if (auto i = std::get_if<std::int64_t>(&v.v)) { out += std::to_string(*i); return; }
    if (auto d = std::get_if<double>(&v.v)) {
        if (std::isnan(*d)) out += "NaN";
        else if (std::isinf(*d)) out += *d > 0 ? "Infinity" : "-Infinity";
        else out += repr(*d);
        return;
    }
    if (auto s = std::get_if<std::string>(&v.v)) { escape(out, *s); return; }
    if (auto l = std::get_if<std::shared_ptr<List>>(&v.v)) {
        if ((*l)->empty()) { out += "[]"; return; }
        out += '[';
        bool first = true;
        for (const auto& e : **l) {
            if (!first) out += indent >= 0 ? "," : ", ";
            first = false;
            if (indent >= 0) newline(out, indent, level + 1);
            write(out, e, indent, sort_keys, level + 1);
        }
        if (indent >= 0) newline(out, indent, level);
        out += ']';
        return;
    }
    const auto& d = *std::get<std::shared_ptr<Dict>>(v.v);
    if (d.empty()) { out += "{}"; return; }
    std::vector<const std::pair<std::string, Value>*> items;
    for (const auto& kv : d) items.push_back(&kv);
    if (sort_keys) std::stable_sort(items.begin(), items.end(), [](auto a, auto b) { return a->first < b->first; });
    out += '{';
    bool first = true;
    for (const auto* kv : items) {
        if (!first) out += indent >= 0 ? "," : ", ";
        first = false;
        if (indent >= 0) newline(out, indent, level + 1);
        escape(out, kv->first);
        out += ": ";
        write(out, kv->second, indent, sort_keys, level + 1);
    }
    if (indent >= 0) newline(out, indent, level);
    out += '}';
}

}  // namespace

std::string repr(double d) {
    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
    if (d == 0.0) return std::signbit(d) ? "-0.0" : "0.0";
    // Shortest round-trip digits, then Python's layout rule ('r' format):
    // positional when -4 <= exponent < 16, scientific otherwise.
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
    std::string sci(buf, res.ptr);
    std::string sign;
    if (sci[0] == '-') { sign = "-"; sci.erase(0, 1); }
    const auto e = sci.find('e');
    std::string digits = sci.substr(0, e);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
    const int exp = std::atoi(sci.c_str() + e + 1);
    const int n = int(digits.size());
    std::string out;
    if (exp >= -4 && exp < 16) {
        if (exp >= 0) {
            if (n > exp + 1) out = digits.substr(0, std::size_t(exp) + 1) + "." + digits.substr(std::size_t(exp) + 1);
            else out = digits + std::string(std::size_t(exp + 1 - n), '0') + ".0";
        } else {
            out = "0." + std::string(std::size_t(-exp - 1), '0') + digits;
        }
    } else {
        out = digits.substr(0, 1);
        if (n > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    }
    return sign + out;
}

std::string dumps(const Value& value, int indent, bool sort_keys) {
    std::string out;
    write(out, value, indent, sort_keys, 0);
    return out;
}

}  // namespace rudra::pyjson
