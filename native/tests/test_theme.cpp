// Phase 3 step 1: the app's look is ui/theme.css and nothing else.
//
// The style sheet the app embeds is generated from ui/theme.css at build time
// (cmake/rudra_theme.cmake). These checks hold it to the page: every colour
// in it is a colour ui/theme.css uses, every grey is neutral (R = G = B, the
// theme's first rule: a tinted surround biases the judgement of the picture),
// the only hues are the theme's named ones, the fonts are the tokens' first
// families, and no colour is written in the app's C++.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>

namespace {

std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

std::string strip_comments(std::string s) {
    for (std::size_t a; (a = s.find("/*")) != std::string::npos;) {
        const std::size_t b = s.find("*/", a + 2);
        s.erase(a, b == std::string::npos ? std::string::npos : b + 2 - a);
    }
    return s;
}

// #rgb, #rrggbb and #rrggbbaa as lower-case #rrggbb[aa]; ids such as
// QWidget#railLeft are not colours (the character before is not a value start).
std::string normalise(std::string hex) {
    std::transform(hex.begin(), hex.end(), hex.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (hex.size() == 4) return std::string{'#', hex[1], hex[1], hex[2], hex[2], hex[3], hex[3]};
    return hex;
}

std::set<std::string> colours_in(const std::string& text) {
    static const std::regex re(R"((^|[\s:,(])(#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3}))\b)");
    std::set<std::string> out;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it)
        out.insert(normalise((*it)[2].str()));
    return out;
}

bool neutral(const std::string& c) {
    return c.size() >= 7 && c.substr(1, 2) == c.substr(3, 2) && c.substr(3, 2) == c.substr(5, 2);
}

std::map<std::string, std::string> tokens() {
    std::map<std::string, std::string> t;
    std::ifstream f(std::filesystem::path(RUDRA_THEME_DIR) / "tokens.txt");
    for (std::string line; std::getline(f, line);) {
        const auto sp = line.find(' ');
        if (sp != std::string::npos) t[line.substr(0, sp)] = line.substr(sp + 1);
    }
    return t;
}

const std::string& qss() {
    static const std::string s = strip_comments(slurp(std::filesystem::path(RUDRA_THEME_DIR) / "studio.qss"));
    return s;
}

}  // namespace

TEST(Theme, GeneratedFromThemeCss) {
    ASSERT_FALSE(qss().empty()) << "studio.qss was not generated";
    EXPECT_EQ(qss().find("var("), std::string::npos);
    const auto t = tokens();
    for (const char* name : {"bg", "panel", "panel-2", "raise", "line", "line-2", "ink", "ink-2", "ink-3", "accent",
                             "accent-dim", "accent-line", "gold", "ok", "warn", "bad", "ui-family", "mono-family"})
        EXPECT_TRUE(t.count(name)) << "token --" << name;
}

TEST(Theme, EveryColourIsTheThemes) {
    const std::set<std::string> page = colours_in(strip_comments(slurp(RUDRA_THEME_CSS)));
    const std::set<std::string> ours = colours_in(qss());
    ASSERT_GT(ours.size(), 10u);
    for (const auto& c : ours) EXPECT_TRUE(page.count(c)) << c << " is in studio.qss and not in ui/theme.css";
}

TEST(Theme, GreysAreNeutralAndHuesAreNamed) {
    const auto t = tokens();
    // The surround and the chrome: R = G = B, no exceptions.
    for (const char* name : {"bg", "panel", "panel-2", "raise", "line", "line-2", "ink", "ink-2", "ink-3"})
        EXPECT_TRUE(neutral(normalise(t.at(name)))) << "--" << name << " " << t.at(name);
    // The only hues: the named ones, each meaning something, and the two
    // the page's controls use (the ink on an active control, the slider key).
    std::set<std::string> hues = {"#dceaf5", "#a2b8c7"};
    for (const char* name : {"accent", "accent-dim", "accent-line", "gold", "ok", "warn", "bad"})
        hues.insert(normalise(t.at(name)));
    for (const auto& c : colours_in(qss()))
        if (!neutral(c)) EXPECT_TRUE(hues.count(c)) << c << " is a hue the theme does not name";
}

TEST(Theme, SurroundIsNeutral) {
    // Every background in the sheet is a grey: the picture is the only
    // saturated thing on screen, apart from the selection's accent-dim.
    const auto t = tokens();
    static const std::regex bg(R"((?:^|[\s;{])background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,8}))");
    const std::string& s = qss();
    int n = 0;
    for (auto it = std::sregex_iterator(s.begin(), s.end(), bg); it != std::sregex_iterator(); ++it, ++n) {
        const std::string c = normalise((*it)[1].str());
        const bool allowed = neutral(c) || c == normalise(t.at("accent-dim")) || c == normalise(t.at("accent")) ||
                             c == "#a2b8c7";
        EXPECT_TRUE(allowed) << "background " << c;
    }
    EXPECT_GT(n, 20);
    // The viewer's surround and the window itself are --bg.
    EXPECT_NE(s.find("QWidget#viewerHost { background: " + t.at("bg")), std::string::npos);
}

TEST(Theme, FontsAreTheTokens) {
    const auto t = tokens();
    static const std::regex fam(R"re(font-family:\s*"([^"]+)")re");
    const std::string& s = qss();
    int n = 0;
    for (auto it = std::sregex_iterator(s.begin(), s.end(), fam); it != std::sregex_iterator(); ++it, ++n) {
        const std::string f = (*it)[1].str();
        EXPECT_TRUE(f == t.at("ui-family") || f == t.at("mono-family")) << f;
    }
    EXPECT_GT(n, 5);
    EXPECT_EQ(t.at("ui-family"), "IBM Plex Sans Condensed");
    EXPECT_EQ(t.at("mono-family"), "IBM Plex Mono");
    // Embedded with their licence.
    const std::filesystem::path fonts = std::filesystem::path(RUDRA_APP_DIR) / "fonts";
    for (const char* f : {"IBMPlexSansCondensed-Regular.ttf", "IBMPlexSansCondensed-Medium.ttf",
                          "IBMPlexSansCondensed-SemiBold.ttf", "IBMPlexSansCondensed-Bold.ttf",
                          "IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf", "OFL.txt"})
        EXPECT_TRUE(std::filesystem::exists(fonts / f)) << f;
}

TEST(Theme, NoColourInTheAppSources) {
    // A colour belongs in ui/theme.css; the app reads it from the tokens.
    static const std::regex lit(R"re("[^"\n]*#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b[^"\n]*")re");
    for (const auto& e : std::filesystem::directory_iterator(RUDRA_APP_DIR)) {
        const auto ext = e.path().extension();
        if (ext != ".cpp" && ext != ".hpp") continue;
        const std::string src = slurp(e.path());
        std::smatch m;
        EXPECT_FALSE(std::regex_search(src, m, lit)) << e.path().filename() << ": " << m.str();
    }
}
