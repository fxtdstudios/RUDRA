// The app's look is app/theme/pro.css (the Pro boards) and nothing else.
//
// The style sheet the app embeds is generated from pro.css at build time
// (cmake/rudra_theme.cmake). These checks hold it to that: every colour in it
// is a colour pro.css names, every grey is neutral (R = G = B: a tinted
// surround biases the judgement of the picture), the only hues are the
// named ones, the fonts are the tokens' first families, and no colour is
// written in the app's C++.

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
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
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
    // LF only, whatever the checkout: a CR in a token is a family Qt cannot find.
    EXPECT_EQ(slurp(std::filesystem::path(RUDRA_THEME_DIR) / "tokens.txt").find('\r'), std::string::npos);
    EXPECT_EQ(slurp(std::filesystem::path(RUDRA_THEME_DIR) / "studio.qss").find('\r'), std::string::npos);
    EXPECT_EQ(qss().find("var("), std::string::npos);
    const auto t = tokens();
    for (const char* name : {"bg", "panel", "panel-2", "side", "bar", "card", "raise", "raise-2", "line", "line-2", "ink",
                             "ink-2", "ink-3", "ink-4", "accent", "accent-dim", "accent-line", "gold", "violet", "ok",
                             "warn", "bad", "ui-family", "mono-family"})
        EXPECT_TRUE(t.count(name)) << "token --" << name;
}

TEST(Theme, EveryColourIsTheThemes) {
    const std::set<std::string> page = colours_in(strip_comments(slurp(RUDRA_THEME_CSS)));
    const std::set<std::string> ours = colours_in(qss());
    ASSERT_GT(ours.size(), 10u);
    for (const auto& c : ours) EXPECT_TRUE(page.count(c)) << c << " is in studio.qss and not in app/theme/pro.css";
}

TEST(Theme, GreysAreNeutralAndHuesAreNamed) {
    const auto t = tokens();
    // The surround and the chrome: R = G = B, no exceptions.
    for (const char* name : {"bg", "panel", "panel-2", "side", "bar", "card", "raise", "raise-2", "line", "line-2", "ink",
                             "ink-2", "ink-3", "ink-4", "white", "scope-bg", "scope-plot", "scope-line"})
        EXPECT_TRUE(neutral(normalise(t.at(name)))) << "--" << name << " " << t.at(name);
    // The only hues: the named ones, each meaning something.
    std::set<std::string> hues;
    for (const char* name : {"accent", "accent-dim", "accent-line", "gold", "violet", "ok", "warn", "bad"})
        hues.insert(normalise(t.at(name)));
    for (const auto& c : colours_in(qss())) {
        if (neutral(c)) continue;
        EXPECT_TRUE(hues.count(c)) << c << " is a hue the theme does not name";
    }
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
        // Besides the greys: the accent (the active control, a primary button,
        // its hover, a selected row), the lamp (--ok), and the marks that mean
        // something: gold for highlights and violet for shadows (the lane's
        // dots, the HDR tag, the clip bar's acted part), --bad for what the
        // SDR lost.
        bool allowed = neutral(c);
        bool named = false;
        for (const char* name : {"accent", "accent-dim", "accent-line", "ok", "bad", "gold", "violet"})
            named = named || c == normalise(t.at(name));
        allowed = allowed || named;
        EXPECT_TRUE(allowed) << "background " << c;
    }
    EXPECT_GT(n, 20);
    // The viewer's surround and the window itself are --bg.
    EXPECT_NE(s.find("QStackedWidget#viewer, QWidget#empty { background: " + t.at("bg")), std::string::npos);
    // Every other widget is transparent over its region's own background.
    EXPECT_NE(s.find("QWidget {\n  font-family: \"" + t.at("ui-family") + "\";\n  background: transparent"),
              std::string::npos);
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
    EXPECT_EQ(t.at("ui-family"), "Geist");
    EXPECT_EQ(t.at("mono-family"), "Geist Mono");
    // Embedded with their licences: Geist for the app, Plex for the scope labels.
    const std::filesystem::path fonts = std::filesystem::path(RUDRA_APP_DIR) / "fonts";
    for (const char* f : {"Geist-Regular.ttf", "Geist-Medium.ttf", "Geist-SemiBold.ttf", "Geist-Bold.ttf",
                          "GeistMono-Regular.ttf", "GeistMono-Medium.ttf", "OFL-Geist.txt",
                          "IBMPlexSansCondensed-Regular.ttf", "IBMPlexSansCondensed-Medium.ttf",
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
