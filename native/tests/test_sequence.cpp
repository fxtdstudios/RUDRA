// Sequence open against ui/sequence.py (goldens from tools/emit_sequence_golden.py):
// the same layouts rebuilt here must give the same frame names in the same
// order, and the same messages, as Sequence.open and describe().

#include <gtest/gtest.h>

#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/media/sequence.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

nlohmann::json golden() {
    std::ifstream in(fs::path(RUDRA_GOLDEN_DIR) / "sequence" / "index.json");
    return nlohmann::json::parse(in);
}

fs::path build(const nlohmann::json& g) {
    const fs::path root = fs::temp_directory_path() / "rudra_sequence_test";
    fs::remove_all(root);
    fs::create_directories(root);
    for (const auto& [folder, entries] : g["layouts"].items()) {
        fs::create_directories(root / folder);
        for (const auto& e : entries) {
            const std::string n = e.get<std::string>();
            if (n.back() == '/') fs::create_directories(root / folder / n.substr(0, n.size() - 1));
            else std::ofstream(root / folder / n, std::ios::binary);
        }
    }
    for (const auto& f : g["files"]) std::ofstream(root / f.get<std::string>(), std::ios::binary);
    return fs::canonical(root);
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (std::size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size()))
        s.replace(at, from.size(), to);
    return s;
}

// The Python ran on POSIX paths; on Windows compare with '/' throughout.
std::string portable(std::string s, const fs::path& root) {
    s = replace_all(s, root.string(), "<root>");
#if defined(_WIN32)
    for (char& c : s)
        if (c == '\\') c = '/';
#endif
    return s;
}

}  // namespace

TEST(Sequence, SuffixListsMatchPython) {
    const auto g = golden();
    EXPECT_EQ(g["frame_suffixes"].get<std::vector<std::string>>(), frame_suffixes());
    EXPECT_EQ(g["video_suffixes"].get<std::vector<std::string>>(), video_suffixes());
}

TEST(Sequence, OpensLikeThePython) {
    const auto g = golden();
    const fs::path root = build(g);
    for (const auto& c : g["cases"]) {
        SCOPED_TRACE(c["name"].get<std::string>());
        const std::string raw = replace_all(c["raw"].get<std::string>(), "<root>", root.string());
        const auto r = open_sequence(raw);
        if (c.contains("video")) {
            ASSERT_FALSE(r.ok());
            EXPECT_EQ(r.error().code, ErrorCode::Unsupported);
            continue;
        }
        if (c.contains("error")) {
            ASSERT_FALSE(r.ok());
            EXPECT_EQ(portable(r.error().message, root), c["error"].get<std::string>());
            continue;
        }
        ASSERT_TRUE(r.ok()) << r.error().message;
        const auto& seq = r.value();
        EXPECT_EQ(seq.kind, c["kind"].get<std::string>());
        EXPECT_EQ(seq.count(), c["count"].get<int>());
        EXPECT_EQ(seq.path.filename().string(), c["name_field"].get<std::string>());
        EXPECT_EQ(portable(seq.path.string(), root), c["path"].get<std::string>());
        std::vector<std::string> names;
        for (int i = 0; i < seq.count(); ++i) names.push_back(seq.name_of(i));
        EXPECT_EQ(names, c["names"].get<std::vector<std::string>>());
    }
    fs::remove_all(root);
}

TEST(Sequence, NaturalOrder) {
    EXPECT_TRUE(natural_less("frame_2.png", "frame_10.png"));
    EXPECT_FALSE(natural_less("frame_10.png", "frame_2.png"));
    EXPECT_TRUE(natural_less("Frame_3.PNG", "frame_007.png"));
    EXPECT_TRUE(natural_less("a10b2", "a10b10"));
    EXPECT_FALSE(natural_less("x07", "x7"));   // equal keys: the sort keeps directory order
    EXPECT_FALSE(natural_less("x7", "x07"));
}
