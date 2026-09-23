// QC against rudra/qc.py (goldens from tools/emit_qc_golden.py): the same
// statuses, the same values and the same report text, on five cases that
// walk every branch (pass, fail, unmeasured, NaN).

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iterator>

#include <nlohmann/json.hpp>

#include "rudra/deliver/qc.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {
const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "qc";
std::string text_of(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}
}  // namespace

TEST(Qc, ChecksAndReportTextMatchQcPy) {
    std::ifstream in(kDir / "index.json");
    const auto idx = nlohmann::json::parse(in);
    auto t = load_qc_thresholds(kDir / idx.at("thresholds").get<std::string>());
    ASSERT_TRUE(t);
    ASSERT_GE(idx.at("cases").size(), 5u);
    for (const auto& c : idx.at("cases")) {
        const std::string name = c.at("name");
        SCOPED_TRACE(name);
        auto s = read_npy(kDir / c.at("sdr").get<std::string>());
        auto h = read_npy_f64(kDir / c.at("hdr").get<std::string>());
        ASSERT_TRUE(s && h);
        const int hh = int(s->shape[1]), ww = int(s->shape[2]);
        const SdrImage sdr(PlanarBuffer(3, hh, ww, s->data));
        NitsFrame hdr(hh, ww);
        std::copy(h->data.begin(), h->data.end(), hdr.span().begin());
        auto r = check_frame(hdr, sdr, *t);
        ASSERT_TRUE(r) << r.error().message;
        const auto& want = c.at("report");
        EXPECT_EQ(r->verdict, want.at("verdict").get<std::string>());
        ASSERT_EQ(r->checks.size(), want.at("checks").size());
        for (std::size_t i = 0; i < r->checks.size(); ++i) {
            const auto& g = r->checks[i];
            const auto& w = want.at("checks")[i];
            SCOPED_TRACE(g.name);
            EXPECT_EQ(g.name, w.at("name").get<std::string>());
            EXPECT_EQ(g.status, w.at("status").get<std::string>());
            if (w.at("value").is_null()) {
                EXPECT_FALSE(g.value.has_value());
            } else if (w.at("value").is_string()) {
                ASSERT_TRUE(g.value.has_value());
                EXPECT_TRUE(std::isnan(*g.value));
            } else {
                ASSERT_TRUE(g.value.has_value());
                const double wv = w.at("value").get<double>();
                // The blur-based checks subtract a float32 blur from its input,
                // and OpenCV sums that blur in its own SIMD order: 2e-4 relative
                // at worst on hf_excess. Everything else is float64 and tight.
                const bool blurred = g.name == "hf_excess" || g.name == "clip_structure_pct";
                EXPECT_NEAR(*g.value, wv, (blurred ? 1e-3 : 1e-9) * std::max(1.0, std::abs(wv)));
            }
            EXPECT_EQ(g.detail, w.at("detail").get<std::string>());
        }
        EXPECT_EQ(format_qc_report(*r, name + ".png", std::to_string(ww) + "x" + std::to_string(hh)),
                  text_of(kDir / c.at("report_text").get<std::string>()));
    }
}
