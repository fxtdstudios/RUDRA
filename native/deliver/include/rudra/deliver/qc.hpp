#pragma once
// QC for a reconstruction: PASS, FAIL or UNMEASURED, where a metric that could
// not be measured fails. Port of rudra/qc.py check_frame and format_report,
// thresholds from configs/qc_reconstruction.json (never hard-coded).

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/core/master.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

inline constexpr const char* kQcPass = "PASS";
inline constexpr const char* kQcFail = "FAIL";
inline constexpr const char* kQcUnmeasured = "UNMEASURED";

struct QcCheck {
    std::string name, status;
    std::optional<double> value, threshold;
    std::string detail;
    bool ok() const { return status == kQcPass; }
};

struct QcReport {
    std::vector<QcCheck> checks;
    std::string verdict = kQcPass;
    void add(QcCheck c) {
        if (!c.ok()) verdict = kQcFail;
        checks.push_back(std::move(c));
    }
    const QcCheck* first_failure() const;
};

using QcThresholds = std::map<std::string, double>;

// load_thresholds: every key not starting with "_", its "value".
Result<QcThresholds> load_qc_thresholds(const std::filesystem::path& path);

// check_frame: `hdr` in absolute nits, `sdr` its source code values.
Result<QcReport> check_frame(const NitsFrame& hdr, const SdrImage& sdr, const QcThresholds& t,
                             double ceiling_nits = 40000.0);

// format_report's text, character for character (UTF-8).
std::string format_qc_report(const QcReport& report, const std::string& name, const std::string& resolution);

}  // namespace rudra
