#pragma once
// The page's read-outs as text (Phase 3 steps 7 and 8): showProbePanel and
// showProbe (the probe in the rail and the floating box), showMetrics (the
// Frame panel, the mask and time status, the source line), updatePipe (the
// colour pipeline bar and its warning) and paintClipBar, from ui/app.js. Each
// takes the data the page's does and gives the words, classes and widths it
// writes (tests/golden/readouts, tools/emit_readouts_golden.py).

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace rudra {

// ---- the probe ------------------------------------------------------------

struct ProbeInput {
    int x = 0, y = 0;
    double model_nits = 0.0, baseline_nits = 0.0;   // Rec.2020 luminance at the pixel
    std::optional<std::array<int, 3>> sdr;          // the SDR codes there, when the frame has them
    std::optional<double> hi_mask, sh_mask;         // the highlight and shadow masks there
};

struct ProbePanelText {
    std::string xy, nits, delta, delta_class, src, src_class, base, model, model_class, mask;
    bool idle = true;   // no pixel: the readout's .idle
};
// The rail's probe panel; nullopt is "no pixel".
ProbePanelText probe_panel(const std::optional<ProbeInput>& p);

struct ProbeRow {
    std::string k, v, cls;
};
// The floating box's rows.
std::vector<ProbeRow> probe_box(const ProbeInput& p);

// ---- the frame measurements ---------------------------------------------

struct FrameMetrics {
    double maxcll = 0, maxfall = 0, peak_nits = 0, p99_nits = 0, median_nits = 0;
    double above_diffuse_white_pct = 0, above_1000_nits_pct = 0;
    double headroom_highlight_stops = 0, headroom_shadow_stops = 0, departure_rms_stops = 0;   // NaN: none
    double highlight_mask_pct = 0, shadow_mask_pct = 0;
    double compose_ms = 0;
};
// The page's state.header: what the server said about the frame.
struct FrameHeader {
    std::optional<std::string> source_resolution, resolution;
    std::optional<double> elapsed_s;
    bool tiled = false;
    std::optional<int> source_bits;
};

struct MetricRow {
    std::string k, v, u;
    bool warn = false;
};
struct MetricsText {
    std::vector<MetricRow> a, b;   // #measA, #measB
    std::string status_mask, status_time, src_info;
};
MetricsText metrics_text(const FrameMetrics& m, const std::optional<FrameHeader>& header, double clipped_pct);

// ---- the pipeline bar ----------------------------------------------------

struct PipeText {
    std::string in, working, view, master, warn;
    bool warn_shown = false;   // the text is the page's only while shown
};
// container "aces" or "linear"; display_nits the view peak; maxcll the
// frame's, NaN or none when there is no measurement.
PipeText pipe_text(const std::string& container, double display_nits, std::optional<double> maxcll,
                   const std::optional<FrameHeader>& header);

// ---- the clip bar --------------------------------------------------------

// The red run (what the SDR lost) and the amber one after it (where the
// network acted), as the page's CSS widths ("4.567%").
struct ClipBarText {
    std::string i_width, u_left, u_width;
};
ClipBarText clip_bar(double clipped_pct, double mask_pct);

}  // namespace rudra
