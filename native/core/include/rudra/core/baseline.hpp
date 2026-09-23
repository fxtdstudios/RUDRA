#pragma once
// The analytic baseline: what the SDR means if the corpus's own tone curve made
// it. Port of rudra/sdr2hdr.py sdr_to_baseline_hdr and CurveHead.correction_log2.
// Tested against arrays emitted from that Python (tools/emit_core_golden.py).

#include <span>

#include "rudra/core/image.hpp"

namespace rudra {

// Default exposure convention of every checkpoint up to sdr2hdr_shadow_v1.
inline constexpr float kLegacyCorpusEv = -1.0f;

float srgb_to_linear(float code) noexcept;
float inverse_aces_approx(float display_linear) noexcept;

// Per value: srgb_to_linear -> inverse ACES (Narkowicz) -> * 2^-ev * 203/10000.
float baseline_value(float sdr_code, float corpus_ev) noexcept;

// The whole image, into the network's linear convention (1.0 = 10 000 nits).
NetworkLinearImage analytic_baseline(const SdrImage& sdr, float corpus_ev);

// CurveHead correction in log2 radiance for one code value: params[0] is the
// global exposure, params[1..] the knots sampled at evenly spaced codes.
float curve_correction_log2(float sdr_code, std::span<const float> params) noexcept;

// Baseline with the frame's curve applied: analytic * 2^correction per channel.
// An empty or single-element params span means "no curve head".
NetworkLinearImage corrected_baseline(const SdrImage& sdr, float corpus_ev, std::span<const float> params);

}  // namespace rudra
