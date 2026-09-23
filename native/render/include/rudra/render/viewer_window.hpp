#pragma once
// The viewer (Phase 2 step 9, ADR-010): a QWindow with its own QRhi swapchain,
// embedded in the widget layout with QWidget::createWindowContainer. It owns
// the frame on the GPU and runs the passes every other part of render/ is
// held to: composite.frag into an fp32 target, display.frag into the picture
// (SDR, scRGB, HDR10 or EDR, whatever the swapchain is), and blit into the
// window at the place core/viewport.hpp computes. The controls of the browser
// Studio's plate work the same: wheel zooms about the cursor, middle drag
// pans, double click fits; W toggles the wipe and left drag moves it; B or a
// held left button shows the baseline; arrows nudge the wipe; Esc leaves it.
//
// QRhi stays behind the pimpl: this header includes only QtGui.

#include <QWindow>

#include <functional>
#include <memory>
#include <string>

#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/view.hpp"
#include "rudra/core/viewport.hpp"
#include "rudra/render/gpu_composite.hpp"

namespace rudra {

struct ViewerStatus {
    std::string backend;     // QRhi backend name
    std::string swapchain;   // SDR, scRGB, HDR10, EDR
    DisplayTarget target;
    int zoom_percent = 100;
    bool has_frame = false;
    bool wiping = false;
};

// One frame on the GPU: the SDR the network saw, its fields, and the
// per-frame scalars. The baseline is computed from them once per frame.
struct ViewerFrame {
    SdrImage sdr;
    Fields fields;
    FrameScalars scalars;
    ModelConstants model;
};

class ViewerWindow : public QWindow {
public:
    // Auto: D3D12 on Windows, Metal on macOS, Vulkan then OpenGL elsewhere.
    // prefer_hdr: take an HDR swapchain format when the display offers one.
    explicit ViewerWindow(GpuApi api = GpuApi::Auto, bool prefer_hdr = true);
    ~ViewerWindow() override;

    void set_frame(ViewerFrame frame);
    // A picture that is already composited (a master, a test card): shown
    // through the display pass as it is, with no composite pass.
    void set_composited(NetworkLinearImage model, NetworkLinearImage baseline);
    void clear_frame();
    void set_composite(const CompositeParams& params);
    // Mode, view peak, source, wipe and difference gain; the target is the
    // swapchain's own and is filled in by the window.
    void set_view(const ViewParams& params);
    ViewParams view() const;
    const ViewportState& viewport() const;
    void set_viewport(const ViewportState& v);
    void zoom_fit();
    void zoom_actual();

    ViewerStatus status() const;
    void on_status(std::function<void(const ViewerStatus&)> cb);

    // The next presented frame, read back from the swapchain (tests, Gate B).
    // Pixels are top-down RGBA in the swapchain's format: 8-bit sRGB codes
    // for SDR, half floats for scRGB and EDR, 10-bit PQ words for HDR10.
    struct Grab {
        int width = 0, height = 0;
        std::string format;         // RGBA8, BGRA8, RGBA16F, RGB10A2
        std::vector<unsigned char> bytes;
        int bytes_per_pixel = 4;
    };
    void grab(std::function<void(const Grab&)> done);

protected:
    void exposeEvent(QExposeEvent*) override;
    bool event(QEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void mouseDoubleClickEvent(QMouseEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
    void keyReleaseEvent(QKeyEvent* e) override;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace rudra
