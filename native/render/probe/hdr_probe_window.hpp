#pragma once
// Gate B: a bare QRhi window that puts known luminance on an HDR swapchain and
// reads the backbuffer back, so "is the 1 000-nit patch above SDR white" is a
// number in a report before it is a judgement by eye or a meter reading.

#include <QJsonObject>
#include <QString>
#include <QWindow>

#include <memory>

#include <rhi/qrhi.h>

namespace rudra::probe {

enum class Api { D3D12, D3D11, Metal, Vulkan, OpenGL, Null };
enum class Want { ScRgb, Hdr10, DisplayP3, Sdr };

struct Options {
    Api api = Api::OpenGL;
    Want want = Want::ScRgb;
    QString report_path;         // empty: stdout only
    int exit_after_frames = 0;   // 0: stay open (Esc quits, R re-reads)
    int screen = -1;             // QGuiApplication::screens() index, -1: primary
};

QString to_string(Api a);
QString to_string(Want w);
QSurface::SurfaceType surface_type(Api a);

class HdrProbeWindow : public QWindow {
public:
    explicit HdrProbeWindow(const Options& opt);
    ~HdrProbeWindow() override;

protected:
    void exposeEvent(QExposeEvent*) override;
    bool event(QEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;

private:
    bool init();
    void resize_swapchain();
    void release_swapchain();
    void render();
    void choose_format();
    void update_encoding();
    void finish_report(const QRhiReadbackResult& rb);

    Options opt_;
    std::unique_ptr<QOffscreenSurface> fallback_;
    std::unique_ptr<QRhi> rhi_;
    std::unique_ptr<QRhiSwapChain> sc_;
    std::unique_ptr<QRhiRenderPassDescriptor> rp_;
    std::unique_ptr<QRhiBuffer> ubuf_;
    std::unique_ptr<QRhiShaderResourceBindings> srb_;
    std::unique_ptr<QRhiGraphicsPipeline> pipe_;
    QRhiSwapChain::Format format_ = QRhiSwapChain::SDR;
    bool initialized_ = false, has_swapchain_ = false, not_exposed_ = false;
    int frame_ = 0;
    bool readback_pending_ = false, readback_requested_ = true;
    QRhiReadbackResult rb_;
    float scale_ = 1.0f, mode_ = 2.0f, peak_nits_ = 0.0f;
    QJsonObject report_;
};

}  // namespace rudra::probe
