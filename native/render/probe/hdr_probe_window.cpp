#include "hdr_probe_window.hpp"
#include "rudra/render/gl_format.hpp"

#include <algorithm>
#include <QCoreApplication>
#include <QFile>
#include <QGuiApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QKeyEvent>
#include <QOffscreenSurface>
#include <QPlatformSurfaceEvent>
#include <QScreen>
#include <QTextStream>
#include <QFloat16>

#include <cmath>
#include <cstring>

#include "rudra/render/viewer_backend.hpp"

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX   // windows.h (via dxgi) would otherwise turn std::max into a macro
#endif
#include <dxgi1_6.h>
#pragma comment(lib, "dxgi.lib")
#endif

namespace rudra::probe {
namespace {

constexpr float kPatches[6] = {100.0f, 203.0f, 400.0f, 600.0f, 1000.0f, 2000.0f};
constexpr float kSdrWhite = 203.0f;

QShader load_shader(const QString& name) {
    QFile f(name);
    if (f.open(QIODevice::ReadOnly)) return QShader::fromSerialized(f.readAll());
    return {};
}

double pq_eotf(double code) {   // ST 2084, code in [0,1] -> nits
    const double m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    const double p = std::pow(std::clamp(code, 0.0, 1.0), 1.0 / m2);
    return 10000.0 * std::pow(std::max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
}

double srgb_eotf(double v) { return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4); }

QString format_name(QRhiSwapChain::Format f) {
    switch (f) {
        case QRhiSwapChain::SDR: return "SDR";
        case QRhiSwapChain::HDRExtendedSrgbLinear: return "HDRExtendedSrgbLinear";
        case QRhiSwapChain::HDR10: return "HDR10";
        case QRhiSwapChain::HDRExtendedDisplayP3Linear: return "HDRExtendedDisplayP3Linear";
    }
    return "?";
}

#ifdef Q_OS_WIN
// What Windows itself says about the output under the window, straight from
// DXGI: the colour space is G2084/P2020 only when "Use HDR" is on for it.
// Independent of Qt, so a FAIL can be told apart from a probe bug.
QJsonObject dxgi_output_for(const QRect& screen_geometry) {
    QJsonObject out;
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return out;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
        IDXGIOutput* output = nullptr;
        for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
            IDXGIOutput6* output6 = nullptr;
            if (SUCCEEDED(output->QueryInterface(__uuidof(IDXGIOutput6), reinterpret_cast<void**>(&output6)))) {
                DXGI_OUTPUT_DESC1 d{};
                if (SUCCEEDED(output6->GetDesc1(&d))) {
                    const RECT r = d.DesktopCoordinates;
                    const QRect g(r.left, r.top, r.right - r.left, r.bottom - r.top);
                    if (out.isEmpty() || g.intersects(screen_geometry)) {
                        const bool hdr = d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
                        out = QJsonObject{};
                        out["device"] = QString::fromWCharArray(d.DeviceName);
                        out["color_space"] = hdr ? "G2084_P2020 (HDR on)"
                                           : d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 ? "G22_P709 (HDR off)"
                                           : QString::number(int(d.ColorSpace));
                        out["windows_hdr_on"] = hdr;
                        out["max_luminance"] = d.MaxLuminance;
                        out["max_full_frame_luminance"] = d.MaxFullFrameLuminance;
                        out["bits_per_color"] = int(d.BitsPerColor);
                    }
                }
                output6->Release();
            }
            output->Release();
        }
        adapter->Release();
    }
    factory->Release();
    return out;
}
#endif

QString texture_format_name(QRhiTexture::Format f) {
    switch (f) {
        case QRhiTexture::RGBA8: return "RGBA8";
        case QRhiTexture::BGRA8: return "BGRA8";
        case QRhiTexture::RGBA16F: return "RGBA16F";
        case QRhiTexture::RGB10A2: return "RGB10A2";
        default: return QString::number(int(f));
    }
}

}  // namespace

QString to_string(Api a) {
    switch (a) {
        case Api::D3D12: return "d3d12";
        case Api::D3D11: return "d3d11";
        case Api::Metal: return "metal";
        case Api::Vulkan: return "vulkan";
        case Api::OpenGL: return "gl";
        case Api::Null: return "null";
    }
    return "?";
}

QString to_string(Want w) {
    switch (w) {
        case Want::ScRgb: return "scrgb";
        case Want::Hdr10: return "hdr10";
        case Want::DisplayP3: return "p3";
        case Want::Sdr: return "sdr";
    }
    return "?";
}

QSurface::SurfaceType surface_type(Api a) {
    switch (a) {
        case Api::D3D12:
        case Api::D3D11: return QSurface::Direct3DSurface;
        case Api::Metal: return QSurface::MetalSurface;
        case Api::Vulkan: return QSurface::VulkanSurface;
        case Api::OpenGL: return QSurface::OpenGLSurface;
        case Api::Null: return QSurface::RasterSurface;
    }
    return QSurface::OpenGLSurface;
}

HdrProbeWindow::HdrProbeWindow(const Options& opt) : opt_(opt) {
    setSurfaceType(surface_type(opt.api));
#ifdef __APPLE__
    if (opt.api == Api::OpenGL) setFormat(rhi_gl_format());
#endif
    setTitle(QStringLiteral("RUDRA HDR probe (Gate B)"));
    resize(1280, 720);
}

HdrProbeWindow::~HdrProbeWindow() {
    pipe_.reset();
    srb_.reset();
    ubuf_.reset();
    rp_.reset();
    sc_.reset();
    rhi_.reset();
}

void HdrProbeWindow::exposeEvent(QExposeEvent*) {
    if (isExposed() && !initialized_) {
        if (!init()) {
            QCoreApplication::exit(2);
            return;
        }
        resize_swapchain();
        initialized_ = true;
    }
    const QSize size = sc_ ? sc_->surfacePixelSize() : QSize();
    if (initialized_ && (!isExposed() || size.isEmpty())) not_exposed_ = true;
    if (initialized_ && isExposed() && !size.isEmpty()) {
        not_exposed_ = false;
        render();
    }
}

bool HdrProbeWindow::event(QEvent* e) {
    switch (e->type()) {
        case QEvent::UpdateRequest: render(); break;
        case QEvent::PlatformSurface:
            if (static_cast<QPlatformSurfaceEvent*>(e)->surfaceEventType() ==
                QPlatformSurfaceEvent::SurfaceAboutToBeDestroyed)
                release_swapchain();
            break;
        default: break;
    }
    return QWindow::event(e);
}

void HdrProbeWindow::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Escape) close();
    if (e->key() == Qt::Key_R) readback_requested_ = true;
}

bool HdrProbeWindow::init() {
    QRhi::Flags flags;
    switch (opt_.api) {
        case Api::OpenGL: {
            fallback_.reset(QRhiGles2InitParams::newFallbackSurface(rhi_gl_format()));
            QRhiGles2InitParams p;
            p.format = rhi_gl_format();
            p.fallbackSurface = fallback_.get();
            p.window = this;
            rhi_.reset(QRhi::create(QRhi::OpenGLES2, &p, flags));
            break;
        }
        case Api::Vulkan: {
#if QT_CONFIG(vulkan)
            QRhiVulkanInitParams p;
            p.inst = vulkanInstance();
            p.window = this;
            rhi_.reset(QRhi::create(QRhi::Vulkan, &p, flags));
#endif
            break;
        }
        case Api::D3D11: {
#ifdef Q_OS_WIN
            QRhiD3D11InitParams p;
            rhi_.reset(QRhi::create(QRhi::D3D11, &p, flags));
#endif
            break;
        }
        case Api::D3D12: {
#ifdef Q_OS_WIN
            QRhiD3D12InitParams p;
            rhi_.reset(QRhi::create(QRhi::D3D12, &p, flags));
#endif
            break;
        }
        case Api::Metal: {
#if QT_CONFIG(metal)
            QRhiMetalInitParams p;
            rhi_.reset(QRhi::create(QRhi::Metal, &p, flags));
#endif
            break;
        }
        case Api::Null: {
            QRhiNullInitParams p;
            rhi_.reset(QRhi::create(QRhi::Null, &p, flags));
            break;
        }
    }
    if (!rhi_) {
        QTextStream(stderr) << "rudra-hdr-probe: could not create a QRhi for " << to_string(opt_.api) << "\n";
        return false;
    }

    sc_.reset(rhi_->newSwapChain());
    sc_->setWindow(this);
    choose_format();
    rp_.reset(sc_->newCompatibleRenderPassDescriptor());
    sc_->setRenderPassDescriptor(rp_.get());

    ubuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 80));
    ubuf_->create();
    srb_.reset(rhi_->newShaderResourceBindings());
    srb_->setBindings({QRhiShaderResourceBinding::uniformBuffer(
        0, QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage, ubuf_.get())});
    srb_->create();

    pipe_.reset(rhi_->newGraphicsPipeline());
    pipe_->setShaderStages({{QRhiShaderStage::Vertex, load_shader(":/shaders/probe.vert.qsb")},
                            {QRhiShaderStage::Fragment, load_shader(":/shaders/probe.frag.qsb")}});
    pipe_->setVertexInputLayout({});
    pipe_->setShaderResourceBindings(srb_.get());
    pipe_->setRenderPassDescriptor(rp_.get());
    if (!pipe_->create()) {
        QTextStream(stderr) << "rudra-hdr-probe: pipeline creation failed\n";
        return false;
    }

    const auto driver = rhi_->driverInfo();
    report_["api"] = to_string(opt_.api);
    report_["backend"] = QString::fromLatin1(rhi_->backendName());
    report_["device"] = QString::fromUtf8(driver.deviceName);
    report_["qt"] = QString::fromLatin1(qVersion());
    report_["requested"] = to_string(opt_.want);
    QJsonObject supported;
    for (auto f : {QRhiSwapChain::SDR, QRhiSwapChain::HDRExtendedSrgbLinear, QRhiSwapChain::HDR10,
                   QRhiSwapChain::HDRExtendedDisplayP3Linear})
        supported[format_name(f)] = sc_->isFormatSupported(f);
    report_["formats_supported"] = supported;
    report_["format"] = format_name(format_);
    if (const QScreen* sc = screen()) {
        report_["screen"] = sc->name();
        report_["screen_model"] = sc->model();
#ifdef Q_OS_WIN
        const qreal dpr = sc->devicePixelRatio();
        const QRect native_geometry(sc->geometry().topLeft() * dpr, sc->geometry().size() * dpr);
        report_["windows_output"] = dxgi_output_for(native_geometry);
#endif
    }
    return true;
}

void HdrProbeWindow::choose_format() {
    // What was asked for if the swapchain can do it, else the next HDR format,
    // else SDR: the report says which, so a silent fallback cannot pass the gate.
    QList<QRhiSwapChain::Format> order;
    switch (opt_.want) {
        case Want::ScRgb: order = {QRhiSwapChain::HDRExtendedSrgbLinear, QRhiSwapChain::HDR10}; break;
        case Want::Hdr10: order = {QRhiSwapChain::HDR10, QRhiSwapChain::HDRExtendedSrgbLinear}; break;
        case Want::DisplayP3:
            order = {QRhiSwapChain::HDRExtendedDisplayP3Linear, QRhiSwapChain::HDRExtendedSrgbLinear};
            break;
        case Want::Sdr: break;
    }
    format_ = QRhiSwapChain::SDR;
    for (auto f : order)
        if (sc_->isFormatSupported(f)) {
            format_ = f;
            break;
        }
    sc_->setFormat(format_);
}

void HdrProbeWindow::update_encoding() {
    // One scalar per path, from render/viewer_backend.hpp output_scale, so the
    // probe and the viewer cannot disagree about what 1.0 means.
    const QRhiSwapChainHdrInfo info = sc_->hdrInfo();
    const bool display_referred = info.luminanceBehavior == QRhiSwapChainHdrInfo::DisplayReferred;
    peak_nits_ = 0.0f;
    QJsonObject hdr;
    if (info.limitsType == QRhiSwapChainHdrInfo::LuminanceInNits) {
        hdr["limits"] = "nits";
        hdr["min_luminance"] = info.limits.luminanceInNits.minLuminance;
        hdr["max_luminance"] = info.limits.luminanceInNits.maxLuminance;
        peak_nits_ = info.limits.luminanceInNits.maxLuminance;
    } else {
        hdr["limits"] = "color_component_value";
        hdr["max_color_component_value"] = info.limits.colorComponentValue.maxColorComponentValue;
        hdr["max_potential_color_component_value"] = info.limits.colorComponentValue.maxPotentialColorComponentValue;
        // EDR: 1.0 is the SDR white, so the live headroom times RUDRA's SDR white.
        peak_nits_ = info.limits.colorComponentValue.maxColorComponentValue * kSdrWhite;
    }
    hdr["luminance_behavior"] = display_referred ? "display_referred" : "scene_referred";
    hdr["sdr_white_level"] = info.sdrWhiteLevel;
    if (format_ == QRhiSwapChain::SDR) {
        // No peak tick on an SDR card. What the limits mean depends on the
        // API: OpenGL and Vulkan report Qt's placeholders (1000 / 0 / 200),
        // D3D reports the display's own DXGI numbers even with HDR off.
        const bool placeholder = info.limitsType == QRhiSwapChainHdrInfo::LuminanceInNits &&
                                 info.limits.luminanceInNits.maxLuminance == 1000.0f &&
                                 info.limits.luminanceInNits.minLuminance == 0.0f;
        peak_nits_ = 0.0f;
        if (placeholder) {
            hdr["note"] = "SDR swapchain: Qt's placeholder limits, not the display's";
        } else {
            hdr["note"] = "the display reports these limits, but no HDR swapchain format is available on it";
            const QJsonObject wo = report_.value("windows_output").toObject();
            if (wo.value("windows_hdr_on").toBool()) {
                report_["hint"] = "Windows says HDR is ON for this output, yet Qt offered no HDR swapchain format. "
                                  "That points at the probe or the driver, not the settings: send the report JSON.";
            } else {
                report_["hint"] =
                    "The display this window opened on is not in HDR mode. Windows: Settings > System > Display, "
                    "select that display, turn on Use HDR (or press Win+Alt+B), then run again (--screen N picks "
                    "another display). macOS: run on the XDR panel or an HDR display with High Dynamic Range on.";
            }
        }
    }
    report_["hdr_info"] = hdr;

    OutputPath path = OutputPath::SdrPqSimulation;
    switch (format_) {
        case QRhiSwapChain::HDRExtendedSrgbLinear:
            path = display_referred ? OutputPath::Edr : OutputPath::ScRgb;
            mode_ = 0.0f;
            break;
        case QRhiSwapChain::HDRExtendedDisplayP3Linear:
            path = OutputPath::Edr;
            mode_ = 0.0f;
            break;
        case QRhiSwapChain::HDR10:
            path = OutputPath::Hdr10;
            mode_ = 1.0f;
            break;
        case QRhiSwapChain::SDR:
            mode_ = 2.0f;
            break;
    }
    scale_ = output_scale(path, Nits{1.0f}, Nits{kSdrWhite});
    report_["output_path"] = path == OutputPath::ScRgb ? "scRGB" : path == OutputPath::Edr ? "EDR"
                            : path == OutputPath::Hdr10 ? "HDR10" : "SDR";
    report_["linear_scale_per_nit"] = mode_ == 0.0f ? QJsonValue(double(scale_)) : QJsonValue();
}

void HdrProbeWindow::resize_swapchain() {
    has_swapchain_ = sc_->createOrResize();
    if (has_swapchain_) update_encoding();
}

void HdrProbeWindow::release_swapchain() {
    if (has_swapchain_) {
        has_swapchain_ = false;
        sc_->destroy();
    }
}

void HdrProbeWindow::render() {
    if (!has_swapchain_ || not_exposed_) return;
    if (sc_->currentPixelSize() != sc_->surfacePixelSize()) resize_swapchain();
    if (!has_swapchain_) return;

    QRhi::FrameOpResult r = rhi_->beginFrame(sc_.get());
    if (r == QRhi::FrameOpSwapChainOutOfDate) {
        resize_swapchain();
        if (!has_swapchain_) return;
        r = rhi_->beginFrame(sc_.get());
    }
    if (r != QRhi::FrameOpSuccess) {
        requestUpdate();
        return;
    }

    QRhiCommandBuffer* cb = sc_->currentFrameCommandBuffer();
    QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
    const QMatrix4x4 corr = rhi_->clipSpaceCorrMatrix();
    const float encode[4] = {scale_, mode_, peak_nits_, 0.0f};
    u->updateDynamicBuffer(ubuf_.get(), 0, 64, corr.constData());
    u->updateDynamicBuffer(ubuf_.get(), 64, 16, encode);

    const QSize px = sc_->currentPixelSize();
    cb->beginPass(sc_->currentFrameRenderTarget(), Qt::black, {1.0f, 0}, u);
    cb->setGraphicsPipeline(pipe_.get());
    cb->setViewport({0, 0, float(px.width()), float(px.height())});
    cb->setShaderResources();
    cb->draw(3);

    QRhiResourceUpdateBatch* after = nullptr;
    // A few frames in, so a compositor that switches the output mode on the
    // first HDR present has done so before the read.
    if (readback_requested_ && !readback_pending_ && frame_ >= 3) {
        readback_requested_ = false;
        readback_pending_ = true;
        rb_ = {};
        rb_.completed = [this] { finish_report(rb_); };
        after = rhi_->nextResourceUpdateBatch();
        after->readBackTexture(QRhiReadbackDescription(), &rb_);   // the backbuffer
    }
    cb->endPass(after);
    rhi_->endFrame(sc_.get());
    ++frame_;
    if (opt_.exit_after_frames > 0 && frame_ >= opt_.exit_after_frames && !readback_pending_) {
        QCoreApplication::exit(report_.value("verdict").toString() == "PASS" ? 0 : 1);
        return;
    }
    requestUpdate();
}

void HdrProbeWindow::finish_report(const QRhiReadbackResult& rb) {
    readback_pending_ = false;
    const int w = rb.pixelSize.width(), h = rb.pixelSize.height();
    if (w <= 0 || h <= 0 || rb.data.isEmpty()) return;
    const bool y_up = rhi_->isYUpInFramebuffer();
    const int bpp = rb.format == QRhiTexture::RGBA16F ? 8 : 4;
    const int row = int(rb.data.size()) / h;

    auto read = [&](double u, double v) -> double {
        const int x = std::clamp(int(u * w), 0, w - 1);
        int y = std::clamp(int(v * h), 0, h - 1);
        if (y_up) y = h - 1 - y;
        const char* p = rb.data.constData() + std::size_t(y) * row + std::size_t(x) * bpp;
        switch (rb.format) {
            case QRhiTexture::RGBA16F: {
                qfloat16 c;
                std::memcpy(&c, p, 2);
                return double(float(c)) / scale_;
            }
            case QRhiTexture::RGB10A2: {
                std::uint32_t word;
                std::memcpy(&word, p, 4);
                return pq_eotf(double(word & 0x3ffu) / 1023.0);
            }
            case QRhiTexture::RGBA8:
            case QRhiTexture::BGRA8:
                return srgb_eotf(double(static_cast<unsigned char>(p[1])) / 255.0) * kSdrWhite;
            default: return std::nan("");
        }
    };

    QJsonArray patches;
    double at_white = 0.0, at_1000 = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double nits = read((i + 0.5) / 6.0, 0.62 * 0.525);
        QJsonObject pch;
        pch["target_nits"] = kPatches[i];
        pch["swapchain_nits"] = std::round(nits * 10.0) / 10.0;
        patches.append(pch);
        if (kPatches[i] == 203.0f) at_white = nits;
        if (kPatches[i] == 1000.0f) at_1000 = nits;
    }
    report_["readback_format"] = texture_format_name(rb.format);
    report_["readback_size"] = QJsonArray{w, h};
    report_["patches"] = patches;
    // The swapchain carries the 1 000-nit patch at least a stop above SDR
    // white. Necessary, not sufficient: the glass is checked by eye or meter.
    const bool pass = at_1000 > 2.0 * at_white && format_ != QRhiSwapChain::SDR;
    report_["verdict"] = pass ? "PASS" : "FAIL";
    report_["verdict_note"] = pass
        ? "swapchain carries the 1000-nit patch above SDR white; confirm on the glass"
        : "the 1000-nit patch does not reach the swapchain above SDR white";

    const QByteArray json = QJsonDocument(report_).toJson(QJsonDocument::Indented);
    QTextStream(stdout) << json;
    if (!opt_.report_path.isEmpty()) {
        QFile f(opt_.report_path);
        if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) f.write(json);
    }
}

}  // namespace rudra::probe
