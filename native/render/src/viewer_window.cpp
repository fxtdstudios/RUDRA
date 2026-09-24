#include "rudra/render/viewer_window.hpp"
#include "rudra/render/gl_format.hpp"

#include <QDropEvent>
#include <QMimeData>
#include <QUrl>
#include <QExposeEvent>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QOffscreenSurface>
#include <QPlatformSurfaceEvent>
#include <QWheelEvent>
#include <rhi/qrhi.h>

#include "rudra/render/qt_vulkan.hpp"
#if RUDRA_QT_VULKAN
#include <QVulkanInstance>
#endif
#include <QScreen>

#ifdef Q_OS_WIN
#include <dxgi1_6.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>

#include "passes.hpp"
#include "rudra/core/baseline.hpp"
#include "rudra/core/hdr10.hpp"

namespace rudra {
namespace {

using detail::load_shader;

// The Blit block of shaders/blit.vert.
struct BlitUbo {
    float clip_corr[16];
    float rect[4];
    float window[4];   // width, height, y-up, SDR white encoded
    float guides[4];   // action, title, centre, aspect
};
static_assert(sizeof(BlitUbo) == 28 * sizeof(float));

// The surround: neutral grey #121212, a graphic at the SDR white.
constexpr float kSurroundCode = 0x12 / 255.0f;

float code_to_linear(float c) {
    return c > 0.04045f ? std::pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
}

QSurface::SurfaceType surface_for(GpuApi api) {
    switch (api) {
        case GpuApi::D3D12:
        case GpuApi::D3D11: return QSurface::Direct3DSurface;
        case GpuApi::Metal: return QSurface::MetalSurface;
        case GpuApi::Vulkan: return QSurface::VulkanSurface;
        default: return QSurface::OpenGLSurface;
    }
}

GpuApi resolve(GpuApi api) {
    if (api != GpuApi::Auto) return api;
#if defined(Q_OS_WIN)
    return GpuApi::D3D12;
#elif defined(Q_OS_MACOS)
    return GpuApi::Metal;
#elif RUDRA_QT_VULKAN
    return GpuApi::Vulkan;
#else
    return GpuApi::OpenGL;
#endif
}

const char* swapchain_name(QRhiSwapChain::Format f, bool display_referred) {
    switch (f) {
        case QRhiSwapChain::HDRExtendedSrgbLinear: return display_referred ? "EDR" : "scRGB";
        case QRhiSwapChain::HDR10: return "HDR10";
        case QRhiSwapChain::HDRExtendedDisplayP3Linear: return "EDR";
        default: return "SDR";
    }
}

#ifdef Q_OS_WIN
// The peak Windows reports for the output under `native_geometry`, from DXGI.
// Qt's Vulkan and OpenGL swapchains report placeholder limits (1 000 nits),
// so on Windows the viewer asks the OS directly (as rudra-hdr-probe does).
std::optional<double> dxgi_peak(const QRect& native_geometry) {
    std::optional<double> peak;
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return peak;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
        IDXGIOutput* output = nullptr;
        for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
            IDXGIOutput6* output6 = nullptr;
            if (SUCCEEDED(output->QueryInterface(__uuidof(IDXGIOutput6), reinterpret_cast<void**>(&output6)))) {
                DXGI_OUTPUT_DESC1 d{};
                if (SUCCEEDED(output6->GetDesc1(&d))) {
                    const RECT r = d.DesktopCoordinates;
                    if (QRect(r.left, r.top, r.right - r.left, r.bottom - r.top).intersects(native_geometry) && !peak)
                        peak = double(d.MaxLuminance);
                }
                output6->Release();
            }
            output->Release();
        }
        adapter->Release();
    }
    factory->Release();
    return peak;
}
#endif

}  // namespace

struct ViewerWindow::Impl {
    ViewerWindow* w = nullptr;
    GpuApi api = GpuApi::OpenGL;
    bool prefer_hdr = true;

#if RUDRA_QT_VULKAN
    std::unique_ptr<QVulkanInstance> vk;
#endif
    std::unique_ptr<QOffscreenSurface> fallback;
    std::unique_ptr<QRhi> rhi;
    std::unique_ptr<QRhiSwapChain> sc;
    std::unique_ptr<QRhiRenderPassDescriptor> sc_rp;
    QRhiSwapChain::Format format = QRhiSwapChain::SDR;
    bool initialized = false, has_swapchain = false;

    QShader vert, composite_frag, display_frag, blit_vert, blit_frag;
    std::unique_ptr<QRhiSampler> nearest, trilinear, fetch;
    std::unique_ptr<QRhiBuffer> composite_ubo, view_ubo, blit_ubo;

    // Frame-sized resources, rebuilt when the frame size changes.
    QSize frame_size;
    std::unique_ptr<QRhiTexture> in_a, in_b, model_t, base_t, picture;
    std::unique_ptr<QRhiTextureRenderTarget> model_rt, picture_rt;
    std::unique_ptr<QRhiRenderPassDescriptor> model_rp, picture_rp;
    std::unique_ptr<QRhiShaderResourceBindings> composite_srb, display_srb, blit_srb_nearest, blit_srb_trilinear;
    std::unique_ptr<QRhiGraphicsPipeline> composite_pipe, display_pipe, blit_pipe;

    // What is shown, kept on the host so a lost device is rebuilt from it.
    bool has_frame = false;
    bool precomposited = false;   // set_composited: model_t is uploaded, not rendered
    NetworkLinearImage precomposite;
    ViewerFrame frame;
    NetworkLinearImage baseline;
    CompositeParams composite;
    ViewParams view;
    ViewportState viewport;
    GuideOptions guides;
    bool upload_dirty = false, composite_dirty = false, view_dirty = true;

    // Interaction.
    bool input = true;
    bool wipe_dragging = false, flip_held_key = false, flip_held_mouse = false;
    bool panning = false;
    QPointF pan_from;
    double pan_x0 = 0, pan_y0 = 0;

    std::string peak_from = "swapchain";   // swapchain, DXGI, or placeholder (unknown)

    std::function<void(const ViewerStatus&)> status_cb;
    std::function<void(const ViewerWindow::Hover&)> hover_cb;
    std::function<void(const QStringList&)> drop_cb;
    void hover(const QPointF& at, bool alt) {
        if (!hover_cb) return;
        ViewerWindow::Hover h;
        h.window = at;
        h.alt = alt;
        const PlacedRect r = placed();
        const ViewSize f = frame_view_size();
        if (has_frame && r.width > 0 && r.height > 0 && at.x() >= r.left && at.y() >= r.top &&
            at.x() < r.left + r.width && at.y() < r.top + r.height) {
            h.x = (at.x() - r.left) / r.width * f.width;
            h.y = (at.y() - r.top) / r.height * f.height;
        }
        hover_cb(h);
    }
    std::function<void(const Grab&)> grab_cb;
    QRhiReadbackResult grab_rb;
    bool grab_pending = false, grab_y_up = false;
    long long updates = 0, frames = 0, begin_failures = 0;
    int last_begin = 0;

    ViewSize viewer_size() const { return {double(w->width()), double(w->height())}; }
    ViewSize frame_view_size() const {
        return has_frame ? ViewSize{double(baseline.width()), double(baseline.height())} : ViewSize{};
    }
    PlacedRect placed() const { return place(viewport, viewer_size(), frame_view_size()); }

    ViewParams effective_view() const {
        ViewParams v = view;
        if (flip_held_key || flip_held_mouse) v.show = ViewSource::Baseline;
        return v;
    }

    bool init() {
        switch (api) {
            case GpuApi::OpenGL: {
                fallback.reset(QRhiGles2InitParams::newFallbackSurface(rhi_gl_format()));
                QRhiGles2InitParams p;
                p.format = rhi_gl_format();
                p.fallbackSurface = fallback.get();
                p.window = w;
                rhi.reset(QRhi::create(QRhi::OpenGLES2, &p));
                break;
            }
            case GpuApi::Vulkan: {
#if RUDRA_QT_VULKAN
                QRhiVulkanInitParams p;
                p.inst = w->vulkanInstance();
                p.window = w;
                rhi.reset(QRhi::create(QRhi::Vulkan, &p));
#endif
                break;
            }
            case GpuApi::D3D11: {
#ifdef Q_OS_WIN
                QRhiD3D11InitParams p;
                rhi.reset(QRhi::create(QRhi::D3D11, &p));
#endif
                break;
            }
            case GpuApi::D3D12: {
#ifdef Q_OS_WIN
                QRhiD3D12InitParams p;
                rhi.reset(QRhi::create(QRhi::D3D12, &p));
#endif
                break;
            }
            case GpuApi::Metal: {
#if QT_CONFIG(metal)
                QRhiMetalInitParams p;
                rhi.reset(QRhi::create(QRhi::Metal, &p));
#endif
                break;
            }
            case GpuApi::Auto: break;
        }
        if (!rhi) return false;
        vert = load_shader(":/rudra/shaders/fullscreen.vert.qsb");
        composite_frag = load_shader(":/rudra/shaders/composite.frag.qsb");
        display_frag = load_shader(":/rudra/shaders/display.frag.qsb");
        blit_vert = load_shader(":/rudra/shaders/blit.vert.qsb");
        blit_frag = load_shader(":/rudra/shaders/blit.frag.qsb");
        if (!vert.isValid() || !composite_frag.isValid() || !display_frag.isValid() || !blit_vert.isValid() ||
            !blit_frag.isValid())
            return false;
        fetch.reset(rhi->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None,
                                    QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        nearest.reset(rhi->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None,
                                      QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        trilinear.reset(rhi->newSampler(QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::Linear,
                                        QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        composite_ubo.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(detail::CompositeUbo)));
        view_ubo.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(detail::ViewUbo)));
        blit_ubo.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(BlitUbo)));
        if (!fetch->create() || !nearest->create() || !trilinear->create() || !composite_ubo->create() ||
            !view_ubo->create() || !blit_ubo->create())
            return false;

        sc.reset(rhi->newSwapChain());
        sc->setWindow(w);
        choose_format();
        sc_rp.reset(sc->newCompatibleRenderPassDescriptor());
        sc->setRenderPassDescriptor(sc_rp.get());
        return build_blit_pipeline();
    }

    void choose_format() {
        // HDR when the display offers it: scRGB (Windows, Linux Vulkan) or
        // EDR (macOS) first, then HDR10, else SDR. The status says which.
        format = QRhiSwapChain::SDR;
        if (prefer_hdr)
            for (auto f : {QRhiSwapChain::HDRExtendedSrgbLinear, QRhiSwapChain::HDRExtendedDisplayP3Linear,
                           QRhiSwapChain::HDR10})
                if (sc->isFormatSupported(f)) {
                    format = f;
                    break;
                }
        sc->setFormat(format);
    }

    void update_target() {
        const QRhiSwapChainHdrInfo info = sc->hdrInfo();
        const bool display_referred = info.luminanceBehavior == QRhiSwapChainHdrInfo::DisplayReferred;
        double nits = info.limitsType == QRhiSwapChainHdrInfo::LuminanceInNits
                          ? double(info.limits.luminanceInNits.maxLuminance)
                          : double(info.limits.colorComponentValue.maxColorComponentValue) * kDiffuseWhite.v;
        peak_from = "swapchain";
        // Qt's placeholder limits (Vulkan, OpenGL): 1 000 / 0 nits, not the display's.
        const bool placeholder = info.limitsType == QRhiSwapChainHdrInfo::LuminanceInNits &&
                                 info.limits.luminanceInNits.maxLuminance == 1000.0f &&
                                 info.limits.luminanceInNits.minLuminance == 0.0f;
        if (placeholder && format != QRhiSwapChain::SDR) {
            peak_from = "placeholder";
#ifdef Q_OS_WIN
            if (const QScreen* scr = w->screen()) {
                const qreal dpr = scr->devicePixelRatio();
                const QRect native(scr->geometry().topLeft() * dpr, scr->geometry().size() * dpr);
                if (auto p = dxgi_peak(native)) {
                    nits = *p;
                    peak_from = "DXGI";
                }
            }
#endif
        }
        DisplayTarget t = DisplayTarget::sdr();
        switch (format) {
            case QRhiSwapChain::HDRExtendedSrgbLinear:
                t = display_referred ? DisplayTarget::edr(nits, Primaries::Rec709) : DisplayTarget::scrgb(nits);
                break;
            case QRhiSwapChain::HDRExtendedDisplayP3Linear: t = DisplayTarget::edr(nits, Primaries::P3D65); break;
            case QRhiSwapChain::HDR10: t = DisplayTarget::hdr10(nits); break;
            default: break;
        }
        if (t.path != view.target.path || t.peak_nits != view.target.peak_nits || t.primaries != view.target.primaries) {
            view.target = t;
            view_dirty = true;
            notify();
        }
    }

    bool build_blit_pipeline() {
        blit_pipe.reset();
        if (!picture) return true;
        blit_srb_nearest.reset(rhi->newShaderResourceBindings());
        blit_srb_nearest->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
                                                     blit_ubo.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, picture.get(), nearest.get()),
        });
        blit_srb_trilinear.reset(rhi->newShaderResourceBindings());
        blit_srb_trilinear->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
                                                     blit_ubo.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, picture.get(), trilinear.get()),
        });
        if (!blit_srb_nearest->create() || !blit_srb_trilinear->create()) return false;
        blit_pipe.reset(rhi->newGraphicsPipeline());
        blit_pipe->setTopology(QRhiGraphicsPipeline::TriangleStrip);
        blit_pipe->setShaderStages({{QRhiShaderStage::Vertex, blit_vert}, {QRhiShaderStage::Fragment, blit_frag}});
        blit_pipe->setVertexInputLayout({});
        blit_pipe->setShaderResourceBindings(blit_srb_nearest.get());
        blit_pipe->setRenderPassDescriptor(sc_rp.get());
        return blit_pipe->create();
    }

    bool build_frame_resources(QSize size) {
        composite_pipe.reset();
        display_pipe.reset();
        blit_pipe.reset();
        composite_srb.reset();
        display_srb.reset();
        model_rt.reset();
        picture_rt.reset();
        model_rp.reset();
        picture_rp.reset();
        in_a.reset(rhi->newTexture(QRhiTexture::RGBA32F, size));
        in_b.reset(rhi->newTexture(QRhiTexture::RGBA32F, size));
        base_t.reset(rhi->newTexture(QRhiTexture::RGBA32F, size));
        model_t.reset(rhi->newTexture(QRhiTexture::RGBA32F, size, 1, QRhiTexture::RenderTarget));   // uploads too (set_composited)
        picture.reset(rhi->newTexture(QRhiTexture::RGBA16F, size, 1,
                                      QRhiTexture::RenderTarget | QRhiTexture::MipMapped |
                                          QRhiTexture::UsedWithGenerateMips));
        if (!in_a->create() || !in_b->create() || !base_t->create() || !model_t->create() || !picture->create())
            return false;
        model_rt.reset(rhi->newTextureRenderTarget({QRhiColorAttachment(model_t.get())}));
        model_rp.reset(model_rt->newCompatibleRenderPassDescriptor());
        model_rt->setRenderPassDescriptor(model_rp.get());
        picture_rt.reset(rhi->newTextureRenderTarget({QRhiColorAttachment(picture.get())}));
        picture_rp.reset(picture_rt->newCompatibleRenderPassDescriptor());
        picture_rt->setRenderPassDescriptor(picture_rp.get());
        if (!model_rt->create() || !picture_rt->create()) return false;

        composite_srb.reset(rhi->newShaderResourceBindings());
        composite_srb->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::FragmentStage, composite_ubo.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, in_a.get(), fetch.get()),
            QRhiShaderResourceBinding::sampledTexture(2, QRhiShaderResourceBinding::FragmentStage, in_b.get(), fetch.get()),
        });
        display_srb.reset(rhi->newShaderResourceBindings());
        display_srb->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::FragmentStage, view_ubo.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, model_t.get(), fetch.get()),
            QRhiShaderResourceBinding::sampledTexture(2, QRhiShaderResourceBinding::FragmentStage, base_t.get(), fetch.get()),
        });
        if (!composite_srb->create() || !display_srb->create()) return false;
        auto pipeline = [&](const QShader& frag, QRhiShaderResourceBindings* srb, QRhiRenderPassDescriptor* rp) {
            std::unique_ptr<QRhiGraphicsPipeline> p(rhi->newGraphicsPipeline());
            p->setShaderStages({{QRhiShaderStage::Vertex, vert}, {QRhiShaderStage::Fragment, frag}});
            p->setVertexInputLayout({});
            p->setShaderResourceBindings(srb);
            p->setRenderPassDescriptor(rp);
            return p->create() ? std::move(p) : nullptr;
        };
        composite_pipe = pipeline(composite_frag, composite_srb.get(), model_rp.get());
        display_pipe = pipeline(display_frag, display_srb.get(), picture_rp.get());
        if (!composite_pipe || !display_pipe) return false;
        frame_size = size;
        return build_blit_pipeline();
    }

    void resize_swapchain() {
        has_swapchain = sc->createOrResize();
        if (has_swapchain) update_target();
    }

    void release_swapchain() {
        if (has_swapchain) {
            has_swapchain = false;
            sc->destroy();
        }
    }

    // White as a graphic (the SDR white) in the swapchain's encoding.
    float graphic_white() const {
        if (view.target.path == OutputPath::SdrPqSimulation) return 1.0f;
        const float nits = float(kDiffuseWhite.v);
        return view.target.path == OutputPath::Hdr10 ? pq_oetf(nits) : nits / float(view.target.unit_nits);
    }

    // The surround as the swapchain wants it: a graphic at the SDR white.
    QColor surround() const {
        if (view.target.path == OutputPath::SdrPqSimulation) return QColor::fromRgbF(kSurroundCode, kSurroundCode, kSurroundCode);
        const float nits = float(kDiffuseWhite.v) * code_to_linear(kSurroundCode);
        const float v = view.target.path == OutputPath::Hdr10 ? pq_oetf(nits) : nits / float(view.target.unit_nits);
        return QColor::fromRgbF(v, v, v);
    }

    void render() {
        if (!has_swapchain) return;
        if (sc->currentPixelSize() != sc->surfacePixelSize()) resize_swapchain();
        if (!has_swapchain) return;
        ++updates;
        QRhi::FrameOpResult r = rhi->beginFrame(sc.get());
        if (r == QRhi::FrameOpSwapChainOutOfDate) {
            resize_swapchain();
            if (!has_swapchain) return;
            r = rhi->beginFrame(sc.get());
        }
        last_begin = int(r);
        if (r != QRhi::FrameOpSuccess) ++begin_failures;
        if (r == QRhi::FrameOpDeviceLost) {
            recover_device();
            return;
        }
        if (r != QRhi::FrameOpSuccess) {
            w->requestUpdate();
            return;
        }
        QRhiCommandBuffer* cb = sc->currentFrameCommandBuffer();
        QRhiResourceUpdateBatch* u = rhi->nextResourceUpdateBatch();

        if (has_frame && upload_dirty) {
            const QSize size(baseline.width(), baseline.height());
            if (size != frame_size && !build_frame_resources(size)) {
                has_frame = false;
            } else {
                auto upload = [&](QRhiTexture* t, const std::vector<float>& v) {
                    u->uploadTexture(t, QRhiTextureUploadDescription(QRhiTextureUploadEntry(
                        0, 0, QRhiTextureSubresourceUploadDescription(v.data(), quint32(v.size() * sizeof(float))))));
                };
                upload(base_t.get(), detail::rgba_of(baseline.buffer()));
                if (precomposited) {
                    upload(model_t.get(), detail::rgba_of(precomposite.buffer()));
                    composite_dirty = false;
                } else {
                    std::vector<float> a, b;
                    detail::interleave_inputs(frame.sdr, frame.fields, a, b);
                    upload(in_a.get(), a);
                    upload(in_b.get(), b);
                    composite_dirty = true;
                }
                view_dirty = true;
            }
            upload_dirty = false;
        }

        const QSize px = sc->currentPixelSize();
        if (has_frame && frame_size.isValid()) {
            if (composite_dirty && !precomposited) {
                const detail::CompositeUbo cu = detail::composite_ubo(frame.scalars, frame.model, composite);
                u->updateDynamicBuffer(composite_ubo.get(), 0, sizeof(cu), &cu);
                cb->beginPass(model_rt.get(), Qt::black, {1.0f, 0}, u);
                u = nullptr;
                cb->setGraphicsPipeline(composite_pipe.get());
                cb->setViewport({0, 0, float(frame_size.width()), float(frame_size.height())});
                cb->setShaderResources(composite_srb.get());
                cb->draw(3);
                cb->endPass();
                composite_dirty = false;
                view_dirty = true;
            }
            if (view_dirty) {
                if (!u) u = rhi->nextResourceUpdateBatch();
                const detail::ViewUbo vu = detail::view_ubo(effective_view(), frame_size.width());
                u->updateDynamicBuffer(view_ubo.get(), 0, sizeof(vu), &vu);
                cb->beginPass(picture_rt.get(), Qt::black, {1.0f, 0}, u);
                u = nullptr;
                cb->setGraphicsPipeline(display_pipe.get());
                cb->setViewport({0, 0, float(frame_size.width()), float(frame_size.height())});
                cb->setShaderResources(display_srb.get());
                cb->draw(3);
                QRhiResourceUpdateBatch* mips = rhi->nextResourceUpdateBatch();
                mips->generateMips(picture.get());
                cb->endPass(mips);
                view_dirty = false;
            }
        }

        if (!u) u = rhi->nextResourceUpdateBatch();
        const double dpr = w->devicePixelRatio();
        const PlacedRect rect = placed();
        BlitUbo bu{};
        const QMatrix4x4 corr = rhi->clipSpaceCorrMatrix();
        std::memcpy(bu.clip_corr, corr.constData(), sizeof(bu.clip_corr));
        bu.rect[0] = float(rect.left * dpr);
        bu.rect[1] = float(rect.top * dpr);
        bu.rect[2] = float((rect.left + rect.width) * dpr);
        bu.rect[3] = float((rect.top + rect.height) * dpr);
        bu.window[0] = float(px.width());
        bu.window[1] = float(px.height());
        bu.window[2] = rhi->isYUpInFramebuffer() ? 1.0f : 0.0f;
        bu.window[3] = graphic_white();
        bu.guides[0] = guides.action_safe ? 1.0f : 0.0f;
        bu.guides[1] = guides.title_safe ? 1.0f : 0.0f;
        bu.guides[2] = guides.centre ? 1.0f : 0.0f;
        bu.guides[3] = float(guides.aspect);
        u->updateDynamicBuffer(blit_ubo.get(), 0, sizeof(bu), &bu);

        cb->beginPass(sc->currentFrameRenderTarget(), surround(), {1.0f, 0}, u);
        if (has_frame && blit_pipe) {
            cb->setGraphicsPipeline(blit_pipe.get());
            cb->setViewport({0, 0, float(px.width()), float(px.height())});
            // Pixels stay pixels from 1:1 up; below it, trilinear over the mips.
            cb->setShaderResources(rect.scale * dpr >= 1.0 - 1e-9 ? blit_srb_nearest.get() : blit_srb_trilinear.get());
            cb->draw(4);
        }
        QRhiResourceUpdateBatch* after = nullptr;
        if (grab_cb && !grab_pending) {
            grab_pending = true;
            grab_rb = {};
            grab_y_up = rhi->isYUpInFramebuffer();
            grab_rb.completed = [this] { finish_grab(); };
            after = rhi->nextResourceUpdateBatch();
            after->readBackTexture(QRhiReadbackDescription(), &grab_rb);
        }
        cb->endPass(after);
        rhi->endFrame(sc.get());
        ++frames;
        // With frames in flight (D3D12, Vulkan, Metal) a readback completes
        // only when its frame slot comes round again, so a grab would report
        // an older frame, or never. Wait for it here: grabs are for tests and
        // Gate B, never the interactive path. On D3D12 even that is not
        // enough: the result is handed over at a later beginFrame (Gate B,
        // 24 Sep: every window check waited two frames that nothing asked
        // for), so while a readback is out the viewer keeps drawing.
        if (grab_pending) rhi->finish();
        if (grab_pending) w->requestUpdate();
    }

    void finish_grab() {
        grab_pending = false;
        auto cb = std::move(grab_cb);
        grab_cb = nullptr;
        if (!cb) return;
        Grab g;
        g.width = grab_rb.pixelSize.width();
        g.height = grab_rb.pixelSize.height();
        switch (grab_rb.format) {
            case QRhiTexture::RGBA16F: g.format = "RGBA16F"; g.bytes_per_pixel = 8; break;
            case QRhiTexture::RGB10A2: g.format = "RGB10A2"; break;
            case QRhiTexture::BGRA8: g.format = "BGRA8"; break;
            default: g.format = "RGBA8"; break;
        }
        const int row = g.height > 0 ? int(grab_rb.data.size()) / g.height : 0;
        g.bytes.resize(std::size_t(g.width) * std::size_t(g.height) * std::size_t(g.bytes_per_pixel));
        const bool y_up = grab_y_up;
        for (int y = 0; y < g.height; ++y) {
            const int src = y_up ? g.height - 1 - y : y;
            std::memcpy(g.bytes.data() + std::size_t(y) * std::size_t(g.width) * std::size_t(g.bytes_per_pixel),
                        grab_rb.data.constData() + std::size_t(src) * std::size_t(row),
                        std::size_t(g.width) * std::size_t(g.bytes_per_pixel));
        }
        cb(g);
    }

    void recover_device() {
        // Everything GPU-side goes; the host copies stay, and the next frame
        // rebuilds from them.
        release_all();
        initialized = false;
        if (init()) {
            initialized = true;
            resize_swapchain();
            if (has_frame) upload_dirty = true;
            frame_size = {};
            view_dirty = true;
        }
        w->requestUpdate();
    }

    void release_all() {
        // A readback still in flight completes while the QRhi is torn down:
        // it must not call back into a half-destroyed viewer.
        grab_rb.completed = nullptr;
        grab_cb = nullptr;
        grab_pending = false;
        blit_pipe.reset();
        composite_pipe.reset();
        display_pipe.reset();
        blit_srb_nearest.reset();
        blit_srb_trilinear.reset();
        composite_srb.reset();
        display_srb.reset();
        model_rt.reset();
        picture_rt.reset();
        model_rp.reset();
        picture_rp.reset();
        in_a.reset();
        in_b.reset();
        base_t.reset();
        model_t.reset();
        picture.reset();
        composite_ubo.reset();
        view_ubo.reset();
        blit_ubo.reset();
        nearest.reset();
        trilinear.reset();
        fetch.reset();
        release_swapchain();
        sc_rp.reset();
        sc.reset();
        rhi.reset();
        frame_size = {};
    }

    void notify() {
        if (status_cb) status_cb(status());
    }

    ViewerStatus status() const {
        ViewerStatus s;
        s.backend = rhi ? rhi->backendName() : "";
        const bool display_referred =
            sc && has_swapchain && sc->hdrInfo().luminanceBehavior == QRhiSwapChainHdrInfo::DisplayReferred;
        s.swapchain = swapchain_name(format, display_referred);
        s.target = view.target;
        // In device pixels: 100 % is one frame pixel per screen pixel.
        s.zoom_percent = int(std::floor(displayed_scale(viewport, viewer_size(), frame_view_size()) *
                                        w->devicePixelRatio() * 100.0 + 0.5));
        s.peak_from = peak_from;
        s.device_pixel_ratio = w->devicePixelRatio();
        s.has_frame = has_frame;
        s.wiping = view.wipe >= 0.0;
        s.updates = updates;
        s.frames = frames;
        s.begin_failures = begin_failures;
        s.last_begin = last_begin;
        s.grab_waiting = bool(grab_cb);
        s.grab_pending = grab_pending;
        return s;
    }

    void changed_view() {
        view_dirty = true;
        w->requestUpdate();
    }
    void moved() {
        notify();
        w->requestUpdate();
    }
};

ViewerWindow::ViewerWindow(GpuApi api, bool prefer_hdr) : d_(std::make_unique<Impl>()) {
    d_->w = this;
    d_->api = resolve(api);
    d_->prefer_hdr = prefer_hdr;
#if RUDRA_QT_VULKAN
    if (d_->api == GpuApi::Vulkan) {
        d_->vk = std::make_unique<QVulkanInstance>();
        d_->vk->setExtensions(QRhiVulkanInitParams::preferredInstanceExtensions());
        d_->vk->setApiVersion(d_->vk->supportedApiVersion());   // 0 is refused by the validation layer
        if (d_->vk->create()) setVulkanInstance(d_->vk.get());
        else d_->api = GpuApi::OpenGL;   // no Vulkan here: the fallback
    }
#else
    if (d_->api == GpuApi::Vulkan) d_->api = GpuApi::OpenGL;
#endif
    setSurfaceType(surface_for(d_->api));
#ifdef __APPLE__
    if (d_->api == GpuApi::OpenGL) setFormat(rhi_gl_format());
#endif
}

ViewerWindow::~ViewerWindow() {
    d_->release_all();
    // The platform window's Vulkan surface is destroyed with the window; the
    // instance it was made from is ours and goes with d_, so the surface goes
    // first. Without this ~QWindow destroys it against a freed instance.
    destroy();
}

void ViewerWindow::set_frame(ViewerFrame frame) {
    d_->baseline = corrected_baseline(frame.sdr, frame.model.corpus_ev, frame.scalars.curve_params);
    d_->frame = std::move(frame);
    d_->precomposited = false;
    d_->has_frame = true;
    d_->upload_dirty = true;
    d_->moved();
}

void ViewerWindow::set_composited(NetworkLinearImage model, NetworkLinearImage baseline) {
    d_->precomposite = std::move(model);
    d_->baseline = std::move(baseline);
    d_->precomposited = true;
    d_->has_frame = true;
    d_->upload_dirty = true;
    d_->moved();
}

void ViewerWindow::clear_frame() {
    d_->has_frame = false;
    d_->moved();
}

void ViewerWindow::set_composite(const CompositeParams& params) {
    d_->composite = params;
    d_->composite_dirty = true;
    requestUpdate();
}

void ViewerWindow::set_view(const ViewParams& params) {
    const DisplayTarget target = d_->view.target;
    d_->view = params;
    d_->view.target = target;
    d_->changed_view();
    d_->notify();
}

ViewParams ViewerWindow::view() const { return d_->view; }
const ViewportState& ViewerWindow::viewport() const { return d_->viewport; }

void ViewerWindow::set_viewport(const ViewportState& v) {
    d_->viewport = v;
    d_->moved();
}

void ViewerWindow::set_guides(const GuideOptions& g) {
    d_->guides = g;
    requestUpdate();
}

GuideOptions ViewerWindow::guides() const { return d_->guides; }

void ViewerWindow::zoom_fit() {
    rudra::zoom_fit(d_->viewport);
    d_->moved();
}

void ViewerWindow::zoom_actual() {
    // Actual pixels are the screen's: one frame pixel per device pixel, so a
    // 150 % display scale does not turn 1:1 into 1.5:1.
    d_->viewport = {1.0 / devicePixelRatio(), 0.0, 0.0};
    d_->moved();
}

ViewerStatus ViewerWindow::status() const { return d_->status(); }
void ViewerWindow::on_status(std::function<void(const ViewerStatus&)> cb) { d_->status_cb = std::move(cb); }

void ViewerWindow::set_input_enabled(bool on) {
    d_->input = on;
    if (!on) d_->wipe_dragging = d_->panning = d_->flip_held_key = d_->flip_held_mouse = false;
}

void ViewerWindow::grab(std::function<void(const Grab&)> done) {
    d_->grab_cb = std::move(done);
    requestUpdate();
}

void ViewerWindow::exposeEvent(QExposeEvent*) {
    if (isExposed() && !d_->initialized) {
        if (!d_->init()) return;
        d_->initialized = true;
        d_->resize_swapchain();
    }
    if (d_->initialized && isExposed() && !size().isEmpty()) d_->render();
}

bool ViewerWindow::event(QEvent* e) {
    switch (e->type()) {
        case QEvent::UpdateRequest:
            if (d_->initialized && isExposed()) d_->render();
            break;
        case QEvent::PlatformSurface:
            if (static_cast<QPlatformSurfaceEvent*>(e)->surfaceEventType() ==
                QPlatformSurfaceEvent::SurfaceAboutToBeDestroyed)
                d_->release_swapchain();
            break;
        case QEvent::Leave:
            if (d_->hover_cb) d_->hover_cb(Hover{});
            break;
        case QEvent::DragEnter:
        case QEvent::DragMove: {
            auto* de = static_cast<QDragMoveEvent*>(e);
            if (d_->drop_cb && de->mimeData()->hasUrls()) {
                de->acceptProposedAction();
                return true;
            }
            break;
        }
        case QEvent::Drop: {
            auto* de = static_cast<QDropEvent*>(e);
            if (!d_->drop_cb) break;
            QStringList paths;
            for (const auto& u : de->mimeData()->urls())
                if (u.isLocalFile()) paths << u.toLocalFile();
            de->acceptProposedAction();
            if (!paths.isEmpty()) d_->drop_cb(paths);
            return true;
        }
        case QEvent::Resize:
            // Fit follows the window; a zoom keeps its scale and pan.
            d_->notify();
            requestUpdate();
            break;
        default: break;
    }
    return QWindow::event(e);
}

void ViewerWindow::wheelEvent(QWheelEvent* e) {
    if (!d_->input) return;
    if (!d_->has_frame) return;
    const double notches = e->angleDelta().y() / 120.0;
    if (notches == 0.0) return;
    // One wheel notch is the Studio's step; a trackpad's fractions scale it.
    zoom_about(d_->viewport, d_->viewer_size(), d_->frame_view_size(), e->position().x(), e->position().y(),
               std::pow(kWheelStep, notches));
    d_->moved();
}

void ViewerWindow::mousePressEvent(QMouseEvent* e) {
    if (!d_->input) return;
    if (e->button() == Qt::MiddleButton && d_->has_frame) {
        d_->panning = true;
        d_->pan_from = e->position();
        if (!d_->viewport.scale) pan_by(d_->viewport, d_->viewer_size(), d_->frame_view_size(), 0, 0);
        d_->pan_x0 = d_->viewport.pan_x;
        d_->pan_y0 = d_->viewport.pan_y;
        return;
    }
    if (e->button() != Qt::LeftButton || !d_->has_frame) return;
    // While the wipe is up, left drag moves the seam; otherwise holding shows
    // the baseline. One gesture, one meaning (ui/app.js).
    if (d_->view.wipe >= 0.0) {
        d_->wipe_dragging = true;
        d_->view.wipe = wipe_at(d_->placed(), e->position().x());
        d_->changed_view();
        return;
    }
    d_->flip_held_mouse = true;
    d_->changed_view();
}

void ViewerWindow::on_hover(std::function<void(const Hover&)> cb) { d_->hover_cb = std::move(cb); }

void ViewerWindow::on_drop(std::function<void(const QStringList&)> cb) { d_->drop_cb = std::move(cb); }

void ViewerWindow::mouseMoveEvent(QMouseEvent* e) {
    if (!d_->input) return;
    if (!d_->panning && !d_->wipe_dragging) d_->hover(e->position(), e->modifiers() & Qt::AltModifier);
    if (d_->panning) {
        d_->viewport.pan_x = d_->pan_x0 + (e->position().x() - d_->pan_from.x());
        d_->viewport.pan_y = d_->pan_y0 + (e->position().y() - d_->pan_from.y());
        d_->moved();
        return;
    }
    if (d_->wipe_dragging) {
        d_->view.wipe = wipe_at(d_->placed(), e->position().x());
        d_->changed_view();
    }
}

void ViewerWindow::mouseReleaseEvent(QMouseEvent* e) {
    if (!d_->input) return;
    if (e->button() == Qt::MiddleButton) d_->panning = false;
    if (e->button() == Qt::LeftButton) {
        d_->wipe_dragging = false;
        if (d_->flip_held_mouse) {
            d_->flip_held_mouse = false;
            d_->changed_view();
        }
    }
}

void ViewerWindow::mouseDoubleClickEvent(QMouseEvent* e) {
    if (!d_->input) return;
    if (e->button() == Qt::LeftButton) zoom_fit();
}

void ViewerWindow::keyPressEvent(QKeyEvent* e) {
    if (!d_->input) return;
    switch (e->key()) {
        case Qt::Key_B:
            if (!e->isAutoRepeat()) {
                d_->flip_held_key = true;
                d_->changed_view();
            }
            return;
        case Qt::Key_W:
            d_->view.wipe = d_->view.wipe >= 0.0 ? -1.0 : 0.5;
            d_->flip_held_key = d_->flip_held_mouse = false;
            d_->changed_view();
            d_->notify();
            return;
        case Qt::Key_Left:
        case Qt::Key_Right:
            if (d_->view.wipe >= 0.0) {
                const double step = (e->modifiers() & Qt::ShiftModifier) ? 0.01 : 0.05;
                d_->view.wipe = std::clamp(d_->view.wipe + (e->key() == Qt::Key_Right ? step : -step), 0.0, 1.0);
                d_->changed_view();
                return;
            }
            break;
        case Qt::Key_Escape:
            if (d_->view.wipe >= 0.0) {
                d_->view.wipe = -1.0;
                d_->changed_view();
                d_->notify();
                return;
            }
            break;
        default: break;
    }
    QWindow::keyPressEvent(e);
}

void ViewerWindow::keyReleaseEvent(QKeyEvent* e) {
    if (!d_->input) return;
    if (e->key() == Qt::Key_B && !e->isAutoRepeat()) {
        d_->flip_held_key = false;
        d_->changed_view();
        return;
    }
    QWindow::keyReleaseEvent(e);
}

}  // namespace rudra
