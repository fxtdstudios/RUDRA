#include "rudra/render/gpu_composite.hpp"

#include <QFile>
#include <QFloat16>
#include <QGuiApplication>
#include <QOffscreenSurface>
#include <rhi/qrhi.h>

#if QT_CONFIG(vulkan)
#include <QVulkanInstance>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

namespace rudra {
namespace {

QShader load_shader(const QString& name) {
    QFile f(name);
    return f.open(QIODevice::ReadOnly) ? QShader::fromSerialized(f.readAll()) : QShader();
}

// std140 layout of the Composite block in shaders/composite.frag.
struct Ubo {
    float model[4];
    float control[4];
    float counts[4];
    float curve[36];
    float bands[32];
};
static_assert(sizeof(Ubo) == (3 * 4 + 36 + 32) * sizeof(float));

// std140 layout of the View block in shaders/display.frag.
struct ViewUbo {
    float view[4];
    float extra[4];
};

// A composite target as an RGBA32F upload: rgb, alpha 1.
std::vector<float> rgba_of(const PlanarBuffer& rgb) {
    const std::size_t n = rgb.plane_size();
    std::vector<float> out(n * 4, 1.0f);
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) out[i * 4 + std::size_t(c)] = rgb.plane(c)[i];
    return out;
}

class RhiCompositor final : public GpuCompositor {
public:
    ~RhiCompositor() override {
        sampler_.reset();
        vbuf_.reset();
        ubuf_.reset();
        rhi_.reset();
    }

    static Result<std::unique_ptr<GpuCompositor>> make(GpuApi api) {
        auto self = std::unique_ptr<RhiCompositor>(new RhiCompositor());
        if (auto r = self->init(api); !r) return r.error();
        return std::unique_ptr<GpuCompositor>(std::move(self));
    }

    GpuCompositorInfo info() const override { return info_; }

    bool supports(GpuPrecision p) const override {
        return rhi_->isTextureFormatSupported(p == GpuPrecision::Fp32 ? QRhiTexture::RGBA32F : QRhiTexture::RGBA16F);
    }

    Result<PlanarBuffer> composite(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                                   const ModelConstants& model, const CompositeParams& params,
                                   GpuPrecision precision) override {
        const int w = sdr.width(), h = sdr.height();
        if (!supports(precision))
            return make_error(ErrorCode::Unsupported, "This GPU cannot render to that float format.");
        if (scalars.curve_params.size() > 36)
            return make_error(ErrorCode::Unsupported, "The curve has more parameters than the shader holds (36).");

        // Inputs: two RGBA32F textures, interleaved on the CPU.
        std::vector<float> a(std::size_t(w) * h * 4), b(std::size_t(w) * h * 4);
        const std::size_t n = std::size_t(w) * h;
        for (std::size_t i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c) {
                a[i * 4 + c] = sdr.buffer().plane(c)[i];
                b[i * 4 + c] = fields.residual.plane(c)[i];
            }
            a[i * 4 + 3] = fields.shadow.plane(0)[i];
            b[i * 4 + 3] = fields.highlight.plane(0)[i];
        }
        std::unique_ptr<QRhiTexture> ta(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        std::unique_ptr<QRhiTexture> tb(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        const QRhiTexture::Format out_fmt = precision == GpuPrecision::Fp32 ? QRhiTexture::RGBA32F : QRhiTexture::RGBA16F;
        std::unique_ptr<QRhiTexture> out(rhi_->newTexture(out_fmt, QSize(w, h), 1,
                                                          QRhiTexture::RenderTarget | QRhiTexture::UsedAsTransferSource));
        if (!ta->create() || !tb->create() || !out->create())
            return make_error(ErrorCode::BackendError, "GPU texture creation failed.");
        std::unique_ptr<QRhiTextureRenderTarget> rt(rhi_->newTextureRenderTarget({QRhiColorAttachment(out.get())}));
        std::unique_ptr<QRhiRenderPassDescriptor> rp(rt->newCompatibleRenderPassDescriptor());
        rt->setRenderPassDescriptor(rp.get());
        if (!rt->create()) return make_error(ErrorCode::BackendError, "GPU render target creation failed.");

        std::unique_ptr<QRhiShaderResourceBindings> srb(rhi_->newShaderResourceBindings());
        srb->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::FragmentStage, ubuf_.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, ta.get(), sampler_.get()),
            QRhiShaderResourceBinding::sampledTexture(2, QRhiShaderResourceBinding::FragmentStage, tb.get(), sampler_.get()),
        });
        if (!srb->create()) return make_error(ErrorCode::BackendError, "GPU resource bindings failed.");
        // Per call: this is the parity and batch path. The viewer (Phase 2)
        // keeps its pipeline and textures alive across frames.
        std::unique_ptr<QRhiGraphicsPipeline> pipe(rhi_->newGraphicsPipeline());
        pipe->setShaderStages({{QRhiShaderStage::Vertex, vert_}, {QRhiShaderStage::Fragment, frag_}});
        pipe->setVertexInputLayout({});
        pipe->setShaderResourceBindings(srb.get());
        pipe->setRenderPassDescriptor(rp.get());
        if (!pipe->create()) return make_error(ErrorCode::BackendError, "GPU pipeline creation failed.");

        Ubo u{};
        u.model[0] = model.log_scale;
        u.model[1] = model.max_hdr;
        u.model[2] = static_cast<float>(std::exp2(-static_cast<double>(model.corpus_ev)) * (203.0 / 10000.0));
        u.model[3] = params.strength;
        u.control[0] = float(int(params.mode));
        u.control[1] = params.preserve_outside ? 1.0f : 0.0f;
        u.control[2] = scalars.shadow_weight;
        u.control[3] = static_cast<float>(params.region_softness_stops);
        const std::size_t np = scalars.curve_params.size();
        u.counts[0] = np >= 3 ? float(np - 1) : 0.0f;
        std::copy(scalars.curve_params.begin(), scalars.curve_params.end(), u.curve);
        int bands = 0;
        for (const auto& band : params.regions) {
            if (band.ev == 0.0 || bands == 8) continue;
            u.bands[bands * 4 + 0] = static_cast<float>(std::log2(band.low_nits));
            u.bands[bands * 4 + 1] = static_cast<float>(std::log2(band.high_nits));
            u.bands[bands * 4 + 2] = static_cast<float>(band.ev);
            ++bands;
        }
        u.counts[1] = float(bands);

        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess)
            return make_error(ErrorCode::BackendError, "GPU frame could not start.");
        QRhiResourceUpdateBatch* up = rhi_->nextResourceUpdateBatch();
        up->updateDynamicBuffer(ubuf_.get(), 0, sizeof(Ubo), &u);
        up->uploadTexture(ta.get(), QRhiTextureUploadDescription(QRhiTextureUploadEntry(
            0, 0, QRhiTextureSubresourceUploadDescription(a.data(), quint32(a.size() * sizeof(float))))));
        up->uploadTexture(tb.get(), QRhiTextureUploadDescription(QRhiTextureUploadEntry(
            0, 0, QRhiTextureSubresourceUploadDescription(b.data(), quint32(b.size() * sizeof(float))))));
        cb->beginPass(rt.get(), Qt::black, {1.0f, 0}, up);
        cb->setGraphicsPipeline(pipe.get());
        cb->setViewport({0, 0, float(w), float(h)});
        cb->setShaderResources(srb.get());
        cb->draw(3);
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* down = rhi_->nextResourceUpdateBatch();
        down->readBackTexture(QRhiReadbackDescription(out.get()), &rb);
        cb->endPass(down);
        rhi_->endOffscreenFrame();   // waits: the readback is complete on return

        const int bpp = precision == GpuPrecision::Fp32 ? 16 : 8;
        if (rb.data.size() < qsizetype(std::size_t(w) * h * bpp))
            return make_error(ErrorCode::BackendError, "GPU readback returned too little data.");
        const std::size_t row = std::size_t(rb.data.size()) / std::size_t(h);
        PlanarBuffer o(3, h, w);
        for (int yy = 0; yy < h; ++yy) {
            const char* r = rb.data.constData() + std::size_t(yy) * row;
            for (int xx = 0; xx < w; ++xx)
                for (int c = 0; c < 3; ++c) {
                    float v;
                    if (bpp == 16) {
                        std::memcpy(&v, r + std::size_t(xx) * 16 + std::size_t(c) * 4, 4);
                    } else {
                        qfloat16 hv;
                        std::memcpy(&hv, r + std::size_t(xx) * 8 + std::size_t(c) * 2, 2);
                        v = float(hv);
                    }
                    o.at(c, yy, xx) = v;
                }
        }
        return o;
    }

    Result<GpuTiming> benchmark(int w, int h, int iterations) override {
        if (!supports(GpuPrecision::Fp16))
            return make_error(ErrorCode::Unsupported, "This GPU cannot render to RGBA16F.");
        std::vector<float> a(std::size_t(w) * h * 4), b(std::size_t(w) * h * 4);
        for (std::size_t i = 0; i < std::size_t(w) * h; ++i) {
            const float v = float(i % 1021) / 1020.0f;
            a[i * 4 + 0] = v; a[i * 4 + 1] = 1.0f - v; a[i * 4 + 2] = v * v; a[i * 4 + 3] = 0.3f;
            b[i * 4 + 0] = 0.2f; b[i * 4 + 1] = -0.1f; b[i * 4 + 2] = 0.05f; b[i * 4 + 3] = 0.6f;
        }
        std::unique_ptr<QRhiTexture> ta(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        std::unique_ptr<QRhiTexture> tb(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        std::unique_ptr<QRhiTexture> out(rhi_->newTexture(QRhiTexture::RGBA16F, QSize(w, h), 1, QRhiTexture::RenderTarget));
        if (!ta->create() || !tb->create() || !out->create())
            return make_error(ErrorCode::BackendError, "GPU texture creation failed.");
        std::unique_ptr<QRhiTextureRenderTarget> rt(rhi_->newTextureRenderTarget({QRhiColorAttachment(out.get())}));
        std::unique_ptr<QRhiRenderPassDescriptor> rp(rt->newCompatibleRenderPassDescriptor());
        rt->setRenderPassDescriptor(rp.get());
        if (!rt->create()) return make_error(ErrorCode::BackendError, "GPU render target creation failed.");
        std::unique_ptr<QRhiShaderResourceBindings> srb(rhi_->newShaderResourceBindings());
        srb->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::FragmentStage, ubuf_.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, ta.get(), sampler_.get()),
            QRhiShaderResourceBinding::sampledTexture(2, QRhiShaderResourceBinding::FragmentStage, tb.get(), sampler_.get()),
        });
        if (!srb->create()) return make_error(ErrorCode::BackendError, "GPU resource bindings failed.");
        std::unique_ptr<QRhiGraphicsPipeline> pipe(rhi_->newGraphicsPipeline());
        pipe->setShaderStages({{QRhiShaderStage::Vertex, vert_}, {QRhiShaderStage::Fragment, frag_}});
        pipe->setVertexInputLayout({});
        pipe->setShaderResourceBindings(srb.get());
        pipe->setRenderPassDescriptor(rp.get());
        if (!pipe->create()) return make_error(ErrorCode::BackendError, "GPU pipeline creation failed.");

        // A typical grade: all modes on, preserve, three graded bands.
        Ubo u{};
        u.model[0] = 16.0f; u.model[1] = 4.0f; u.model[2] = 0.0406f; u.model[3] = 1.0f;
        u.control[1] = 1.0f; u.control[2] = 1.0f; u.control[3] = 1.0f;
        u.counts[0] = 8.0f; u.counts[1] = 3.0f;
        const float bands[3][3] = {{8.64f, 10.97f, 0.5f}, {10.97f, 12.97f, -0.3f}, {-4.32f, 3.58f, 0.4f}};
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) u.bands[i * 4 + k] = bands[i][k];

        using clock = std::chrono::steady_clock;
        std::vector<double> gpu, wall;
        for (int i = 0; i <= iterations; ++i) {
            QRhiCommandBuffer* cb = nullptr;
            const auto t0 = clock::now();
            if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess)
                return make_error(ErrorCode::BackendError, "GPU frame could not start.");
            QRhiResourceUpdateBatch* up = rhi_->nextResourceUpdateBatch();
            u.control[0] = float(i % 2);   // a slider move each frame: the UBO changes, nothing else
            up->updateDynamicBuffer(ubuf_.get(), 0, sizeof(Ubo), &u);
            if (i == 0) {   // fields arrive once per inference, not per slider move
                up->uploadTexture(ta.get(), QRhiTextureUploadDescription(QRhiTextureUploadEntry(
                    0, 0, QRhiTextureSubresourceUploadDescription(a.data(), quint32(a.size() * sizeof(float))))));
                up->uploadTexture(tb.get(), QRhiTextureUploadDescription(QRhiTextureUploadEntry(
                    0, 0, QRhiTextureSubresourceUploadDescription(b.data(), quint32(b.size() * sizeof(float))))));
            }
            cb->beginPass(rt.get(), Qt::black, {1.0f, 0}, up);
            cb->setGraphicsPipeline(pipe.get());
            cb->setViewport({0, 0, float(w), float(h)});
            cb->setShaderResources(srb.get());
            cb->draw(3);
            cb->endPass();
            rhi_->endOffscreenFrame();
            const auto t1 = clock::now();
            if (i == 0) continue;   // warm-up, and the upload
            wall.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            const double g = cb->lastCompletedGpuTime();
            if (g > 0.0) gpu.push_back(g * 1000.0);
        }
        auto median = [](std::vector<double> v) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        GpuTiming t;
        t.wall_ms = median(wall);
        t.gpu_ms = median(gpu);
        t.has_gpu_timestamps = !gpu.empty();
        return t;
    }

    Result<Rgb8Image> view(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                           const ViewParams& params) override {
        const int w = model.width(), h = model.height();
        if (baseline.width() != w || baseline.height() != h)
            return make_error(ErrorCode::InvalidArgument, "The two composite targets differ in size.");
        const std::vector<float> a = rgba_of(model.buffer()), b = rgba_of(baseline.buffer());
        std::unique_ptr<QRhiTexture> ta(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        std::unique_ptr<QRhiTexture> tb(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        std::unique_ptr<QRhiTexture> out(rhi_->newTexture(QRhiTexture::RGBA8, QSize(w, h), 1,
                                                          QRhiTexture::RenderTarget | QRhiTexture::UsedAsTransferSource));
        if (!ta->create() || !tb->create() || !out->create())
            return make_error(ErrorCode::BackendError, "GPU texture creation failed.");
        std::unique_ptr<QRhiTextureRenderTarget> rt(rhi_->newTextureRenderTarget({QRhiColorAttachment(out.get())}));
        std::unique_ptr<QRhiRenderPassDescriptor> rp(rt->newCompatibleRenderPassDescriptor());
        rt->setRenderPassDescriptor(rp.get());
        if (!rt->create()) return make_error(ErrorCode::BackendError, "GPU render target creation failed.");
        std::unique_ptr<QRhiShaderResourceBindings> srb(rhi_->newShaderResourceBindings());
        srb->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::FragmentStage, vbuf_.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, ta.get(), sampler_.get()),
            QRhiShaderResourceBinding::sampledTexture(2, QRhiShaderResourceBinding::FragmentStage, tb.get(), sampler_.get()),
        });
        if (!srb->create()) return make_error(ErrorCode::BackendError, "GPU resource bindings failed.");
        std::unique_ptr<QRhiGraphicsPipeline> pipe(rhi_->newGraphicsPipeline());
        pipe->setShaderStages({{QRhiShaderStage::Vertex, vert_}, {QRhiShaderStage::Fragment, display_}});
        pipe->setVertexInputLayout({});
        pipe->setShaderResourceBindings(srb.get());
        pipe->setRenderPassDescriptor(rp.get());
        if (!pipe->create()) return make_error(ErrorCode::BackendError, "GPU pipeline creation failed.");

        // The uniforms as core/view.cpp rounds them to fp32.
        ViewUbo u{};
        u.view[0] = float(int(params.mode));
        u.view[1] = float(10000.0 / std::max(params.display_nits, 1e-3));
        u.view[2] = params.wipe >= 0.0 ? float(std::clamp(params.wipe, 0.0, 1.0)) : -1.0f;
        u.view[3] = float(params.wipe_half_width);
        u.extra[0] = std::log2(1.0f + float(std::max(params.diff_gain, 1.0)));
        u.extra[1] = float(w);
        u.extra[2] = params.show == ViewSource::Baseline ? 1.0f : 0.0f;

        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess)
            return make_error(ErrorCode::BackendError, "GPU frame could not start.");
        QRhiResourceUpdateBatch* up = rhi_->nextResourceUpdateBatch();
        up->updateDynamicBuffer(vbuf_.get(), 0, sizeof(ViewUbo), &u);
        up->uploadTexture(ta.get(), QRhiTextureUploadDescription(QRhiTextureUploadEntry(
            0, 0, QRhiTextureSubresourceUploadDescription(a.data(), quint32(a.size() * sizeof(float))))));
        up->uploadTexture(tb.get(), QRhiTextureUploadDescription(QRhiTextureUploadEntry(
            0, 0, QRhiTextureSubresourceUploadDescription(b.data(), quint32(b.size() * sizeof(float))))));
        cb->beginPass(rt.get(), Qt::black, {1.0f, 0}, up);
        cb->setGraphicsPipeline(pipe.get());
        cb->setViewport({0, 0, float(w), float(h)});
        cb->setShaderResources(srb.get());
        cb->draw(3);
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* down = rhi_->nextResourceUpdateBatch();
        down->readBackTexture(QRhiReadbackDescription(out.get()), &rb);
        cb->endPass(down);
        rhi_->endOffscreenFrame();

        if (rb.data.size() < qsizetype(std::size_t(w) * h * 4))
            return make_error(ErrorCode::BackendError, "GPU readback returned too little data.");
        const std::size_t row = std::size_t(rb.data.size()) / std::size_t(h);
        Rgb8Image o{w, h, std::vector<std::uint8_t>(std::size_t(w) * h * 3)};
        for (int yy = 0; yy < h; ++yy) {
            const auto* r = reinterpret_cast<const std::uint8_t*>(rb.data.constData()) + std::size_t(yy) * row;
            for (int xx = 0; xx < w; ++xx)
                for (int c = 0; c < 3; ++c)
                    o.rgb[(std::size_t(yy) * w + xx) * 3 + std::size_t(c)] = r[std::size_t(xx) * 4 + std::size_t(c)];
        }
        return o;
    }

private:
    Result<void> init(GpuApi api) {
        if (!QGuiApplication::instance())
            return make_error(ErrorCode::Unsupported, "The GPU compositor needs a QGuiApplication.");
        if (api == GpuApi::Auto) {
#if defined(Q_OS_WIN)
            api = GpuApi::D3D12;
#elif defined(Q_OS_MACOS)
            api = GpuApi::Metal;
#else
            api = GpuApi::Vulkan;
#endif
        }
        switch (api) {
            case GpuApi::D3D12: {
#ifdef Q_OS_WIN
                QRhiD3D12InitParams p;
                rhi_.reset(QRhi::create(QRhi::D3D12, &p, QRhi::EnableTimestamps));
#endif
                break;
            }
            case GpuApi::D3D11: {
#ifdef Q_OS_WIN
                QRhiD3D11InitParams p;
                rhi_.reset(QRhi::create(QRhi::D3D11, &p, QRhi::EnableTimestamps));
#endif
                break;
            }
            case GpuApi::Metal: {
#if QT_CONFIG(metal)
                QRhiMetalInitParams p;
                rhi_.reset(QRhi::create(QRhi::Metal, &p, QRhi::EnableTimestamps));
#endif
                break;
            }
            case GpuApi::Vulkan: {
#if QT_CONFIG(vulkan)
                vk_ = std::make_unique<QVulkanInstance>();
                vk_->setExtensions(QRhiVulkanInitParams::preferredInstanceExtensions());
                if (vk_->create()) {
                    QRhiVulkanInitParams p;
                    p.inst = vk_.get();
                    rhi_.reset(QRhi::create(QRhi::Vulkan, &p, QRhi::EnableTimestamps));
                }
#endif
                break;
            }
            case GpuApi::OpenGL: {
                fallback_.reset(QRhiGles2InitParams::newFallbackSurface());
                QRhiGles2InitParams p;
                p.fallbackSurface = fallback_.get();
                rhi_.reset(QRhi::create(QRhi::OpenGLES2, &p, QRhi::EnableTimestamps));
                break;
            }
            case GpuApi::Auto: break;
        }
        if (!rhi_) return make_error(ErrorCode::Unsupported, "No GPU device for that API.", to_string(api));
        info_.backend = rhi_->backendName();
        info_.device = rhi_->driverInfo().deviceName.toStdString();

        vert_ = load_shader(":/rudra/shaders/fullscreen.vert.qsb");
        frag_ = load_shader(":/rudra/shaders/composite.frag.qsb");
        display_ = load_shader(":/rudra/shaders/display.frag.qsb");
        if (!vert_.isValid() || !frag_.isValid() || !display_.isValid())
            return make_error(ErrorCode::NotFound, "The composite shaders are missing from the build.");
        ubuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Ubo)));
        vbuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(ViewUbo)));
        sampler_.reset(rhi_->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None,
                                        QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        if (!ubuf_->create() || !vbuf_->create() || !sampler_->create())
            return make_error(ErrorCode::BackendError, "GPU buffer creation failed.");
        return {};
    }

#if QT_CONFIG(vulkan)
    std::unique_ptr<QVulkanInstance> vk_;
#endif
    std::unique_ptr<QOffscreenSurface> fallback_;
    std::unique_ptr<QRhi> rhi_;
    std::unique_ptr<QRhiBuffer> ubuf_;
    std::unique_ptr<QRhiBuffer> vbuf_;
    std::unique_ptr<QRhiSampler> sampler_;
    QShader vert_, frag_, display_;
    GpuCompositorInfo info_;
};

}  // namespace

const char* to_string(GpuApi a) noexcept {
    switch (a) {
        case GpuApi::Auto: return "auto";
        case GpuApi::D3D12: return "d3d12";
        case GpuApi::D3D11: return "d3d11";
        case GpuApi::Metal: return "metal";
        case GpuApi::Vulkan: return "vulkan";
        case GpuApi::OpenGL: return "gl";
    }
    return "?";
}

Result<std::unique_ptr<GpuCompositor>> GpuCompositor::create(GpuApi api) { return RhiCompositor::make(api); }

}  // namespace rudra
