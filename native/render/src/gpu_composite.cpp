#include "rudra/render/gpu_composite.hpp"

#include "rudra/core/gamut.hpp"
#include "passes.hpp"

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

using detail::CompositeUbo;
using detail::ReduceUbo;
using detail::ViewUbo;
using detail::load_shader;
using detail::rgba_of;

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
        std::vector<float> a, b;
        detail::interleave_inputs(sdr, fields, a, b);
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

        const CompositeUbo u = detail::composite_ubo(scalars, model, params);

        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess)
            return make_error(ErrorCode::BackendError, "GPU frame could not start.");
        QRhiResourceUpdateBatch* up = rhi_->nextResourceUpdateBatch();
        up->updateDynamicBuffer(ubuf_.get(), 0, sizeof(CompositeUbo), &u);
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
        CompositeUbo u{};
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
            up->updateDynamicBuffer(ubuf_.get(), 0, sizeof(CompositeUbo), &u);
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

    Result<GpuTiming> benchmark_view(int w, int h, int iterations) override {
        if (!supports(GpuPrecision::Fp32) || !supports(GpuPrecision::Fp16))
            return make_error(ErrorCode::Unsupported, "This GPU cannot render to RGBA32F and RGBA16F.");
        std::vector<float> a(std::size_t(w) * h * 4), b(std::size_t(w) * h * 4), base(std::size_t(w) * h * 4, 1.0f);
        for (std::size_t i = 0; i < std::size_t(w) * h; ++i) {
            const float v = float(i % 1021) / 1020.0f;
            a[i * 4 + 0] = v; a[i * 4 + 1] = 1.0f - v; a[i * 4 + 2] = v * v; a[i * 4 + 3] = 0.3f;
            b[i * 4 + 0] = 0.2f; b[i * 4 + 1] = -0.1f; b[i * 4 + 2] = 0.05f; b[i * 4 + 3] = 0.6f;
            base[i * 4 + 0] = base[i * 4 + 1] = base[i * 4 + 2] = v * 0.02f;
        }
        auto tex = [&](QRhiTexture::Format f, QRhiTexture::Flags fl = {}) {
            return std::unique_ptr<QRhiTexture>(rhi_->newTexture(f, QSize(w, h), 1, fl));
        };
        auto ta = tex(QRhiTexture::RGBA32F), tb = tex(QRhiTexture::RGBA32F), tbase = tex(QRhiTexture::RGBA32F);
        auto model = tex(QRhiTexture::RGBA32F, QRhiTexture::RenderTarget);
        auto pic = tex(QRhiTexture::RGBA16F, QRhiTexture::RenderTarget);
        if (!ta->create() || !tb->create() || !tbase->create() || !model->create() || !pic->create())
            return make_error(ErrorCode::BackendError, "GPU texture creation failed.");
        std::unique_ptr<QRhiTextureRenderTarget> mrt(rhi_->newTextureRenderTarget({QRhiColorAttachment(model.get())}));
        std::unique_ptr<QRhiRenderPassDescriptor> mrp(mrt->newCompatibleRenderPassDescriptor());
        mrt->setRenderPassDescriptor(mrp.get());
        std::unique_ptr<QRhiTextureRenderTarget> prt(rhi_->newTextureRenderTarget({QRhiColorAttachment(pic.get())}));
        std::unique_ptr<QRhiRenderPassDescriptor> prp(prt->newCompatibleRenderPassDescriptor());
        prt->setRenderPassDescriptor(prp.get());
        if (!mrt->create() || !prt->create()) return make_error(ErrorCode::BackendError, "GPU render target creation failed.");
        std::unique_ptr<QRhiShaderResourceBindings> csrb(rhi_->newShaderResourceBindings()), dsrb(rhi_->newShaderResourceBindings());
        csrb->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::FragmentStage, ubuf_.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, ta.get(), sampler_.get()),
            QRhiShaderResourceBinding::sampledTexture(2, QRhiShaderResourceBinding::FragmentStage, tb.get(), sampler_.get()),
        });
        dsrb->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::FragmentStage, vbuf_.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, model.get(), sampler_.get()),
            QRhiShaderResourceBinding::sampledTexture(2, QRhiShaderResourceBinding::FragmentStage, tbase.get(), sampler_.get()),
        });
        if (!csrb->create() || !dsrb->create()) return make_error(ErrorCode::BackendError, "GPU resource bindings failed.");
        auto pipeline = [&](const QShader& frag, QRhiShaderResourceBindings* srb, QRhiRenderPassDescriptor* rp) {
            std::unique_ptr<QRhiGraphicsPipeline> p(rhi_->newGraphicsPipeline());
            p->setShaderStages({{QRhiShaderStage::Vertex, vert_}, {QRhiShaderStage::Fragment, frag}});
            p->setVertexInputLayout({});
            p->setShaderResourceBindings(srb);
            p->setRenderPassDescriptor(rp);
            return p->create() ? std::move(p) : nullptr;
        };
        auto cpipe = pipeline(frag_, csrb.get(), mrp.get());
        auto dpipe = pipeline(display_, dsrb.get(), prp.get());
        if (!cpipe || !dpipe) return make_error(ErrorCode::BackendError, "GPU pipeline creation failed.");

        CompositeParams cp;
        cp.regions = {{400.0, 2000.0, 0.5}, {2000.0, 8000.0, -0.3}, {0.05, 12.0, 0.4}};
        FrameScalars sc;
        const ModelConstants mc{16.0f, 4.0f, -1.0f};
        using clock = std::chrono::steady_clock;
        std::vector<double> gpu, wall;
        for (int i = 0; i <= iterations; ++i) {
            QRhiCommandBuffer* cb = nullptr;
            const auto t0 = clock::now();
            if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess)
                return make_error(ErrorCode::BackendError, "GPU frame could not start.");
            QRhiResourceUpdateBatch* up = rhi_->nextResourceUpdateBatch();
            cp.strength = 0.9f + 0.1f * float(i % 2);   // a slider move: both UBOs change, nothing else
            const CompositeUbo cu = detail::composite_ubo(sc, mc, cp);
            ViewParams vp = view_params(ViewMode::Image, 1000.0);
            vp.target = DisplayTarget::scrgb(1000.0);
            const ViewUbo vu = detail::view_ubo(vp, w);
            up->updateDynamicBuffer(ubuf_.get(), 0, sizeof(cu), &cu);
            up->updateDynamicBuffer(vbuf_.get(), 0, sizeof(vu), &vu);
            if (i == 0) {
                for (auto [t, v] : {std::pair{ta.get(), &a}, std::pair{tb.get(), &b}, std::pair{tbase.get(), &base}})
                    up->uploadTexture(t, QRhiTextureUploadDescription(QRhiTextureUploadEntry(
                        0, 0, QRhiTextureSubresourceUploadDescription(v->data(), quint32(v->size() * sizeof(float))))));
            }
            cb->beginPass(mrt.get(), Qt::black, {1.0f, 0}, up);
            cb->setGraphicsPipeline(cpipe.get());
            cb->setViewport({0, 0, float(w), float(h)});
            cb->setShaderResources(csrb.get());
            cb->draw(3);
            cb->endPass();
            cb->beginPass(prt.get(), Qt::black, {1.0f, 0});
            cb->setGraphicsPipeline(dpipe.get());
            cb->setViewport({0, 0, float(w), float(h)});
            cb->setShaderResources(dsrb.get());
            cb->draw(3);
            cb->endPass();
            rhi_->endOffscreenFrame();
            const auto t1 = clock::now();
            if (i == 0) continue;
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
        if (params.target.path != OutputPath::SdrPqSimulation)
            return make_error(ErrorCode::InvalidArgument, "The 8-bit view is the SDR path; HDR paths read back as floats.");
        QByteArray data;
        if (auto r = view_pass(model, baseline, params, QRhiTexture::RGBA8, data); !r) return r.error();
        const int w = model.width(), h = model.height();
        const std::size_t row = std::size_t(data.size()) / std::size_t(h);
        Rgb8Image o{w, h, std::vector<std::uint8_t>(std::size_t(w) * h * 3)};
        for (int yy = 0; yy < h; ++yy) {
            const auto* r = reinterpret_cast<const std::uint8_t*>(data.constData()) + std::size_t(yy) * row;
            for (int xx = 0; xx < w; ++xx)
                for (int c = 0; c < 3; ++c)
                    o.rgb[(std::size_t(yy) * w + xx) * 3 + std::size_t(c)] = r[std::size_t(xx) * 4 + std::size_t(c)];
        }
        return o;
    }

    Result<PlanarBuffer> view_values(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                                     const ViewParams& params, GpuPrecision precision) override {
        if (!supports(precision))
            return make_error(ErrorCode::Unsupported, "This GPU cannot render to that float format.");
        const bool f32 = precision == GpuPrecision::Fp32;
        QByteArray data;
        if (auto r = view_pass(model, baseline, params, f32 ? QRhiTexture::RGBA32F : QRhiTexture::RGBA16F, data); !r)
            return r.error();
        const int w = model.width(), h = model.height();
        const std::size_t row = std::size_t(data.size()) / std::size_t(h);
        PlanarBuffer o(3, h, w);
        for (int yy = 0; yy < h; ++yy) {
            const char* r = data.constData() + std::size_t(yy) * row;
            for (int xx = 0; xx < w; ++xx)
                for (int c = 0; c < 3; ++c) {
                    float v;
                    if (f32) {
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

    Result<Reductions> reduce(const NetworkLinearImage& image) override {
        if (!supports(GpuPrecision::Fp32))
            return make_error(ErrorCode::Unsupported, "This GPU cannot render to RGBA32F.");
        const int w = image.width(), h = image.height();
        const std::vector<float> a = rgba_of(image.buffer());
        std::unique_ptr<QRhiTexture> src(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        if (!src->create()) return make_error(ErrorCode::BackendError, "GPU texture creation failed.");

        // Both ladders (max, sum), one pass per level each, one UBO per pass:
        // a dynamic buffer may be written once per frame.
        struct Step {
            std::unique_ptr<QRhiTexture> tex;
            std::unique_ptr<QRhiTextureRenderTarget> rt;
            std::unique_ptr<QRhiRenderPassDescriptor> rp;
            std::unique_ptr<QRhiBuffer> ubo;
            std::unique_ptr<QRhiShaderResourceBindings> srb;
            std::unique_ptr<QRhiGraphicsPipeline> pipe;
            int w = 0, h = 0;
            ReduceUbo u{};
        };
        std::vector<Step> steps[2];
        for (int op = 0; op < 2; ++op) {
            int sw = w, sh = h;
            QRhiTexture* in = src.get();
            bool first = true;
            do {
                Step st;
                st.w = std::max(1, (sw + 1) / 2);
                st.h = std::max(1, (sh + 1) / 2);
                st.tex.reset(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(st.w, st.h), 1,
                                              QRhiTexture::RenderTarget | QRhiTexture::UsedAsTransferSource));
                st.ubo.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(ReduceUbo)));
                if (!st.tex->create() || !st.ubo->create())
                    return make_error(ErrorCode::BackendError, "GPU reduction target creation failed.");
                st.rt.reset(rhi_->newTextureRenderTarget({QRhiColorAttachment(st.tex.get())}));
                st.rp.reset(st.rt->newCompatibleRenderPassDescriptor());
                st.rt->setRenderPassDescriptor(st.rp.get());
                if (!st.rt->create()) return make_error(ErrorCode::BackendError, "GPU render target creation failed.");
                st.srb.reset(rhi_->newShaderResourceBindings());
                st.srb->setBindings({
                    QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::FragmentStage, st.ubo.get()),
                    QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, in, sampler_.get()),
                });
                if (!st.srb->create()) return make_error(ErrorCode::BackendError, "GPU resource bindings failed.");
                st.pipe.reset(rhi_->newGraphicsPipeline());
                st.pipe->setShaderStages({{QRhiShaderStage::Vertex, vert_}, {QRhiShaderStage::Fragment, reduce_}});
                st.pipe->setVertexInputLayout({});
                st.pipe->setShaderResourceBindings(st.srb.get());
                st.pipe->setRenderPassDescriptor(st.rp.get());
                if (!st.pipe->create()) return make_error(ErrorCode::BackendError, "GPU pipeline creation failed.");
                st.u.sizes[0] = float(sw);
                st.u.sizes[1] = float(sh);
                st.u.sizes[2] = float(op);
                st.u.sizes[3] = first ? 1.0f : 0.0f;
                sw = st.w;
                sh = st.h;
                first = false;
                steps[op].push_back(std::move(st));
                in = steps[op].back().tex.get();
            } while (sw > 1 || sh > 1);
        }

        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess)
            return make_error(ErrorCode::BackendError, "GPU frame could not start.");
        QRhiResourceUpdateBatch* up = rhi_->nextResourceUpdateBatch();
        up->uploadTexture(src.get(), QRhiTextureUploadDescription(QRhiTextureUploadEntry(
            0, 0, QRhiTextureSubresourceUploadDescription(a.data(), quint32(a.size() * sizeof(float))))));
        for (auto& ladder : steps)
            for (auto& st : ladder) up->updateDynamicBuffer(st.ubo.get(), 0, sizeof(ReduceUbo), &st.u);
        QRhiReadbackResult rb[2];
        for (int op = 0; op < 2; ++op) {
            auto& ladder = steps[op];
            for (std::size_t k = 0; k < ladder.size(); ++k) {
                auto& st = ladder[k];
                cb->beginPass(st.rt.get(), Qt::black, {1.0f, 0}, up);
                up = nullptr;
                cb->setGraphicsPipeline(st.pipe.get());
                cb->setViewport({0, 0, float(st.w), float(st.h)});
                cb->setShaderResources(st.srb.get());
                cb->draw(3);
                QRhiResourceUpdateBatch* down = nullptr;
                if (k + 1 == ladder.size()) {
                    down = rhi_->nextResourceUpdateBatch();
                    down->readBackTexture(QRhiReadbackDescription(st.tex.get()), &rb[op]);
                }
                cb->endPass(down);
            }
        }
        rhi_->endOffscreenFrame();
        Reductions r;
        for (int op = 0; op < 2; ++op) {
            if (rb[op].data.size() < 16) return make_error(ErrorCode::BackendError, "GPU readback returned too little data.");
            float v;
            std::memcpy(&v, rb[op].data.constData(), 4);
            (op == 0 ? r.peak : r.sum) = v;
        }
        return r;
    }

private:
    // One display pass into a `fmt` target, read back raw.
    Result<void> view_pass(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                           const ViewParams& params, QRhiTexture::Format fmt, QByteArray& data) {
        const int w = model.width(), h = model.height();
        if (baseline.width() != w || baseline.height() != h)
            return make_error(ErrorCode::InvalidArgument, "The two composite targets differ in size.");
        const std::vector<float> a = rgba_of(model.buffer()), b = rgba_of(baseline.buffer());
        std::unique_ptr<QRhiTexture> ta(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        std::unique_ptr<QRhiTexture> tb(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(w, h)));
        std::unique_ptr<QRhiTexture> out(rhi_->newTexture(fmt, QSize(w, h), 1,
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

        const ViewUbo u = detail::view_ubo(params, w);

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

        const int bpp = fmt == QRhiTexture::RGBA32F ? 16 : fmt == QRhiTexture::RGBA16F ? 8 : 4;
        if (rb.data.size() < qsizetype(std::size_t(w) * h * bpp))
            return make_error(ErrorCode::BackendError, "GPU readback returned too little data.");
        data = rb.data;
        return {};
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
                vk_->setApiVersion(vk_->supportedApiVersion());
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
        reduce_ = load_shader(":/rudra/shaders/reduce.frag.qsb");
        if (!vert_.isValid() || !frag_.isValid() || !display_.isValid() || !reduce_.isValid())
            return make_error(ErrorCode::NotFound, "The composite shaders are missing from the build.");
        ubuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(CompositeUbo)));
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
    QShader vert_, frag_, display_, reduce_;
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
