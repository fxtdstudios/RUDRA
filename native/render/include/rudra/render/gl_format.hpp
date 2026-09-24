#pragma once
// The OpenGL surface format for QRhi: QRhi's own default, and on macOS a 4.1
// core profile, set on the window too. macOS hands out a GL 2.1 legacy
// context unless a 3.2+ core profile is asked for, and the shaders are GLSL
// 330 (texelFetch, float render targets): without it nothing is drawn there
// but the clear colour. Elsewhere QRhi's default already gives GL 3.3+.
#include <rhi/qrhi.h>

namespace rudra {

inline QSurfaceFormat rhi_gl_format() {
    QSurfaceFormat f = QRhiGles2InitParams().format;
#ifdef __APPLE__
    f.setVersion(4, 1);
    f.setProfile(QSurfaceFormat::CoreProfile);
#endif
    return f;
}

}  // namespace rudra
