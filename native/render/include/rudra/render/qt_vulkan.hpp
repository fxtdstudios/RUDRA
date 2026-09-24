#pragma once
// Whether this build can use Qt's Vulkan classes. QT_CONFIG(vulkan) says Qt was
// built with Vulkan; QVulkanInstance and QRhiVulkanInitParams are declared only
// when the Vulkan headers are installed too, which a machine without the SDK
// (GitHub's Windows runner, say) does not have. Test RUDRA_QT_VULKAN, never
// QT_CONFIG(vulkan) alone.
#include <QtGui/qtguiglobal.h>

#if QT_CONFIG(vulkan) && __has_include(<vulkan/vulkan.h>)
#define RUDRA_QT_VULKAN 1
#else
#define RUDRA_QT_VULKAN 0
#endif
