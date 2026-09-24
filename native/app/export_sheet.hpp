#pragma once
// The export sheet (the "Pro direction: export sheet" board): the shot at the
// top, the formats as tiles, what the file will carry, what is checked, and
// Export. It drives the Deliver tab's own fields (the container, Output,
// the folder and the name) and Master EXR, so a render from the sheet is the
// same render as one from the tab. EXR (ACES or linear) is written here;
// HDR10, HLG and ProRes arrive with video delivery and are shown, not offered.

#include <QDialog>

class QLabel;

namespace rudra::app {

class MainWindow;

class ExportSheet : public QDialog {
public:
    explicit ExportSheet(MainWindow* w);
    void refresh();              // from the window's state now
    void pick(const QString& id);  // "exr-aces" or "exr-linear"
    QString picked() const { return picked_; }
    void export_now();           // Master EXR, then close

private:
    MainWindow* w_;
    QString picked_ = "exr-aces";
    QLabel *title_ = nullptr, *sub_ = nullptr, *signal_ = nullptr, *frames_ = nullptr, *dest_ = nullptr;
};

}  // namespace rudra::app
