#pragma once
// The main window (Phase 3). Its menubar is the Studio's, built from
// engine/actions (the page's menus, labels, keys and check states, step 2);
// every action runs through run(id), which the session (step 3) takes over.
// The rails, toolbar and panels arrive with step 4.

#include <QMainWindow>
#include <QString>

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <QTimer>

#include "rudra/core/composite.hpp"
#include "rudra/core/model_manifest.hpp"

class QAction;
class QLabel;
class QMenu;

namespace rudra {
class FrameEngine;
class InferenceBackend;
struct ReadyFrame;
struct ModelConstants;
struct ViewerStatus;
class ViewerWindow;
}  // namespace rudra

namespace rudra::app {

class MainWindow : public QMainWindow {
public:
    // viewer: false builds the window without the QRhi viewer (tests, and
    // builds without Qt Shader Tools).
    explicit MainWindow(bool viewer = true);
    ~MainWindow() override;

    void open_package(const QString& preset = {});
    // A still, or a folder of frames: both are a sequence to the engine.
    void open_source(const QString& preset = {}, bool folder = false);

    // The QAction for an action id (engine/actions), or nullptr.
    QAction* action(std::string_view id) const;
    // Run an action by id, as a menu item or its key does.
    void run(std::string_view id);
    // Enable each action as the page's refreshMenu() does, for the state now.
    void refresh_enabled();

    // Why an action is off in this build or at this step ("" when it is live).
    QString pending_reason(std::string_view id) const;

    // For the tests: the recovery settings the actions have set.
    const CompositeParams& composite() const { return composite_; }
    const std::string& container() const { return container_; }

private:
    void build_menus();
    void bind_handlers();
    void show_sheet(const QString& title, const std::vector<std::pair<QString, QString>>& rows);
    void set_mode(RecoveryMode m);
    void nudge_strength(double d);
    void sync_checks();

    void start_engine(std::vector<std::filesystem::path> frames);
    void close_frames();
    void step_to(int i);
    void toggle_play();
    void frame_ready(const ReadyFrame& f, const ModelConstants& model);
    void show_status();

    std::map<std::string, QAction*, std::less<>> actions_;
    std::map<std::string, std::function<void()>, std::less<>> handlers_;
    std::map<std::string, QString, std::less<>> pending_;

    ViewerWindow* viewer_ = nullptr;
    CompositeParams composite_;
    std::string container_ = "aces";
    QTimer play_;
    int current_ = 0;
    bool waiting_ = false;
    QString frame_info_;
    QLabel* model_ = nullptr;
    QString runtimes_;
    std::unique_ptr<ModelManifest> manifest_;
    std::unique_ptr<InferenceBackend> backend_;   // used only from the engine's worker
    std::vector<std::filesystem::path> frames_;
    std::unique_ptr<FrameEngine> engine_;         // declared after the backend: destroyed first
};

}  // namespace rudra::app
