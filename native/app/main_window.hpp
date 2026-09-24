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
#include <optional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <QDialog>
#include <QImage>
#include <QPointer>
#include <QTimer>

#include <thread>

#include "rudra/core/composite.hpp"
#include "rudra/core/model_manifest.hpp"
#include "rudra/core/scopes.hpp"
#include "rudra/engine/master_job.hpp"
#include "rudra/engine/measure.hpp"
#include "rudra/engine/model_catalog.hpp"
#include "rudra/infer/self_test.hpp"
#include "rudra/engine/session.hpp"

class QAction;
class QLabel;
class QLineEdit;
class QMenu;
class QPlainTextEdit;
class QPushButton;
class QSlider;
class QStackedWidget;

namespace rudra {
class FrameEngine;
class InferenceBackend;
struct ReadyFrame;
struct ModelConstants;
struct ViewerStatus;
class ViewerWindow;
}  // namespace rudra

namespace rudra::app {

class CheckRow;
class IconButton;
class RegionEditor;
class ScopePlot;
class VectorscopeView;
class ScrubBar;
class Seg;

class MainWindow : public QMainWindow {
public:
    // viewer: false builds the window without the QRhi viewer (tests, and
    // builds without Qt Shader Tools).
    explicit MainWindow(bool viewer = true);
    ~MainWindow() override;

    // A model package from a folder picker (or `preset`): use_model on it.
    void open_package(const QString& preset = {});

    // The checkpoint manager (Phase 3 step 10). The roots searched, the
    // catalog they give, and a switch that keeps the session: frames, grade,
    // undo and the frame on screen stay; the new model's fields replace the
    // old ones when they arrive. Files are verified, the backend opened and,
    // the first time this package runs on this backend, the goldens checked
    // (infer/self_test), all off the UI thread; a package that fails keeps
    // the model in use. `done` gets whether it switched and why not.
    void set_model_roots(std::vector<std::filesystem::path> roots);   // instead of package_roots() (tests)
    std::vector<std::filesystem::path> model_roots() const;
    void add_model_root(const std::filesystem::path& root);              // remembered between runs
    void rescan_models();
    const Catalog& catalog() const { return catalog_; }
    void use_model(const std::filesystem::path& package, std::optional<BackendChoice> choice = std::nullopt,
                   std::function<void(bool, QString)> done = {});
    bool loading_model() const { return loading_model_; }
    std::filesystem::path model_package() const { return manifest_ ? manifest_->root : std::filesystem::path(); }
    std::optional<BackendChoice> model_backend() const { return backend_choice_; }
    // Stand-ins for the runtimes (the tests): how a package is opened and tested.
    struct ModelHooks {
        std::function<Result<OpenedBackend>(const ModelManifest&, std::optional<BackendChoice>)> open;
        std::function<Result<SelfTestReport>(const ModelManifest&, InferenceBackend&)> test;
        bool verify_files = true;
    };
    void set_model_hooks(ModelHooks h) { hooks_ = std::move(h); }
    // What RUDRA does on a bare start: the first-run check once, then the
    // package used last (on its backend), else the catalog's pick.
    void boot();
    void open_model_manager();
    // The export sheet (the Pro boards): format, frames, destination, then Master EXR.
    void open_export_sheet();
    void open_first_run();
    // A still, or a folder of frames: both are a sequence to the engine.
    void open_source(const QString& preset = {}, bool folder = false);
    // The page's addFiles (Open, a drop of images): appended to the frames,
    // the first of them shown ("added N frames").
    void add_files(const std::vector<std::filesystem::path>& files);
    // What a drop brings (step 11): a model package folder is used, a folder
    // is opened as a shot, image files are added.
    void open_paths(const QStringList& paths);
    // Open recent: the shots opened last, newest first, kept between runs.
    QStringList recent_sources() const;
    void fill_recent();
    // Settings kept between runs: the window's place and size, the workspace,
    // the rails, the inspector tab, the container and the Render fields.
    void save_settings() const;
    void restore_settings();
    // The sheet on screen (show_sheet), if any.
    QDialog* sheet() const { return sheet_; }
    // What Deliver > Copy writes (deliveryRecord()).
    std::string delivery_text() const;

    // The QAction for an action id (engine/actions), or nullptr.
    QAction* action(std::string_view id) const;
    // Run an action by id, as a menu item or its key does.
    void run(std::string_view id);
    // Enable each action as the page's refreshMenu() does, for the state now.
    void refresh_enabled();

    // Why an action is off in this build or at this step ("" when it is live).
    QString pending_reason(std::string_view id) const;

    // The page's shell (shell.js): the workspace and the inspector tab.
    void set_workspace(const QString& mode);     // "full" or "simple"
    QString workspace() const { return workspace_; }
    void show_tab(const QString& tab);           // "rec", "grade", "deliver"
    QString tab() const { return tab_; }
    // A line in the Log, as the page's log() writes it.
    void log(const QString& line);
    // The scopes of the frame on screen (drawScopes, drawVector); cleared
    // when there is none.
    void show_scopes(const ScopeData& s, std::optional<double> maxcll, const std::vector<std::uint8_t>& vector_rgba);
    void clear_scopes();

    // A frame to show and measure (the engine's, or a test's): the viewer
    // gets it now, the measurements follow on their own thread.
    void present_frame(SdrImage sdr, Fields fields, FrameScalars scalars, ModelConstants model, FrameHeader header);
    // Measure the frame on screen now, on this thread (the tests).
    void measure_now();
    const FrameMeasure* measurement() const { return measure_.get(); }
    // The probe at a frame pixel, as a hover with Probe on shows it; nullopt
    // clears it. `global` places the floating box.
    void probe_pixel(std::optional<std::pair<double, double>> px, QPoint global = {});
    QWidget* probe_box() const { return probe_box_; }

    // The Deliver tab's Master EXR (the page's master()): the render plan from
    // the Render fields, then a background job over the frames. `prepare` and
    // `count` stand in for the opened frames and the network (the tests).
    void master(PrepareMasterFrame prepare = {}, std::size_t count = 0);
    bool mastering() const { return master_job_ && master_job_->running(); }
    // What a master renders with now (the session, the Deliver checks, the model).
    MasterRequest master_request() const;
    const SdrImage* frame_sdr() const;
    const std::vector<std::filesystem::path>& frames() const { return frames_; }
    ViewerWindow* viewer() const { return viewer_; }
    // The preview's long side (the Studio's 1600; 0 for full size). Applies to frames opened after it.
    void set_preview_max_side(int side) { preview_max_side_ = side; }
    int preview_max_side() const { return preview_max_side_; }
    // The frame on screen (the tests): its fields and which frame it is.
    const Fields* frame_fields() const;
    int current_index() const { return current_; }
    std::size_t frame_count() const { return frames_.size(); }

    // The page's state: grade, undo, peak, wipe, container (engine/session).
    Session& session() { return session_; }
    const Session& session() const { return session_; }
    CompositeParams composite() const { return session_.composite_params(); }
    const std::string& container() const { return session_.container; }

protected:
    // The keys the page's keydown takes before its map: the wipe nudge,
    // Escape, and B held to flip (the viewer takes them itself when it has
    // the focus).
    void keyPressEvent(class QKeyEvent* e) override;
    void keyReleaseEvent(class QKeyEvent* e) override;
    void dragEnterEvent(class QDragEnterEvent* e) override;
    void dragLeaveEvent(class QDragLeaveEvent* e) override;
    void dropEvent(class QDropEvent* e) override;
    void closeEvent(class QCloseEvent* e) override;
    bool eventFilter(QObject* o, QEvent* e) override;   // a shot row in the sidebar reopens its shot

private:
    void build_menus();
    void bind_handlers();
    void show_sheet(const QString& title, const std::vector<std::pair<QString, QString>>& rows);
    void sync_checks();
    void sync_ui();
    void session_changed(std::uint32_t what);
    void build_ui(bool with_viewer);
    QWidget* build_left_rail();
    QWidget* build_centre(bool with_viewer);
    QWidget* build_right_rail();
    QWidget* build_pipe();
    QWidget* build_toolbar();
    // The Pro chrome: the toolbar's shot title, the sidebar's shot rows and
    // collections, the viewer's badges, the filmstrip and its clipping lane.
    void refresh_library();
    void update_badges();
    void note_frame_in_strip(int index);
    int delivered_ = 0;
    std::optional<std::pair<QString, bool>> pending_source_;   // a shot asked for while the model loads
    QImage first_thumb_;
    QImage thumb_of_first() const;
    QPointer<QDialog> export_sheet_;

    void start_engine(std::vector<std::filesystem::path> frames, int at = 0);
    struct LoadedModel;
    void adopt_model(std::shared_ptr<LoadedModel> loaded, const std::function<void(bool, QString)>& done);
    void set_model_pills();
    void close_frames();
    void step_to(int i);
    void toggle_play();
    void frame_ready(const ReadyFrame& f, const ModelConstants& model);
    void show_status();
    void schedule_stats();
    void run_stats();
    void apply_measure(std::shared_ptr<const FrameMeasure> m);
    void update_pipe();
    void fill_rows(QWidget* ms, const std::vector<MetricRow>& rows);

    std::map<std::string, QAction*, std::less<>> actions_;
    std::map<std::string, std::function<void()>, std::less<>> handlers_;
    std::map<std::string, QString, std::less<>> pending_;

    ViewerWindow* viewer_ = nullptr;
    QString workspace_ = "full", tab_ = "rec";
    bool guides_on_ = false;
    // The widgets the state is drawn into (the page's ids).
    QWidget *rail_left_ = nullptr, *rail_right_ = nullptr, *scopes_ = nullptr;
    QStackedWidget* viewer_stack_ = nullptr;
    QWidget* panels_ = nullptr;          // the inspector's tab page (#ipanels)
    std::vector<QWidget*> pages_;        // Reconstruct, Grade, Deliver
    Seg *view_mode_ = nullptr, *view_layer_ = nullptr, *zoom_seg_ = nullptr, *mode_seg_ = nullptr, *tabs_ = nullptr,
        *ws_ = nullptr;
    CheckRow *preserve_ = nullptr, *anchor_ = nullptr, *carry_chroma_ = nullptr;
    QSlider *strength_ = nullptr, *peak_ = nullptr;
    QLabel *strength_val_ = nullptr, *peak_val_ = nullptr, *shot_count_ = nullptr, *frames_empty_ = nullptr,
           *tc_ = nullptr, *src_info_ = nullptr, *zoom_val_ = nullptr, *container_field_ = nullptr,
           *primaries_field_ = nullptr, *ckpt_ = nullptr, *device_ = nullptr, *lamp_ = nullptr,
           *region_count_ = nullptr, *view_transform_ = nullptr;
    QPushButton *guide_btn_ = nullptr, *probe_btn_ = nullptr, *btn_master_ = nullptr, *btn_reprocess_ = nullptr;
    IconButton *i_media_ = nullptr, *i_scopes_ = nullptr, *i_inspector_ = nullptr, *btn_prev_ = nullptr,
               *btn_play_ = nullptr, *btn_next_ = nullptr;
    ScrubBar* scrub_ = nullptr;
    ScopePlot *wave_ = nullptr, *hist_ = nullptr;
    RegionEditor* regions_ = nullptr;
    VectorscopeView* vector_ = nullptr;
    QLineEdit* seq_path_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    std::vector<QWidget*> notes_;
    // The frame on screen and its measurements (computeStats).
    struct Current {
        SdrImage sdr;
        Fields fields;
        FrameScalars scalars;
        ModelConstants model;
        std::shared_ptr<const NetworkLinearImage> baseline;
        FrameHeader header;
    };
    std::shared_ptr<const Current> current_frame_;
    std::shared_ptr<const FrameMeasure> measure_;
    QTimer stats_timer_;
    int stats_gen_ = 0;
    bool stats_running_ = false, stats_again_ = false;
    class ClipBar* clip_bar_ = nullptr;
    QWidget* probe_box_ = nullptr;
    std::unique_ptr<MasterJob> master_job_;
    std::shared_ptr<std::mutex> backend_mutex_ = std::make_shared<std::mutex>();   // the engine and the master share it
    void master_finished(const MasterOutcome& o, const QString& folder);
    bool probe_on_ = false;
    Session session_;
    QTimer play_;
    int current_ = 0;
    bool waiting_ = false;
    QString frame_info_;
    QString runtimes_;
    Catalog catalog_;
    std::optional<std::vector<std::filesystem::path>> roots_override_;
    ModelHooks hooks_;
    bool loading_model_ = false;
    std::thread model_worker_;
    std::optional<BackendChoice> backend_choice_;
    QPointer<QDialog> manager_, first_run_, sheet_;
    QMenu* recent_menu_ = nullptr;
    void remember_source(const QString& path);
    void copy_text(const QString& label, const std::string& text);
    void set_drop_hot(bool hot);
    std::unique_ptr<ModelManifest> manifest_;
    std::unique_ptr<InferenceBackend> backend_;   // used only from the engine's worker
    std::vector<std::filesystem::path> frames_;
    std::unique_ptr<FrameEngine> engine_;         // declared after the backend: destroyed first
    int engine_gen_ = 0;                          // which engine a delivered frame came from
    // Each frame's size as decoded, before the preview downscale (the header's source_resolution).
    struct SourceSizes {
        std::mutex mu;
        std::vector<std::pair<int, int>> size;
    };
    std::shared_ptr<SourceSizes> source_sizes_;
    int preview_max_side_ = 1600;                 // media/still.hpp kPreviewMaxSide
};

}  // namespace rudra::app
