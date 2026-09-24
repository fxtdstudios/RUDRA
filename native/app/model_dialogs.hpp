#pragma once
// The checkpoint manager and the first-run check (Phase 3 step 10).
//
// ModelManager lists what the catalog found (engine/model_catalog: the
// server's discovery, for packages), the backends this build can run, and
// switches the window's model in place. FirstRun is shown once: the HDR card
// through the viewer on the display the window opened on, what that display
// can show (core/hdr_card display_report), and the model loaded and tested.

#include <QDialog>

#include <optional>
#include <string>

#include "rudra/core/view.hpp"
#include "rudra/engine/model_catalog.hpp"

class QComboBox;
class QLabel;
class QPushButton;
class QTreeWidget;

namespace rudra::app {

class MainWindow;

// "Automatic" first, then every backend_choices() entry; the data is its key.
void fill_backend_combo(QComboBox* c, std::optional<BackendChoice> selected);
std::optional<BackendChoice> backend_from_combo(const QComboBox* c);

class ModelManager : public QDialog {
public:
    explicit ModelManager(MainWindow* w);
    void refresh();                          // the catalog and the model in use
    void select(std::size_t entry);
    void use_selected();
    QTreeWidget* list() const { return list_; }

private:
    MainWindow* w_;
    QTreeWidget* list_ = nullptr;
    QLabel *note_ = nullptr, *status_ = nullptr;
    QComboBox* backend_ = nullptr;
    QPushButton* use_ = nullptr;
};

class FirstRun : public QDialog {
public:
    // with_card: show the card in a viewer of its own (off in the tests).
    FirstRun(MainWindow* w, bool with_card);
    // What the viewer's swapchain says (the card's window, or a test).
    void set_display(const DisplayTarget& target, const std::string& swapchain, const std::string& peak_from);
    void test_model();                       // loads the picked package on the chosen backend
    void start();                            // done: remembered, not shown again

private:
    MainWindow* w_;
    QLabel *display_ = nullptr, *display_detail_ = nullptr, *model_ = nullptr, *model_status_ = nullptr;
    QComboBox* backend_ = nullptr;
    QWidget* card_ = nullptr;
};

}  // namespace rudra::app
