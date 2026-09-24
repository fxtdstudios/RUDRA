// The main window's layout (Phase 3 step 4): ui/index.html as Qt widgets.
//
// The frame is the page's: the menubar row (lockup, menus, the workspace
// switch, the model lamp and names), the icon rail, the media rail with the
// probe pinned to its foot, the centre (viewer toolbar, viewer, transport),
// the right rail (scopes, the Reconstruct / Grade / Deliver tabs, the frame
// measurements, Render, Log, and the actions pinned below the scrolling
// column) and the colour pipeline bar. Every element the page names has a
// widget of that object name (tests/golden/layout, checked by rudra_app_tests),
// and the sizes are the page's at 1600 x 1000.

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenuBar>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QVBoxLayout>

#include "main_window.hpp"
#include "widgets.hpp"

#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif

namespace rudra::app {
namespace {

QLabel* label(const QString& text, const QString& id = {}, const QString& role = {}, QWidget* parent = nullptr) {
    auto* l = new QLabel(text, parent);
    if (!id.isEmpty()) l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    return l;
}

QVBoxLayout* column(QWidget* w, int margin = 0, int spacing = 0) {
    auto* v = new QVBoxLayout(w);
    v->setContentsMargins(margin, margin, margin, margin);
    v->setSpacing(spacing);
    return v;
}

QWidget* styled(const QString& id = {}, const QString& role = {}) {
    auto* w = new QWidget;
    if (!id.isEmpty()) w->setObjectName(id);
    if (!role.isEmpty()) w->setProperty("role", role);
    w->setAttribute(Qt::WA_StyledBackground, true);
    return w;
}

// A .ctl block: padded, spaced.
QWidget* ctl(QWidget* parent = nullptr) {
    auto* w = new QWidget(parent);
    w->setProperty("role", "ctl");
    return w;
}

// A .prow / .field / .ro: a key on the left, a value on the right.
QWidget* row(const QString& key, QLabel* value, const QString& role = "prow") {
    auto* w = new QWidget;
    w->setProperty("role", role);
    auto* h = new QHBoxLayout(w);
    h->setContentsMargins(0, 3, 0, 3);
    h->setSpacing(6);
    h->addWidget(label(key, {}, "key"));
    h->addStretch(1);
    value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    h->addWidget(value);
    return w;
}

}  // namespace

void MainWindow::build_ui(bool with_viewer) {
    auto* app = styled("app");
    auto* v = column(app);
    auto* body = styled("body");
    auto* h = new QHBoxLayout(body);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(0);

    // ── icon rail ──
    auto* icons = styled("iconRail", "iconrail");
    icons->setFixedWidth(36);
    auto* iv = column(icons, 0, 2);
    iv->setContentsMargins(4, 6, 4, 6);
    i_media_ = new IconButton("iMedia", IconButton::Glyph::Media, "Frames", icons);
    i_scopes_ = new IconButton("iScopes", IconButton::Glyph::Scopes, "Scopes", icons);
    i_inspector_ = new IconButton("iInspector", IconButton::Glyph::Inspector, "Inspector", icons);
    auto* i_help = new IconButton("iHelp", IconButton::Glyph::Help, "Keyboard shortcuts", icons);
    for (auto* b : {i_media_, i_scopes_, i_inspector_}) {
        b->setFixedSize(28, 28);
        iv->addWidget(b, 0, Qt::AlignHCenter);
    }
    iv->addStretch(1);
    i_help->setFixedSize(28, 28);
    iv->addWidget(i_help, 0, Qt::AlignHCenter);
    connect(i_media_, &QToolButton::clicked, this, [this] { run("rail-left"); });
    connect(i_inspector_, &QToolButton::clicked, this, [this] { run("rail-right"); });
    connect(i_scopes_, &QToolButton::clicked, this, [this] { run("scopes"); });
    connect(i_help, &QToolButton::clicked, this, [this] { run("shortcuts"); });

    rail_left_ = build_left_rail();
    rail_right_ = build_right_rail();
    h->addWidget(icons);
    h->addWidget(rail_left_);
    h->addWidget(build_centre(with_viewer), 1);
    h->addWidget(rail_right_);
    v->addWidget(body, 1);
    v->addWidget(build_pipe());
    setCentralWidget(app);
    build_menubar_corners();
}

void MainWindow::build_menubar_corners() {
    auto* lockup = styled("lockup", "lockup");
    auto* lh = new QHBoxLayout(lockup);
    lh->setContentsMargins(9, 0, 13, 0);
    lh->setSpacing(7);
    auto* mark = label({}, "lkMark", "lk-mark");
    mark->setPixmap(QPixmap(":/assets/rudra-mark.png").scaled(16, 16, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    lh->addWidget(mark);
    lh->addWidget(label("RUDRA", "lkName", "lk-name"));
    menuBar()->setCornerWidget(lockup, Qt::TopLeftCorner);

    auto* right = styled("mstat", "mstat");
    auto* rh = new QHBoxLayout(right);
    rh->setContentsMargins(8, 0, 9, 0);
    rh->setSpacing(8);
    ws_ = new Seg("wsswitch", {{"simple", "Simple"}, {"full", "Full"}}, right);
    ws_->button("simple")->setObjectName("wsSimple");
    ws_->button("full")->setObjectName("wsFull");
    ws_->button("simple")->setToolTip("Hide the scopes and the advanced groups");
    ws_->button("full")->setToolTip("Everything");
    ws_->clicked = [this](const QString& k) { set_workspace(k); };
    rh->addWidget(ws_);
    lamp_ = label("", "lamp", "lamp");
    lamp_->setFixedSize(6, 6);
    ckpt_ = label("no model", "ckpt", "pill");
    device_ = label("\u2014", "device", "pill");
    rh->addWidget(lamp_);
    rh->addWidget(ckpt_);
    rh->addWidget(device_);
    rh->addWidget(label("RADIANCE STUDIO", "studiomark", "studiomark"));
    menuBar()->setCornerWidget(right, Qt::TopRightCorner);
    menuBar()->setFixedHeight(30);   // the page's one 30 px row
}

QWidget* MainWindow::build_left_rail() {
    auto* rail = styled("railLeft", "rail-left");
    rail->setFixedWidth(208);
    auto* v = column(rail);
    v->addWidget(panel_label("Media", "0", "shotCount", rail));
    shot_count_ = rail->findChild<QLabel*>("shotCount");

    auto* dz = new QFrame(rail);
    dz->setObjectName("dropzone");
    auto* dv = column(dz, 10, 3);
    dv->addWidget(label("Drop footage here", {}, "drop-b"), 0, Qt::AlignHCenter);
    dv->addWidget(label("a frame, a folder of frames, or a movie", {}, "drop-t"), 0, Qt::AlignHCenter);
    dv->addWidget(label("or", {}, "drop-or"), 0, Qt::AlignHCenter);
    auto* browse = new QPushButton("Open\u200a\u2026", dz);
    browse->setObjectName("browseBtn");
    browse->setProperty("role", "primary");
    connect(browse, &QPushButton::clicked, this, [this] { run("open"); });
    dv->addWidget(browse, 0, Qt::AlignHCenter);
    auto* dzwrap = new QWidget(rail);
    column(dzwrap, 8)->addWidget(dz);
    v->addWidget(dzwrap);

    auto* seq = new QWidget(rail);
    auto* sh = new QHBoxLayout(seq);
    sh->setContentsMargins(8, 0, 8, 8);
    sh->setSpacing(4);
    seq_path_ = new QLineEdit(seq);
    seq_path_->setObjectName("seqPath");
    seq_path_->setPlaceholderText("D:\\plates\\shot_010  or  shot.mov");
    seq_path_->setToolTip("A folder of frames, or a video file. Nothing is uploaded \u2014 the server reads it where it sits.");
    auto* seq_open = new QPushButton("Open shot", seq);
    seq_open->setObjectName("seqOpen");
    seq_open->setToolTip("Open the shot at that path");
    auto open_path = [this] {
        const QString p = seq_path_->text().trimmed();
        if (!p.isEmpty()) open_source(p);
    };
    connect(seq_open, &QPushButton::clicked, this, open_path);
    connect(seq_path_, &QLineEdit::returnPressed, this, open_path);
    sh->addWidget(seq_path_, 1);
    sh->addWidget(seq_open);
    v->addWidget(seq);
    auto* note = label("", "seqNote", "seq-note", rail);
    note->hide();
    v->addWidget(note);

    auto* head = styled({}, "shot-head");
    auto* hh = new QHBoxLayout(head);
    hh->setContentsMargins(11, 5, 11, 5);
    hh->setSpacing(6);
    hh->addWidget(label("NAME"), 1);
    hh->addWidget(label("SIZE"));
    auto* pk = label("PEAK");
    pk->setFixedWidth(40);
    pk->setAlignment(Qt::AlignRight);
    hh->addWidget(pk);
    hh->addSpacing(26);
    v->addWidget(head);
    auto* shots = styled("shots", "shots");
    column(shots);
    shots->hide();   // the shot list is filled as frames open (drawFrames)
    v->addWidget(shots);
    frames_empty_ = label("Nothing open yet", "framesEmpty", "empty-rail", rail);
    frames_empty_->setAlignment(Qt::AlignCenter);
    v->addWidget(frames_empty_);
    v->addStretch(1);

    // The probe, pinned to the foot of the rail.
    auto* probe = styled("probePanelWrap", "probepanel");
    auto* pv = column(probe);
    pv->addWidget(panel_label("Probe", "\u2014", "probeXY", probe));
    auto* body = styled("probePanel", "pbody");
    auto* bv = column(body, 9, 0);
    auto* nits = new QWidget(body);
    auto* nh = new QHBoxLayout(nits);
    nh->setContentsMargins(0, 0, 0, 0);
    nh->setSpacing(5);
    auto* nit = label("\u2014", "probeNits", "pnit", nits);
    nit->setProperty("state", "idle");
    nh->addWidget(nit, 0, Qt::AlignBottom);
    auto* unit = label("nits", "probeNitsUnit", "pnit-unit", nits);
    unit->hide();   // .pnit.idle small: no unit on an idle readout
    nh->addWidget(unit, 0, Qt::AlignBottom);
    nh->addStretch(1);
    bv->addWidget(nits);
    auto* delta = label("pick a pixel with Probe, or hold Alt", "probeDelta", "pdelta", body);
    bv->addWidget(delta);
    bv->addSpacing(6);
    for (const auto& [k, id] : std::vector<std::pair<QString, QString>>{
             {"Source", "probeSrc"}, {"Baseline", "probeBase"}, {"RUDRA", "probeModel"}, {"Masks", "probeMask"}})
        bv->addWidget(row(k, label("\u2014", id, "value")));
    pv->addWidget(body);
    v->addWidget(probe);
    return rail;
}

QWidget* MainWindow::build_centre(bool with_viewer) {
    auto* centre = styled("centre", "centre");
    auto* v = column(centre);

    // ── viewer toolbar ──
    auto* tools = styled("viewerTools", "vtools");
    tools->setFixedHeight(28);
    auto* th = new QHBoxLayout(tools);
    th->setContentsMargins(8, 0, 8, 0);
    th->setSpacing(6);
    th->addWidget(label("Compare", {}, "vlabel"));
    view_mode_ = new Seg("viewMode", {{"model", "RUDRA"}, {"baseline", "Baseline"}, {"#wipeBtn", "Wipe"}}, tools, false);
    view_mode_->button("model")->setToolTip("The reconstruction (1)");
    view_mode_->button("baseline")->setToolTip("The analytic baseline (2)");
    view_mode_->button("#wipeBtn")->setToolTip("Wipe: baseline left, RUDRA right (W)");
    view_mode_->clicked = [this](const QString& k) {
        if (k == "#wipeBtn") run("wipe");
        else session_.set_show(k.toStdString());
    };
    th->addWidget(view_mode_);
    th->addWidget(styled({}, "vsep"));
    th->addWidget(label("Layer", {}, "vlabel"));
    view_layer_ = new Seg("viewLayer", {{"0", "Image"}, {"1", "False colour"}, {"2", "Difference"}}, tools);
    view_layer_->button("0")->setToolTip("The composed picture");
    view_layer_->button("1")->setToolTip("Luminance zones, in nits");
    view_layer_->button("2")->setToolTip("What the network changed, and where it did not");
    view_layer_->clicked = [this](const QString& k) { session_.set_view_layer(k.toInt()); };
    th->addWidget(view_layer_);
    th->addWidget(styled({}, "vsep"));
    probe_btn_ = new QPushButton("Probe", tools);
    probe_btn_->setObjectName("probeBtn");
    probe_btn_->setCheckable(true);
    probe_btn_->setToolTip("Read one pixel: baseline, RUDRA, and whether the SDR clipped there");
    guide_btn_ = new QPushButton("Guides", tools);
    guide_btn_->setObjectName("guideBtn");
    guide_btn_->setCheckable(true);
    guide_btn_->setToolTip("Framing guides, 90% and 80%");
    connect(guide_btn_, &QPushButton::clicked, this, [this](bool on) {
        guides_on_ = on;
#ifdef RUDRA_APP_VIEWER
        if (viewer_) {
            auto g = viewer_->guides();
            g.action_safe = g.title_safe = on;
            viewer_->set_guides(g);
        }
#endif
        sync_checks();
    });
    th->addWidget(probe_btn_);
    th->addWidget(guide_btn_);
    th->addStretch(1);
    th->addWidget(label("Zoom", {}, "vlabel"));
    zoom_seg_ = new Seg("zoomSeg", {{"fit", "Fit"}, {"actual", "100%"}}, tools);
    zoom_seg_->clicked = [this](const QString& k) { run(k == "actual" ? "zoom-actual" : "zoom-fit"); };
    th->addWidget(zoom_seg_);
    zoom_val_ = label("100%", "zoomVal", "vstatic", tools);
    zoom_val_->setToolTip("Scroll to zoom, middle-drag to pan, double-click to fit");
    th->addWidget(zoom_val_);
    v->addWidget(tools);

    // ── viewer ──
    viewer_stack_ = new QStackedWidget(centre);
    viewer_stack_->setObjectName("viewer");
    auto* empty = new QWidget(viewer_stack_);
    empty->setObjectName("empty");
    auto* ev = column(empty);
    ev->addStretch(1);
    ev->addWidget(label("Drop an SDR frame", {}, "empty-b"), 0, Qt::AlignHCenter);
    ev->addWidget(label("PNG, JPEG or TIFF \u2014 one, or a whole sequence", {}, "empty-t"), 0, Qt::AlignHCenter);
    ev->addStretch(1);
    viewer_stack_->addWidget(empty);
#ifdef RUDRA_APP_VIEWER
    if (with_viewer) {
        viewer_ = new ViewerWindow();
        auto* container = QWidget::createWindowContainer(viewer_, viewer_stack_);
        container->setObjectName("viewerHost");
        container->setFocusPolicy(Qt::StrongFocus);
        viewer_stack_->addWidget(container);
    }
#else
    (void)with_viewer;
#endif
    v->addWidget(viewer_stack_, 1);

    // ── transport ──
    auto* transport = styled("transport", "transport");
    transport->setFixedHeight(40);
    auto* trh = new QHBoxLayout(transport);
    trh->setContentsMargins(10, 0, 12, 0);
    trh->setSpacing(10);
    auto* seg = new QWidget(transport);
    seg->setProperty("role", "seg");
    auto* sgh = new QHBoxLayout(seg);
    sgh->setContentsMargins(0, 0, 0, 0);
    sgh->setSpacing(0);
    btn_prev_ = new IconButton("btnPrev", IconButton::Glyph::Prev, "Previous frame", seg);
    btn_play_ = new IconButton("btnPlay", IconButton::Glyph::Play, "Play", seg);
    btn_next_ = new IconButton("btnNext", IconButton::Glyph::Next, "Next frame", seg);
    for (auto* b : {btn_prev_, btn_play_, btn_next_}) {
        b->setFixedSize(28, 24);
        b->setProperty("role", "tbtn");
        sgh->addWidget(b);
    }
    connect(btn_prev_, &QToolButton::clicked, this, [this] {
        play_.stop();
        run("prev");
    });
    connect(btn_next_, &QToolButton::clicked, this, [this] {
        play_.stop();
        run("next");
    });
    connect(btn_play_, &QToolButton::clicked, this, [this] { run("play"); });
    trh->addWidget(seg);
    tc_ = label("01:00:00:00", "tc", "tc", transport);
    trh->addWidget(tc_);
    scrub_ = new ScrubBar(transport);
    scrub_->seek = [this](double f) {
        if (frames_.size() < 2) return;
        play_.stop();
        step_to(int(std::lround(f * double(frames_.size() - 1))));
    };
    trh->addWidget(scrub_, 1);
    src_info_ = label("\u2014", "srcInfo", "srcinfo", transport);
    trh->addWidget(src_info_);
    v->addWidget(transport);
    return centre;
}

QWidget* MainWindow::build_right_rail() {
    auto* rail = styled("railRight", "rail-right");
    rail->setFixedWidth(272);
    auto* rv = column(rail);
    auto* scroll = new QScrollArea(rail);
    scroll->setObjectName("railscroll");
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* col = styled("railColumn");
    auto* v = column(col);

    // ── scopes ──
    scopes_ = styled("scopes", "scopes");
    auto* sv = column(scopes_);
    const struct { const char* title; const char* note; const char* id; int h; } scopes[] = {
        {"Waveform", "luma percentile \u00b7 nits/log", "wave", 112},
        {"Histogram", "absolute nits \u00b7 log2 stops \u00b7 DW", "hist", 84},
        {"Vectorscope", "Rec.2020", "vector", 176}};
    for (const auto& s : scopes) {
        sv->addWidget(panel_label(s.title, s.note, {}, scopes_));
        auto* plot = styled({}, "plot");
        auto* pl = column(plot);
        pl->setContentsMargins(8, 7, 8, 9);
        auto* area = styled(s.id, "plot-area");
        area->setFixedHeight(s.h);
        pl->addWidget(area);
        sv->addWidget(plot);
    }
    v->addWidget(scopes_);

    // ── inspector tabs ──
    tabs_ = new Seg("itabs", {{"rec", "Reconstruct"}, {"grade", "Grade"}, {"deliver", "Deliver"}}, col);
    tabs_->setProperty("role", "itabs");
    tabs_->button("rec")->setObjectName("tabRec");
    tabs_->button("grade")->setObjectName("tabGrade");
    tabs_->button("deliver")->setObjectName("tabDeliver");
    tabs_->clicked = [this](const QString& k) { show_tab(k); };
    v->addWidget(tabs_);
    panels_ = new QStackedWidget(col);
    panels_->setObjectName("ipanels");

    // Reconstruct
    auto* rec = ctl();
    auto* recv = column(rec, 0, 8);
    recv->setContentsMargins(10, 10, 10, 10);
    mode_seg_ = new Seg("mode", {{"all", "All"}, {"highlights", "Highlights"}, {"shadows", "Shadows"}, {"off", "Off"}},
                        rec);
    mode_seg_->clicked = [this](const QString& k) { session_.set_mode(k.toStdString()); };
    recv->addWidget(mode_seg_);
    auto slider = [&](const QString& lab, const QString& val_id, const QString& unit, const QString& id, int lo,
                      int hi, QLabel*& val_out, QSlider*& out) {
        auto* w = new QWidget(rec);
        w->setProperty("role", "slider");
        auto* sv2 = column(w, 0, 2);
        auto* top = new QWidget(w);
        auto* tph = new QHBoxLayout(top);
        tph->setContentsMargins(0, 0, 0, 0);
        tph->setSpacing(6);
        tph->addWidget(label(lab, {}, "slider-lab"));
        tph->addStretch(1);
        val_out = label("", val_id, "slider-val");
        tph->addWidget(val_out);
        tph->addWidget(label(unit, {}, "unit"));
        sv2->addWidget(top);
        out = new QSlider(Qt::Horizontal, w);
        out->setObjectName(id);
        out->setRange(lo, hi);
        sv2->addWidget(out);
        recv->addWidget(w);
    };
    // The page's range inputs, in their own steps: strength 0 to 2 by 0.05,
    // peak -1 to 5 by 0.5. A step n reads back as n / 20 (n / 2): the double
    // the page's parseFloat of the same value gives.
    slider("Residual strength", "strengthVal", "\u00d7", "strength", 0, 40, strength_val_, strength_);
    slider("Display peak", "peakVal", "nits", "peak", -2, 10, peak_val_, peak_);
    connect(strength_, &QSlider::sliderPressed, this, [this] { session_.strength_press(); });
    connect(strength_, &QSlider::valueChanged, this, [this](int n) {
        if (std::lround(session_.grade.strength * 20.0) == n) return;   // sync_ui, not a move
        if (!strength_->isSliderDown()) session_.strength_press();      // a key or a click on the track
        session_.strength_input(n / 20.0);
    });
    connect(peak_, &QSlider::valueChanged, this, [this](int n) {
        if (std::lround(session_.peak_ev * 2.0) == n) return;
        session_.peak_input(n / 2.0);
    });
    preserve_ = new CheckRow("preserve", "Preserve outside masks", "preserveHint", "do-no-harm", rec);
    preserve_->clicked = [this] { session_.toggle_preserve(); };
    recv->addWidget(preserve_);
    auto* n1 = label("The reconstruction is applied where clipping or crushed shadows made the SDR mapping "
                     "non-invertible. Everywhere else the analytic baseline is preserved unchanged.",
                     {}, "note-p", rec);
    n1->setWordWrap(true);
    notes_.push_back(n1);
    recv->addWidget(n1);
    panels_->addWidget(rec);

    // Grade
    auto* grade = new QWidget;
    auto* gv = column(grade);
    gv->addWidget(panel_label("Region EV", "3", "regionCount", grade));
    region_count_ = grade->findChild<QLabel*>("regionCount");
    auto* regions = styled("regions", "regions");
    column(regions, 4);
    gv->addWidget(regions);
    auto* gctl = ctl(grade);
    auto* gcv = column(gctl);
    gcv->setContentsMargins(10, 0, 10, 10);
    auto* n2 = label("A region is a soft luminance qualifier, not a mask, so it follows the picture. Drag a value to "
                     "scrub it, double-click to zero it. Master applies the identical qualifier and gain to the file.",
                     {}, "note-p", gctl);
    n2->setWordWrap(true);
    notes_.push_back(n2);
    gcv->addWidget(n2);
    gv->addWidget(gctl);
    panels_->addWidget(grade);

    // Deliver
    auto* del = ctl();
    auto* dv = column(del, 0, 6);
    dv->setContentsMargins(10, 10, 10, 10);
    anchor_ = new CheckRow("anchor", "Anchor to source exposure", "anchorHint", "conform", del);
    anchor_->clicked = [this] {
        session_.toggle_anchor();
        log(session_.anchor ? "master will anchor to the source exposure"
                            : "master will keep the inverse tone map's own level");
    };
    carry_chroma_ = new CheckRow("carryChroma", "Carry source chroma", "chromaHint", "below the clip", del);
    carry_chroma_->clicked = [this] {
        session_.toggle_carry_chroma();
        log(session_.carry_chroma ? "master will take hue from the source below the clip"
                                  : "master will keep the per-channel expansion's own hue");
    };
    dv->addWidget(anchor_);
    dv->addWidget(carry_chroma_);
    container_field_ = label("OpenEXR \u2014 ACES 2065-1", "containerField", "value");
    primaries_field_ = label("AP0 (ST 2065-4)", "primariesField", "value");
    dv->addWidget(row("Container", container_field_, "field"));
    dv->addWidget(row("Transfer", label("linear (scene-referred)", {}, "value"), "field"));
    dv->addWidget(row("Primaries", primaries_field_, "field"));
    dv->addWidget(row("White", label("diffuse 1.0 = 203 nits", {}, "value"), "field"));
    auto* n3 = label("Switch the container from the <b>Deliver</b> menu. ProRes, HDR10 and HLG are written by "
                     "<span style=\"font-family:'IBM Plex Mono'\">rudra deliver</span> from the command line; they "
                     "are not wired to this panel yet.",
                     {}, "note-p", del);
    n3->setTextFormat(Qt::RichText);
    n3->setWordWrap(true);
    notes_.push_back(n3);
    dv->addWidget(n3);
    panels_->addWidget(del);
    v->addWidget(panels_);

    // ── frame measurements ──
    auto* stats = styled("frameStats", "framestats");
    auto* fv = column(stats);
    fv->addWidget(panel_label("Frame", "measured", {}, stats));
    for (const char* id : {"measA", "measB"}) {
        auto* ms = styled(id, "ms");
        column(ms)->setContentsMargins(9, 2, 9, 2);
        fv->addWidget(ms);
    }
    auto* clip = styled("clipBar", "clipbar");
    clip->setFixedHeight(5);
    clip->hide();
    fv->addWidget(clip);
    v->addWidget(stats);

    // ── render ──
    v->addWidget(panel_label("Render", {}, {}, col));
    auto* render = ctl(col);
    auto* form = column(render, 0, 3);
    form->setContentsMargins(12, 10, 12, 12);
    auto* dir = new QLineEdit(render);
    dir->setObjectName("renderDir");
    dir->setPlaceholderText("C:\\Renders\\Shot01");
    auto* mode = new QComboBox(render);
    mode->setObjectName("renderMode");
    mode->addItem("Current image", "image");
    mode->addItem("All loaded frames \u2014 sequence", "sequence");
    auto* name = new QLineEdit("master", render);
    name->setObjectName("renderName");
    auto* start = new QSpinBox(render);
    start->setObjectName("renderStart");
    start->setRange(0, 9999999);
    start->setValue(1);
    for (const auto& [text, field] : std::vector<std::pair<QString, QWidget*>>{
             {"Render folder", dir}, {"Output", mode}, {"Render name", name}, {"Start frame", start}}) {
        form->addWidget(label(text, {}, "key", render));
        form->addWidget(field);
        form->addSpacing(4);
    }
    auto* small = label("Folder on the Studio computer. Sequence uses loaded frame order and the current grade for every "
                        "frame.",
                        {}, "small", render);
    small->setWordWrap(true);
    form->addWidget(small);
    auto* status = label("", "renderStatus", "render-status", render);
    status->hide();
    form->addWidget(status);
    v->addWidget(render);

    // ── log ──
    v->addWidget(panel_label("Log", {}, {}, col));
    log_ = new QPlainTextEdit(col);
    log_->setObjectName("log");
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(400);
    log_->setFixedHeight(68);
    v->addWidget(log_);
    v->addStretch(1);
    scroll->setWidget(col);
    rv->addWidget(scroll, 1);

    // ── the actions, below the scrolling column ──
    auto* actions = styled("actions", "actions");
    auto* ah = new QHBoxLayout(actions);
    ah->setContentsMargins(9, 5, 9, 5);
    ah->setSpacing(6);
    btn_reprocess_ = new QPushButton("Reprocess", actions);
    btn_reprocess_->setObjectName("btnReprocess");
    btn_master_ = new QPushButton("Master EXR", actions);
    btn_master_->setObjectName("btnMaster");
    btn_master_->setProperty("role", "primary");
    connect(btn_reprocess_, &QPushButton::clicked, this, [this] {
        if (!frames_.empty()) step_to(current_);
    });
    connect(btn_master_, &QPushButton::clicked, this, [this] { run("master"); });
    ah->addWidget(btn_reprocess_, 1);
    ah->addWidget(btn_master_, 1);
    rv->addWidget(actions);
    return rail;
}

QWidget* MainWindow::build_pipe() {
    auto* pipe = styled("pipe", "pipe");
    pipe->setFixedHeight(26);
    auto* h = new QHBoxLayout(pipe);
    h->setContentsMargins(11, 0, 11, 0);
    h->setSpacing(0);
    auto seg = [&](const QString& em, const QString& id, const QString& text) {
        auto* w = new QWidget(pipe);
        w->setProperty("role", "pseg");
        auto* sh = new QHBoxLayout(w);
        sh->setContentsMargins(h->count() == 0 ? 0 : 12, 0, 12, 0);
        sh->setSpacing(8);
        sh->addWidget(label(em, {}, "pipe-em"));
        auto* b = label(text, id, "pipe-b");
        sh->addWidget(b);
        h->addWidget(w);
        return b;
    };
    auto arrow = [&] { h->addWidget(label("\u25b8", {}, "pipe-arrow", pipe)); };
    seg("in", "pipeIn", "sRGB \u00b7 Rec.709");
    arrow();
    seg("working", "pipeWorking", "scene-linear \u00b7 203 nits = 1.0");
    arrow();
    view_transform_ = seg("view", "viewTransform", "PQ \u00b7 Rec.2020");
    arrow();
    seg("master", "pipeMaster", "ACES 2065-1 EXR, half");
    h->addStretch(1);
    h->addWidget(label("\u2014", "statusMask", "pipe-s", pipe));
    auto* rule = styled({}, "vrule");
    rule->setFixedSize(1, 12);
    h->addSpacing(10);
    h->addWidget(rule);
    h->addSpacing(10);
    h->addWidget(label("\u2014", "statusTime", "pipe-s", pipe));
    auto* warn = label("", "pipeWarn", {}, pipe);
    warn->hide();
    h->addWidget(warn);
    return pipe;
}

}  // namespace rudra::app
