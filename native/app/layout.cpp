// The main window's layout: the "Pro direction" boards (RUDRA Studio Desktop
// canvas, "Pro direction: main window").
//
// A unified toolbar (the sidebar toggle, the shot's name, the Reconstruct /
// Grade / Deliver tabs in the middle, the machine pill, Probe, Scopes, the
// inspector and Export on the right); the sidebar library (open by path, the
// shots, collections from the clipping lane, the probe card and the model
// card); the viewer on its neutral surround with its compare bar and badges,
// the colour pipeline under it and the timeline (timecode, round transport,
// the filmstrip with its clipping lane); and the inspector's grouped cards.
// Every control the Studio page names keeps that page's id as its object
// name, so every action, handler and read-out behaves as before.

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
#include "region_editor.hpp"
#include "scope_widgets.hpp"
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
    v->addWidget(build_toolbar());
    auto* body = styled("body");
    auto* h = new QHBoxLayout(body);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(0);
    rail_left_ = build_left_rail();
    rail_right_ = build_right_rail();
    h->addWidget(rail_left_);
    h->addWidget(build_centre(with_viewer), 1);
    h->addWidget(rail_right_);
    v->addWidget(body, 1);
    setCentralWidget(app);
    menuBar()->setFixedHeight(28);
}

QWidget* MainWindow::build_toolbar() {
    auto* bar = styled("toolbar");
    bar->setFixedHeight(52);
    auto* h = new QHBoxLayout(bar);
    h->setContentsMargins(12, 0, 14, 0);
    h->setSpacing(10);
    i_media_ = new IconButton("iMedia", IconButton::Glyph::Sidebar, "Show or hide the sidebar", bar);
    i_media_->setFixedSize(32, 28);
    connect(i_media_, &QToolButton::clicked, this, [this] { run("rail-left"); });
    h->addWidget(i_media_);
    auto* title = new QWidget(bar);
    auto* tv = column(title, 0, 0);
    tv->addStretch(1);
    tv->addWidget(label("No shot open", "shotTitle", {}, title));
    tv->addWidget(label("Drop frames, or open a folder", "shotSub", {}, title));
    tv->addStretch(1);
    title->setMinimumWidth(200);
    h->addWidget(title);
    h->addStretch(1);
    tabs_ = new Seg("itabs", {{"rec", "Reconstruct"}, {"grade", "Grade"}, {"deliver", "Deliver"}}, bar);
    tabs_->button("rec")->setObjectName("tabRec");
    tabs_->button("grade")->setObjectName("tabGrade");
    tabs_->button("deliver")->setObjectName("tabDeliver");
    tabs_->clicked = [this](const QString& k) {
        if (!session_.rail_right) run("rail-right");   // a tab brings its inspector back
        show_tab(k);
    };
    tabs_->setFixedHeight(30);
    h->addWidget(tabs_, 0, Qt::AlignVCenter);
    h->addStretch(1);

    auto* pill = styled("devicePill");
    pill->setFixedHeight(24);
    auto* ph = new QHBoxLayout(pill);
    ph->setContentsMargins(10, 0, 10, 0);
    ph->setSpacing(6);
    lamp_ = label("", "lamp", "lamp", pill);
    lamp_->setFixedSize(6, 6);
    device_ = label("\u2014", "device", "pill", pill);
    ph->addWidget(lamp_, 0, Qt::AlignVCenter);
    ph->addWidget(device_, 0, Qt::AlignVCenter);
    h->addWidget(pill, 0, Qt::AlignVCenter);

    ws_ = new Seg("wsswitch", {{"simple", "Simple"}, {"full", "Full"}}, bar);
    ws_->button("simple")->setObjectName("wsSimple");
    ws_->button("full")->setObjectName("wsFull");
    ws_->button("simple")->setToolTip("Hide the scopes and the advanced groups");
    ws_->button("full")->setToolTip("Everything");
    ws_->clicked = [this](const QString& k) { set_workspace(k); };
    ws_->setFixedHeight(26);
    h->addWidget(ws_, 0, Qt::AlignVCenter);

    probe_btn_ = new QPushButton("Probe", bar);
    probe_btn_->setObjectName("probeBtn");
    probe_btn_->setCheckable(true);
    probe_btn_->setToolTip("Read one pixel: baseline, RUDRA, and whether the SDR clipped there");
    connect(probe_btn_, &QPushButton::clicked, this, [this](bool on) {
        probe_on_ = on;
        if (!on) probe_pixel(std::nullopt);
    });
    probe_btn_->setFixedHeight(28);
    h->addWidget(probe_btn_, 0, Qt::AlignVCenter);
    i_scopes_ = new IconButton("iScopes", IconButton::Glyph::Scopes, "Scopes", bar);
    i_inspector_ = new IconButton("iInspector", IconButton::Glyph::Inspector, "Inspector", bar);
    auto* i_export = new IconButton("iExport", IconButton::Glyph::Export, "Export\u2026", bar);
    auto* i_help = new IconButton("iHelp", IconButton::Glyph::Help, "Keyboard shortcuts", bar);
    for (auto* b : {i_scopes_, i_inspector_, i_export, i_help}) {
        b->setFixedSize(32, 28);
        h->addWidget(b);
    }
    connect(i_inspector_, &QToolButton::clicked, this, [this] { run("rail-right"); });
    connect(i_scopes_, &QToolButton::clicked, this, [this] { run("scopes"); });
    connect(i_export, &QToolButton::clicked, this, [this] { open_export_sheet(); });
    connect(i_help, &QToolButton::clicked, this, [this] { run("shortcuts"); });
    return bar;
}

QWidget* MainWindow::build_left_rail() {
    auto* rail = styled("railLeft", "rail-left");
    rail->setFixedWidth(240);
    auto* v = column(rail, 0, 4);
    v->setContentsMargins(10, 12, 10, 12);

    // Open by path: the board's search field.
    auto* seq = new QWidget(rail);
    auto* sh = new QHBoxLayout(seq);
    sh->setContentsMargins(0, 0, 0, 0);
    sh->setSpacing(6);
    seq_path_ = new QLineEdit(seq);
    seq_path_->setObjectName("seqPath");
    seq_path_->setPlaceholderText("Open a folder of frames\u2026");
    seq_path_->setToolTip("A folder of frames, read where it sits: nothing is copied.");
    auto* seq_open = new QPushButton("Open", seq);
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

    auto* shots_head = new QWidget(rail);
    auto* shh = new QHBoxLayout(shots_head);
    shh->setContentsMargins(0, 0, 8, 0);
    shh->addWidget(label("Shots", {}, "section"));
    shh->addStretch(1);
    shot_count_ = label("0", "shotCount", {}, shots_head);
    shh->addWidget(shot_count_, 0, Qt::AlignBottom);
    v->addWidget(shots_head);

    auto* dz = new QFrame(rail);
    dz->setObjectName("dropzone");
    auto* dv = column(dz, 12, 4);
    dv->addWidget(label("Drop footage here", {}, "drop-b"), 0, Qt::AlignHCenter);
    dv->addWidget(label("a frame or a folder of frames", {}, "drop-t"), 0, Qt::AlignHCenter);
    dv->addWidget(label("or", {}, "drop-or"), 0, Qt::AlignHCenter);
    auto* browse = new QPushButton("Open\u200a\u2026", dz);
    browse->setObjectName("browseBtn");
    browse->setProperty("role", "primary");
    connect(browse, &QPushButton::clicked, this, [this] { run("open"); });
    dv->addWidget(browse, 0, Qt::AlignHCenter);
    v->addWidget(dz);

    auto* shots = styled("shots", "shots");
    column(shots, 0, 2);
    v->addWidget(shots);
    frames_empty_ = label("Nothing open yet", "framesEmpty", "empty-rail", rail);
    v->addWidget(frames_empty_);

    // Collections: the frames the clipping lane has marked.
    v->addWidget(label("Collections", {}, "section"));
    const struct { const char* id; const char* dot; const char* text; } cols[] = {
        {"colHighlights", "dot-gold", "Clipped highlights"},
        {"colShadows", "dot-violet", "Crushed shadows"},
        {"colDelivered", "dot-ok", "Delivered"}};
    for (const auto& c : cols) {
        auto* r = styled({}, "collection");
        auto* rh = new QHBoxLayout(r);
        rh->setContentsMargins(8, 5, 8, 5);
        rh->setSpacing(10);
        auto* dot = label({}, {}, c.dot, r);
        dot->setFixedSize(8, 8);
        rh->addWidget(dot, 0, Qt::AlignVCenter);
        rh->addWidget(label(c.text, {}, {}, r));
        rh->addStretch(1);
        rh->addWidget(label("0", c.id, "count", r));
        v->addWidget(r);
    }
    v->addStretch(1);

    // The probe, as a card.
    auto* probe = styled("probePanelWrap", "probepanel");
    auto* pv = column(probe, 0, 0);
    pv->setContentsMargins(12, 10, 12, 10);
    auto* phead = new QWidget(probe);
    auto* phh = new QHBoxLayout(phead);
    phh->setContentsMargins(0, 0, 0, 4);
    phh->addWidget(label("Probe", {}, "card-head"));
    phh->addStretch(1);
    phh->addWidget(label("\u2014", "probeXY", {}, phead));
    pv->addWidget(phead);
    auto* body = styled("probePanel", "pbody");
    auto* bv = column(body, 0, 0);
    auto* nits = new QWidget(body);
    auto* nh = new QHBoxLayout(nits);
    nh->setContentsMargins(0, 0, 0, 0);
    nh->setSpacing(5);
    auto* nit = label("\u2014", "probeNits", "pnit", nits);
    nit->setProperty("state", "idle");
    nh->addWidget(nit, 0, Qt::AlignBottom);
    auto* unit = label("nits", "probeNitsUnit", "pnit-unit", nits);
    unit->hide();
    nh->addWidget(unit, 0, Qt::AlignBottom);
    nh->addStretch(1);
    bv->addWidget(nits);
    auto* delta = label("pick a pixel with Probe, or hold Alt", "probeDelta", "pdelta", body);
    delta->setProperty("state", "idle");
    delta->setWordWrap(true);
    bv->addWidget(delta);
    bv->addSpacing(4);
    for (const auto& [k, id] : std::vector<std::pair<QString, QString>>{
             {"Source", "probeSrc"}, {"Baseline", "probeBase"}, {"RUDRA", "probeModel"}, {"Masks", "probeMask"}})
        bv->addWidget(row(k, label("\u2014", id, "value")));
    pv->addWidget(body);
    v->addWidget(probe);

    // The model card.
    auto* model = styled("modelCard");
    auto* mv = column(model, 0, 2);
    mv->setContentsMargins(12, 10, 12, 10);
    mv->addWidget(label("Model", "modelCardHead", {}, model));
    ckpt_ = label("no model", "ckpt", "pill", model);
    mv->addWidget(ckpt_);
    mv->addWidget(label("File > Model packages\u2026", "modelCardState", {}, model));
    v->addWidget(model);
    return rail;
}

QWidget* MainWindow::build_centre(bool with_viewer) {
    auto* centre = styled("centre", "centre");
    auto* v = column(centre);

    // ── over the picture: the HDR badge, the compare bar, the MaxCLL badge ──
    // (a strip on the surround: widgets cannot float over the viewer's own
    // swapchain window on every platform)
    auto* tools = styled("viewerTools", "vtools");
    tools->setFixedHeight(52);
    auto* th = new QHBoxLayout(tools);
    th->setContentsMargins(16, 8, 16, 8);
    th->setSpacing(10);
    auto* hdr = styled("hdrBadge", "badge");
    hdr->setFixedHeight(26);
    auto* hh = new QHBoxLayout(hdr);
    hh->setContentsMargins(12, 0, 12, 0);
    hh->setSpacing(7);
    auto* dot = label({}, "hdrDot", "dot", hdr);
    dot->setFixedSize(7, 7);
    dot->setProperty("state", "sdr");
    hh->addWidget(dot, 0, Qt::AlignVCenter);
    hh->addWidget(label("SDR", "hdrBadgeText", {}, hdr));
    hh->addWidget(label("display not measured", "hdrBadgeSub", {}, hdr));
    // Badges in two equal sides, so the compare bar sits in the middle.
    auto* left = new QWidget(tools);
    auto* lh = new QHBoxLayout(left);
    lh->setContentsMargins(0, 0, 0, 0);
    lh->addWidget(hdr);
    lh->addStretch(1);
    th->addWidget(left, 1);

    auto* hud = styled("hud");
    hud->setFixedHeight(38);
    auto* hudh = new QHBoxLayout(hud);
    hudh->setContentsMargins(5, 4, 5, 4);
    hudh->setSpacing(8);
    view_mode_ = new Seg("viewMode", {{"model", "RUDRA"}, {"baseline", "Baseline"}, {"#wipeBtn", "Wipe"}}, hud, false);
    view_mode_->button("model")->setToolTip("The reconstruction (1)");
    view_mode_->button("baseline")->setToolTip("The analytic baseline (2)");
    view_mode_->button("#wipeBtn")->setToolTip("Wipe: baseline left, RUDRA right (W)");
    view_mode_->clicked = [this](const QString& k) {
        if (k == "#wipeBtn") run("wipe");
        else session_.set_show(k.toStdString());
    };
    hudh->addWidget(view_mode_);
    hudh->addWidget(styled({}, "hud-sep"));
    view_layer_ = new Seg("viewLayer", {{"0", "Image"}, {"1", "False colour"}, {"2", "Difference"}}, hud);
    view_layer_->button("0")->setToolTip("The composed picture");
    view_layer_->button("1")->setToolTip("Luminance zones, in nits");
    view_layer_->button("2")->setToolTip("What the network changed, and where it did not");
    view_layer_->clicked = [this](const QString& k) { session_.set_view_layer(k.toInt()); };
    hudh->addWidget(view_layer_);
    hudh->addWidget(styled({}, "hud-sep"));
    zoom_seg_ = new Seg("zoomSeg", {{"fit", "Fit"}, {"actual", "100%"}}, hud);
    zoom_seg_->clicked = [this](const QString& k) { run(k == "actual" ? "zoom-actual" : "zoom-fit"); };
    hudh->addWidget(zoom_seg_);
    zoom_val_ = label("100%", "zoomVal", "vstatic", hud);
    zoom_val_->setToolTip("Scroll to zoom, middle-drag to pan, double-click to fit");
    hudh->addWidget(zoom_val_);
    guide_btn_ = new QPushButton("Guides", hud);
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
    hudh->addWidget(guide_btn_);
    th->addWidget(hud);
    auto* right = new QWidget(tools);
    auto* rh = new QHBoxLayout(right);
    rh->setContentsMargins(0, 0, 0, 0);
    rh->addStretch(1);
    th->addWidget(right, 1);
    auto* cll = styled("cllBadgeWrap", "badge");
    cll->setFixedHeight(26);
    auto* ch = new QHBoxLayout(cll);
    ch->setContentsMargins(12, 0, 12, 0);
    ch->addWidget(label("MaxCLL \u2014", "cllBadge", {}, cll));
    rh->addWidget(cll);
    v->addWidget(tools);

    // ── the viewer ──
    viewer_stack_ = new QStackedWidget(centre);
    viewer_stack_->setObjectName("viewer");
    auto* empty = new QWidget(viewer_stack_);
    empty->setObjectName("empty");
    auto* ev = column(empty, 0, 6);
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
    v->addWidget(build_pipe());

    // ── the timeline ──
    auto* transport = styled("transport", "transport");
    transport->setFixedHeight(128);
    auto* tv = column(transport, 0, 8);
    tv->setContentsMargins(16, 10, 16, 10);
    auto* top = new QWidget(transport);
    auto* trh = new QHBoxLayout(top);
    trh->setContentsMargins(0, 0, 0, 0);
    trh->setSpacing(14);
    tc_ = label("01:00:00:00", "tc", "tc", top);
    trh->addWidget(tc_);
    auto* seg = new QWidget(top);
    auto* sgh = new QHBoxLayout(seg);
    sgh->setContentsMargins(8, 0, 0, 0);
    sgh->setSpacing(6);
    btn_prev_ = new IconButton("btnPrev", IconButton::Glyph::Prev, "Previous frame", seg);
    btn_play_ = new IconButton("btnPlay", IconButton::Glyph::Play, "Play", seg);
    btn_next_ = new IconButton("btnNext", IconButton::Glyph::Next, "Next frame", seg);
    for (auto* b : {btn_prev_, btn_play_, btn_next_}) {
        b->setFixedSize(30, 30);
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
    trh->addStretch(1);
    src_info_ = label("\u2014", "srcInfo", "srcinfo", top);
    trh->addWidget(src_info_);
    tv->addWidget(top);
    scrub_ = new ScrubBar(transport);
    scrub_->seek = [this](double f) {
        if (frames_.size() < 2) return;
        play_.stop();
        step_to(int(std::lround(f * double(frames_.size() - 1))));
    };
    tv->addWidget(scrub_, 1);
    v->addWidget(transport);
    return centre;
}

QWidget* MainWindow::build_right_rail() {
    auto* rail = styled("railRight", "rail-right");
    rail->setFixedWidth(320);
    auto* rv = column(rail);
    auto* scroll = new QScrollArea(rail);
    scroll->setObjectName("railscroll");
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* col = styled("railColumn");
    auto* v = column(col, 0, 12);
    v->setContentsMargins(14, 14, 14, 14);
    auto card = [](QWidget* parent, int pad = 12, int spacing = 12) {
        auto* c = styled({}, "card");
        c->setParent(parent);
        auto* cv = column(c, pad, spacing);
        return std::pair{c, cv};
    };

    // The tab's page: the three live in one column and only the tab's shows,
    // so the cards below follow its height.
    panels_ = styled("ipanels");
    panels_->setParent(col);
    auto* pages = column(panels_);

    // Reconstruct
    auto* rec = ctl();
    auto* recv = column(rec, 0, 12);
    recv->addWidget(label("Reconstruction", {}, "title", rec));
    auto [rcard, rcv] = card(rec, 12, 14);
    mode_seg_ = new Seg("mode", {{"all", "All"}, {"highlights", "Highlights"}, {"shadows", "Shadows"}, {"off", "Off"}},
                        rcard);
    mode_seg_->clicked = [this](const QString& k) { session_.set_mode(k.toStdString()); };
    rcv->addWidget(mode_seg_);
    auto slider = [&](const QString& lab, const QString& val_id, const QString& unit, const QString& id, int lo,
                      int hi, QLabel*& val_out, QSlider*& out) {
        auto* w = new QWidget(rcard);
        w->setProperty("role", "slider");
        auto* sv2 = column(w, 0, 6);
        auto* top = new QWidget(w);
        auto* tph = new QHBoxLayout(top);
        tph->setContentsMargins(0, 0, 0, 0);
        tph->setSpacing(4);
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
        rcv->addWidget(w);
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
    preserve_ = new CheckRow("preserve", "Preserve outside masks", "preserveHint", "do-no-harm", rcard);
    preserve_->clicked = [this] { session_.toggle_preserve(); };
    rcv->addWidget(preserve_);
    recv->addWidget(rcard);
    auto* n1 = label("The reconstruction is applied where clipping or crushed shadows made the SDR mapping "
                     "non-invertible. Everywhere else the analytic baseline is preserved unchanged.",
                     {}, "note-p", rec);
    n1->setWordWrap(true);
    notes_.push_back(n1);
    recv->addWidget(n1);
    pages->addWidget(rec);
    pages_.push_back(rec);

    // Grade
    auto* grade = new QWidget;
    auto* gv = column(grade, 0, 12);
    gv->addWidget(label("Region exposure", {}, "title", grade));
    auto [gcard, gcv] = card(grade, 4, 0);
    gcv->addWidget(panel_label("Region EV", "3", "regionCount", gcard));
    region_count_ = gcard->findChild<QLabel*>("regionCount");
    regions_ = new RegionEditor(session_, gcard);
    gcv->addWidget(regions_);
    gv->addWidget(gcard);
    auto* n2 = label("A region is a soft luminance qualifier, not a mask, so it follows the picture. Drag a value to "
                     "scrub it, double-click to zero it. Master applies the identical qualifier and gain to the file.",
                     {}, "note-p", grade);
    n2->setWordWrap(true);
    notes_.push_back(n2);
    gv->addWidget(n2);
    pages->addWidget(grade);
    pages_.push_back(grade);

    // Deliver
    auto* del = ctl();
    auto* dv = column(del, 0, 12);
    dv->addWidget(label("Delivery", {}, "title", del));
    auto [fcard, fcv] = card(del, 12, 0);
    container_field_ = label("OpenEXR \u2014 ACES 2065-1", "containerField", "value");
    primaries_field_ = label("AP0 (ST 2065-4)", "primariesField", "value");
    for (auto* r : {row("Container", container_field_, "field"),
                    row("Transfer", label("linear (scene-referred)", {}, "value"), "field"),
                    row("Primaries", primaries_field_, "field"),
                    row("White", label("diffuse 1.0 = 203 nits", {}, "value"), "field")}) {
        r->layout()->setContentsMargins(0, 9, 0, 9);
        fcv->addWidget(r);
    }
    dv->addWidget(fcard);
    auto [ocard, ocv] = card(del, 12, 10);
    anchor_ = new CheckRow("anchor", "Anchor to source exposure", "anchorHint", "conform", ocard);
    anchor_->clicked = [this] {
        session_.toggle_anchor();
        log(session_.anchor ? "master will anchor to the source exposure"
                            : "master will keep the inverse tone map's own level");
    };
    carry_chroma_ = new CheckRow("carryChroma", "Carry source chroma", "chromaHint", "below the clip", ocard);
    carry_chroma_->clicked = [this] {
        session_.toggle_carry_chroma();
        log(session_.carry_chroma ? "master will take hue from the source below the clip"
                                  : "master will keep the per-channel expansion's own hue");
    };
    ocv->addWidget(anchor_);
    ocv->addWidget(carry_chroma_);
    dv->addWidget(ocard);

    auto [rendc, form] = card(del, 12, 4);
    form->addWidget(label("Render", {}, "card-head", rendc));
    auto* dir = new QLineEdit(rendc);
    dir->setObjectName("renderDir");
    dir->setPlaceholderText("C:\\Renders\\Shot01");
    auto* mode = new QComboBox(rendc);
    mode->setObjectName("renderMode");
    mode->addItem("Current image", "image");
    mode->addItem("All loaded frames \u2014 sequence", "sequence");
    auto* name = new QLineEdit("master", rendc);
    name->setObjectName("renderName");
    auto* start = new QSpinBox(rendc);
    start->setObjectName("renderStart");
    start->setRange(0, 9999999);
    start->setValue(1);
    for (const auto& [text, field] : std::vector<std::pair<QString, QWidget*>>{
             {"Render folder", dir}, {"Output", mode}, {"Render name", name}, {"Start frame", start}}) {
        form->addWidget(label(text, {}, "note", rendc));
        form->addWidget(field);
        form->addSpacing(2);
    }
    auto* small = label("Sequence uses loaded frame order and the current grade for every frame. Nothing is ever "
                        "replaced.",
                        {}, "small", rendc);
    small->setWordWrap(true);
    form->addWidget(small);
    auto* status = label("", "renderStatus", "render-status", rendc);
    status->setWordWrap(true);
    status->hide();
    form->addWidget(status);
    auto* acts = new QWidget(rendc);
    auto* ah = new QHBoxLayout(acts);
    ah->setContentsMargins(0, 6, 0, 0);
    ah->setSpacing(8);
    btn_reprocess_ = new QPushButton("Reprocess", acts);
    btn_reprocess_->setObjectName("btnReprocess");
    btn_master_ = new QPushButton("Master EXR", acts);
    btn_master_->setObjectName("btnMaster");
    btn_master_->setProperty("role", "primary");
    connect(btn_reprocess_, &QPushButton::clicked, this, [this] {
        if (!frames_.empty()) step_to(current_);
    });
    connect(btn_master_, &QPushButton::clicked, this, [this] { run("master"); });
    ah->addWidget(btn_reprocess_, 1);
    ah->addWidget(btn_master_, 1);
    form->addWidget(acts);
    dv->addWidget(rendc);
    auto* n3 = label("ProRes, HDR10 and HLG are written by <span style=\"font-family:'Geist Mono'\">rudra deliver</span> "
                     "from the command line until video delivery arrives in the app.",
                     {}, "note-p", del);
    n3->setTextFormat(Qt::RichText);
    n3->setWordWrap(true);
    notes_.push_back(n3);
    dv->addWidget(n3);
    pages->addWidget(del);
    pages_.push_back(del);
    v->addWidget(panels_);

    // ── this frame ──
    auto* stats = styled("frameStats", "card");
    auto* fv = column(stats, 12, 6);
    auto* shead = new QWidget(stats);
    auto* shh = new QHBoxLayout(shead);
    shh->setContentsMargins(0, 0, 0, 0);
    shh->addWidget(label("This frame", {}, "card-head"));
    shh->addStretch(1);
    shh->addWidget(label("measured", {}, "note"));
    fv->addWidget(shead);
    for (const char* id : {"measA", "measB"}) {
        auto* ms = styled(id, "ms");
        column(ms);
        fv->addWidget(ms);
    }
    clip_bar_ = new ClipBar(stats);
    clip_bar_->hide();   // until a frame is measured
    fv->addWidget(clip_bar_);
    v->addWidget(stats);

    // ── scopes, each a card ──
    scopes_ = styled("scopes", "scopes");
    auto* sv = column(scopes_, 0, 12);
    const struct { const char* title; const char* note; const char* id; int h; } scopes[] = {
        {"Waveform", "luma percentile \u00b7 nits/log", "wave", 112},
        {"Histogram", "absolute nits \u00b7 log2 stops \u00b7 DW", "hist", 84},
        {"Vectorscope", "Rec.2020", "vector", 0}};
    for (const auto& s : scopes) {
        auto [sc, scv] = card(scopes_, 12, 8);
        auto* head = new QWidget(sc);
        auto* hh = new QHBoxLayout(head);
        hh->setContentsMargins(0, 0, 0, 0);
        hh->addWidget(label(s.title, {}, "card-head"));
        hh->addStretch(1);
        hh->addWidget(label(s.note, {}, "note"));
        scv->addWidget(head);
        auto* plot = styled({}, "plot");
        auto* pl = column(plot);
        if (s.h > 0) {
            auto* area = new ScopePlot(s.id, QString(s.id) == "wave" ? ScopePlot::Kind::Waveform
                                                                      : ScopePlot::Kind::Histogram,
                                       s.h, plot);
            (QString(s.id) == "wave" ? wave_ : hist_) = area;
            pl->addWidget(area);
        } else {
            vector_ = new VectorscopeView(plot);
            pl->addWidget(vector_);
        }
        scv->addWidget(plot);
        sv->addWidget(sc);
    }
    v->addWidget(scopes_);

    // ── log ──
    auto [lcard, lcv] = card(col, 12, 6);
    lcv->addWidget(label("Log", {}, "card-head", lcard));
    log_ = new QPlainTextEdit(lcard);
    log_->setObjectName("log");
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(400);
    log_->setFixedHeight(88);
    lcv->addWidget(log_);
    v->addWidget(lcard);
    v->addStretch(1);
    scroll->setWidget(col);
    rv->addWidget(scroll, 1);
    return rail;
}

QWidget* MainWindow::build_pipe() {
    auto* pipe = styled("pipe", "pipe");
    pipe->setFixedHeight(30);
    auto* h = new QHBoxLayout(pipe);
    h->setContentsMargins(12, 0, 12, 0);
    h->setSpacing(8);
    h->addWidget(label("\u2014", "statusMask", "pipe-s", pipe));
    h->addStretch(1);
    auto seg = [&](const QString& id, const QString& text) {
        auto* b = label(text, id, "pipe-b", pipe);
        h->addWidget(b);
        return b;
    };
    auto arrow = [&] { h->addWidget(label("\u203a", {}, "pipe-arrow", pipe)); };
    seg("pipeIn", "sRGB \u00b7 Rec.709");
    arrow();
    seg("pipeWorking", "scene-linear \u00b7 203 nits = 1.0");
    arrow();
    view_transform_ = seg("viewTransform", "exposure + clip \u00b7 203 nits");
    arrow();
    seg("pipeMaster", "ACES 2065-1 EXR, half");
    auto* warn = label("", "pipeWarn", {}, pipe);
    warn->hide();
    h->addWidget(warn);
    h->addStretch(1);
    h->addWidget(label("\u2014", "statusTime", "pipe-s", pipe));
    return pipe;
}

}  // namespace rudra::app
