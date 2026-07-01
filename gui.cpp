#include "gui.h"
#include "model.h"
#include "dataset.h"
#include <QApplication>
#include <QMessageBox>
#include <QMutexLocker>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>

constexpr int GUI_MAX_ITER = 256;

// ---- AccuracyPlot ----
void AccuracyPlot::addPoint(float train, float test) {
    train_errs.push_back(1.f - train);
    test_errs.push_back(1.f - test);
    if (train_errs.size() > 5000) {
        train_errs.pop_front();
        test_errs.pop_front();
    }
    update();
}

static void drawTextRight(QPainter& p, int x, int y, const QString& text) {
    p.drawText(x - 5, y + 4, text);
}

static void drawTextCenter(QPainter& p, int x, int y, const QString& text) {
    QRect r(x - 40, y + 2, 80, 20);
    p.drawText(r, Qt::AlignCenter, text);
}

void AccuracyPlot::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int w = width(), h = height();
    int pad = 50, rw = w - 2 * pad, rh = h - 2 * pad;
    if (rw < 10 || rh < 10) return;

    p.fillRect(rect(), QColor(245, 245, 250));
    p.setPen(Qt::black);
    p.drawRect(pad, pad, rw, rh);

    int n = (int)train_errs.size();
    if (n < 2) {
        p.drawText(rect(), Qt::AlignCenter, "Training...");
        return;
    }

    p.setPen(QPen(QColor(220, 220, 220), 1));
    for (int i = 0; i <= 4; ++i) {
        int y = pad + rh - (rh * i / 4);
        p.drawLine(pad, y, pad + rw, y);
        p.setPen(Qt::black);
        drawTextRight(p, pad, y, QString::number(i / 4.0, 'f', 2));
        p.setPen(QPen(QColor(220, 220, 220), 1));
    }
    for (int i = 0; i <= 4; ++i) {
        int x = pad + (rw * i / 4);
        p.drawLine(x, pad, x, pad + rh);
        p.setPen(Qt::black);
        drawTextCenter(p, x, pad + rh, QString::number(i * n / 4));
        p.setPen(QPen(QColor(220, 220, 220), 1));
    }

    auto draw_line = [&](const std::deque<float>& data, const QColor& color) {
        if (data.size() < 2) return;
        QPainterPath path;
        for (int i = 0; i < (int)data.size(); ++i) {
            float fx = pad + (float)i / (n - 1) * rw;
            float fy = pad + rh - data[i] * rh;
            if (i == 0) path.moveTo(fx, fy);
            else path.lineTo(fx, fy);
        }
        p.setPen(QPen(color, 2));
        p.drawPath(path);
    };
    draw_line(train_errs, QColor(50, 120, 220));
    draw_line(test_errs, QColor(220, 80, 50));

    p.setPen(Qt::black);
    drawTextCenter(p, pad + rw / 2, h - 20, "Epoch");
    QTransform tf = p.transform();
    p.translate(12, pad + rh / 2);
    p.rotate(-90);
    p.drawText(0, 0, "Error");
    p.setTransform(tf);

    p.fillRect(w - 140, 8, 130, 40, QColor(255, 255, 255, 200));
    p.setPen(QPen(QColor(50, 120, 220), 2));
    p.drawLine(w - 130, 20, w - 110, 20);
    p.setPen(Qt::black);
    p.drawText(w - 105, 24, "Train");
    p.setPen(QPen(QColor(220, 80, 50), 2));
    p.drawLine(w - 130, 38, w - 110, 38);
    p.setPen(Qt::black);
    p.drawText(w - 105, 42, "Test");
}

// ---- HeatmapWidget ----
void HeatmapWidget::setData(const std::vector<float>& data, int rows, int cols,
                            Type type, float vmin, float vmax) {
    m_data = data; m_rows = rows; m_cols = cols;
    m_type = type; m_vmin = vmin; m_vmax = vmax;
    if (m_vmax <= m_vmin) m_vmax = m_vmin + 1.f;
    update();
}

void HeatmapWidget::clear() {
    m_data.clear(); m_rows = 0; m_cols = 0;
    update();
}

void HeatmapWidget::setViewport(float cx, float cy, float zoom) {
    m_vp_cx = cx;
    m_vp_cy = cy;
    m_vp_zoom = zoom;
    update();
}

void HeatmapWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor(240, 240, 240));
    if (m_data.empty() || m_rows < 1 || m_cols < 1) {
        p.drawText(rect(), Qt::AlignCenter, "Waiting...");
        return;
    }

    auto colormap = [](float v, float vmin, float vmax, HeatmapWidget::Type type) -> QRgb {
        float t = (v - vmin) / (vmax - vmin);
        t = std::max(0.f, std::min(1.f, t));
        if (type == HeatmapWidget::Weight) {
            float tt = 2.f * t - 1.f;
            float r, g, b;
            if (tt < 0) { r = 1.f + tt; g = 1.f + tt; b = 1.f; }
            else { r = 1.f; g = 1.f - tt; b = 1.f - tt; }
            return qRgba(int(r * 255), int(g * 255), int(b * 255), 255);
        }
        struct Stop { float pos; float r, g, b; };
        static const Stop stops[] = {
            {0.00f, 0.00f, 0.00f, 0.50f},
            {0.15f, 0.00f, 0.00f, 1.00f},
            {0.30f, 0.00f, 1.00f, 1.00f},
            {0.50f, 0.00f, 0.80f, 0.00f},
            {0.70f, 1.00f, 1.00f, 0.00f},
            {0.85f, 1.00f, 0.40f, 0.00f},
            {1.00f, 0.60f, 0.00f, 0.00f},
        };
        int n = sizeof(stops) / sizeof(stops[0]);
        float r, g, b;
        auto lerp = [&](int i, float f) {
            r = stops[i].r + (stops[i+1].r - stops[i].r) * f;
            g = stops[i].g + (stops[i+1].g - stops[i].g) * f;
            b = stops[i].b + (stops[i+1].b - stops[i].b) * f;
        };
        if (t <= stops[0].pos) { r = stops[0].r; g = stops[0].g; b = stops[0].b; }
        else if (t >= stops[n-1].pos) { r = stops[n-1].r; g = stops[n-1].g; b = stops[n-1].b; }
        else {
            for (int i = 0; i < n - 1; ++i) {
                if (t >= stops[i].pos && t < stops[i+1].pos) {
                    lerp(i, (t - stops[i].pos) / (stops[i+1].pos - stops[i].pos));
                    break;
                }
            }
        }
        return qRgba(int(r * 255), int(g * 255), int(b * 255), 255);
    };

    int W = width(), H = height();
    QImage img(W, H, QImage::Format_ARGB32);
    float vp_w = 1.0f / m_vp_zoom;
    float vp_h = 1.0f / m_vp_zoom;
    float x0 = m_vp_cx - vp_w * 0.5f;
    float x1 = m_vp_cx + vp_w * 0.5f;
    float y0 = m_vp_cy - vp_h * 0.5f;
    float y1 = m_vp_cy + vp_h * 0.5f;

    for (int py = 0; py < H; ++py) {
        auto* scan = (QRgb*)img.scanLine(py);
        float ny = (float)py / H;
        float dy = y0 + ny * (y1 - y0);
        if (dy < 0.f) dy = 0.f; else if (dy > 1.f) dy = 1.f;
        int ri = (int)(dy * (m_rows - 1));
        if (ri >= m_rows) ri = m_rows - 1;
        int base = ri * m_cols;
        for (int px = 0; px < W; ++px) {
            float nx = (float)px / W;
            float dx = x0 + nx * (x1 - x0);
            if (dx < 0.f) dx = 0.f; else if (dx > 1.f) dx = 1.f;
            int ci = (int)(dx * (m_cols - 1));
            if (ci >= m_cols) ci = m_cols - 1;
            scan[px] = colormap(m_data[base + ci], m_vmin, m_vmax, m_type);
        }
    }
    p.drawImage(0, 0, img);
}

// ---- MainWindow ----
MainWindow::MainWindow() {
    setWindowTitle("Gandalf - Training GUI");
    resize(1100, 750);

    auto* central = new QWidget;
    setCentralWidget(central);
    auto* hsplit = new QSplitter(Qt::Horizontal, central);
    auto* main_lo = new QVBoxLayout(central);
    main_lo->setContentsMargins(0,0,0,0);
    main_lo->addWidget(hsplit);

    auto* left = new QWidget;
    auto* left_lo = new QVBoxLayout(left);
    left_lo->setContentsMargins(6,6,6,6);

    auto* ctrl_group = new QGroupBox("Controls");
    auto* form = new QFormLayout(ctrl_group);

    resolution_spin = new QSpinBox; resolution_spin->setRange(2, 1024); resolution_spin->setValue(16);
    form->addRow("Resolution:", resolution_spin);
    epochs_spin = new QSpinBox; epochs_spin->setRange(1, 999999999); epochs_spin->setValue(200000000);
    form->addRow("Epochs:", epochs_spin);
    batch_spin = new QSpinBox; batch_spin->setRange(1, 8192); batch_spin->setValue(128);
    form->addRow("Batch:", batch_spin);
    lr_spin = new QDoubleSpinBox; lr_spin->setRange(1e-6, 1);
    lr_spin->setDecimals(6); lr_spin->setValue(0.001);
    form->addRow("LR:", lr_spin);
    d_model_spin = new QSpinBox; d_model_spin->setRange(8, 1024); d_model_spin->setValue(64);
    form->addRow("d_model:", d_model_spin);
    neurons_spin = new QSpinBox; neurons_spin->setRange(8, 4096); neurons_spin->setValue(128);
    form->addRow("Neurons:", neurons_spin);
    layers_spin = new QSpinBox; layers_spin->setRange(1, 32); layers_spin->setValue(1);
    form->addRow("Layers:", layers_spin);

    weight_decay_spin = new QDoubleSpinBox; weight_decay_spin->setRange(0, 1);
    weight_decay_spin->setDecimals(6); weight_decay_spin->setValue(0);
    form->addRow("Weight decay:", weight_decay_spin);
    activation_combo = new QComboBox;
    activation_combo->addItems({"gelu","relu","silu","tanh","identity"});
    form->addRow("Activation:", activation_combo);
    heatmap_check = new QCheckBox("Enable heatmaps");
    heatmap_check->setChecked(true);
    form->addRow(heatmap_check);
    heatmap_every_spin = new QSpinBox; heatmap_every_spin->setRange(1, 1000); heatmap_every_spin->setValue(5);
    form->addRow("Heatmap every N:", heatmap_every_spin);

    left_lo->addWidget(ctrl_group);

    auto* btn_row = new QHBoxLayout;
    start_btn = new QPushButton("Start");
    pause_btn = new QPushButton("Pause");
    stop_btn = new QPushButton("Stop");
    stop_btn->setEnabled(false);
    pause_btn->setEnabled(false);
    btn_row->addWidget(start_btn);
    btn_row->addWidget(pause_btn);
    btn_row->addWidget(stop_btn);
    left_lo->addLayout(btn_row);

    log_edit = new QTextEdit;
    log_edit->setReadOnly(true);
    log_edit->setMaximumHeight(200);
    left_lo->addWidget(log_edit, 1);
    hsplit->addWidget(left);

    auto* right = new QWidget;
    auto* right_lo = new QVBoxLayout(right);
    right_lo->setContentsMargins(6,6,6,6);

    accuracy_plot = new AccuracyPlot;
    accuracy_plot->setMinimumHeight(250);
    right_lo->addWidget(accuracy_plot, 1);

    auto* hm_container = new QWidget;
    auto* hm_lo = new QVBoxLayout(hm_container);
    auto* hm_row1 = new QHBoxLayout;

    auto add_hm = [this](const QString& title, QWidget*& w, HeatmapWidget*& hm) {
        w = new QWidget;
        auto* lo = new QVBoxLayout(w);
        lo->setContentsMargins(0,0,0,0);
        auto* lbl = new QLabel(title);
        lbl->setAlignment(Qt::AlignCenter);
        lo->addWidget(lbl);
        hm = new HeatmapWidget;
        lo->addWidget(hm, 1);
    };
    QWidget *wt, *wp, *wd;
    add_hm("Target", wt, hm_target);
    add_hm("Prediction", wp, hm_pred);
    add_hm("|Diff|", wd, hm_diff);
    hm_row1->addWidget(wt); hm_row1->addWidget(wp); hm_row1->addWidget(wd);

    hm_lo->addLayout(hm_row1);
    right_lo->addWidget(hm_container, 1);

    hsplit->addWidget(right);
    hsplit->setStretchFactor(0, 0);
    hsplit->setStretchFactor(1, 1);

    connect(start_btn, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(stop_btn, &QPushButton::clicked, this, &MainWindow::stopTraining);
    connect(pause_btn, &QPushButton::clicked, this, &MainWindow::togglePause);

    heatmap_timer = new QTimer(this);
    connect(heatmap_timer, &QTimer::timeout, this, &MainWindow::pollHeatmap);
    heatmap_timer->setInterval(2000);

    auto* zi = new QShortcut(QKeySequence(Qt::Key_Plus), this);
    connect(zi, &QShortcut::activated, this, [this]() { adjustZoom(1.5f); });
    auto* zi2 = new QShortcut(QKeySequence(Qt::Key_Equal), this);
    connect(zi2, &QShortcut::activated, this, [this]() { adjustZoom(1.5f); });
    auto* zo = new QShortcut(QKeySequence(Qt::Key_Minus), this);
    connect(zo, &QShortcut::activated, this, [this]() { adjustZoom(1.0f / 1.5f); });
    auto* rst = new QShortcut(QKeySequence(Qt::Key_0), this);
    connect(rst, &QShortcut::activated, this, [this]() {
        m_vp_cx = 0.5f; m_vp_cy = 0.5f; m_vp_zoom = 1.0f;
        updateHeatmapViewport();
    });

    auto* sk_left = new QShortcut(QKeySequence(Qt::Key_Left), this);
    connect(sk_left, &QShortcut::activated, this, [this]() { pan(-0.15f, 0.0f); });
    auto* sk_right = new QShortcut(QKeySequence(Qt::Key_Right), this);
    connect(sk_right, &QShortcut::activated, this, [this]() { pan(0.15f, 0.0f); });
    auto* up = new QShortcut(QKeySequence(Qt::Key_Up), this);
    connect(up, &QShortcut::activated, this, [this]() { pan(0.0f, -0.15f); });
    auto* down = new QShortcut(QKeySequence(Qt::Key_Down), this);
    connect(down, &QShortcut::activated, this, [this]() { pan(0.0f, 0.15f); });
}

MainWindow::~MainWindow() {
    stopTraining();
    worker_thread.quit();
    worker_thread.wait(3000);
}

void MainWindow::startTraining() {
    if (worker && worker_thread.isRunning()) {
        onLogMessage("Training already running");
        return;
    }

    worker = new TrainWorker;
    worker->resolution = resolution_spin->value();
    worker->epochs = epochs_spin->value();
    worker->batch = batch_spin->value();
    worker->lr = (float)lr_spin->value();
    worker->d_model = d_model_spin->value();
    worker->ff_hidden = neurons_spin->value();
    worker->n_hidden_layers = layers_spin->value();
    worker->weight_decay = (float)weight_decay_spin->value();
    worker->activation = activation_combo->currentText().toStdString();
    worker->heatmap_enabled = heatmap_check->isChecked();
    worker->heatmap_every = heatmap_every_spin->value();
    worker->moveToThread(&worker_thread);

    connect(&worker_thread, &QThread::started, worker, &TrainWorker::run);
    connect(worker, &TrainWorker::logMessage, this, &MainWindow::onLogMessage);
    connect(worker, &TrainWorker::epochDone, this, &MainWindow::onEpochDone);
    connect(worker, &TrainWorker::finished, this, &MainWindow::onTrainingFinished);
    connect(worker, &TrainWorker::finished, &worker_thread, &QThread::quit);
    connect(worker, &TrainWorker::finished, worker, &QObject::deleteLater);

    start_btn->setEnabled(false);
    stop_btn->setEnabled(true);
    pause_btn->setEnabled(true);
    pause_btn->setText("Pause");

    m_hm_resolution = 0;
    m_hm_target.clear();
    m_vp_cx = 0.5f; m_vp_cy = 0.5f; m_vp_zoom = 1.0f;
    accuracy_plot->train_errs.clear();
    accuracy_plot->test_errs.clear();
    hm_target->clear();
    hm_pred->clear();
    hm_diff->clear();

    worker_thread.start();
    heatmap_timer->start();
}

void MainWindow::stopTraining() {
    if (worker) worker->stop_flag.store(1);
    heatmap_timer->stop();
}

void MainWindow::togglePause() {
    if (!worker) return;
    if (worker->pause_flag.load()) {
        worker->pause_flag.store(0);
        pause_btn->setText("Pause");
    } else {
        worker->pause_flag.store(1);
        pause_btn->setText("Resume");
    }
}

void MainWindow::updateHeatmapViewport() {
    hm_target->setViewport(m_vp_cx, m_vp_cy, m_vp_zoom);
    hm_pred->setViewport(m_vp_cx, m_vp_cy, m_vp_zoom);
    hm_diff->setViewport(m_vp_cx, m_vp_cy, m_vp_zoom);
}

void MainWindow::adjustZoom(float factor) {
    float new_zoom = m_vp_zoom * factor;
    if (new_zoom < 1.0f) new_zoom = 1.0f;
    if (new_zoom > 128.0f) new_zoom = 128.0f;
    m_vp_zoom = new_zoom;
    updateHeatmapViewport();
}

void MainWindow::pan(float dx, float dy) {
    float view_size = 1.0f / m_vp_zoom;
    m_vp_cx += dx * view_size;
    m_vp_cy += dy * view_size;
    if (m_vp_cx < 0.0f) m_vp_cx = 0.0f;
    if (m_vp_cx > 1.0f) m_vp_cx = 1.0f;
    if (m_vp_cy < 0.0f) m_vp_cy = 0.0f;
    if (m_vp_cy > 1.0f) m_vp_cy = 1.0f;
    updateHeatmapViewport();
}

void MainWindow::onEpochDone(int, float train_acc, float test_acc, float) {
    accuracy_plot->addPoint(train_acc, test_acc);
}

void MainWindow::onLogMessage(const QString& msg) {
    log_edit->append(msg);
}

void MainWindow::onTrainingFinished() {
    start_btn->setEnabled(true);
    stop_btn->setEnabled(false);
    pause_btn->setEnabled(false);
    worker = nullptr;
    heatmap_timer->stop();
    onLogMessage("Training finished");
}

void MainWindow::pollHeatmap() {
    if (!worker || !worker->heatmap_enabled) return;

    int R;
    std::vector<float> pred_copy;
    {
        QMutexLocker l(&worker->hm_mutex);
        if (!worker->hm_updated) return;
        R = worker->hm_resolution;
        pred_copy = worker->hm_pred_data;
        worker->hm_updated = false;
    }
    if (R < 2) return;

    if (R != m_hm_resolution) {
        m_hm_target.resize(R * R);
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < R; ++j) {
                float x = (i + 0.5f) / R;
                float y = (j + 0.5f) / R;
                m_hm_target[i * R + j] = (float)mandelbrot_cont(x, y, GUI_MAX_ITER);
            }
        m_hm_resolution = R;
    }

    std::vector<float> pred(R * R), diff(R * R);
    for (int i = 0; i < R * R; ++i) {
        pred[i] = pred_copy.empty() ? 0.f : pred_copy[i];
        diff[i] = std::abs(pred[i] - m_hm_target[i]);
    }

    float max_v = (float)GUI_MAX_ITER;
    hm_target->setData(m_hm_target, R, R, HeatmapWidget::Target, 0, max_v);
    hm_pred->setData(pred, R, R, HeatmapWidget::Prediction, 0, max_v);
    hm_diff->setData(diff, R, R, HeatmapWidget::Diff, 0, max_v);
}

// ---- TrainWorker ----
void TrainWorker::run() {
    int R = resolution;
    int d_dim = d_model;
    int ff_h = ff_hidden;
    int nL = n_hidden_layers;
    int B = batch;
    auto f = [](float x, float y) { return mandelbrot_cont(x, y, GUI_MAX_ITER); };

    auto [train_ds, test_ds] = PointDataset::train_test_split(10000, 0.2, f);

    FFNNAttentionModel model(d_dim, ff_h, nL, activation);

    struct MState { Eigen::MatrixXf m, v; };
    std::vector<MState> mat_states;
    std::vector<Eigen::MatrixXf*> mat_ptrs;
    auto add_mat = [&](Eigen::MatrixXf& p) {
        mat_ptrs.push_back(&p);
        mat_states.push_back({Eigen::MatrixXf::Zero(p.rows(), p.cols()),
                              Eigen::MatrixXf::Zero(p.rows(), p.cols())});
    };
    add_mat(model.W_in); add_mat(model.U); add_mat(model.P);
    add_mat(model.Wq); add_mat(model.Wk); add_mat(model.Wv);
    for (auto& w : model.W) add_mat(w);

    struct VState { Eigen::VectorXf m, v; };
    std::vector<VState> vec_states;
    std::vector<Eigen::VectorXf*> vec_ptrs;
    auto add_vec = [&](Eigen::VectorXf& p) {
        vec_ptrs.push_back(&p);
        vec_states.push_back({Eigen::VectorXf::Zero(p.size()),
                              Eigen::VectorXf::Zero(p.size())});
    };
    add_vec(model.ln1_gamma); add_vec(model.ln1_beta);
    add_vec(model.ln2_gamma); add_vec(model.ln2_beta);
    for (auto& bv : model.b) add_vec(bv);

    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int t = 0;

    std::vector<int> idx(train_ds.get_size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 srng(42);

    emit logMessage(QString("GUI: d=%1 ff=%2 layers=%3 lr=%4 act=%5 batch=%6")
                    .arg(d_dim).arg(ff_h).arg(nL).arg(lr)
                    .arg(QString::fromStdString(activation)).arg(B));

    auto gen_heatmap = [&]() {
        if (!heatmap_enabled) return;
        int HM = std::min(R, 512);
        std::vector<float> grid_pred(HM * HM);
        Eigen::MatrixXf grid_coords(HM * HM, 2);
        for (int i = 0; i < HM; ++i)
            for (int j = 0; j < HM; ++j) {
                grid_coords(i * HM + j, 0) = (i + 0.5f) / HM;
                grid_coords(i * HM + j, 1) = (j + 0.5f) / HM;
            }
        for (int gs = 0; gs < HM * HM; gs += 16384) {
            if (stop_flag.load()) break;
            int ge = std::min(gs + 16384, HM * HM);
            int gB = ge - gs;
            Eigen::MatrixXf g_in = grid_coords.middleRows(gs, gB);
            Eigen::MatrixXf g_pred = model.forward(g_in);
            for (int k = 0; k < gB; ++k)
                grid_pred[gs + k] = g_pred(k, 0);
        }
        {
            QMutexLocker l(&hm_mutex);
            hm_pred_data = std::move(grid_pred);
            hm_resolution = HM;
            hm_updated = true;
        }
    };

    gen_heatmap();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        if (stop_flag.load()) break;

        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(idx.begin(), idx.end(), srng);

        float train_loss = 0.f;
        int correct = 0, total = 0;
        Eigen::MatrixXf coords;

        for (int bi = 0; bi < train_ds.get_size(); bi += B) {
            if (stop_flag.load()) break;
            while (pause_flag.load()) {
                if (stop_flag.load()) break;
                QThread::msleep(100);
            }
            if (stop_flag.load()) break;

            int end = std::min(bi + B, train_ds.get_size());
            int B_actual = end - bi;

            coords.resize(B_actual, 2);
            Eigen::VectorXi targets(B_actual);
            for (int si = 0; si < B_actual; ++si) {
                auto [x, y, tgt] = train_ds.get(idx[bi + si]);
                coords(si, 0) = x; coords(si, 1) = y;
                targets(si) = tgt;
            }

            model.zero_grad();
            Eigen::MatrixXf pred = model.forward(coords);

            Eigen::VectorXf diff = pred.col(0) - targets.cast<float>();
            float batch_loss = diff.array().square().mean();
            train_loss += batch_loss * B_actual;

            Eigen::MatrixXf dlogits(B_actual, 1);
            dlogits.col(0) = 2.f * diff / (float)B_actual;
            for (int si = 0; si < B_actual; ++si) {
                if (std::abs(pred(si, 0) - targets(si)) < 0.5f)
                    ++correct;
            }
            total += B_actual;

            model.backward(dlogits);

            if (weight_decay > 0.f) {
                auto wd = [&](auto& g, const auto& p) { g += weight_decay * p; };
                wd(model.grad.dWq, model.Wq); wd(model.grad.dWk, model.Wk);
                wd(model.grad.dWv, model.Wv);
                wd(model.grad.dU, model.U);
                wd(model.grad.dW_in, model.W_in); wd(model.grad.dP, model.P);
                for (int i = 0; i <= model.n_hidden_layers; ++i)
                    wd(model.grad.dW[i], model.W[i]);
            }

            ++t;
            float m_bias = 1.f - std::pow(beta1, t);
            float v_bias = 1.f - std::pow(beta2, t);
            std::vector<Eigen::MatrixXf*> mat_gs = {
                &model.grad.dW_in, &model.grad.dU, &model.grad.dP,
                &model.grad.dWq, &model.grad.dWk, &model.grad.dWv,
            };
            for (auto& dw : model.grad.dW) mat_gs.push_back(&dw);
            for (size_t i = 0; i < mat_ptrs.size(); ++i) {
                auto& s = mat_states[i];
                const auto& g = *mat_gs[i];
                s.m = beta1 * s.m + (1.f - beta1) * g;
                s.v = beta2 * s.v + (1.f - beta2) * g.cwiseProduct(g);
                *mat_ptrs[i] -= (lr * (s.m.array() / m_bias) / ((s.v.array() / v_bias).sqrt() + eps)).matrix();
            }
            std::vector<Eigen::VectorXf*> vec_gs = {
                &model.grad.dln1_gamma, &model.grad.dln1_beta,
                &model.grad.dln2_gamma, &model.grad.dln2_beta,
            };
            for (auto& db : model.grad.db) vec_gs.push_back(&db);
            for (size_t i = 0; i < vec_ptrs.size(); ++i) {
                auto& s = vec_states[i];
                const auto& g = *vec_gs[i];
                s.m = beta1 * s.m + (1.f - beta1) * g;
                s.v = beta2 * s.v + (1.f - beta2) * g.cwiseProduct(g);
                *vec_ptrs[i] -= (lr * (s.m.array() / m_bias) / ((s.v.array() / v_bias).sqrt() + eps)).matrix();
            }
        }

        float test_loss = 0.f;
        int test_correct = 0;
        for (int si = 0; si < test_ds.get_size(); si += B) {
            if (stop_flag.load()) break;
            int end = std::min(si + B, test_ds.get_size());
            int B_actual = end - si;
            Eigen::MatrixXf coords(B_actual, 2);
            Eigen::VectorXi targets(B_actual);
            for (int k = 0; k < B_actual; ++k) {
                auto [x, y, tgt] = test_ds.get(si + k);
                coords(k, 0) = x; coords(k, 1) = y;
                targets(k) = tgt;
            }
            Eigen::MatrixXf pred = model.forward(coords);
            Eigen::VectorXf diff = pred.col(0) - targets.cast<float>();
            test_loss += diff.array().square().sum();
            for (int k = 0; k < B_actual; ++k) {
                if (std::abs(pred(k, 0) - targets(k)) < 0.5f)
                    ++test_correct;
            }
        }
        test_loss /= test_ds.get_size();
        float test_acc = (float)test_correct / test_ds.get_size();
        float train_acc = (float)correct / total;

        auto t1 = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(t1 - t0).count();

        emit epochDone(epoch, train_acc, test_acc, elapsed);

        std::ostringstream oss;
        oss << "Epoch " << std::setw(3) << epoch
            << " | test acc " << std::fixed << std::setprecision(5) << test_acc
            << " | time " << std::fixed << std::setprecision(2) << elapsed << "s";
        emit logMessage(QString::fromStdString(oss.str()));

        if (heatmap_enabled && (epoch % heatmap_every == 0) && !stop_flag.load())
            gen_heatmap();
    }

    stop_flag.store(1);
    emit finished();
}

int run_gui(int argc, char** argv) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
