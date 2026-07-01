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

// ---- AccuracyPlot ----
void AccuracyPlot::addPoint(float train, float test) {
    train_accs.push_back(train);
    test_accs.push_back(test);
    if (train_accs.size() > 5000) {
        train_accs.pop_front();
        test_accs.pop_front();
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

    int n = (int)train_accs.size();
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
    draw_line(train_accs, QColor(50, 120, 220));
    draw_line(test_accs, QColor(220, 80, 50));

    p.setPen(Qt::black);
    drawTextCenter(p, pad + rw / 2, h - 20, "Epoch");
    QTransform tf = p.transform();
    p.translate(12, pad + rh / 2);
    p.rotate(-90);
    p.drawText(0, 0, "Accuracy");
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

void HeatmapWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor(240, 240, 240));
    if (m_data.empty() || m_rows < 1 || m_cols < 1) {
        p.drawText(rect(), Qt::AlignCenter, "Waiting...");
        return;
    }
    int cw = width() / m_cols;
    int ch = height() / m_rows;

    auto colormap = [&](float v) {
        float t = (v - m_vmin) / (m_vmax - m_vmin);
        t = std::max(0.f, std::min(1.f, t));
        if (m_type == Diff) {
            float u = t * 2.f - 1.f;
            if (u < 0)
                return QColor::fromRgbF(0.5f + 0.5f * u, 0.5f + 0.5f * u, 1.f);
            else
                return QColor::fromRgbF(1.f, 0.5f - 0.5f * u, 0.5f - 0.5f * u);
        }
        if (m_type == Weight) {
            float tt = 2.f * t - 1.f;
            if (tt < 0) return QColor::fromRgbF(1.f + tt, 1.f + tt, 1.f);
            return QColor::fromRgbF(1.f, 1.f - tt, 1.f - tt);
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
        if (t <= stops[0].pos) return QColor::fromRgbF(stops[0].r, stops[0].g, stops[0].b);
        if (t >= stops[n-1].pos) return QColor::fromRgbF(stops[n-1].r, stops[n-1].g, stops[n-1].b);
        for (int i = 0; i < n - 1; ++i) {
            if (t >= stops[i].pos && t < stops[i+1].pos) {
                float f = (t - stops[i].pos) / (stops[i+1].pos - stops[i].pos);
                return QColor::fromRgbF(
                    stops[i].r + (stops[i+1].r - stops[i].r) * f,
                    stops[i].g + (stops[i+1].g - stops[i].g) * f,
                    stops[i].b + (stops[i+1].b - stops[i].b) * f);
            }
        }
        return QColor::fromRgbF(stops[n-1].r, stops[n-1].g, stops[n-1].b);
    };

    for (int r = 0; r < m_rows; ++r)
        for (int c = 0; c < m_cols; ++c)
            p.fillRect(c * cw, r * ch, cw, ch, colormap(m_data[r * m_cols + c]));
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

    accuracy_plot->train_accs.clear();
    accuracy_plot->test_accs.clear();
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
    std::vector<int> pred_copy;
    {
        QMutexLocker l(&worker->hm_mutex);
        if (!worker->hm_updated) return;
        R = worker->hm_resolution;
        pred_copy = worker->hm_pred_data;
        worker->hm_updated = false;
    }
    if (R < 2) return;

    std::vector<float> tgt(R * R), pred(R * R), diff(R * R);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < R; ++j) {
            float x = (i + 0.5f) / R;
            float y = (j + 0.5f) / R;
            tgt[i * R + j] = (float)mandelbrot_cont(x, y, R - 1);
        }

    for (int i = 0; i < R; ++i)
        for (int j = 0; j < R; ++j) {
            int pv = pred_copy.empty() ? 0 : pred_copy[i * R + j];
            pred[i * R + j] = (float)pv;
            diff[i * R + j] = std::abs(pred[i * R + j] - tgt[i * R + j]);
        }

    float max_v = (float)(R - 1);
    hm_target->setData(tgt, R, R, HeatmapWidget::Target, 0, max_v);
    hm_pred->setData(pred, R, R, HeatmapWidget::Prediction, 0, max_v);
    hm_diff->setData(diff, R, R, HeatmapWidget::Diff, 0, max_v);
}

// ---- TrainWorker ----
void TrainWorker::run() {
    int R = resolution;
    int d_dim = d_model;
    int ff_h = d_dim * 2;
    int B = batch;
    auto f = [R](float x, float y) { return mandelbrot_cont(x, y, R - 1); };

    auto [train_ds, test_ds] = PointDataset::grid_split(R, 0.2, f);

    FFNNAttentionModel model(R, d_dim, ff_h, activation);

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
    add_mat(model.W1); add_mat(model.W2);

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
    add_vec(model.b1); add_vec(model.b2);

    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int t = 0;

    std::vector<int> idx(train_ds.get_size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 srng(42);

    emit logMessage(QString("GUI: res=%1 d=%2 lr=%3 act=%4 batch=%5")
                    .arg(R).arg(d_dim).arg(lr).arg(QString::fromStdString(activation))
                    .arg(B));

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        if (stop_flag.load()) break;

        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(idx.begin(), idx.end(), srng);

        int correct = 0, total = 0;
        Eigen::MatrixXf coords, dlogits;
        Eigen::VectorXi targets;
        Eigen::VectorXf max_vals, sum_exp;

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
            targets.resize(B_actual);
            for (int si = 0; si < B_actual; ++si) {
                auto [x, y, tgt] = train_ds.get(idx[bi + si]);
                coords(si, 0) = x; coords(si, 1) = y;
                targets(si) = tgt;
            }

            model.zero_grad();
            Eigen::MatrixXf logits = model.forward(coords);

            max_vals = logits.rowwise().maxCoeff();
            Eigen::MatrixXf exps = (logits.colwise() - max_vals).array().exp();
            sum_exp = exps.rowwise().sum();

            dlogits = exps.array().colwise() / sum_exp.array();
            for (int si = 0; si < B_actual; ++si) {
                dlogits(si, targets(si)) -= 1.f;
                int pred;
                logits.row(si).maxCoeff(&pred);
                if (pred == targets(si)) ++correct;
            }
            dlogits /= (float)B_actual;
            total += B_actual;

            model.backward(dlogits);

            if (weight_decay > 0.f) {
                auto wd = [&](auto& g, const auto& p) { g += weight_decay * p; };
                wd(model.grad.dWq, model.Wq); wd(model.grad.dWk, model.Wk);
                wd(model.grad.dWv, model.Wv); wd(model.grad.dW1, model.W1);
                wd(model.grad.dW2, model.W2); wd(model.grad.dU, model.U);
                wd(model.grad.dW_in, model.W_in); wd(model.grad.dP, model.P);
            }

            ++t;
            float m_bias = 1.f - std::pow(beta1, t);
            float v_bias = 1.f - std::pow(beta2, t);
            std::vector<Eigen::MatrixXf*> mat_gs = {
                &model.grad.dW_in, &model.grad.dU, &model.grad.dP,
                &model.grad.dWq, &model.grad.dWk, &model.grad.dWv,
                &model.grad.dW1, &model.grad.dW2
            };
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
                &model.grad.db1, &model.grad.db2
            };
            for (size_t i = 0; i < vec_ptrs.size(); ++i) {
                auto& s = vec_states[i];
                const auto& g = *vec_gs[i];
                s.m = beta1 * s.m + (1.f - beta1) * g;
                s.v = beta2 * s.v + (1.f - beta2) * g.cwiseProduct(g);
                *vec_ptrs[i] -= (lr * (s.m.array() / m_bias) / ((s.v.array() / v_bias).sqrt() + eps)).matrix();
            }
        }

        int test_correct = 0;
        for (int si = 0; si < test_ds.get_size(); si += B) {
            if (stop_flag.load()) break;
            int end = std::min(si + B, test_ds.get_size());
            int B_actual = end - si;
            Eigen::MatrixXf coords(B_actual, 2);
            for (int k = 0; k < B_actual; ++k) {
                auto [x, y, tgt] = test_ds.get(si + k);
                coords(k, 0) = x; coords(k, 1) = y;
            }
            Eigen::MatrixXf logits = model.forward(coords);
            for (int k = 0; k < B_actual; ++k) {
                int pred; logits.row(k).maxCoeff(&pred);
                auto [x, y, tgt] = test_ds.get(si + k);
                if (pred == tgt) ++test_correct;
            }
        }
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

        if (heatmap_enabled && (epoch % heatmap_every == 0) && !stop_flag.load()) {
            std::vector<int> grid_pred(R * R);
            Eigen::MatrixXf grid_coords(R * R, 2);
            for (int i = 0; i < R; ++i)
                for (int j = 0; j < R; ++j) {
                    grid_coords(i * R + j, 0) = (i + 0.5f) / R;
                    grid_coords(i * R + j, 1) = (j + 0.5f) / R;
                }
            for (int gs = 0; gs < R * R; gs += 4096) {
                if (stop_flag.load()) break;
                int ge = std::min(gs + 4096, R * R);
                int gB = ge - gs;
                Eigen::MatrixXf g_in = grid_coords.middleRows(gs, gB);
                Eigen::MatrixXf g_logits = model.forward(g_in);
                for (int k = 0; k < gB; ++k) {
                    int pred; g_logits.row(k).maxCoeff(&pred);
                    grid_pred[gs + k] = pred;
                }
            }
            {
                QMutexLocker l(&hm_mutex);
                hm_pred_data = std::move(grid_pred);
                hm_resolution = R;
                hm_updated = true;
            }
        }
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
