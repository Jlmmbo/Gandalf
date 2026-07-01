#pragma once
#include "model.h"
#include "dataset.h"
#include <QCoreApplication>
#include <QMainWindow>
#include <QWidget>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QTextEdit>
#include <QPushButton>
#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QScrollArea>
#include <QThread>
#include <QMutex>
#include <QPainter>
#include <QPainterPath>
#include <QTimer>
#include <QLabel>
#include <QPointer>
#include <atomic>
#include <vector>
#include <deque>

int run_gui(int argc, char** argv);

class AccuracyPlot : public QWidget {
    Q_OBJECT
public:
    std::deque<float> train_errs, test_errs;
    void addPoint(float train, float test);
protected:
    void paintEvent(QPaintEvent*) override;
    QSize minimumSizeHint() const override { return {400, 250}; }
};

class HeatmapWidget : public QWidget {
    Q_OBJECT
public:
    enum Type { Target, Prediction, Diff, Weight };
    void setData(const std::vector<float>& data, int rows, int cols, Type type, float vmin, float vmax);
    void clear();
protected:
    void paintEvent(QPaintEvent*) override;
    QSize minimumSizeHint() const override { return {200, 200}; }
private:
    std::vector<float> m_data;
    int m_rows = 0, m_cols = 0;
    Type m_type = Target;
    float m_vmin = 0, m_vmax = 1;
};

class TrainWorker : public QObject {
    Q_OBJECT
public:
    std::atomic<int> stop_flag{0};
    std::atomic<int> pause_flag{0};
    int resolution = 16, epochs = 200000000, batch = 128, d_model = 64;
    int ff_hidden = 128, n_hidden_layers = 1;
    float lr = 1e-3f, weight_decay = 0.f;
    std::string activation = "gelu";
    bool heatmap_enabled = true;
    int heatmap_every = 1;
    QMutex hm_mutex;
    std::vector<float> hm_pred_data;
    int hm_resolution = 0;
    bool hm_updated = false;

public slots:
    void run();

signals:
    void logMessage(const QString& msg);
    void epochDone(int epoch, float train_acc, float test_acc, float elapsed);
    void finished();
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow();
    ~MainWindow();

private slots:
    void startTraining();
    void stopTraining();
    void togglePause();
    void onEpochDone(int epoch, float train_acc, float test_acc, float elapsed);
    void onLogMessage(const QString& msg);
    void onTrainingFinished();
    void pollHeatmap();

private:
    QSpinBox *resolution_spin, *epochs_spin, *batch_spin, *d_model_spin;
    QSpinBox *neurons_spin, *layers_spin;
    QSpinBox *heatmap_every_spin;
    QDoubleSpinBox *lr_spin, *weight_decay_spin;
    QComboBox *activation_combo;
    QCheckBox *heatmap_check;
    QPushButton *start_btn, *pause_btn, *stop_btn;
    QTextEdit *log_edit;
    AccuracyPlot *accuracy_plot;
    HeatmapWidget *hm_target, *hm_pred, *hm_diff;

    QThread worker_thread;
    QPointer<TrainWorker> worker;
    QTimer* heatmap_timer = nullptr;
};
