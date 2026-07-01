#include "model.h"
#include "dataset.h"
#ifdef GANDALF_GUI
#include "gui.h"
#endif
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

constexpr int MAX_ITER = 256;

class Adam {
public:
    float lr, beta1, beta2, eps;
    int t;
    std::vector<Eigen::MatrixXf*> mat_params;
    std::vector<Eigen::VectorXf*> vec_params;
    std::vector<Eigen::MatrixXf> mat_m, mat_v;
    std::vector<Eigen::VectorXf> vec_m, vec_v;

    Adam(float lr_ = 1e-3f, float b1 = 0.9f, float b2 = 0.999f, float e = 1e-8f)
        : lr(lr_), beta1(b1), beta2(b2), eps(e), t(0) {}

    void add(Eigen::MatrixXf& p) {
        mat_params.push_back(&p);
        mat_m.emplace_back(p.rows(), p.cols());
        mat_m.back().setZero();
        mat_v.emplace_back(p.rows(), p.cols());
        mat_v.back().setZero();
    }
    void add(Eigen::VectorXf& p) {
        vec_params.push_back(&p);
        vec_m.emplace_back(p.size());
        vec_m.back().setZero();
        vec_v.emplace_back(p.size());
        vec_v.back().setZero();
    }

    void step(Gradients& g) {
        ++t;
        float m_bias = 1.f - std::pow(beta1, t);
        float v_bias = 1.f - std::pow(beta2, t);

        auto step_mat = [&](Eigen::MatrixXf* p, Eigen::MatrixXf& m, Eigen::MatrixXf& v, const Eigen::MatrixXf& g_) {
            m = beta1 * m + (1.f - beta1) * g_;
            v = beta2 * v + (1.f - beta2) * g_.cwiseProduct(g_);
            *p -= (lr * (m.array() / m_bias) / ((v.array() / v_bias).sqrt() + eps)).matrix();
        };
        auto step_vec = [&](Eigen::VectorXf* p, Eigen::VectorXf& m, Eigen::VectorXf& v, const Eigen::VectorXf& g_) {
            m = beta1 * m + (1.f - beta1) * g_;
            v = beta2 * v + (1.f - beta2) * g_.cwiseProduct(g_);
            *p -= (lr * (m.array() / m_bias) / ((v.array() / v_bias).sqrt() + eps)).matrix();
        };

        std::vector<Eigen::MatrixXf*> mat_gs = {&g.dW_in, &g.dU, &g.dP, &g.dWq, &g.dWk, &g.dWv};
        std::vector<Eigen::VectorXf*> vec_gs = {&g.dln1_gamma, &g.dln1_beta, &g.dln2_gamma, &g.dln2_beta};
        for (auto& dw : g.dW) mat_gs.push_back(&dw);
        for (auto& db : g.db) vec_gs.push_back(&db);

        for (size_t i = 0; i < mat_params.size(); ++i)
            step_mat(mat_params[i], mat_m[i], mat_v[i], *mat_gs[i]);
        for (size_t i = 0; i < vec_params.size(); ++i)
            step_vec(vec_params[i], vec_m[i], vec_v[i], *vec_gs[i]);
    }
};

int main(int argc, char** argv) {
    int epochs = 200000000, batch = 128, d_model = 64;
    int ff_hidden = 128, n_hidden_layers = 1;
    float lr = 1e-3f, weight_decay = 0.f;
    std::string activation = "gelu";
    int dataset_size = 10000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (arg == "--epochs" && i + 1 < argc) epochs = std::stoi(next());
        else if (arg == "--batch" && i + 1 < argc) batch = std::stoi(next());
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(next());
        else if (arg == "--d_model" && i + 1 < argc) d_model = std::stoi(next());
        else if (arg == "--dataset-size" && i + 1 < argc) dataset_size = std::stoi(next());
        else if (arg == "--weight-decay" && i + 1 < argc) weight_decay = std::stof(next());
        else if (arg == "--activation" && i + 1 < argc) activation = next();
        else if (arg == "--neurons" && i + 1 < argc) ff_hidden = std::stoi(next());
        else if (arg == "--layers" && i + 1 < argc) n_hidden_layers = std::stoi(next());
        else if (arg == "--gui") {
#ifdef GANDALF_GUI
            return run_gui(argc, argv);
#else
            std::cerr << "GUI not available (compile with gandalf-gui target)\n";
            return 1;
#endif
        }
    }

    auto f = [](float x, float y) { return mandelbrot_cont(x, y, MAX_ITER); };
    auto datasets = PointDataset::train_test_split(dataset_size, 0.2, f);
    auto& train_ds = datasets.first;
    auto& test_ds = datasets.second;

    FFNNAttentionModel model(d_model, ff_hidden, n_hidden_layers, activation);
    Adam optim(lr);
    optim.add(model.W_in); optim.add(model.U); optim.add(model.P);
    optim.add(model.Wq); optim.add(model.Wk); optim.add(model.Wv);
    optim.add(model.ln1_gamma); optim.add(model.ln1_beta);
    optim.add(model.ln2_gamma); optim.add(model.ln2_beta);
    for (auto& w : model.W) optim.add(w);
    for (auto& b : model.b) optim.add(b);

    std::vector<int> idx(train_ds.get_size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 srng(42);

    std::cout << "Gandalf C++ | d=" << d_model
              << " ff=" << ff_hidden << " layers=" << n_hidden_layers
              << " lr=" << lr << " act=" << activation
              << " batch=" << batch << " max_iter=" << MAX_ITER << "\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(idx.begin(), idx.end(), srng);

        float train_loss = 0.f;
        int correct = 0, total = 0;
        Eigen::MatrixXf coords;

        for (int bi = 0; bi < train_ds.get_size(); bi += batch) {
            int end = std::min(bi + batch, train_ds.get_size());
            int B = end - bi;

            coords.resize(B, 2);
            Eigen::VectorXi targets(B);
            for (int si = 0; si < B; ++si) {
                auto [x, y, t] = train_ds.get(idx[bi + si]);
                coords(si, 0) = x; coords(si, 1) = y;
                targets(si) = t;
            }

            model.zero_grad();
            Eigen::MatrixXf pred = model.forward(coords);

            Eigen::VectorXf diff = pred.col(0) - targets.cast<float>();
            float batch_loss = diff.array().square().mean();
            train_loss += batch_loss * B;

            Eigen::MatrixXf dlogits(B, 1);
            dlogits.col(0) = 2.f * diff / (float)B;
            for (int si = 0; si < B; ++si) {
                if (std::abs(pred(si, 0) - targets(si)) < 0.5f)
                    ++correct;
            }
            total += B;

            model.backward(dlogits);

            if (weight_decay > 0.f) {
                auto apply_wd = [&](auto& g, const auto& p) { g += weight_decay * p; };
                apply_wd(model.grad.dWq, model.Wq); apply_wd(model.grad.dWk, model.Wk);
                apply_wd(model.grad.dWv, model.Wv);
                apply_wd(model.grad.dU, model.U);
                apply_wd(model.grad.dW_in, model.W_in); apply_wd(model.grad.dP, model.P);
                for (int i = 0; i <= model.n_hidden_layers; ++i)
                    apply_wd(model.grad.dW[i], model.W[i]);
            }
            optim.step(model.grad);
        }
        train_loss /= total;

        float test_loss = 0.f;
        int test_correct = 0;
        for (int si = 0; si < test_ds.get_size(); si += batch) {
            int end = std::min(si + batch, test_ds.get_size());
            int B = end - si;
            Eigen::MatrixXf coords(B, 2);
            Eigen::VectorXi targets(B);
            for (int k = 0; k < B; ++k) {
                auto [x, y, t] = test_ds.get(si + k);
                coords(k, 0) = x; coords(k, 1) = y;
                targets(k) = t;
            }
            Eigen::MatrixXf pred = model.forward(coords);
            Eigen::VectorXf diff = pred.col(0) - targets.cast<float>();
            test_loss += diff.array().square().sum();
            for (int k = 0; k < B; ++k) {
                if (std::abs(pred(k, 0) - targets(k)) < 0.5f)
                    ++test_correct;
            }
        }
        test_loss /= test_ds.get_size();
        float test_acc = static_cast<float>(test_correct) / test_ds.get_size();

        auto t1 = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(t1 - t0).count();

        std::cout << "Epoch " << std::setw(3) << epoch
                  << " | test loss " << std::fixed << std::setprecision(5) << test_loss
                  << " | test acc " << std::fixed << std::setprecision(5) << test_acc
                  << " | time " << std::fixed << std::setprecision(2) << elapsed << "s"
                  << std::endl;
    }
    return 0;
}
