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

        std::vector<Eigen::MatrixXf*> mat_gs = {&g.dW_in, &g.dU, &g.dP, &g.dWq, &g.dWk, &g.dWv, &g.dW1, &g.dW2};
        std::vector<Eigen::VectorXf*> vec_gs = {&g.dln1_gamma, &g.dln1_beta, &g.dln2_gamma, &g.dln2_beta, &g.db1, &g.db2};

        for (size_t i = 0; i < mat_params.size(); ++i)
            step_mat(mat_params[i], mat_m[i], mat_v[i], *mat_gs[i]);
        for (size_t i = 0; i < vec_params.size(); ++i)
            step_vec(vec_params[i], vec_m[i], vec_v[i], *vec_gs[i]);
    }
};

int main(int argc, char** argv) {
    int resolution = 16, epochs = 200000000, batch = 128, d_model = 64;
    float lr = 1e-3f, weight_decay = 0.f;
    std::string activation = "gelu";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (arg == "--resolution" && i + 1 < argc) resolution = std::stoi(next());
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoi(next());
        else if (arg == "--batch" && i + 1 < argc) batch = std::stoi(next());
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(next());
        else if (arg == "--d_model" && i + 1 < argc) d_model = std::stoi(next());
        else if (arg == "--weight-decay" && i + 1 < argc) weight_decay = std::stof(next());
        else if (arg == "--activation" && i + 1 < argc) activation = next();
        else if (arg == "--gui") {
#ifdef GANDALF_GUI
            return run_gui(argc, argv);
#else
            std::cerr << "GUI not available (compile with gandalf-gui target)\n";
            return 1;
#endif
        }
    }

    auto f = [resolution](float x, float y) { return mandelbrot_cont(x, y, resolution - 1); };
    auto datasets = PointDataset::grid_split(resolution, 0.2, f);
    auto& train_ds = datasets.first;
    auto& test_ds = datasets.second;

    FFNNAttentionModel model(resolution, d_model, d_model * 2, activation);
    Adam optim(lr);
    optim.add(model.W_in); optim.add(model.U); optim.add(model.P);
    optim.add(model.Wq); optim.add(model.Wk); optim.add(model.Wv);
    optim.add(model.ln1_gamma); optim.add(model.ln1_beta);
    optim.add(model.ln2_gamma); optim.add(model.ln2_beta);
    optim.add(model.W1); optim.add(model.b1);
    optim.add(model.W2); optim.add(model.b2);

    std::vector<int> idx(train_ds.get_size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 srng(42);

    std::cout << "Gandalf C++ | res=" << resolution << " d=" << d_model
              << " lr=" << lr << " act=" << activation
              << " batch=" << batch << "\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(idx.begin(), idx.end(), srng);

        float train_loss = 0.f;
        int correct = 0, total = 0;

        for (int bi = 0; bi < train_ds.get_size(); bi += batch) {
            int end = std::min(bi + batch, train_ds.get_size());
            int B = end - bi;

            Eigen::MatrixXf coords(B, 2);
            Eigen::VectorXi targets(B);
            for (int si = 0; si < B; ++si) {
                auto [x, y, t] = train_ds.get(idx[bi + si]);
                coords(si, 0) = x; coords(si, 1) = y;
                targets(si) = t;
            }

            model.zero_grad();
            Eigen::MatrixXf logits = model.forward(coords);

            float batch_loss = 0.f;
            Eigen::MatrixXf dlogits(B, resolution);
            for (int si = 0; si < B; ++si) {
                Eigen::VectorXf l_row = logits.row(si);
                float mx = l_row.maxCoeff();
                Eigen::VectorXf shifted = l_row.array() - mx;
                Eigen::VectorXf exps = shifted.array().exp();
                float sum_exp = exps.sum();
                int tgt = targets(si);
                batch_loss -= std::log(exps(tgt) / sum_exp + 1e-38f);
                dlogits.row(si) = (exps / sum_exp).transpose();
                dlogits(si, tgt) -= 1.f;
                int pred;
                l_row.maxCoeff(&pred);
                if (pred == tgt) ++correct;
            }
            dlogits /= (float)B;
            train_loss += batch_loss;
            total += B;

            model.backward(dlogits);

            if (weight_decay > 0.f) {
                auto apply_wd = [&](auto& g, const auto& p) { g += weight_decay * p; };
                apply_wd(model.grad.dWq, model.Wq); apply_wd(model.grad.dWk, model.Wk);
                apply_wd(model.grad.dWv, model.Wv); apply_wd(model.grad.dW1, model.W1);
                apply_wd(model.grad.dW2, model.W2); apply_wd(model.grad.dU, model.U);
                apply_wd(model.grad.dW_in, model.W_in); apply_wd(model.grad.dP, model.P);
            }
            optim.step(model.grad);
        }
        train_loss /= total;

        int test_correct = 0;
        for (int si = 0; si < test_ds.get_size(); si += batch) {
            int end = std::min(si + batch, test_ds.get_size());
            int B = end - si;
            Eigen::MatrixXf coords(B, 2);
            for (int k = 0; k < B; ++k) {
                auto [x, y, t] = test_ds.get(si + k);
                coords(k, 0) = x; coords(k, 1) = y;
            }
            Eigen::MatrixXf logits = model.forward(coords);
            for (int k = 0; k < B; ++k) {
                int pred; logits.row(k).maxCoeff(&pred);
                auto [x, y, t] = test_ds.get(si + k);
                if (pred == t) ++test_correct;
            }
        }
        float test_acc = static_cast<float>(test_correct) / test_ds.get_size();

        auto t1 = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(t1 - t0).count();

        std::cout << "Epoch " << std::setw(3) << epoch
                  << " | test acc " << std::fixed << std::setprecision(5) << test_acc
                  << " | time " << std::fixed << std::setprecision(2) << elapsed << "s"
                  << std::endl;
    }
    return 0;
}
