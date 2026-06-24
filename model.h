#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

struct Gradients {
    Eigen::MatrixXf dE, dU, dP;
    Eigen::MatrixXf dWq, dWk, dWv;
    Eigen::VectorXf dln1_gamma, dln1_beta;
    Eigen::VectorXf dln2_gamma, dln2_beta;
    Eigen::MatrixXf dW1;
    Eigen::VectorXf db1;
    Eigen::MatrixXf dW2;
    Eigen::VectorXf db2;

    void setZero(int V, int d, int ff_hidden) {
        dE.setZero(V, d); dU.setZero(d, V); dP.setZero(2, d);
        dWq.setZero(d, d); dWk.setZero(d, d); dWv.setZero(d, d);
        dln1_gamma.setZero(d); dln1_beta.setZero(d);
        dln2_gamma.setZero(d); dln2_beta.setZero(d);
        dW1.setZero(ff_hidden, d); db1.setZero(ff_hidden);
        dW2.setZero(d, ff_hidden); db2.setZero(d);
    }
};

template<typename T>
struct ActFunctor {
    const std::string* act;
    ActFunctor(const std::string& a) : act(&a) {}
    T operator()(T x) const {
        if (*act == "relu") return x > 0 ? x : 0;
        if (*act == "silu" || *act == "swish") {
            T s = 1 / (1 + std::exp(-x));
            return x * s;
        }
        if (*act == "tanh") return std::tanh(x);
        if (*act == "identity" || *act == "none") return x;
        return x * T(0.5) * (1 + std::erff(x * T(0.7071067811865475)));
    }
};

template<typename T>
struct ActGradFunctor {
    const std::string* act;
    ActGradFunctor(const std::string& a) : act(&a) {}
    T operator()(T x) const {
        if (*act == "relu") return x > 0 ? 1 : 0;
        if (*act == "silu" || *act == "swish") {
            T s = 1 / (1 + std::exp(-x));
            return s + x * s * (1 - s);
        }
        if (*act == "tanh") { T t = std::tanh(x); return 1 - t * t; }
        if (*act == "identity" || *act == "none") return 1;
        T c = T(0.7071067811865475);
        return T(0.5) * (1 + std::erff(x * c)) +
               x * std::exp(-x * x * T(0.5)) * T(0.3989422804014327) * c;
    }
};

class FFNNAttentionModel {
public:
    int V, d, ff_hidden;
    std::string activation;

    Eigen::MatrixXf E, U, P;
    Eigen::MatrixXf Wq, Wk, Wv;
    Eigen::VectorXf ln1_gamma, ln1_beta;
    Eigen::VectorXf ln2_gamma, ln2_beta;
    Eigen::MatrixXf W1;
    Eigen::VectorXf b1;
    Eigen::MatrixXf W2;
    Eigen::VectorXf b2;

    Gradients grad;

    FFNNAttentionModel(int vocab_size, int d_model = 64, int ff_hidden_ = 128,
                       const std::string& act = "gelu")
        : V(vocab_size), d(d_model), ff_hidden(ff_hidden_), activation(act) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        auto randn = [&](int r, int c) {
            Eigen::MatrixXf m(r, c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    m(i, j) = dist(rng);
            return m;
        };
        float sd = 1.f / std::sqrt(static_cast<float>(d));
        E = randn(V, d) * sd; U = randn(d, V) * sd; P = randn(2, d) * sd;
        float sk = std::sqrt(1.f / static_cast<float>(d));
        Wq = randn(d, d) * sk; Wk = randn(d, d) * sk; Wv = randn(d, d) * sk;
        W1 = randn(ff_hidden, d) * sk; b1.setZero(ff_hidden);
        W2 = randn(d, ff_hidden) * std::sqrt(1.f / ff_hidden); b2.setZero(d);
        ln1_gamma.setOnes(d); ln1_beta.setZero(d);
        ln2_gamma.setOnes(d); ln2_beta.setZero(d);
        grad.dE.resize(V, d); grad.dU.resize(d, V); grad.dP.resize(2, d);
        grad.dWq.resize(d, d); grad.dWk.resize(d, d); grad.dWv.resize(d, d);
        grad.dln1_gamma.resize(d); grad.dln1_beta.resize(d);
        grad.dln2_gamma.resize(d); grad.dln2_beta.resize(d);
        grad.dW1.resize(ff_hidden, d); grad.db1.resize(ff_hidden);
        grad.dW2.resize(d, ff_hidden); grad.db2.resize(d);
        zero_grad();
    }

    void zero_grad() { grad.setZero(V, d, ff_hidden); }

    Eigen::MatrixXf forward(const Eigen::MatrixXi& indices) {
        int B = indices.rows();
        auto& c = cache;
        c.B = B;
        c.indices = indices;

        c.x0.resize(B, d); c.x1.resize(B, d);
        for (int b = 0; b < B; ++b) {
            c.x0.row(b) = E.row(indices(b, 0));
            c.x1.row(b) = E.row(indices(b, 1));
        }
        c.x0.rowwise() += P.row(0);
        c.x1.rowwise() += P.row(1);

        c.q0 = c.x0 * Wq.transpose(); c.q1 = c.x1 * Wq.transpose();
        c.k0 = c.x0 * Wk.transpose(); c.k1 = c.x1 * Wk.transpose();
        c.v0 = c.x0 * Wv.transpose(); c.v1 = c.x1 * Wv.transpose();

        float inv_sd = 1.f / std::sqrt(static_cast<float>(d));
        c.s00 = (c.q0.cwiseProduct(c.k0)).rowwise().sum() * inv_sd;
        c.s01 = (c.q0.cwiseProduct(c.k1)).rowwise().sum() * inv_sd;
        c.s10 = (c.q1.cwiseProduct(c.k0)).rowwise().sum() * inv_sd;
        c.s11 = (c.q1.cwiseProduct(c.k1)).rowwise().sum() * inv_sd;

        // Vectorized 2-element softmax
        softmax2(c.s00, c.s01, c.a00, c.a01);
        softmax2(c.s10, c.s11, c.a10, c.a11);

        c.att0 = c.a00.asDiagonal() * c.v0 + c.a01.asDiagonal() * c.v1;
        c.att1 = c.a10.asDiagonal() * c.v0 + c.a11.asDiagonal() * c.v1;

        c.res1_0 = c.x0 + c.att0;
        c.res1_1 = c.x1 + c.att1;

        layernorm_fwd(c.res1_0, ln1_gamma, ln1_beta,
                      c.mu1_0, c.var1_0, c.x_hat1_0, c.ln1_out_0);
        layernorm_fwd(c.res1_1, ln1_gamma, ln1_beta,
                      c.mu1_1, c.var1_1, c.x_hat1_1, c.ln1_out_1);

        ffwd_batch(c.ln1_out_0, c.ff_pre_0, c.ff_act_0, c.ff_out_0);
        ffwd_batch(c.ln1_out_1, c.ff_pre_1, c.ff_act_1, c.ff_out_1);

        c.res2_0 = c.ln1_out_0 + c.ff_out_0;
        c.res2_1 = c.ln1_out_1 + c.ff_out_1;

        layernorm_fwd(c.res2_0, ln2_gamma, ln2_beta,
                      c.mu2_0, c.var2_0, c.x_hat2_0, c.ln2_out_0);
        layernorm_fwd(c.res2_1, ln2_gamma, ln2_beta,
                      c.mu2_1, c.var2_1, c.x_hat2_1, c.ln2_out_1);

        c.pooled = (c.ln2_out_0 + c.ln2_out_1) * 0.5f;
        c.logits = c.pooled * U;
        return c.logits;
    }

    void backward(const Eigen::MatrixXf& dlogits) {
        auto& c = cache;
        int B = c.B;

        grad.dU += c.pooled.transpose() * dlogits;
        Eigen::MatrixXf dpooled = dlogits * U.transpose();

        Eigen::MatrixXf dln2_out_0 = dpooled * 0.5f;
        Eigen::MatrixXf dln2_out_1 = dpooled * 0.5f;

        Eigen::MatrixXf dres2_0 = layernorm_bwd(
            dln2_out_0, c.res2_0, ln2_gamma, ln2_beta,
            c.mu2_0, c.var2_0, c.x_hat2_0, grad.dln2_gamma, grad.dln2_beta);
        Eigen::MatrixXf dres2_1 = layernorm_bwd(
            dln2_out_1, c.res2_1, ln2_gamma, ln2_beta,
            c.mu2_1, c.var2_1, c.x_hat2_1, grad.dln2_gamma, grad.dln2_beta);

        Eigen::MatrixXf dln1_out_0 = dres2_0;
        Eigen::MatrixXf dln1_out_1 = dres2_1;
        Eigen::MatrixXf dff_out_0 = dres2_0;
        Eigen::MatrixXf dff_out_1 = dres2_1;

        // FFN backward
        Eigen::MatrixXf dff_act_0 = dff_out_0 * W2;
        Eigen::MatrixXf dff_act_1 = dff_out_1 * W2;

        Eigen::MatrixXf dff_pre_0 = dff_act_0.cwiseProduct(
            c.ff_pre_0.unaryExpr(ActGradFunctor<float>(activation)));
        Eigen::MatrixXf dff_pre_1 = dff_act_1.cwiseProduct(
            c.ff_pre_1.unaryExpr(ActGradFunctor<float>(activation)));

        grad.dW2 += dff_out_0.transpose() * c.ff_act_0 + dff_out_1.transpose() * c.ff_act_1;
        grad.db2 += (dff_out_0.colwise().sum() + dff_out_1.colwise().sum()).transpose();
        grad.dW1 += dff_pre_0.transpose() * c.ln1_out_0 + dff_pre_1.transpose() * c.ln1_out_1;
        grad.db1 += (dff_pre_0.colwise().sum() + dff_pre_1.colwise().sum()).transpose();

        dln1_out_0 += dff_pre_0 * W1;
        dln1_out_1 += dff_pre_1 * W1;

        Eigen::MatrixXf dres1_0 = layernorm_bwd(
            dln1_out_0, c.res1_0, ln1_gamma, ln1_beta,
            c.mu1_0, c.var1_0, c.x_hat1_0, grad.dln1_gamma, grad.dln1_beta);
        Eigen::MatrixXf dres1_1 = layernorm_bwd(
            dln1_out_1, c.res1_1, ln1_gamma, ln1_beta,
            c.mu1_1, c.var1_1, c.x_hat1_1, grad.dln1_gamma, grad.dln1_beta);

        Eigen::MatrixXf dx0 = dres1_0;
        Eigen::MatrixXf dx1 = dres1_1;
        Eigen::MatrixXf datt0 = dres1_0;
        Eigen::MatrixXf datt1 = dres1_1;

        // Attention backward — vectorized
        Eigen::VectorXf da00 = (c.v0.cwiseProduct(datt0)).rowwise().sum();
        Eigen::VectorXf da01 = (c.v1.cwiseProduct(datt0)).rowwise().sum();
        Eigen::VectorXf da10 = (c.v0.cwiseProduct(datt1)).rowwise().sum();
        Eigen::VectorXf da11 = (c.v1.cwiseProduct(datt1)).rowwise().sum();

        Eigen::MatrixXf dv0 = c.a00.asDiagonal() * datt0 + c.a10.asDiagonal() * datt1;
        Eigen::MatrixXf dv1 = c.a01.asDiagonal() * datt0 + c.a11.asDiagonal() * datt1;

        // Softmax backward — vectorized
        softmax2_bwd(c.s00, c.s01, da00, da01, c.a00, c.a01, c.ds00, c.ds01);
        softmax2_bwd(c.s10, c.s11, da10, da11, c.a10, c.a11, c.ds10, c.ds11);

        float inv_sd = 1.f / std::sqrt(static_cast<float>(d));
        c.ds00 *= inv_sd; c.ds01 *= inv_sd; c.ds10 *= inv_sd; c.ds11 *= inv_sd;

        Eigen::MatrixXf dq0 = c.ds00.asDiagonal() * c.k0 + c.ds01.asDiagonal() * c.k1;
        Eigen::MatrixXf dq1 = c.ds10.asDiagonal() * c.k0 + c.ds11.asDiagonal() * c.k1;
        Eigen::MatrixXf dk0 = c.ds00.asDiagonal() * c.q0 + c.ds10.asDiagonal() * c.q1;
        Eigen::MatrixXf dk1 = c.ds01.asDiagonal() * c.q0 + c.ds11.asDiagonal() * c.q1;

        grad.dWq += dq0.transpose() * c.x0 + dq1.transpose() * c.x1;
        grad.dWk += dk0.transpose() * c.x0 + dk1.transpose() * c.x1;
        grad.dWv += dv0.transpose() * c.x0 + dv1.transpose() * c.x1;

        dx0 += dq0 * Wq + dk0 * Wk + dv0 * Wv;
        dx1 += dq1 * Wq + dk1 * Wk + dv1 * Wv;

        grad.dP.row(0) += dx0.colwise().sum();
        grad.dP.row(1) += dx1.colwise().sum();

        for (int b = 0; b < B; ++b) {
            grad.dE.row(c.indices(b, 0)) += dx0.row(b);
            grad.dE.row(c.indices(b, 1)) += dx1.row(b);
        }
    }

    struct BatchCache {
        int B;
        Eigen::MatrixXi indices;
        Eigen::MatrixXf x0, x1;
        Eigen::MatrixXf q0, q1, k0, k1, v0, v1;
        Eigen::VectorXf s00, s01, s10, s11;
        Eigen::VectorXf a00, a01, a10, a11;
        Eigen::VectorXf ds00, ds01, ds10, ds11;
        Eigen::MatrixXf att0, att1;
        Eigen::MatrixXf res1_0, res1_1;
        Eigen::VectorXf mu1_0, var1_0, mu1_1, var1_1;
        Eigen::MatrixXf x_hat1_0, x_hat1_1, ln1_out_0, ln1_out_1;
        Eigen::MatrixXf ff_pre_0, ff_pre_1, ff_act_0, ff_act_1, ff_out_0, ff_out_1;
        Eigen::MatrixXf res2_0, res2_1;
        Eigen::VectorXf mu2_0, var2_0, mu2_1, var2_1;
        Eigen::MatrixXf x_hat2_0, x_hat2_1, ln2_out_0, ln2_out_1;
        Eigen::MatrixXf pooled;
        Eigen::MatrixXf logits;
    };
    BatchCache cache;

private:
    static void softmax2(Eigen::VectorXf& s0, Eigen::VectorXf& s1,
                         Eigen::VectorXf& a0, Eigen::VectorXf& a1) {
        int B = s0.size();
        a0.resize(B); a1.resize(B);
        Eigen::ArrayXf m = s0.array().max(s1.array());
        Eigen::ArrayXf e0 = (s0.array() - m).exp();
        Eigen::ArrayXf e1 = (s1.array() - m).exp();
        Eigen::ArrayXf sum = e0 + e1 + 1e-38f;
        a0 = (e0 / sum).matrix();
        a1 = (e1 / sum).matrix();
    }

    static void softmax2_bwd(const Eigen::VectorXf& s0, const Eigen::VectorXf& s1,
                             const Eigen::VectorXf& da0, const Eigen::VectorXf& da1,
                             const Eigen::VectorXf& a0, const Eigen::VectorXf& a1,
                             Eigen::VectorXf& ds0, Eigen::VectorXf& ds1) {
        int B = s0.size();
        ds0.resize(B); ds1.resize(B);
        Eigen::ArrayXf dot = da0.array() * a0.array() + da1.array() * a1.array();
        ds0 = (a0.array() * (da0.array() - dot)).matrix();
        ds1 = (a1.array() * (da1.array() - dot)).matrix();
    }

    void ffwd_batch(const Eigen::MatrixXf& input,
                    Eigen::MatrixXf& pre, Eigen::MatrixXf& act, Eigen::MatrixXf& out) {
        pre = input * W1.transpose();
        pre.rowwise() += b1.transpose();
        act = pre.unaryExpr(ActFunctor<float>(activation));
        out = act * W2.transpose();
        out.rowwise() += b2.transpose();
    }

    static void layernorm_fwd(const Eigen::MatrixXf& x,
                              const Eigen::VectorXf& gamma, const Eigen::VectorXf& beta,
                              Eigen::VectorXf& mu, Eigen::VectorXf& var,
                              Eigen::MatrixXf& x_hat, Eigen::MatrixXf& y) {
        int B = x.rows(), n = x.cols();
        mu.resize(B); var.resize(B); x_hat.resize(B, n); y.resize(B, n);
        Eigen::MatrixXf centered = x;
        Eigen::RowVectorXf mean = x.colwise().mean();
        centered.rowwise() -= mean;
        mu = mean.transpose();
        var = centered.array().square().rowwise().sum() / n;
        Eigen::ArrayXf inv_std = (var.array() + 1e-5f).sqrt().inverse();
        x_hat = centered.array().colwise() * inv_std;
        y = x_hat.array().rowwise() * gamma.transpose().array();
        y.rowwise() += beta.transpose();
    }

    static Eigen::MatrixXf layernorm_bwd(const Eigen::MatrixXf& dy,
                                          const Eigen::MatrixXf& x,
                                          const Eigen::VectorXf& gamma,
                                          const Eigen::VectorXf& beta,
                                          const Eigen::VectorXf& mu,
                                          const Eigen::VectorXf& var,
                                          const Eigen::MatrixXf& x_hat,
                                          Eigen::VectorXf& dgamma,
                                          Eigen::VectorXf& dbeta) {
        int B = x.rows(), n = x.cols();
        Eigen::MatrixXf centered = x;
        centered.rowwise() -= mu.transpose();
        Eigen::ArrayXf inv_std = (var.array() + 1e-5f).sqrt().inverse();

        Eigen::MatrixXf dx_hat = dy.array().rowwise() * gamma.transpose().array();

        // dvar = (-0.5) * inv_std^3 * sum(dx_hat * centered, axis=1)
        Eigen::VectorXf dvar = (dx_hat.cwiseProduct(centered)).rowwise().sum();
        dvar.array() *= (-0.5f) * inv_std.cube();

        // dmu = -inv_std * sum(dx_hat, axis=1)  (dvar/dmu = 0 because centered sums to 0)
        Eigen::VectorXf dmu = (dx_hat.array().colwise() * (-inv_std)).rowwise().sum();

        // dx = dx_hat * inv_std + centered * (2/n * dvar) + dmu/n
        Eigen::MatrixXf dx = (dx_hat.array().colwise() * inv_std).matrix();
        dx += (centered.array().colwise() * ((2.f / n) * dvar.array())).matrix();
        dx += (dmu / n) * Eigen::RowVectorXf::Ones(n);

        dgamma += (dy.cwiseProduct(x_hat)).colwise().sum().transpose();
        dbeta += dy.colwise().sum().transpose();
        return dx;
    }
};
