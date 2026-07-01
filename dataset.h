#pragma once
#include <Eigen/Dense>
#include <numeric>
#include <random>
#include <vector>
#include <tuple>

class PointDataset {
public:
    int resolution, size;
    std::vector<float> xs, ys;
    std::vector<int> targets;

    PointDataset(int res, int sz, std::function<int(float, float)> func)
        : resolution(res), size(sz) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        xs.resize(size);
        ys.resize(size);
        targets.resize(size);
        for (int k = 0; k < size; ++k) {
            xs[k] = dist(rng);
            ys[k] = dist(rng);
            targets[k] = func(xs[k], ys[k]);
        }
    }

private:
    PointDataset(int res, int sz) : resolution(res), size(sz) {
        xs.resize(size);
        ys.resize(size);
        targets.resize(size);
    }

public:

    int get_size() const { return size; }

    std::tuple<float, float, int> get(int idx) const {
        return {xs[idx], ys[idx], targets[idx]};
    }

    static std::pair<PointDataset, PointDataset>
    train_test_split(int resolution, int total_size, double test_ratio,
                     std::function<int(float, float)> func) {
        int test_sz = static_cast<int>(total_size * test_ratio);
        int train_sz = total_size - test_sz;
        PointDataset full(resolution, total_size, std::move(func));
        PointDataset train(resolution, train_sz);
        PointDataset test(resolution, test_sz);
        train.xs.assign(full.xs.begin(), full.xs.begin() + train_sz);
        train.ys.assign(full.ys.begin(), full.ys.begin() + train_sz);
        train.targets.assign(full.targets.begin(), full.targets.begin() + train_sz);
        test.xs.assign(full.xs.begin() + train_sz, full.xs.end());
        test.ys.assign(full.ys.begin() + train_sz, full.ys.end());
        test.targets.assign(full.targets.begin() + train_sz, full.targets.end());
        return {train, test};
    }

    static std::pair<PointDataset, PointDataset>
    grid_split(int resolution, double test_ratio, std::function<int(float, float)> func) {
        int n = resolution * resolution;
        int test_sz = static_cast<int>(n * test_ratio);
        int train_sz = n - test_sz;
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), std::mt19937(42));

        PointDataset train(resolution, train_sz);
        PointDataset test(resolution, test_sz);
        for (int k = 0; k < train_sz; ++k) {
            int idx = order[k];
            int i = idx / resolution, j = idx % resolution;
            float x = (i + 0.5f) / resolution;
            float y = (j + 0.5f) / resolution;
            train.xs[k] = x;
            train.ys[k] = y;
            train.targets[k] = func(x, y);
        }
        for (int k = 0; k < test_sz; ++k) {
            int idx = order[train_sz + k];
            int i = idx / resolution, j = idx % resolution;
            float x = (i + 0.5f) / resolution;
            float y = (j + 0.5f) / resolution;
            test.xs[k] = x;
            test.ys[k] = y;
            test.targets[k] = func(x, y);
        }
        return {train, test};
    }
};
