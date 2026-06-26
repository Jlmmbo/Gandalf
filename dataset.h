#pragma once
#include <Eigen/Dense>
#include <numeric>
#include <random>
#include <vector>
#include <tuple>

class IntegerPairDataset {
public:
    int vocab_size, size;
    std::vector<int> a, b;
    std::vector<int> targets;
    std::function<int(int, int)> f;

    IntegerPairDataset(int vocab, int sz, std::function<int(int, int)> func)
        : vocab_size(vocab), size(sz), f(std::move(func)) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, vocab_size - 1);
        a.resize(size);
        b.resize(size);
        targets.resize(size);
        for (int k = 0; k < size; ++k) {
            a[k] = dist(rng);
            b[k] = dist(rng);
            targets[k] = f(a[k], b[k]);
        }
    }

private:
    IntegerPairDataset(int vocab, int sz) : vocab_size(vocab), size(sz) {
        a.resize(size);
        b.resize(size);
        targets.resize(size);
    }

public:

    int get_size() const { return size; }

    std::tuple<int, int, int> get(int idx) const {
        return {a[idx], b[idx], targets[idx]};
    }

    static std::pair<IntegerPairDataset, IntegerPairDataset>
    train_test_split(int vocab, int total_size, double test_ratio,
                     std::function<int(int, int)> func) {
        int test_sz = static_cast<int>(total_size * test_ratio);
        int train_sz = total_size - test_sz;
        IntegerPairDataset full(vocab, total_size, std::move(func));
        IntegerPairDataset train(vocab, train_sz);
        IntegerPairDataset test(vocab, test_sz);
        train.a.assign(full.a.begin(), full.a.begin() + train_sz);
        train.b.assign(full.b.begin(), full.b.begin() + train_sz);
        train.targets.assign(full.targets.begin(), full.targets.begin() + train_sz);
        test.a.assign(full.a.begin() + train_sz, full.a.end());
        test.b.assign(full.b.begin() + train_sz, full.b.end());
        test.targets.assign(full.targets.begin() + train_sz, full.targets.end());
        return {train, test};
    }

    static std::pair<IntegerPairDataset, IntegerPairDataset>
    grid_split(int vocab, double test_ratio, std::function<int(int, int)> func) {
        int n = vocab * vocab;
        int test_sz = static_cast<int>(n * test_ratio);
        int train_sz = n - test_sz;
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), std::mt19937(42));

        IntegerPairDataset train(vocab, train_sz);
        IntegerPairDataset test(vocab, test_sz);
        for (int k = 0; k < train_sz; ++k) {
            int idx = order[k];
            int i = idx / vocab, j = idx % vocab;
            train.a[k] = i; train.b[k] = j;
            train.targets[k] = func(i, j);
        }
        for (int k = 0; k < test_sz; ++k) {
            int idx = order[train_sz + k];
            int i = idx / vocab, j = idx % vocab;
            test.a[k] = i; test.b[k] = j;
            test.targets[k] = func(i, j);
        }
        return {train, test};
    }
};
