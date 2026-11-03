#pragma once

#include <Eigen/Dense>
#include <sentencepiece_processor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class ShellyEngine {
public:
    using Matrix = Eigen::Matrix<float, Eigen::Dynamic,
        Eigen::Dynamic, Eigen::RowMajor>;
    using Vector = Eigen::VectorXf;

    static constexpr int VOCAB = 2000;
    static constexpr int DIM = 384;
    static constexpr int LAYERS = 6;
    static constexpr int HEADS = 8;
    static constexpr int HEAD_DIM = 48;
    static constexpr int HIDDEN = 1024;
    static constexpr int CONTEXT = 256;

private:
    sentencepiece::SentencePieceProcessor tokenizer;
    std::unordered_map<std::string, Matrix> weights;
    std::mt19937 rng{std::random_device{}()};

    Matrix& W(const std::string& name) {
        auto it = weights.find(name);

        if (it == weights.end())
            throw std::runtime_error("Missing tensor: " + name);

        return it->second;
    }

    Vector rmsnorm(const Vector& x, const Matrix& weight) {
        float mean = x.squaredNorm() / DIM;

        return (x.array() / std::sqrt(mean + 1e-6f))
            * weight.row(0).transpose().array();
    }

    Vector linear(const Vector& x, const Matrix& weight) {
        return weight * x;
    }

    static float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }

    void rope(Vector& x, int position) {
        for (int i = 0; i < HEAD_DIM / 2; ++i) {
            float frequency = 1.0f /
                std::pow(10000.0f, float(i) / (HEAD_DIM / 2));

            float angle = position * frequency;
            float c = std::cos(angle);
            float s = std::sin(angle);

            float even = x[2 * i];
            float odd = x[2 * i + 1];

            x[2 * i] = even * c - odd * s;
            x[2 * i + 1] = even * s + odd * c;
        }
    }

    Vector forward(const std::vector<int>& tokens) {
        if (tokens.empty() || tokens.size() > CONTEXT)
            throw std::runtime_error("Invalid context length");

        int length = static_cast<int>(tokens.size());

        std::vector<Vector> states;
        states.reserve(length);

        for (int token : tokens) {
            if (token < 0 || token >= VOCAB)
                throw std::runtime_error("Invalid token ID");

            states.push_back(
                W("embedding.weight").row(token).transpose()
            );
        }

        for (int layer = 0; layer < LAYERS; ++layer) {
            std::string prefix =
                "layers." + std::to_string(layer) + ".";

            std::vector<Vector> queries(length);
            std::vector<Vector> keys(length);
            std::vector<Vector> values(length);

            for (int t = 0; t < length; ++t) {
                Vector normalized = rmsnorm(
                    states[t], W(prefix + "norm1.weight")
                );

                Vector qkv = linear(
                    normalized,
                    W(prefix + "attention.qkv.weight")
                );

                for (int h = 0; h < HEADS; ++h) {
                    Vector q = qkv.segment(
                        h * HEAD_DIM, HEAD_DIM
                    );

                    Vector k = qkv.segment(
                        DIM + h * HEAD_DIM, HEAD_DIM
                    );

                    Vector v = qkv.segment(
                        2 * DIM + h * HEAD_DIM, HEAD_DIM
                    );

                    rope(q, t);
                    rope(k, t);

                    if (h == 0) {
                        queries[t] = Vector::Zero(DIM);
                        keys[t] = Vector::Zero(DIM);
                        values[t] = Vector::Zero(DIM);
                    }

                    queries[t].segment(
                        h * HEAD_DIM, HEAD_DIM
                    ) = q;

                    keys[t].segment(
                        h * HEAD_DIM, HEAD_DIM
                    ) = k;

                    values[t].segment(
                        h * HEAD_DIM, HEAD_DIM
                    ) = v;
                }
            }

            std::vector<Vector> attention_outputs(length);

            for (int t = 0; t < length; ++t) {
                Vector merged = Vector::Zero(DIM);

                for (int h = 0; h < HEADS; ++h) {
                    std::vector<float> scores(t + 1);

                    float maximum =
                        -std::numeric_limits<float>::infinity();

                    for (int j = 0; j <= t; ++j) {
                        scores[j] =
                            queries[t].segment(
                                h * HEAD_DIM, HEAD_DIM
                            ).dot(
                                keys[j].segment(
                                    h * HEAD_DIM, HEAD_DIM
                                )
                            ) / std::sqrt(float(HEAD_DIM));

                        maximum = std::max(maximum, scores[j]);
                    }

                    float denominator = 0.0f;

                    for (float& score : scores) {
                        score = std::exp(score - maximum);
                        denominator += score;
                    }

                    Vector result = Vector::Zero(HEAD_DIM);

                    for (int j = 0; j <= t; ++j) {
                        float probability =
                            scores[j] / denominator;

                        result += probability *
                            values[j].segment(
                                h * HEAD_DIM, HEAD_DIM
                            );
                    }

                    merged.segment(
                        h * HEAD_DIM, HEAD_DIM
                    ) = result;
                }

                attention_outputs[t] = linear(
                    merged,
                    W(prefix + "attention.output.weight")
                );
            }

            for (int t = 0; t < length; ++t) {
                states[t] += attention_outputs[t];

                Vector normalized = rmsnorm(
                    states[t], W(prefix + "norm2.weight")
                );

                Vector gate = linear(
                    normalized, W(prefix + "mlp.gate.weight")
                );

                Vector up = linear(
                    normalized, W(prefix + "mlp.up.weight")
                );

                for (int i = 0; i < HIDDEN; ++i)
                    gate[i] = silu(gate[i]) * up[i];

                Vector down = linear(
                    gate, W(prefix + "mlp.down.weight")
                );

                states[t] += down;
            }
        }

        Vector final_state = rmsnorm(
            states.back(), W("norm.weight")
        );

        return W("embedding.weight") * final_state;
    }

    int sample_token(
        Vector& logits,
        const std::vector<int>& history
    ) {
        constexpr float TEMPERATURE = 0.8f;
        constexpr float REPETITION_PENALTY = 1.18f;
        constexpr int TOP_K = 40;
        constexpr int REPEAT_WINDOW = 48;

        int start = std::max(
            0,
            static_cast<int>(history.size()) - REPEAT_WINDOW
        );

        for (int i = start; i < static_cast<int>(history.size()); ++i) {
            int id = history[i];

            if (id < 0 || id >= VOCAB)
                continue;

            if (logits[id] > 0)
                logits[id] /= REPETITION_PENALTY;
            else
                logits[id] *= REPETITION_PENALTY;
        }

        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(VOCAB);

        for (int i = 0; i < VOCAB; ++i)
            candidates.emplace_back(logits[i], i);

        int k = std::min(TOP_K, VOCAB);

        std::partial_sort(
            candidates.begin(),
            candidates.begin() + k,
            candidates.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            }
        );

        float maximum = candidates[0].first;
        std::vector<double> probabilities(k);

        for (int i = 0; i < k; ++i) {
            probabilities[i] = std::exp(
                (candidates[i].first - maximum) / TEMPERATURE
            );
        }

        std::discrete_distribution<int> distribution(
            probabilities.begin(),
            probabilities.end()
        );

        return candidates[distribution(rng)].second;
    }

public:
    bool load(
        const std::string& tokenizer_path,
        const std::string& weights_path
    ) {
        auto status = tokenizer.Load(tokenizer_path);

        if (!status.ok()) {
            std::cerr << status.ToString() << '\n';
            return false;
        }

        if (tokenizer.GetPieceSize() != VOCAB) {
            std::cerr << "Vocabulary mismatch!\n";
            return false;
        }

        std::ifstream file(weights_path, std::ios::binary);

        if (!file) {
            std::cerr << "Cannot open weights!\n";
            return false;
        }

        char magic[8];
        file.read(magic, 8);

        if (!file ||
            std::memcmp(magic, "SHELLYQ1", 8) != 0) {
            std::cerr << "Invalid weights file!\n";
            return false;
        }

        uint32_t count;
        file.read(reinterpret_cast<char*>(&count), 4);

        if (!file || count != 45) {
            std::cerr << "Unexpected tensor count!\n";
            return false;
        }

        weights.clear();

        for (uint32_t n = 0; n < count; ++n) {
            uint32_t name_length;
            file.read(reinterpret_cast<char*>(&name_length), 4);

            if (!file || name_length == 0 || name_length > 4096)
                return false;

            std::string name(name_length, '\0');
            file.read(name.data(), name_length);

            uint32_t dimensions;
            file.read(reinterpret_cast<char*>(&dimensions), 4);

            if (!file ||
                (dimensions != 1 && dimensions != 2))
                return false;

            std::vector<uint32_t> shape(dimensions);
            uint64_t elements = 1;

            for (auto& dimension : shape) {
                file.read(
                    reinterpret_cast<char*>(&dimension), 4
                );

                if (!file || dimension == 0 ||
                    elements > 100000000 / dimension)
                    return false;

                elements *= dimension;
            }

            Matrix tensor;

            if (dimensions == 1)
                tensor.resize(1, shape[0]);
            else
                tensor.resize(shape[0], shape[1]);

            file.read(
                reinterpret_cast<char*>(tensor.data()),
                elements * sizeof(float)
            );

            if (!file)
                return false;

            weights.emplace(name, std::move(tensor));
        }

        std::cout << "Loaded tokenizer and "
                  << weights.size() << " tensors!\n";

        return true;
    }

    std::string generate(
        const std::string& prompt,
        int max_new_tokens = 60
    ) {
        std::vector<int> tokens;

        auto status = tokenizer.Encode(prompt, &tokens);

        if (!status.ok())
            throw std::runtime_error(status.ToString());

        if (tokens.empty())
            return "";

        std::vector<int> generated;

        for (int step = 0; step < max_new_tokens; ++step) {
            if (tokens.size() > CONTEXT) {
                tokens.erase(
                    tokens.begin(),
                    tokens.begin() + (tokens.size() - CONTEXT)
                );
            }

            Vector logits = forward(tokens);

            int next = sample_token(logits, tokens);

            tokens.push_back(next);
            generated.push_back(next);

            if (next == tokenizer.eos_id())
                break;
        }

        std::string output;

        status = tokenizer.Decode(generated, &output);

        if (!status.ok())
            throw std::runtime_error(status.ToString());

        return output;
    }
};
