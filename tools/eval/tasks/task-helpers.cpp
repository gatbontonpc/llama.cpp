#include "common.h"
#include "log.h"
#include "task-helpers.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>


std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

void compute_logprobs(const float * batch_logits, int n_vocab, std::vector<std::thread>& workers,
        const std::vector<std::pair<size_t, llama_token>>& eval_pairs, std::vector<float>& eval_results) {
    if (eval_results.size() != eval_pairs.size()) {
        eval_results.resize(eval_pairs.size());
    }
    if (eval_pairs.empty()) {
        return;
    }

    size_t max_threads = std::min((eval_pairs.size() + K_TOKEN_CHUNK - 1)/K_TOKEN_CHUNK, workers.size());

    std::atomic<int> counter(0);
    auto compute = [&counter, &eval_pairs, &eval_results, batch_logits, n_vocab] () {
        float local_logprobs[K_TOKEN_CHUNK];
        while (true) {
            const size_t first = counter.fetch_add(K_TOKEN_CHUNK, std::memory_order_relaxed);
            if (first >= eval_results.size()) {
                break;
            }
            const size_t last = std::min(first + K_TOKEN_CHUNK, eval_results.size());
            for (size_t i = first; i < last; ++i) {
                const auto * logits = batch_logits + eval_pairs[i].first * n_vocab;
                float max_logit = logits[0];
                for (int j = 1; j < n_vocab; ++j) {
                    max_logit = std::max(max_logit, logits[j]);
                }
                float sum_p = 0.f;
                for (int j = 0; j < n_vocab; ++j) {
                    sum_p += expf(logits[j] - max_logit);
                }
                local_logprobs[i - first] = logits[eval_pairs[i].second] - max_logit - std::log(sum_p);
            }
            std::memcpy(eval_results.data() + first, local_logprobs, (last - first)*sizeof(float));
        }
    };

    for (size_t it = 0; it < max_threads; ++it) {
        workers[it] = std::thread(compute);
    }
    for (size_t it = 0; it < max_threads; ++it) {
        workers[it].join();
    }
}

bool decode_helper(llama_context * ctx, llama_batch & batch, std::vector<float> & batch_logits, int n_batch, int n_vocab) {
    int prev_outputs = 0;
    for (int i = 0; i < (int) batch.n_tokens; i += n_batch) {
        const int n_tokens = std::min<int>(n_batch, batch.n_tokens - i);

        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
        };

        const int ret = llama_decode(ctx, batch_view);
        if (ret != 0) {
            LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
            return false;
        }

        int n_outputs = 0;
        for (int i = 0; i < n_tokens; ++i) {
            n_outputs += batch_view.logits[i] != 0;
        }

        memcpy(batch_logits.data() + size_t(prev_outputs)*n_vocab, llama_get_logits(ctx), size_t(n_outputs)*n_vocab*sizeof(float));

        prev_outputs += n_outputs;
    }

    return true;
}

bool deserialize_string(std::istream & in, std::string & str) {
    uint32_t size;
    if (!in.read((char *)&size, sizeof(size)).fail()) {
        str.resize(size);
        if (!in.read((char *)&str[0], size).fail()) return true;
    }
    return false;
}