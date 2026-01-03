#include "log.h"
#include "task-helpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "../eval-task.h"


// Calculates hellaswag score (acc_norm) from prompt
//
// Data extracted from the HellaSwag validation dataset (MIT license) https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl
// All used data fields are preprocessed as in https://github.com/EleutherAI/lm-evaluation-harness/blob/df3da98c5405deafd519c2ddca52bb7c3fe36bef/lm_eval/tasks/hellaswag.py#L62-L68
//
// All 10042 tasks should be extracted to keep the results standardized like other implementations.
//
// Datafile layout:
// ['??'] denotes json fields
// 6 lines per task:
// ['activity_label'] + ": " +['ctx']  - The first part of the query, the context
// ['label'] - The index the best common sense ending aka gold ending
// ['endings'][0] - Endings added to the first part of the query
// ['endings'][1]
// ['endings'][2]
// ['endings'][3]

hellaswag_task::hellaswag_task(common_params p) : eval_task(std::move(p)) {
    params.n_batch = std::min(params.n_batch, params.n_ctx);
    if (params.kl_divergence) {
        params.n_parallel = 1;
    } else {
        // ensure there's at least enough seq_ids for HellaSwag
        params.n_parallel = std::max(4, params.n_parallel);
    }
}

void hellaswag_task::prepare(llama_context* ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_ctx_train = llama_model_n_ctx_train(model);

    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    LOG_INF("%s: hellaswag prepare\n", __func__);

    std::mt19937 rng(1);

    std::vector<std::string> prompt_lines;
    std::istringstream strstream(params.prompt);
    std::string line;

    while (std::getline(strstream, line, '\n')) {
        prompt_lines.push_back(line);
    }

    if (prompt_lines.size() % 6 != 0) {
        LOG_ERR("%s : number of lines in prompt not a multiple of 6.\n", __func__);
        return;
    }

    size_t hs_task_count = prompt_lines.size()/6;
    LOG_INF("%s : loaded %zu tasks from prompt.\n", __func__, hs_task_count);

    const bool is_spm = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;
    LOG_INF("================================= is_spm = %d\n", is_spm);

    bool randomize_tasks = true;

    if (params.hellaswag_tasks < hs_task_count) {
        hs_task_count = params.hellaswag_tasks;
    }

    LOG_INF("%s : selecting %zu %s tasks.\n", __func__, hs_task_count, (randomize_tasks ? "randomized" : "the first"));

    hs_data.resize(hs_task_count);
    LOG_INF("%s : hs_data size %d\n", __func__, hs_data.size());

    for (size_t i = 0; i < hs_task_count; i++) {
        size_t idx = i;

        auto & hs_cur = hs_data[i];

        if (randomize_tasks) {
            std::uniform_int_distribution<size_t> dist(0, prompt_lines.size()/6 - 1);
            idx = dist(rng);
        }

        hs_cur.context = prompt_lines[idx*6];
        hs_cur.gold_ending_idx = std::stoi(prompt_lines[idx*6+1]);
        for (size_t j = 0; j < 4; j++) {
            hs_cur.ending[j] = prompt_lines[idx*6+2+j];
            hs_cur.seq_tokens[j] = common_tokenize(ctx, hs_cur.context + " " + hs_cur.ending[j], true);
        }

        hs_cur.common_prefix = 0;
        for (size_t k = 0; k < hs_cur.seq_tokens[0].size(); k++) {
            if (hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[1][k] ||
                hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[2][k] ||
                hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[3][k]) {
                break;
            }
            hs_cur.common_prefix++;
        }
        hs_cur.required_tokens = hs_cur.common_prefix +
            hs_cur.seq_tokens[0].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[1].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[2].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[3].size() - hs_cur.common_prefix;

        if (randomize_tasks) {
            prompt_lines.erase(std::next(prompt_lines.begin(), idx*6), std::next(prompt_lines.begin(), idx*6 + 6));
        }
    }
}

void hellaswag_task::run(llama_context* ctx) {
    LOG_INF("%s: hellaswag run\n", __func__);
    size_t hs_task_count = hs_data.size();
    LOG_INF("%s: \n", __func__);
    if (hs_data.empty()) {
        LOG_ERR("%s: no tasks loaded\n", __func__);
        return;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    LOG_INF("%s : calculating hellaswag score over selected tasks.\n", __func__);

    LOG("\ntask\tacc_norm\t95%% confidence interval\n");

    double acc = 0.0;

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 32;
    const int max_seq = std::min(4*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, 4);

    std::vector<float> tok_logits(n_vocab);
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());

   for (size_t i0 = 0; i0 < hs_task_count; i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0; // this tells us how many logits were needed before this point in the batch

        common_batch_clear(batch);

        // batch as much tasks as possible into the available context
        // each task has 4 unique sequence ids - one for each ending
        // the common prefix is shared among the 4 sequences to save tokens
        // we extract logits only from the last common token and from all ending tokens of each sequence
        while (n_cur + (int) hs_data[i1].required_tokens <= n_ctx) {
            auto & hs_cur = hs_data[i1];
            int n_logits = 0;

            const int s0 = 4*(i1 - i0);
            if (s0 + 4 > max_seq) {
                break;
            }

            for (size_t i = 0; i < hs_cur.common_prefix; ++i) {
                common_batch_add(batch, hs_cur.seq_tokens[0][i], i, { s0 + 0, s0 + 1, s0 + 2, s0 + 3 }, false);
            }
            batch.logits[batch.n_tokens - 1] = true; // we need logits for the last token of the common prefix
            n_logits += 1;

            for (int s = 0; s < 4; ++s) {
                const size_t seq_tokens_size = hs_cur.seq_tokens[s].size();
                // TODO: don't evaluate the last token of each sequence
                for (size_t i = hs_cur.common_prefix; i < seq_tokens_size; ++i) {
                    const bool needs_logits = i < seq_tokens_size - 1;
                    common_batch_add(batch, hs_cur.seq_tokens[s][i], i, { s0 + s }, needs_logits);
                    n_logits += needs_logits;
                }
            }

            hs_cur.i_logits = i_logits;
            i_logits += n_logits;

            n_cur += hs_data[i1].required_tokens;
            if (++i1 == hs_task_count) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %lu tokens)\n", __func__, i0, hs_data[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto & hs_cur = hs_data[i];
            size_t li = 1;
            for (int s = 0; s < 4; ++s) {
                for (size_t j = hs_cur.common_prefix; j < hs_cur.seq_tokens[s].size() - 1; j++) {
                    eval_pairs.emplace_back(hs_cur.i_logits + li++, hs_cur.seq_tokens[s][j + 1]);
                }
            }
        }

        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;

        for (size_t i = i0; i < i1; ++i) {
            auto & hs_cur = hs_data[i];

            std::memcpy(tok_logits.data(), batch_logits.data() + hs_cur.i_logits*n_vocab, n_vocab*sizeof(float));

            const auto first_probs = softmax(tok_logits);

            for (int s = 0; s < 4; ++s) {
                hs_cur.ending_logprob_count[s] = 1;
                hs_cur.ending_logprob[s] = std::log(first_probs[hs_cur.seq_tokens[s][hs_cur.common_prefix]]);
                for (size_t j = hs_cur.common_prefix; j < hs_cur.seq_tokens[s].size() - 1; j++) {
                    hs_cur.ending_logprob[s] += eval_results[ir++];
                    hs_cur.ending_logprob_count[s]++;
                }
                hs_cur.ending_logprob[s] /= hs_cur.ending_logprob_count[s];
            }

            size_t ending_logprob_max_idx = 0;
            double ending_logprob_max_val = hs_cur.ending_logprob[0];
            for (size_t s = 1; s < 4; s++) {
                if (hs_cur.ending_logprob[s] > ending_logprob_max_val) {
                    ending_logprob_max_idx = s;
                    ending_logprob_max_val =  hs_cur.ending_logprob[s];
                }
            }

            if (ending_logprob_max_idx == hs_cur.gold_ending_idx) {
                acc += 1.0;
            }

            double freq = acc / double(i + 1);

            const double za = 1.95996398454;

            double z   = za * za / double(i + 1);
            double cnf = z * sqrt(double(i + 1) * (4.0 * freq * (1 - freq) + z)) / (za + za);
            double a   = (freq + z * 0.5 - cnf) / (1.0 + z);
            double b   = (freq + z * 0.5 + cnf) / (1.0 + z);

            LOG("%zu\t%3.8lf%%\t[%3.4lf%%, %3.4lf%%]\n", i + 1, freq * 100.0, a * 100.0, b * 100.0);
        }

        i0 = i1 - 1;
    }

    llama_batch_free(batch);

    LOG("\n");
}

eval_result hellaswag_task::result() {
    LOG_INF("%s: hellawswag get result\n", __func__);
    return {0};
}
