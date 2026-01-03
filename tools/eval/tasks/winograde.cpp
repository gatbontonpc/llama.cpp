#include "../eval-task.h"
#include "log.h"
#include "task-helpers.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

std::vector<winograde_tasks::winogrande_entry> winograde_tasks::load_winogrande_from_csv(const std::string & prompt) {
    std::vector<winogrande_entry> result;
    std::istringstream            in(prompt);
    std::string                   line;
    std::array<int, 4>            comma_pos;
    while (true) {
        std::getline(in, line);
        if (in.fail() || in.eof()) {
            break;
        }
        int  ipos       = 0;
        bool quote_open = false;
        for (int i = 0; i < int(line.size()); ++i) {
            if (!quote_open) {
                if (line[i] == ',') {
                    comma_pos[ipos++] = i;
                    if (ipos == 4) {
                        break;
                    }
                } else if (line[i] == '"') {
                    quote_open = true;
                }
            } else {
                if (line[i] == '"') {
                    quote_open = false;
                }
            }
        }
        if (ipos != 4) {
            LOG_ERR("%s: failed to find comma separators in <%s>\n", __func__, line.c_str());
            continue;
        }
        auto sentence = line[comma_pos[0] + 1] == '"' ? line.substr(comma_pos[0] + 2, comma_pos[1] - comma_pos[0] - 3) :
                                                        line.substr(comma_pos[0] + 1, comma_pos[1] - comma_pos[0] - 1);
        auto choice1  = line.substr(comma_pos[1] + 1, comma_pos[2] - comma_pos[1] - 1);
        auto choice2  = line.substr(comma_pos[2] + 1, comma_pos[3] - comma_pos[2] - 1);
        auto answer   = line.substr(comma_pos[3] + 1, line.size() - comma_pos[3] - 1);
        auto index    = line.substr(0, comma_pos[0]);
        int  where    = 0;
        for (; where < int(sentence.size()); ++where) {
            if (sentence[where] == '_') {
                break;
            }
        }
        if (where == int(sentence.size())) {
            LOG_ERR("%s: no _ in <%s>\n", __func__, sentence.c_str());
            continue;
        }
        std::istringstream stream(answer.c_str());
        int                i_answer;
        stream >> i_answer;
        if (stream.fail() || i_answer < 1 || i_answer > 2) {
            LOG_ERR("%s: failed to parse answer <%s>\n", __func__, answer.c_str());
            continue;
        }
        result.emplace_back();
        auto & wg     = result.back();
        wg.first      = sentence.substr(0, where);
        wg.second     = sentence.substr(where + 1, sentence.size() - where - 1);
        wg.choices[0] = std::move(choice1);
        wg.choices[1] = std::move(choice2);
        wg.answer     = i_answer;
    }
    return result;
}

winograde_tasks::winograde_tasks(common_params p) : eval_task(std::move(p)) {}

void winograde_tasks::prepare(llama_context * ctx) {}

void winograde_tasks::run(llama_context * ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    constexpr int k_min_trailing_ctx = 3;

    auto data = load_winogrande_from_csv(params.prompt);
    if (data.empty()) {
        LOG_ERR("%s: no tasks\n", __func__);
        return;
    }

    LOG_INF("%s : loaded %zu tasks from prompt.\n", __func__, data.size());

    if (params.winogrande_tasks > 0 && params.winogrande_tasks < data.size()) {
        LOG_INF("%s : selecting %zu random tasks\n", __func__, params.winogrande_tasks);
        std::mt19937 rng(1);
        std::vector<int> aux(data.size());
        for (int i = 0; i < int(data.size()); ++i) {
            aux[i] = i;
        }
        float scale = 1/(1.f + (float)rng.max());
        std::vector<winogrande_entry> selected;
        selected.resize(params.winogrande_tasks);
        for (int i = 0; i < int(params.winogrande_tasks); ++i) {
            int j = int(scale*rng()*aux.size());
            selected[i] = std::move(data[aux[j]]);
            aux[j] = aux.back();
            aux.pop_back();
        }
        data = std::move(selected);
    }

    LOG_INF("%s : tokenizing selected tasks\n", __func__);

    for (auto & task : data) {
        task.seq_tokens[0] = common_tokenize(ctx, task.first + task.choices[0] + task.second, true);
        task.seq_tokens[1] = common_tokenize(ctx, task.first + task.choices[1] + task.second, true);

        task.common_prefix = 0;
        for (size_t k = 0; k < task.seq_tokens[0].size(); k++) {
            if (task.seq_tokens[0][k] != task.seq_tokens[1][k]) {
                break;
            }
            task.common_prefix++;
        }

        // TODO: the last token of each of the sequences don't need to be evaluated
        task.required_tokens = task.common_prefix +
            task.seq_tokens[0].size() - task.common_prefix +
            task.seq_tokens[1].size() - task.common_prefix;

        task.n_base1 = common_tokenize(ctx, task.first + task.choices[0], true).size();
        task.n_base2 = common_tokenize(ctx, task.first + task.choices[1], true).size();
    }

    LOG_INF("%s : calculating winogrande score over selected tasks.\n", __func__);

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 128;
    const int max_seq = std::min(2*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, 2);

    std::vector<float> tok_logits(n_vocab);
    // TODO: this could be made smaller; it's currently the worst-case size
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());

    int n_correct = 0;
    int n_done    = 0;

    for (size_t i0 = 0; i0 < data.size(); i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0;

        common_batch_clear(batch);

        while (n_cur + (int) data[i1].required_tokens <= n_ctx) {
            int n_logits = 0;
            const int s0 = 2*(i1 - i0);
            if (s0 + 2 > max_seq) {
                break;
            }

            for (size_t i = 0; i < data[i1].common_prefix; ++i) {
                common_batch_add(batch, data[i1].seq_tokens[0][i], i, { s0 + 0, s0 + 1 }, false);
            }
            batch.logits[batch.n_tokens - 1] = true;
            n_logits += 1;

            for (int s = 0; s < 2; ++s) {
                // TODO: end before the last token, no need to predict past the end of the sequences
                for (size_t i = data[i1].common_prefix; i < data[i1].seq_tokens[s].size(); ++i) {
                    common_batch_add(batch, data[i1].seq_tokens[s][i], i, { s0 + s }, true);
                    n_logits += 1;
                }
            }

            data[i1].i_logits = i_logits;
            i_logits += n_logits;

            n_cur += data[i1].required_tokens;
            if (++i1 == data.size()) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %lu tokens)\n", __func__, i0, data[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto & task = data[i];

            const bool skip_choice =
                task.seq_tokens[0].size() - task.common_prefix > k_min_trailing_ctx &&
                task.seq_tokens[1].size() - task.common_prefix > k_min_trailing_ctx;

            const auto& n_base1 = skip_choice ? task.n_base1 : task.common_prefix;
            const int last_1st = task.seq_tokens[0].size() - n_base1 > 1 ? 1 : 0;
            size_t li = n_base1 - task.common_prefix;
            for (size_t j = n_base1-1; j < task.seq_tokens[0].size()-1-last_1st; ++j) {
                eval_pairs.emplace_back(task.i_logits + li++, task.seq_tokens[0][j+1]);
            }
            const auto& n_base2 = skip_choice ? task.n_base2 : task.common_prefix;
            const int last_2nd = task.seq_tokens[1].size() - n_base2 > 1 ? 1 : 0;
            // FIXME: this uses the wrong first logits when not skipping the choice word
            li = task.seq_tokens[0].size() - task.common_prefix + n_base2 - task.common_prefix;
            for (size_t j = n_base2-1; j < task.seq_tokens[1].size()-1-last_2nd; ++j) {
                eval_pairs.emplace_back(task.i_logits + li++, task.seq_tokens[1][j+1]);
            }
        }
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;
        for (size_t i = i0; i < i1; ++i) {
            auto & task = data[i];

            const bool skip_choice =
                task.seq_tokens[0].size() - task.common_prefix > k_min_trailing_ctx &&
                task.seq_tokens[1].size() - task.common_prefix > k_min_trailing_ctx;

            float score_1st = 0;
            const auto& n_base1 = skip_choice ? task.n_base1 : task.common_prefix;
            const int last_1st = task.seq_tokens[0].size() - n_base1 > 1 ? 1 : 0;
            for (size_t j = n_base1-1; j < task.seq_tokens[0].size()-1-last_1st; ++j) {
                score_1st += eval_results[ir++];
            }
            score_1st /= (task.seq_tokens[0].size() - n_base1 - last_1st);

            float score_2nd = 0;
            const auto& n_base2 = skip_choice ? task.n_base2 : task.common_prefix;
            const int last_2nd = task.seq_tokens[1].size() - n_base2 > 1 ? 1 : 0;
            for (size_t j = n_base2-1; j < task.seq_tokens[1].size()-1-last_2nd; ++j) {
                score_2nd += eval_results[ir++];
            }
            score_2nd /= (task.seq_tokens[1].size() - n_base2 - last_2nd);

            int result = score_1st > score_2nd ? 1 : 2;

            if (result == task.answer) {
                ++n_correct;
            }
            ++n_done;

            // print the accumulated accuracy mean x 100
            LOG("%zu\t%.4lf\t%10.6f  %10.6f  %d  %d\n", i+1, 100.0 * n_correct/n_done, score_1st, score_2nd, result, task.answer);
        }

        i0 = i1 - 1;
    }

    LOG("\n");

    if (n_done < 100) return;

    const float p = 1.f*n_correct/n_done;
    const float sigma = 100.f*sqrt(p*(1-p)/(n_done-1));

    LOG_INF("Final Winogrande score(%d tasks): %.4lf +/- %.4lf\n", n_done, 100*p, sigma);
}

eval_result winograde_tasks::result() {
    LOG_INF("%s: hellawswag get result\n", __func__);
    return { 0 };
}

/*
 * Evaluates the Winogrande score.
 * Uses a CSV containing task index, dentence, choice 1, choice 2, answer (1 or 2)
 * You can get one such dataset from e.g. https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp
 * As an example, the 1st row in the above dataset is
 *
 *    0,Sarah was a much better surgeon than Maria so _ always got the easier cases.,Sarah,Maria,2
 *
 */
