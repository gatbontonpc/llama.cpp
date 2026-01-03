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


// Calculates score for multiple choice tasks with single correct answer from prompt.
// Commonly used LLM evaluation metrics of this type are
//   * ARC
//   * HellaSwag
//   * MMLU
//   * TruthfulQA
//
// Validation datasets for these 4 tests can be found at
//     https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp
// The data for these datasets was extracted from
//     git@hf.co:datasets/allenai/ai2_arc
//     https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl
//     git@hf.co:datasets/Stevross/mmlu
//     https://huggingface.co/datasets/truthful_qa
//
multiple_choice_tasks::multiple_choice_tasks(common_params p) : eval_task(std::move(p)) {
    params.n_batch = std::min(params.n_batch, params.n_ctx);
}

void multiple_choice_tasks::prepare(llama_context* ctx) {
}


bool multiple_choice_tasks::multiple_choice_prepare_one_task(llama_context * ctx, multiple_choice_task& task, bool log_error) {
    if (task.question.empty() || task.mc1.answers.empty()) {
        if (log_error) {
            LOG_ERR("%s: found bad task with empty question and/or answers\n", __func__);
        }
        return false;
    }
    task.seq_tokens.reserve(task.mc1.answers.size());
    for (auto& answer : task.mc1.answers) {
        if (answer.empty()) {
            if (log_error) {
                LOG_ERR("%s: found empty answer\n", __func__);
            }
            return false;
        }
        task.seq_tokens.emplace_back(::common_tokenize(ctx, task.question + " " + answer, true));
    }
    auto min_len = task.seq_tokens.front().size();
    for (auto& seq : task.seq_tokens) {
        min_len = std::min(min_len, seq.size());
    }
    task.common_prefix = 0;
    for (size_t k = 0; k < min_len; ++k) {
        auto token = task.seq_tokens[0][k];
        bool all_same = true;
        for (size_t i = 1; i < task.seq_tokens.size(); ++i) {
            if (task.seq_tokens[i][k] != token) {
                all_same = false;
                break;
            }
        }
        if (!all_same) {
            break;
        }
        ++task.common_prefix;
    }
    task.required_tokens = task.common_prefix;
    for (auto& seq : task.seq_tokens) {
        task.required_tokens += seq.size() - task.common_prefix;
    }
    return true;
}


void multiple_choice_tasks::run(llama_context* ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::istringstream strstream(params.prompt);
    uint32_t n_task;
    strstream.read((char *)&n_task, sizeof(n_task));
    if (strstream.fail() || n_task == 0) {
        LOG_ERR("%s: no tasks\n", __func__);
        return;
    }
    LOG_INF("%s: there are %u tasks in prompt\n", __func__, n_task);
    std::vector<uint32_t> task_pos(n_task);
    strstream.read((char *)task_pos.data(), task_pos.size()*sizeof(uint32_t));
    if (strstream.fail()) {
        LOG_ERR("%s: failed to read task positions from prompt\n", __func__);
        return;
    }

    std::vector<multiple_choice_task> tasks;
    if (params.multiple_choice_tasks == 0 || params.multiple_choice_tasks >= (size_t)n_task) {
        // Use all tasks
        tasks.resize(n_task);
        LOG_INF("%s: reading tasks", __func__);
        int n_dot = std::max((int) n_task/100, 1);
        int i = 0;
        for (auto& task : tasks) {
            ++i;
            if (!task.deserialize(strstream)) {
                LOG_ERR("%s: failed to read task %d of %u\n", __func__, i, n_task);
                return;
            }
            if (i%n_dot == 0) LOG(".");
        }
        LOG("done\n");
    }
    else {
        LOG_INF("%s: selecting %zu random tasks from %u tasks available\n", __func__, params.multiple_choice_tasks, n_task);
        std::mt19937 rng(1);
        std::vector<int> aux(n_task);
        for (uint32_t i = 0; i < n_task; ++i) aux[i] = i;
        float scale = 1.f/(1.f + (float)std::mt19937::max());
        tasks.resize(params.multiple_choice_tasks);
        for (auto& task : tasks) {
            int j = (int)(scale * rng() * aux.size());
            int idx = aux[j];
            aux[j] = aux.back();
            aux.pop_back();
            strstream.seekg(task_pos[idx], std::ios::beg);
            if (!task.deserialize(strstream)) {
                LOG_ERR("%s: failed to read task %d at position %u\n", __func__, idx, task_pos[idx]);
                return;
            }
        }
        n_task = params.multiple_choice_tasks;
    }

    LOG_INF("%s: preparing task data", __func__);
    if (n_task > 500) {
        LOG("...");
        std::atomic<int> counter(0);
        std::atomic<int> n_bad(0);
        auto prepare = [this, &counter, &n_bad, &tasks, ctx] () {
            int num_tasks = tasks.size();
            int n_bad_local = 0;
            while (true) {
                int first = counter.fetch_add(K_TOKEN_CHUNK);
                if (first >= num_tasks) {
                    if (n_bad_local > 0) n_bad += n_bad_local;
                    break;
                }
                int last = std::min(first + K_TOKEN_CHUNK, num_tasks);
                for (int i = first; i < last; ++i) {
                    if (!multiple_choice_prepare_one_task(ctx, tasks[i], false)) ++n_bad_local;
                }
            }
        };
        size_t max_thread = std::thread::hardware_concurrency();
        max_thread = std::min(max_thread, (tasks.size() + K_TOKEN_CHUNK - 1)/K_TOKEN_CHUNK);
        std::vector<std::thread> workers(max_thread-1);
        for (auto& w : workers) w = std::thread(prepare);
        prepare();
        for (auto& w : workers) w.join();
        LOG("done\n");
        int nbad = n_bad;
        if (nbad > 0) {
            LOG_ERR("%s: found %d malformed tasks\n", __func__, nbad);
            return;
        }
    } else {
        int n_dot = std::max((int) n_task/100, 1);
        int i_task = 0;
        for (auto& task : tasks) {
            ++i_task;
            if (!multiple_choice_prepare_one_task(ctx, task, true)) {
                return;
            }
            if (i_task%n_dot == 0) {
                LOG(".");
            }
        }
        LOG("done\n");
    }

    LOG_INF("%s : calculating TruthfulQA score over %zu tasks.\n", __func__, tasks.size());

    LOG("\ntask\tacc_norm\n");

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 32;
    const int max_seq = std::min(4*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, max_seq);

    std::vector<float> tok_logits(n_vocab);
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());
    std::vector<int> batch_indeces;

    int n_done = 0;
    int n_correct = 0;
    int n_tot_answers = 0;

    for (size_t i0 = 0; i0 < tasks.size(); i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0; // this tells us how many logits were needed before this point in the batch

        common_batch_clear(batch);

        // batch as much tasks as possible into the available context
        // each task has 4 unique sequence ids - one for each ending
        // the common prefix is shared among the 4 sequences to save tokens
        // we extract logits only from the last common token and from all ending tokens of each sequence
        int s0 = 0;
        while (n_cur + (int) tasks[i1].required_tokens <= n_ctx) {
            auto& cur_task = tasks[i1];
            int n_logits = 0;

            int num_answers = cur_task.seq_tokens.size();
            if (s0 + num_answers > max_seq) {
                if (s0 == 0) {
                    LOG_ERR("%s : task %zu requires a higher -np|--parallel value (at least %d)\n", __func__, i0, num_answers);
                    return;
                }
                break;
            }

            if (int(batch_indeces.size()) != num_answers) {
                batch_indeces.resize(num_answers);
            }

            for (int s = 0; s < num_answers; ++s) {
                batch_indeces[s] = s0 + s;
            }

            for (size_t i = 0; i < cur_task.common_prefix; ++i) {
                //llama_batch_add(batch, cur_task.seq_tokens[0][i], i, { s0 + 0, s0 + 1, s0 + 2, s0 + 3}, false);
                common_batch_add(batch, cur_task.seq_tokens[0][i], i, batch_indeces, false);
            }
            batch.logits[batch.n_tokens - 1] = true; // we need logits for the last token of the common prefix
            n_logits += 1;

            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                const size_t seq_tokens_size = cur_task.seq_tokens[s].size();
                // TODO: don't evaluate the last token of each sequence
                for (size_t i = cur_task.common_prefix; i < seq_tokens_size; ++i) {
                    const bool needs_logits = i < seq_tokens_size - 1;
                    common_batch_add(batch, cur_task.seq_tokens[s][i], i, { s0 + s }, needs_logits);
                    n_logits += needs_logits;
                }
            }

            s0 += num_answers;

            cur_task.i_logits = i_logits;
            i_logits += n_logits;

            n_cur += cur_task.required_tokens;
            if (++i1 == tasks.size()) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %lu tokens)\n", __func__, i0, tasks[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        // Compute log-probs in parallel
        // First we collect all tasks
        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto& cur_task = tasks[i];
            size_t li = 1; // skip the last logit of the common prefix (computed separately below)
            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                for (size_t j = cur_task.common_prefix; j < cur_task.seq_tokens[s].size() - 1; j++) {
                    eval_pairs.emplace_back(cur_task.i_logits + li++, cur_task.seq_tokens[s][j + 1]);
                }
            }
        }
        // Then we do the actual calculation
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;

        // compute the logprobs for each ending of the decoded tasks
        for (size_t i = i0; i < i1; ++i) {
            auto & cur_task = tasks[i];
            //LOG("==== Evaluating <%s> with correct answer ", cur_task.question.c_str());
            //for (int j = 0; j < int(cur_task.mc1.labels.size()); ++j) {
            //    if (cur_task.mc1.labels[j] == 1) {
            //        LOG("%d", j+1);
            //    }
            //}
            //LOG("\n    common_prefix: %zu\n", cur_task.common_prefix);

            // get the logits of the last token of the common prefix
            std::memcpy(tok_logits.data(), batch_logits.data() + cur_task.i_logits*n_vocab, n_vocab*sizeof(float));

            const auto first_probs = softmax(tok_logits);

            cur_task.log_probs.resize(cur_task.seq_tokens.size());
            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                size_t count = 1;
                float  log_prob  = std::log(first_probs[cur_task.seq_tokens[s][cur_task.common_prefix]]);
                for (size_t j = cur_task.common_prefix; j < cur_task.seq_tokens[s].size() - 1; j++) {
                    //LOG("        %zu  %g\n", ir, eval_results[ir]);
                    ++count;
                    log_prob += eval_results[ir++];
                }
                cur_task.log_probs[s] = log_prob / count;
                //LOG("        Final: %g\n", log_prob / count);
                //LOG("    <%s> : %g\n", cur_task.mc1.answers[s].c_str(), log_prob/count);
            }

            // Find the ending with maximum logprob
            size_t logprob_max_idx = 0;
            float  logprob_max_val = cur_task.log_probs[0];
            for (size_t s = 1; s < cur_task.log_probs.size(); s++) {
                if (cur_task.log_probs[s] > logprob_max_val) {
                    logprob_max_val = cur_task.log_probs[s];
                    logprob_max_idx = s;
                }
            }

            n_tot_answers += cur_task.log_probs.size();
            if (cur_task.mc1.labels[logprob_max_idx] == 1) {
                ++n_correct;
            }
            ++n_done;

            // Print the accumulated accuracy mean x 100
            LOG("%d\t%.8lf\n", n_done, 100.*n_correct/n_done);
        }

        i0 = i1 - 1;
    }

    llama_batch_free(batch);

    if (n_done < 100 && (params.multiple_choice_tasks != 0 && params.multiple_choice_tasks < (size_t)n_task)) return;

    float p = 1.f*n_correct/n_done;
    float sigma = sqrt(p*(1-p)/(n_done-1));
    LOG("\n");
    LOG_INF("Final result: %.4f +/- %.4f\n", 100.f*p, 100.f*sigma);
    p = 1.f*n_done/n_tot_answers;
    sigma = sqrt(p*(1-p)/(n_done-1));
    LOG_INF("Random chance: %.4f +/- %.4f\n", 100.f*p, 100.f*sigma);

    LOG_INF("\n");
}

eval_result multiple_choice_tasks::result() {
    LOG_INF("%s: hellawswag get result\n", __func__);
    return {0};
}

