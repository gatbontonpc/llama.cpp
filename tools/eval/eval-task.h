#ifndef LLAMA_EVAL_H
#define LLAMA_EVAL_H

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "./tasks/task-helpers.h"
#include "common.h"
#include "llama.h"

// command line params
enum output_formats { NONE, CSV, JSON, JSONL, MARKDOWN, SQL };
struct eval_result {
    double result;
};

struct eval_task {
    common_params params;

    eval_task(common_params p) : params(std::move(p)) {}
    virtual ~eval_task() = default;

    virtual void prepare(llama_context*) = 0;
    virtual void run(llama_context*) = 0;
    virtual eval_result result() = 0;

    std::string dataset;
};

struct dummy_task : eval_task {
    dummy_task(common_params p) : eval_task(p){}

    void prepare(llama_context* ctx) override;
    void run(llama_context* ctx) override;
    eval_result result() override;
};


struct hellaswag_task : eval_task {
    struct hs_data_t {
        std::string context;
        size_t gold_ending_idx;
        std::string ending[4];
        size_t ending_logprob_count[4];
        double ending_logprob[4];

        size_t i_logits;        // starting index of logits in the llama_batch
        size_t common_prefix;   // max number of initial tokens that are the same in all sentences
        size_t required_tokens; // needed number of tokens to evaluate all 4 endings
        std::vector<llama_token> seq_tokens[4];
    };

    hellaswag_task(common_params p);

    void prepare(llama_context* ctx) override;
    void run(llama_context* ctx) override;
    eval_result result() override;

private:
    std::vector<hs_data_t> hs_data;
};

struct multiple_choice_tasks : eval_task {
    struct multiple_choice_answers {
        std::vector<std::string> answers;
        std::vector<int>         labels;
        bool deserialize(std::istream& in) {
            uint32_t n;
            in.read((char *)&n, sizeof(n));
            if (in.fail() || n > 100) return false; // 100 as max. number of answers should be good enough for any practical purpose
            answers.resize(n);
            labels.resize(n);
            for (auto& a : answers) {
                if (!deserialize_string(in, a)) return false;
            }
            in.read((char *)labels.data(), n*sizeof(int));
            return !in.fail();
        }
    };

    struct multiple_choice_task {
        std::string question;         // the question (or context that needs to be continued)
        multiple_choice_answers mc1;  // possible answers (continuations) with a single correct answer
        multiple_choice_answers mc2;  // possible answers (continuations) with multiple correct answers - not handled yet
        bool deserialize(std::istream& in) {
            if (!deserialize_string(in, question)) return false;
            return mc1.deserialize(in) && mc2.deserialize(in);
        }

        // For evaluation
        size_t i_logits;        // starting index of logits in the llama_batch
        size_t common_prefix;   // max number of initial tokens that are the same in all sentences
        size_t required_tokens; // needed number of tokens to evaluate all answers
        std::vector<std::vector<llama_token>> seq_tokens;
        std::vector<float> log_probs;
    };

    multiple_choice_tasks(common_params p);

    void prepare(llama_context* ctx) override;
    void run(llama_context* ctx) override;
    eval_result result() override;

private:
    bool multiple_choice_prepare_one_task(llama_context * ctx, multiple_choice_task& task, bool log_error);

};

struct winograde_tasks : eval_task {
    struct winogrande_entry {
        std::string                first;
        std::string                second;
        std::array<std::string, 2> choices;
        int                        answer;

        size_t                   i_logits;
        size_t                   common_prefix;
        size_t                   required_tokens;
        size_t                   n_base1;  // number of tokens for context + choice 1
        size_t                   n_base2;  // number of tokens for context + choice 2
        std::vector<llama_token> seq_tokens[2];
    };

    winograde_tasks(common_params p);

    void        prepare(llama_context * ctx) override;
    void        run(llama_context * ctx) override;
    eval_result result() override;

    std::vector<winogrande_entry> load_winogrande_from_csv(const std::string & prompt);
};

#endif  // LLAMA_EVAL_H
