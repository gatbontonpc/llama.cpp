#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "eval-task.h"

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

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx = 512;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    common_init();

    const int32_t n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        LOG_ERR("%s: perplexity tool requires '--ctx-size' > 0\n", __func__);
        return 1;
    }

    const bool ppl = !params.hellaswag && !params.winogrande && !params.multiple_choice && !params.kl_divergence;

    if (ppl) {
        const int32_t n_seq = std::max(1, params.n_batch / n_ctx);
        const int32_t n_kv = n_seq * n_ctx;

        params.n_parallel = n_seq;
        params.n_ctx      = n_kv;

        params.n_batch = std::min(params.n_batch, n_kv);
    } else {
        params.n_batch = std::min(params.n_batch, params.n_ctx);
        if (params.kl_divergence) {
            params.n_parallel = 1;
        } else {
            // ensure there's at least enough seq_ids for HellaSwag
            params.n_parallel = std::max(4, params.n_parallel);
        }
    }

    if (params.ppl_stride > 0) {
        LOG_INF("Will perform strided perplexity calculation -> adjusting context size from %d to %d\n",
                params.n_ctx, params.n_ctx + params.ppl_stride/2);
        params.n_ctx += params.ppl_stride/2;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model and apply lora adapter, if any
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }
    const int n_ctx_train = llama_model_n_ctx_train(model);

    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    std::vector<std::unique_ptr<eval_task>> task_queue;
    if (params.hellaswag) {
        LOG_INF("Adding HELLASWAG\n");
        task_queue.emplace_back(std::make_unique<hellaswag_task>(params));
    }

    for (auto& task: task_queue) {
        task->prepare(ctx);
        LOG_INF("\nStarting HELLASWAG\n");
        task->run(ctx);
        task->result();
    }


    LOG("\n");
    llama_perf_context_print(ctx);
    llama_memory_breakdown_print(ctx);

    llama_backend_free();

    return 0;
}
