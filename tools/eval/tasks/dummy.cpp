#include "../eval-task.h"
#include "log.h"

void dummy_task::prepare(llama_context* ctx)  {
    LOG_INF("%s: dummy prepare\n", __func__);

}
void dummy_task::run(llama_context* ctxs) {
    LOG_INF("%s: dummy run\n", __func__);

}
eval_result dummy_task::result() {
    LOG_INF("%s: dummy get result\n", __func__);

    return {0};
}

