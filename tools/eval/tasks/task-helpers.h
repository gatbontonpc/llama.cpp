#ifndef LLAMA_TASK_HELPERS_H
#define LLAMA_TASK_HELPERS_H

#include "common.h"
#include "log.h"

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

#define K_TOKEN_CHUNK 4

void compute_logprobs(const float * batch_logits, int n_vocab, std::vector<std::thread>& workers, const std::vector<std::pair<size_t, llama_token>>& eval_pairs, std::vector<float>& eval_results);

bool decode_helper(llama_context * ctx, llama_batch & batch, std::vector<float> & batch_logits, int n_batch, int n_vocab);
bool deserialize_string(std::istream & in, std::string & str);
bool decode_helper(llama_context * ctx, llama_batch & batch, std::vector<float> & batch_logits, int n_batch, int n_vocab);

std::vector<float> softmax(const std::vector<float>& logits);

#endif