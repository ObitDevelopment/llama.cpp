#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_pre_norm;        // also extract the hidden state before the final output norm
    bool embeddings_pre_norm_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // Obit fork: stage-aware execution. When obit_stage_active is true, the
    // per-model graph builder runs only layers [obit_stage_layer_start,
    // obit_stage_layer_end). Stage 0 (obit_stage_layer_start == 0) takes
    // token input; non-stage-0 stages take a hidden-state input via
    // ubatch->embd. When obit_stage_emit_logits is false, the graph emits
    // the post-loop hidden state instead of applying output_norm + lm_head.
    // See LIBLLAMA_STAGE_SLICE2_DESIGN.md.
    bool     obit_stage_active;
    bool     obit_stage_emit_logits;
    uint32_t obit_stage_layer_start;
    uint32_t obit_stage_layer_end;
};
