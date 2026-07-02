// Obit fork: stage-aware ABI implementation. Moved out of llama.cpp so the
// upstream-tracking diff against that file is essentially zero. See
// include/llama-obit-stage.h for the public surface.

#include "llama.h"

#include "llama-impl.h"
#include "llama-context.h"
#include "llama-model.h"

#include "ggml.h"

#include <cstdint>
#include <string>
#include <utility>

uint32_t obit_llama_abi_version(void) {
    return OBIT_LLAMA_ABI_VERSION;
}

const char * obit_llama_build_info(void) {
    return "obit-llama abi=1 stage_abi=2 stage_flags=3 boundary_info=1 single_stage=1 layer_range=qwen3+qwen3moe+qwen35moe clear_seq=1";
}

uint32_t obit_llama_stage_abi_version(void) {
    return OBIT_LLAMA_STAGE_ABI_VERSION;
}

uint64_t obit_llama_stage_capability_flags(void) {
    // Process-global capability: at least one supported architecture exposes
    // layer-range execution. Per-architecture support is queried via
    // obit_llama_stage_get_model_info / obit_llama_stage_init_from_model.
    return OBIT_LLAMA_STAGE_CAPABILITY_LAYER_RANGE
        |  OBIT_LLAMA_STAGE_CAPABILITY_CLEAR_SEQUENCE;
}

const char * obit_llama_stage_unsupported_reason(void) {
    return "obit libllama stage execution hooks support only qwen3, qwen3moe, and qwen35moe architectures today; other architectures fail closed before StageReady";
}

static bool obit_llama_arch_supports_layer_range(llm_arch arch) {
    // Expand as additional per-arch graph builders are taught to honor
    // cparams.obit_stage_* bounds via get_stage_bounds() +
    // build_stage_output_or_boundary().
    return arch == LLM_ARCH_QWEN3
        || arch == LLM_ARCH_QWEN3MOE
        || arch == LLM_ARCH_QWEN35MOE;
}

struct obit_llama_stage_runtime {
    llama_model * model;
    llama_context * ctx;
    obit_llama_stage_params params;
};

static thread_local std::string obit_llama_stage_error;

static void obit_llama_stage_set_error(const std::string & error) {
    obit_llama_stage_error = error;
}

struct obit_llama_stage_params obit_llama_stage_default_params(void) {
    struct obit_llama_stage_params result = {
        /*.stage_index =*/ 0,
        /*.total_stages =*/ 1,
        /*.layer_start  =*/ 0,
        /*.layer_end    =*/ 0,
        /*.emit_logits  =*/ true,
    };

    return result;
}

int32_t obit_llama_stage_validate_params(struct obit_llama_stage_params stage_params) {
    if (stage_params.total_stages == 0) {
        obit_llama_stage_set_error("obit libllama stage params require total_stages > 0");
        return -1;
    }
    if (stage_params.stage_index >= stage_params.total_stages) {
        obit_llama_stage_set_error("obit libllama stage params require stage_index < total_stages");
        return -1;
    }
    if (stage_params.layer_start >= stage_params.layer_end) {
        obit_llama_stage_set_error("obit libllama stage params require layer_start < layer_end");
        return -1;
    }

    obit_llama_stage_set_error("");
    return 0;
}

int32_t obit_llama_stage_get_model_info(
        const struct llama_model * model,
        struct obit_llama_stage_model_info * out_info) {
    if (model == nullptr) {
        obit_llama_stage_set_error("obit libllama stage model info requires a non-null llama_model");
        return -1;
    }
    if (out_info == nullptr) {
        obit_llama_stage_set_error("obit libllama stage model info requires a non-null output pointer");
        return -1;
    }

    *out_info = {
        /*.n_layer      =*/ uint32_t(llama_model_n_layer(model)),
        /*.n_embd       =*/ uint32_t(llama_model_n_embd(model)),
        /*.has_encoder  =*/ llama_model_has_encoder(model),
        /*.has_decoder  =*/ llama_model_has_decoder(model),
        /*.is_recurrent =*/ llama_model_is_recurrent(model),
        /*.is_hybrid    =*/ llama_model_is_hybrid(model),
    };

    obit_llama_stage_set_error("");
    return 0;
}

int32_t obit_llama_stage_get_boundary_info(
        const struct llama_model * model,
        struct obit_llama_stage_params stage_params,
        struct obit_llama_stage_boundary_info * out_info) {
    if (model == nullptr) {
        obit_llama_stage_set_error("obit libllama stage boundary info requires a non-null llama_model");
        return -1;
    }
    if (out_info == nullptr) {
        obit_llama_stage_set_error("obit libllama stage boundary info requires a non-null output pointer");
        return -1;
    }
    if (obit_llama_stage_validate_params(stage_params) != 0) {
        return -1;
    }

    struct obit_llama_stage_model_info model_info = {};
    if (obit_llama_stage_get_model_info(model, &model_info) != 0) {
        return -1;
    }
    if (stage_params.layer_end > model_info.n_layer) {
        obit_llama_stage_set_error(
                "obit libllama stage params require layer_end <= model layer count");
        return -1;
    }

    const bool first_stage = stage_params.stage_index == 0;
    const bool output_logits = stage_params.emit_logits;
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (output_logits && vocab == nullptr) {
        obit_llama_stage_set_error("obit libllama stage boundary info requires model vocab for logits output");
        return -1;
    }

    *out_info = {
        /*.input_kind   =*/ uint32_t(first_stage
                ? OBIT_LLAMA_STAGE_TENSOR_KIND_TOKENS
                : OBIT_LLAMA_STAGE_TENSOR_KIND_HIDDEN_STATE),
        /*.input_dtype  =*/ uint32_t(first_stage
                ? OBIT_LLAMA_STAGE_TENSOR_DTYPE_I32
                : OBIT_LLAMA_STAGE_TENSOR_DTYPE_F32),
        /*.input_width  =*/ uint32_t(first_stage
                ? 1
                : llama_model_n_embd_inp(model)),
        /*.output_kind  =*/ uint32_t(output_logits
                ? OBIT_LLAMA_STAGE_TENSOR_KIND_LOGITS
                : OBIT_LLAMA_STAGE_TENSOR_KIND_HIDDEN_STATE),
        /*.output_dtype =*/ uint32_t(OBIT_LLAMA_STAGE_TENSOR_DTYPE_F32),
        /*.output_width =*/ uint32_t(output_logits
                ? llama_vocab_n_tokens(vocab)
                : llama_model_n_embd_out(model)),
    };

    obit_llama_stage_set_error("");
    return 0;
}

struct obit_llama_stage_runtime * obit_llama_stage_init_from_model(
        struct llama_model * model,
        struct llama_context_params context_params,
        struct obit_llama_stage_params stage_params) {
    if (obit_llama_stage_validate_params(stage_params) != 0) {
        return nullptr;
    }
    if (model == nullptr) {
        obit_llama_stage_set_error("obit libllama stage init requires a non-null llama_model");
        return nullptr;
    }
    struct obit_llama_stage_model_info model_info = {};
    if (obit_llama_stage_get_model_info(model, &model_info) != 0) {
        return nullptr;
    }
    if (stage_params.layer_end > model_info.n_layer) {
        obit_llama_stage_set_error(
                "obit libllama stage params require layer_end <= model layer count");
        return nullptr;
    }

    const bool is_full_single_stage =
            stage_params.total_stages == 1 &&
            stage_params.stage_index  == 0 &&
            stage_params.layer_start  == 0 &&
            stage_params.layer_end    == model_info.n_layer &&
            stage_params.emit_logits;

    const bool is_first_stage = stage_params.stage_index == 0;
    const bool is_last_stage  = stage_params.stage_index + 1 == stage_params.total_stages;

    if (!is_full_single_stage) {
        if (!obit_llama_arch_supports_layer_range(model->arch)) {
            obit_llama_stage_set_error(obit_llama_stage_unsupported_reason());
            return nullptr;
        }
        // For non-degenerate stages the boundary semantics must agree with
        // the stage index: first stage takes token input, last stage emits
        // logits. Reject mismatches early so the harness fails before
        // running a bad graph.
        //
        // Note: a previous version of this check rejected
        // `!is_first_stage && layer_start == 0` on the assumption that
        // every stage loads the same full-model GGUF. That assumption is
        // false for compile-time per-stage GGUFs (each stage loads only
        // its own slice, so layer_start is always 0 relative to the loaded
        // model). The check was removed; emit_logits below is the
        // sufficient role check.
        if (is_last_stage && !stage_params.emit_logits) {
            obit_llama_stage_set_error(
                    "obit libllama stage init: last stage must set emit_logits=true");
            return nullptr;
        }
        if (!is_last_stage && stage_params.emit_logits) {
            obit_llama_stage_set_error(
                    "obit libllama stage init: non-last stage must set emit_logits=false");
            return nullptr;
        }
    }

    llama_context * ctx = llama_init_from_model(model, context_params);
    if (ctx == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage init failed to create llama_context for stage runtime");
        return nullptr;
    }

    if (!is_full_single_stage) {
        ctx->set_obit_stage_params(
                /*active=*/true,
                /*layer_start=*/stage_params.layer_start,
                /*layer_end=*/stage_params.layer_end,
                /*emit_logits=*/stage_params.emit_logits);
    }

    auto * runtime = new obit_llama_stage_runtime;
    runtime->model  = model;
    runtime->ctx    = ctx;
    runtime->params = stage_params;

    obit_llama_stage_set_error("");
    return runtime;
}

void obit_llama_stage_free(struct obit_llama_stage_runtime * runtime) {
    if (runtime == nullptr) {
        return;
    }
    if (runtime->ctx != nullptr) {
        llama_free(runtime->ctx);
    }
    delete runtime;
}

const char * obit_llama_stage_last_error(void) {
    return obit_llama_stage_error.c_str();
}

int32_t obit_llama_stage_decode(
        struct obit_llama_stage_runtime * runtime,
        struct llama_batch batch) {
    if (runtime == nullptr || runtime->ctx == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage decode requires a runtime with an initialized context");
        return -1;
    }

    const int32_t rc = llama_decode(runtime->ctx, batch);
    if (rc != 0) {
        obit_llama_stage_set_error(
                std::string("obit libllama stage decode forwarded a non-zero llama_decode return: ") +
                std::to_string(rc));
    } else {
        obit_llama_stage_set_error("");
    }
    return rc;
}

float * obit_llama_stage_get_logits_ith(
        struct obit_llama_stage_runtime * runtime,
        int32_t i) {
    if (runtime == nullptr || runtime->ctx == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage get_logits requires a runtime with an initialized context");
        return nullptr;
    }
    float * logits = llama_get_logits_ith(runtime->ctx, i);
    if (logits == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage get_logits forwarded a null llama_get_logits_ith result");
    } else {
        obit_llama_stage_set_error("");
    }
    return logits;
}

float * obit_llama_stage_get_embeddings_ith(
        struct obit_llama_stage_runtime * runtime,
        int32_t i) {
    if (runtime == nullptr || runtime->ctx == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage get_embeddings requires a runtime with an initialized context");
        return nullptr;
    }
    float * embd = llama_get_embeddings_ith(runtime->ctx, i);
    if (embd == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage get_embeddings forwarded a null llama_get_embeddings_ith result");
    } else {
        obit_llama_stage_set_error("");
    }
    return embd;
}

bool obit_llama_stage_clear_sequence(
        struct obit_llama_stage_runtime * runtime,
        int32_t seq_id,
        int32_t p0,
        int32_t p1) {
    if (runtime == nullptr || runtime->ctx == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage clear_sequence requires a runtime with an initialized context");
        return false;
    }
    llama_memory_t mem = llama_get_memory(runtime->ctx);
    if (mem == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage clear_sequence: llama_get_memory returned null");
        return false;
    }
    const bool ok = llama_memory_seq_rm(mem, (llama_seq_id) seq_id, (llama_pos) p0, (llama_pos) p1);
    if (!ok) {
        obit_llama_stage_set_error(
                std::string("obit libllama stage clear_sequence: llama_memory_seq_rm failed for seq_id=") +
                std::to_string(seq_id));
    } else {
        obit_llama_stage_set_error("");
    }
    return ok;
}
