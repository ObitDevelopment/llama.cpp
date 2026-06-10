// Obit fork: stage-aware execution ABI.
//
// This header is intentionally self-contained from upstream `llama.h`. It is
// re-included by `llama.h` at the bottom so existing consumers don't need to
// change their includes, but downstream Obit code MAY include it directly
// (e.g. when only the stage ABI is needed). Keeping the fork-specific
// declarations in a dedicated header means upstream merges against
// `llama.h` essentially never conflict.
//
// All declarations use stable C linkage and are versioned via
// `OBIT_LLAMA_ABI_VERSION` / `OBIT_LLAMA_STAGE_ABI_VERSION`. Bump those
// constants any time the wire shape of an exposed function or struct
// changes; consumers check them at load time before calling fork APIs.

#ifndef LLAMA_OBIT_STAGE_H
#define LLAMA_OBIT_STAGE_H

#include <stdbool.h>
#include <stdint.h>

// Forward declarations of upstream types used in the ABI signatures. We
// avoid pulling all of `llama.h` to keep this header light when consumers
// include it directly. `llama.h` includes us *after* defining these.
struct llama_model;
struct llama_context_params;
struct llama_batch;

#ifndef LLAMA_API
#    ifdef LLAMA_SHARED
#        if defined(_WIN32) && !defined(__MINGW32__)
#            ifdef LLAMA_BUILD
#                define LLAMA_API __declspec(dllexport)
#            else
#                define LLAMA_API __declspec(dllimport)
#            endif
#        else
#            define LLAMA_API __attribute__ ((visibility ("default")))
#        endif
#    else
#        define LLAMA_API
#    endif
#endif

#define OBIT_LLAMA_ABI_VERSION 1
#define OBIT_LLAMA_STAGE_ABI_VERSION 1

enum obit_llama_stage_capability_flags {
    OBIT_LLAMA_STAGE_CAPABILITY_NONE = 0,
    // Set when this fork build can execute a contiguous transformer layer range
    // and exchange boundary tensors with the Obit sidecar.
    OBIT_LLAMA_STAGE_CAPABILITY_LAYER_RANGE = 1 << 0,
};

struct obit_llama_stage_runtime;

struct obit_llama_stage_params {
    uint32_t stage_index;
    uint32_t total_stages;
    uint32_t layer_start;
    uint32_t layer_end;
    bool emit_logits;
};

struct obit_llama_stage_model_info {
    uint32_t n_layer;
    uint32_t n_embd;
    bool has_encoder;
    bool has_decoder;
    bool is_recurrent;
    bool is_hybrid;
};

enum obit_llama_stage_tensor_kind {
    OBIT_LLAMA_STAGE_TENSOR_KIND_NONE = 0,
    OBIT_LLAMA_STAGE_TENSOR_KIND_TOKENS = 1,
    OBIT_LLAMA_STAGE_TENSOR_KIND_HIDDEN_STATE = 2,
    OBIT_LLAMA_STAGE_TENSOR_KIND_LOGITS = 3,
};

enum obit_llama_stage_tensor_dtype {
    OBIT_LLAMA_STAGE_TENSOR_DTYPE_NONE = 0,
    OBIT_LLAMA_STAGE_TENSOR_DTYPE_I32 = 1,
    OBIT_LLAMA_STAGE_TENSOR_DTYPE_F32 = 2,
};

struct obit_llama_stage_boundary_info {
    uint32_t input_kind;
    uint32_t input_dtype;
    uint32_t input_width;
    uint32_t output_kind;
    uint32_t output_dtype;
    uint32_t output_width;
};

#ifdef __cplusplus
extern "C" {
#endif

    // Obit fork ABI marker. This lets downstream loaders distinguish the Obit
    // fork from upstream-compatible libllama builds before using fork APIs.
    LLAMA_API uint32_t obit_llama_abi_version(void);
    LLAMA_API const char * obit_llama_build_info(void);
    LLAMA_API uint32_t obit_llama_stage_abi_version(void);
    LLAMA_API uint64_t obit_llama_stage_capability_flags(void);
    LLAMA_API const char * obit_llama_stage_unsupported_reason(void);
    LLAMA_API struct obit_llama_stage_params obit_llama_stage_default_params(void);
    LLAMA_API int32_t obit_llama_stage_validate_params(
            struct obit_llama_stage_params stage_params);
    LLAMA_API int32_t obit_llama_stage_get_model_info(
            const struct llama_model * model,
            struct obit_llama_stage_model_info * out_info);
    LLAMA_API int32_t obit_llama_stage_get_boundary_info(
            const struct llama_model * model,
            struct obit_llama_stage_params stage_params,
            struct obit_llama_stage_boundary_info * out_info);
    LLAMA_API struct obit_llama_stage_runtime * obit_llama_stage_init_from_model(
            struct llama_model * model,
            struct llama_context_params context_params,
            struct obit_llama_stage_params stage_params);
    LLAMA_API void obit_llama_stage_free(struct obit_llama_stage_runtime * runtime);
    LLAMA_API const char * obit_llama_stage_last_error(void);

    // Stage runtime decode/sample surface.
    LLAMA_API int32_t obit_llama_stage_decode(
            struct obit_llama_stage_runtime * runtime,
            struct llama_batch batch);
    LLAMA_API float * obit_llama_stage_get_logits_ith(
            struct obit_llama_stage_runtime * runtime,
            int32_t i);
    // For non-last stages (emit_logits=false), the runtime produces a
    // hidden state at each requested output row. Returns NULL on misuse.
    LLAMA_API float * obit_llama_stage_get_embeddings_ith(
            struct obit_llama_stage_runtime * runtime,
            int32_t i);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_OBIT_STAGE_H
