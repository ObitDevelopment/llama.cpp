// Obit fork: MTP speculative-decoding ABI.
//
// Thin C shim over `common_speculative` for the MTP type only. The Rust
// sidecar links `libllama.so` and drives MTP through these exports, so it does
// not have to link the `common` C++ library. Same conventions as
// `include/llama-obit-stage.h`: self-contained, stable C linkage, and versioned
// via `OBIT_LLAMA_SPEC_ABI_VERSION`. `llama.h` re-includes this header at the
// bottom, but downstream code MAY include it directly.
//
// The shim hides `common_speculative`, not its state machine. The caller must
// drive this exact order per sequence:
//
//   1. obit_spec_mtp_process(prompt_batch)  once, after the target decodes the prompt
//   2. obit_spec_mtp_begin(seq, prompt)     once per sequence, before its first draft
//   3. obit_spec_mtp_draft(seq, id_last, pos_last, out)  after process, to get drafts
//      (or obit_spec_mtp_draft_all to draft several sequences in one call)
//   4. append the drafts to the next target batch and decode it
//   5. obit_spec_mtp_accept(seq, n)         after decoding that batch
//   6. obit_spec_mtp_process(target_batch)  after every later target decode
//   7. repeat from 3
//
// Sequence invariants:
//   - process() sees the prompt batch before begin(); begin() runs once per
//     sequence before any draft() for it. The MTP impl pulls the prompt KV from
//     the prompt batch, and begin() with an unprocessed prompt only warns.
//   - process() sees every later target batch, for every sequence in flight, in
//     order. `batch->token` must be set (token batches, not embedding batches).
//   - draft() is per sequence. It drafts for the requested sequence only; any
//     other sequence's stale drafting flag is cleared first.
//   - draft()/draft_all() take the target's start point explicitly: `id_last`
//     is the token the target just sampled, and `pos_last` is the position it
//     is decoded at. These are `common_speculative`'s `id_last` and `pos0`
//     (the server feeds `pos0 = prompt.pos_next()` and `id_last = sampled`).
//     Passing them per call replaces any shim-side reconstruction.
//   - accept() is called only after draft(), with the number of draft tokens
//     the target accepted. Do NOT count the token the target sampled.
//   - free() after every sequence stops. It also frees the draft context state
//     the shim turned on (nextn embeddings), not the caller's contexts.

#ifndef LLAMA_OBIT_SPEC_H
#define LLAMA_OBIT_SPEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Forward declarations of upstream types used in the ABI. `llama.h` defines
// them; we avoid pulling all of `llama.h` to keep this header light.
// llama_seq_id / llama_token / llama_pos are int32_t in upstream.
struct llama_model;
struct llama_context;
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

#define OBIT_LLAMA_SPEC_ABI_VERSION 1

struct obit_spec_mtp;

#ifdef __cplusplus
extern "C" {
#endif

    // Speculative ABI marker. Lets a loader check the MTP shim ABI before use.
    LLAMA_API uint32_t obit_llama_spec_abi_version(void);

    // Return 1 when `model` has trained MTP layers (n_layer_nextn > 0), 0 when
    // it does not, and negative (-EINVAL) on a null model.
    LLAMA_API int32_t obit_spec_mtp_supported(const struct llama_model * model);

    // Build the MTP draft driver over `ctx_tgt` and `ctx_dft`. `ctx_dft` must
    // already be created by the caller with
    // `ctx_type = LLAMA_CONTEXT_TYPE_MTP`, against the same model as the
    // target. `n_seq` is the number of sequence slots, `n_max` the maximum
    // draft count. On success stores the handle in `*out` and returns 0.
    // The target and draft contexts stay owned by the caller.
    LLAMA_API int32_t obit_spec_mtp_init(
            struct llama_context * ctx_tgt,
            struct llama_context * ctx_dft,
            uint32_t n_seq,
            int32_t n_max,
            struct obit_spec_mtp ** out);

    // Register the prompt of a sequence. Call once before the first draft().
    LLAMA_API int32_t obit_spec_mtp_begin(
            struct obit_spec_mtp * spec,
            int32_t seq_id,
            const int32_t * prompt,
            size_t n);

    // Feed one target decode batch and update the draft driver state. Returns
    // 1 when the batch was ingested and drafting may proceed, negative on
    // failure (a draft-context decode error). Does not produce drafts; the
    // next draft() call does. Returns 0 for a null/empty batch.
    LLAMA_API int32_t obit_spec_mtp_process(
            struct obit_spec_mtp * spec,
            const struct llama_batch * batch);

    // Run the draft for one sequence and copy the tokens into `out`. `id_last`
    // and `pos_last` are the target's freshly sampled token and the position it
    // is decoded at (common_speculative's id_last and pos0). `cap` is the
    // capacity of `out` in tokens; `*n_out` receives the token count. If the
    // draft is longer than `cap`, returns -ENOSPC after setting `*n_out` to the
    // required count and copying nothing.
    LLAMA_API int32_t obit_spec_mtp_draft(
            struct obit_spec_mtp * spec,
            int32_t seq_id,
            int32_t id_last,
            int32_t pos_last,
            int32_t * out,
            size_t cap,
            size_t * n_out);

    // Draft for several sequences in one call, matching the server's batched
    // speculative step. The n_seq entries of `seq_ids`, `id_last` and
    // `pos_last` are parallel; `out` holds n_seq buffers of `cap_per_seq`
    // tokens each, and `n_out_per_seq` receives each count. Drafts every listed
    // sequence in one common_speculative_draft() call. Returns 0 on success; on
    // -ENOSPC every element of `n_out_per_seq` holds the count that was needed.
    LLAMA_API int32_t obit_spec_mtp_draft_all(
            struct obit_spec_mtp * spec,
            const int32_t * seq_ids,
            const int32_t * id_last,
            const int32_t * pos_last,
            size_t n_seq,
            int32_t * out,
            size_t cap_per_seq,
            size_t * n_out_per_seq);

    // Report how many draft tokens the target accepted for a sequence. Call
    // only after draft(). `n_accepted` must not include the sampled token.
    LLAMA_API int32_t obit_spec_mtp_accept(
            struct obit_spec_mtp * spec,
            int32_t seq_id,
            uint16_t n_accepted);

    // Tear the driver down. NULL is accepted and ignored.
    LLAMA_API void obit_spec_mtp_free(struct obit_spec_mtp * spec);

    // Static string for the last failure. Empty after a success.
    LLAMA_API const char * obit_spec_mtp_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_OBIT_SPEC_H
