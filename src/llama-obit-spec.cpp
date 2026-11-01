// Obit fork: MTP speculative-decoding shim. Wraps common_speculative for the
// MTP type only so the sidecar can drive it through libllama. See
// include/llama-obit-spec.h for the public surface and the call order.

#include "llama.h"

#include "llama-obit-spec.h"

#include "common.h"
#include "speculative.h"

#include <cerrno>
#include <cstring>
#include <new>
#include <string>
#include <vector>

struct obit_spec_mtp_seq {
    // prompt passed to begin(), kept alive for common_speculative_draft_params
    std::vector<llama_token> prompt;

    // draft output for the last common_speculative_draft() call
    std::vector<llama_token> result;
};

struct obit_spec_mtp {
    common_speculative * spec = nullptr;
    llama_context * ctx_tgt = nullptr;
    // reusable mirror of the caller's llama_batch, see obit_spec_mtp_process
    common_batch batch;
    uint32_t n_seq = 0;
    int32_t  n_max = 0;
    std::vector<obit_spec_mtp_seq> seqs;
};

static thread_local std::string obit_spec_mtp_error;

static void obit_spec_mtp_set_error(const std::string & error) {
    obit_spec_mtp_error = error;
}

const char * obit_spec_mtp_last_error(void) {
    return obit_spec_mtp_error.c_str();
}

uint32_t obit_llama_spec_abi_version(void) {
    return OBIT_LLAMA_SPEC_ABI_VERSION;
}

int32_t obit_spec_mtp_supported(const struct llama_model * model) {
    if (model == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp supported requires a non-null llama_model");
        return -EINVAL;
    }

    obit_spec_mtp_set_error("");
    return llama_model_n_layer_nextn(model) > 0 ? 1 : 0;
}

int32_t obit_spec_mtp_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft,
        uint32_t n_seq,
        int32_t n_max,
        struct obit_spec_mtp ** out) {
    if (out == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp init requires a non-null output handle");
        return -EINVAL;
    }
    *out = nullptr;

    if (ctx_tgt == nullptr || ctx_dft == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp init requires non-null ctx_tgt and ctx_dft");
        return -EINVAL;
    }
    if (n_seq == 0) {
        obit_spec_mtp_set_error("obit spec mtp init requires n_seq > 0");
        return -EINVAL;
    }
    if (n_max <= 0) {
        obit_spec_mtp_set_error("obit spec mtp init requires n_max > 0");
        return -EINVAL;
    }

    // MTP only. The draft context is built by the caller with
    // ctx_type = LLAMA_CONTEXT_TYPE_MTP, so there is no draft model path here.
    common_params_speculative params;
    params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.draft.n_max   = n_max;
    params.draft.ctx_tgt = ctx_tgt;
    params.draft.ctx_dft = ctx_dft;

    common_speculative * spec = common_speculative_init(params, n_seq);
    if (spec == nullptr) {
        obit_spec_mtp_set_error(
                "common_speculative_init failed for MTP; check that ctx_dft was created with "
                "ctx_type = LLAMA_CONTEXT_TYPE_MTP and that the model has MTP layers");
        return -EIO;
    }

    obit_spec_mtp * handle = new (std::nothrow) obit_spec_mtp{};
    if (handle == nullptr) {
        common_speculative_free(spec);
        obit_spec_mtp_set_error("obit spec mtp init could not allocate the handle");
        return -ENOMEM;
    }

    handle->spec    = spec;
    handle->ctx_tgt = ctx_tgt;
    handle->batch   = common_batch(ctx_tgt);
    handle->n_seq = n_seq;
    handle->n_max = n_max;
    handle->seqs.resize(n_seq);

    *out = handle;
    obit_spec_mtp_set_error("");
    return 0;
}

int32_t obit_spec_mtp_begin(
        struct obit_spec_mtp * spec,
        int32_t seq_id,
        const int32_t * prompt,
        size_t n) {
    if (spec == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp begin requires a non-null handle");
        return -EINVAL;
    }
    if (seq_id < 0 || seq_id >= (int32_t) spec->n_seq) {
        obit_spec_mtp_set_error("obit spec mtp begin seq_id is out of range");
        return -EINVAL;
    }
    if (n > 0 && prompt == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp begin requires a non-null prompt for n > 0");
        return -EINVAL;
    }

    obit_spec_mtp_seq & s = spec->seqs[seq_id];
    if (n == 0) {
        s.prompt.clear();
    } else {
        s.prompt.assign(prompt, prompt + n);
    }

    common_speculative_begin(spec->spec, seq_id, s.prompt);

    obit_spec_mtp_set_error("");
    return 0;
}

int32_t obit_spec_mtp_process(
        struct obit_spec_mtp * spec,
        const struct llama_batch * batch) {
    if (spec == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp process requires a non-null handle");
        return -EINVAL;
    }
    if (batch == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp process requires a non-null batch");
        return -EINVAL;
    }

    // Upstream #29385 moved common_speculative_process() from llama_batch to
    // common_batch (the llama_batch_ext wrapper). The shim's public ABI keeps
    // taking the same llama_batch the caller just handed llama_decode(), so
    // mirror it entry by entry. The MTP impl reads only id / pos / seq_id /
    // embd from it; embd views point into the caller's batch, which outlives
    // this call.
    common_batch & cb = spec->batch;
    cb.clear();
    const int32_t n_embd = llama_model_n_embd_inp(llama_get_model(spec->ctx_tgt));
    for (int32_t i = 0; i < batch->n_tokens; ++i) {
        const llama_seq_id seq0  = batch->seq_id ? batch->seq_id[i][0] : 0;
        const bool         out_i = batch->logits ? batch->logits[i] != 0 : (i == batch->n_tokens - 1);
        const llama_pos    pos0  = batch->pos ? batch->pos[i] : 0;
        int32_t idx;
        if (batch->token) {
            idx = cb.add(batch->token[i], pos0, seq0, out_i);
        } else if (batch->embd) {
            // llama_batch keeps M-RoPE positions section-major: pos[i + s*n_tokens]
            llama_pos pos_i[GGML_MROPE_SECTIONS] = {};
            for (int32_t j = 0; j < cb.n_pos; ++j) {
                pos_i[j] = batch->pos ? batch->pos[i + j * batch->n_tokens] : 0;
            }
            idx = cb.add_embd({ batch->embd + (size_t) i * n_embd, 1, (size_t) n_embd }, pos_i, seq0, out_i);
        } else {
            break;
        }
        if (batch->seq_id && batch->n_seq_id) {
            for (int32_t s = 1; s < batch->n_seq_id[i]; ++s) {
                cb.add_seq(idx, batch->seq_id[i][s]);
            }
        }
    }

    const bool ingested = common_speculative_process(spec->spec, cb);
    if (!ingested) {
        obit_spec_mtp_set_error("common_speculative_process failed; see the log for the draft decode error");
        return -EIO;
    }

    obit_spec_mtp_set_error("");
    return 1;
}

int32_t obit_spec_mtp_draft(
        struct obit_spec_mtp * spec,
        int32_t seq_id,
        int32_t id_last,
        int32_t pos_last,
        int32_t * out,
        size_t cap,
        size_t * n_out) {
    if (spec == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp draft requires a non-null handle");
        return -EINVAL;
    }
    if (seq_id < 0 || seq_id >= (int32_t) spec->n_seq) {
        obit_spec_mtp_set_error("obit spec mtp draft seq_id is out of range");
        return -EINVAL;
    }
    if (n_out == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp draft requires a non-null n_out");
        return -EINVAL;
    }
    if (out == nullptr && cap > 0) {
        obit_spec_mtp_set_error("obit spec mtp draft requires a non-null output buffer for cap > 0");
        return -EINVAL;
    }
    *n_out = 0;

    // Draft for the requested sequence only. common_speculative_draft() walks
    // every sequence with drafting set; clear the others first.
    for (int32_t i = 0; i < (int32_t) spec->n_seq; ++i) {
        common_speculative_get_draft_params(spec->spec, i).drafting = false;
    }

    obit_spec_mtp_seq & s = spec->seqs[seq_id];
    s.result.clear();

    // Feed common_speculative what the server feeds it: id_last is the token
    // the target just sampled, pos0 the position it is decoded at.
    common_speculative_draft_params & dp = common_speculative_get_draft_params(spec->spec, seq_id);
    dp          = common_speculative_draft_params{};
    dp.drafting = true;
    dp.n_max    = -1; // no override; the impl already caps at its n_max
    dp.pos0     = pos_last;
    dp.id_last  = id_last;
    dp.prompt   = &s.prompt;
    dp.result   = &s.result;

    common_speculative_draft(spec->spec);

    *n_out = s.result.size();
    if (s.result.size() > cap) {
        obit_spec_mtp_set_error("obit spec mtp draft output buffer is too small");
        return -ENOSPC;
    }

    if (!s.result.empty()) {
        std::memcpy(out, s.result.data(), s.result.size() * sizeof(llama_token));
    }

    obit_spec_mtp_set_error("");
    return 0;
}

int32_t obit_spec_mtp_draft_all(
        struct obit_spec_mtp * spec,
        const int32_t * seq_ids,
        const int32_t * id_last,
        const int32_t * pos_last,
        size_t n_seq,
        int32_t * out,
        size_t cap_per_seq,
        size_t * n_out_per_seq) {
    if (spec == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp draft_all requires a non-null handle");
        return -EINVAL;
    }
    if (seq_ids == nullptr || id_last == nullptr || pos_last == nullptr || n_out_per_seq == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp draft_all requires non-null per-sequence arrays");
        return -EINVAL;
    }
    if (n_seq == 0) {
        obit_spec_mtp_set_error("obit spec mtp draft_all requires n_seq > 0");
        return -EINVAL;
    }
    if (out == nullptr && cap_per_seq > 0) {
        obit_spec_mtp_set_error("obit spec mtp draft_all requires a non-null output buffer for cap_per_seq > 0");
        return -EINVAL;
    }

    // Clear every drafting flag, then mark the listed sequences. This mirrors
    // the server's speculative step: it sets the draft params for each active
    // slot and calls common_speculative_draft() once for all of them.
    for (int32_t i = 0; i < (int32_t) spec->n_seq; ++i) {
        common_speculative_get_draft_params(spec->spec, i).drafting = false;
    }

    for (size_t j = 0; j < n_seq; ++j) {
        const int32_t sid = seq_ids[j];
        if (sid < 0 || sid >= (int32_t) spec->n_seq) {
            obit_spec_mtp_set_error("obit spec mtp draft_all seq_id is out of range");
            return -EINVAL;
        }

        obit_spec_mtp_seq & s = spec->seqs[sid];
        s.result.clear();

        common_speculative_draft_params & dp = common_speculative_get_draft_params(spec->spec, sid);
        dp          = common_speculative_draft_params{};
        dp.drafting = true;
        dp.n_max    = -1;
        dp.pos0     = pos_last[j];
        dp.id_last  = id_last[j];
        dp.prompt   = &s.prompt;
        dp.result   = &s.result;
    }

    common_speculative_draft(spec->spec);

    bool too_small = false;
    for (size_t j = 0; j < n_seq; ++j) {
        n_out_per_seq[j] = spec->seqs[seq_ids[j]].result.size();
        if (n_out_per_seq[j] > cap_per_seq) {
            too_small = true;
        }
    }
    if (too_small) {
        obit_spec_mtp_set_error("obit spec mtp draft_all output buffer is too small");
        return -ENOSPC;
    }

    for (size_t j = 0; j < n_seq; ++j) {
        const std::vector<llama_token> & r = spec->seqs[seq_ids[j]].result;
        if (!r.empty()) {
            std::memcpy(out + j * cap_per_seq, r.data(), r.size() * sizeof(llama_token));
        }
    }

    obit_spec_mtp_set_error("");
    return 0;
}

int32_t obit_spec_mtp_accept(
        struct obit_spec_mtp * spec,
        int32_t seq_id,
        uint16_t n_accepted) {
    if (spec == nullptr) {
        obit_spec_mtp_set_error("obit spec mtp accept requires a non-null handle");
        return -EINVAL;
    }
    if (seq_id < 0 || seq_id >= (int32_t) spec->n_seq) {
        obit_spec_mtp_set_error("obit spec mtp accept seq_id is out of range");
        return -EINVAL;
    }

    common_speculative_accept(spec->spec, seq_id, n_accepted);

    obit_spec_mtp_set_error("");
    return 0;
}

void obit_spec_mtp_free(struct obit_spec_mtp * spec) {
    if (spec == nullptr) {
        return;
    }

    common_speculative_free(spec->spec);
    delete spec;
}
