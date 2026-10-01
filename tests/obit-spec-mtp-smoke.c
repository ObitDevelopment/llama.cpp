// Obit fork: smoke test for the MTP speculative shim (obit_spec_mtp_*).
//
// Always checks the fail-closed surface: ABI version, the null-argument paths,
// and that a failure leaves a message in obit_spec_mtp_last_error().
//
// When $OBIT_MTP_TEST_GGUF points at an MTP-capable GGUF, it also loads the
// model, checks obit_spec_mtp_supported() == 1, builds the target and draft
// contexts, and runs obit_spec_mtp_init(). When $OBIT_NON_MTP_TEST_GGUF points
// at a model without MTP layers, it checks obit_spec_mtp_supported() == 0.
// Without a fixture the model-backed checks are skipped.

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n_failed = 0;

static void check(int cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        n_failed++;
    }
}

static struct llama_model * load_model(const char * path, int load_mtp) {
    struct llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = load_mtp;
    return llama_model_load_from_file(path, mparams);
}

static void check_non_mtp_model(const char * path) {
    struct llama_model * model = load_model(path, /*load_mtp*/ false);
    if (model == NULL) {
        printf("skip: could not load non-MTP model '%s'\n", path);
        return;
    }

    check(obit_spec_mtp_supported(model) == 0, "supported() is 0 for a non-MTP model");

    llama_model_free(model);
}

static void check_mtp_model(const char * path) {
    struct llama_model * model = load_model(path, /*load_mtp*/ true);
    if (model == NULL) {
        printf("skip: could not load MTP model '%s'\n", path);
        return;
    }

    check(obit_spec_mtp_supported(model) == 1, "supported() is 1 for an MTP model");

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 64;

    struct llama_context * ctx_tgt = llama_init_from_model(model, cparams);
    check(ctx_tgt != NULL, "target context created");
    if (ctx_tgt == NULL) {
        llama_model_free(model);
        return;
    }

    struct llama_context_params dparams = llama_context_default_params();
    dparams.n_ctx   = llama_n_ctx(ctx_tgt);
    dparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;

    struct llama_context * ctx_dft = llama_init_from_model(model, dparams);
    check(ctx_dft != NULL, "MTP draft context created");
    if (ctx_dft != NULL) {
        struct obit_spec_mtp * spec = NULL;
        int rc = obit_spec_mtp_init(ctx_tgt, ctx_dft, 1, 3, &spec);
        check(rc == 0, "obit_spec_mtp_init() succeeded");
        check(spec != NULL, "obit_spec_mtp_init() set the handle");

        if (spec != NULL) {
            // Exercise the full call order once: prefill on the target, then
            // begin/process/draft/accept.
            const char * text = "The capital of France is";
            llama_token toks[32];
            int32_t n_toks = llama_tokenize(
                    llama_model_get_vocab(model), text, (int32_t) strlen(text),
                    toks, 32, /*add_special*/ true, /*parse_special*/ true);

            if (n_toks > 0 && n_toks <= 32) {
                struct llama_batch batch = llama_batch_init(n_toks, 0, 1);
                for (int32_t i = 0; i < n_toks; ++i) {
                    batch.token[i]     = toks[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = 1;
                }
                batch.n_tokens = n_toks;

                rc = llama_decode(ctx_tgt, batch);
                check(rc == 0, "target prefill decode succeeded");

                rc = obit_spec_mtp_process(spec, &batch);
                check(rc >= 0, "obit_spec_mtp_process() succeeded");

                rc = obit_spec_mtp_begin(spec, 0, toks, (size_t) n_toks);
                check(rc == 0, "obit_spec_mtp_begin() succeeded");

                // Sample the target's next token. Its position is pos_next,
                // which is n_toks here. This is what the driver feeds the draft.
                struct llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

                const llama_token sampled = llama_sampler_sample(smpl, ctx_tgt, -1);
                const int32_t pos_last = n_toks;

                // Round 1: batched entry point, one sequence.
                const int32_t seq_ids[1] = { 0 };
                const int32_t id_last[1] = { sampled };
                const int32_t pos_batch[1] = { pos_last };
                llama_token out_all[8];
                size_t n_out_all[1] = { 0 };
                rc = obit_spec_mtp_draft_all(spec, seq_ids, id_last, pos_batch, 1, out_all, 8, n_out_all);
                if (rc != 0) {
                    fprintf(stderr, "draft_all rc = %d: %s\n", rc, obit_spec_mtp_last_error());
                }
                check(rc == 0, "obit_spec_mtp_draft_all() succeeded");
                printf("draft_all drafted %zu token(s) from sampled token %d at pos %d\n",
                        n_out_all[0], sampled, pos_last);

                rc = obit_spec_mtp_accept(spec, 0, 0);
                check(rc == 0, "obit_spec_mtp_accept() succeeded");

                // Round 2: verify the drafts on the target, then use the single
                // entry point from the next sampled token. This keeps the draft
                // KV position moving forward, like the real loop.
                const int32_t n_pass = 1 + (int32_t) n_out_all[0];
                struct llama_batch batch2 = llama_batch_init(n_pass, 0, 1);
                batch2.token[0]     = sampled;
                batch2.pos[0]       = pos_last;
                batch2.n_seq_id[0]  = 1;
                batch2.seq_id[0][0] = 0;
                batch2.logits[0]    = 1;
                for (int32_t i = 0; i < (int32_t) n_out_all[0]; ++i) {
                    batch2.token[1 + i]     = out_all[i];
                    batch2.pos[1 + i]       = pos_last + 1 + i;
                    batch2.n_seq_id[1 + i]  = 1;
                    batch2.seq_id[1 + i][0] = 0;
                    batch2.logits[1 + i]    = 1;
                }
                batch2.n_tokens = n_pass;

                rc = llama_decode(ctx_tgt, batch2);
                check(rc == 0, "target verify decode succeeded");

                // Drop the draft context's draft region before re-feeding, the
                // same reset the server does from its draft checkpoint. The
                // shim does not own this; the caller drives ctx_dft directly.
                llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pos_last, -1);

                rc = obit_spec_mtp_process(spec, &batch2);
                check(rc >= 0, "obit_spec_mtp_process() (round 2) succeeded");

                const llama_token sampled2 = llama_sampler_sample(smpl, ctx_tgt, -1);
                const int32_t pos_last2 = pos_last + n_pass;

                llama_token out[8];
                size_t n_out = 0;
                rc = obit_spec_mtp_draft(spec, 0, sampled2, pos_last2, out, 8, &n_out);
                if (rc != 0) {
                    fprintf(stderr, "draft rc = %d: %s\n", rc, obit_spec_mtp_last_error());
                }
                check(rc == 0, "obit_spec_mtp_draft() succeeded");
                printf("draft drafted %zu token(s) from sampled token %d at pos %d\n",
                        n_out, sampled2, pos_last2);

                rc = obit_spec_mtp_accept(spec, 0, 0);
                check(rc == 0, "obit_spec_mtp_accept() (round 2) succeeded");

                llama_sampler_free(smpl);
                llama_batch_free(batch2);
                llama_batch_free(batch);
            } else {
                printf("skip: tokenize returned %d; draft-cycle check skipped\n", n_toks);
            }
        }

        obit_spec_mtp_free(spec);
        llama_free(ctx_dft);
    }

    llama_free(ctx_tgt);
    llama_model_free(model);
}

int main(void) {
    check(obit_llama_spec_abi_version() == 1, "spec ABI version is 1");

    check(obit_spec_mtp_supported(NULL) < 0, "supported(NULL) returns negative");
    check(obit_spec_mtp_last_error()[0] != '\0', "last_error set after supported(NULL)");

    struct obit_spec_mtp * spec = NULL;
    check(obit_spec_mtp_init(NULL, NULL, 1, 3, &spec) < 0, "init(NULL, NULL, ...) fails");
    check(spec == NULL, "init failure leaves the handle NULL");
    check(obit_spec_mtp_last_error()[0] != '\0', "last_error set after init failure");

    obit_spec_mtp_free(NULL); // must not crash

    const char * non_mtp = getenv("OBIT_NON_MTP_TEST_GGUF");
    if (non_mtp != NULL && non_mtp[0] != '\0') {
        check_non_mtp_model(non_mtp);
    } else {
        printf("skip: $OBIT_NON_MTP_TEST_GGUF not set; non-MTP model check skipped\n");
    }

    const char * mtp = getenv("OBIT_MTP_TEST_GGUF");
    if (mtp != NULL && mtp[0] != '\0') {
        check_mtp_model(mtp);
    } else {
        printf("skip: $OBIT_MTP_TEST_GGUF not set; MTP model check skipped\n");
    }

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }

    printf("obit-spec-mtp-smoke: all checks passed\n");
    return 0;
}
