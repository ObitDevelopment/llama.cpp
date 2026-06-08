#include "llama.h"

#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// returns true on success
static bool llama_prepare_model_devices(const llama_model_params & params, llama_model * model) {
    // create list of devices to use with this model
    if (params.devices) {
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            size_t n_devs = 0;
            while (params.devices[n_devs]) {
                n_devs++;
            }
            if (n_devs == 0) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: creating a Meta device with %zu devices\n", __func__, n_devs);
            for (size_t i = 0; i < n_devs; ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s\n", __func__, i, ggml_backend_dev_name(params.devices[i]));
            }
            model->get_split_state_ud.n_devices = n_devs;
            model->get_split_state_ud.model = model;
            model->devices.push_back({
                true, ggml_backend_meta_device(
                params.devices, n_devs, llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
                model->devices.push_back({false, *dev});
            }
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<llama_device> gpus;
        std::vector<llama_device> igpus;
        std::vector<llama_device> rpc_servers;

        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            std::vector<ggml_backend_dev_t> devs;
            devs.reserve(ggml_backend_dev_count());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_buffer_type(dev) == ggml_backend_cpu_buffer_type()) {
                    LLAMA_LOG_INFO("%s: skipping %s (%s) for tensor parallelism\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                    continue;
                }
                devs.push_back(dev);
            }
            if (devs.empty()) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }

            LLAMA_LOG_INFO("%s: creating a Meta device for tensor parallelism from %zu devices:\n", __func__, devs.size());
            for (size_t i = 0; i < devs.size(); ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s (%s)\n", __func__, i, ggml_backend_dev_name(devs[i]), ggml_backend_dev_description(devs[i]));
            }

            GGML_ASSERT(!devs.empty());
            model->get_split_state_ud.n_devices = devs.size();
            model->get_split_state_ud.model     = model;
            gpus.push_back({
                true, ggml_backend_meta_device(
                devs.data(), devs.size(), llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                        // skip CPU backends since they are handled separately
                        break;

                    case GGML_BACKEND_DEVICE_TYPE_GPU: {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                        if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                            rpc_servers.push_back({false, dev});
                        } else {
                            // check if there is already a GPU with the same device id
                            ggml_backend_dev_props props;
                            ggml_backend_dev_get_props(dev, &props);
                            auto it = std::find_if(gpus.begin(), gpus.end(), [&props](const llama_device & d) {
                                ggml_backend_dev_props d_props;
                                ggml_backend_dev_get_props(d.dev, &d_props);
                                if (props.device_id && d_props.device_id) {
                                    return strcmp(props.device_id, d_props.device_id) == 0;
                                }
                                return false;
                            });

                            if (it != gpus.end()) {
                                LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                        __func__,
                                        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                        props.device_id ? props.device_id : "unknown id",
                                        ggml_backend_dev_name(it->dev), ggml_backend_dev_description(it->dev));
                            } else {
                                gpus.push_back({false, dev});
                            }
                        }
                        break;
                    }

                    case GGML_BACKEND_DEVICE_TYPE_IGPU:
                        if (igpus.empty()) {
                            igpus.push_back({false, dev});
                        }
                        break;
                    case GGML_BACKEND_DEVICE_TYPE_META:
                        GGML_ABORT("fatal error");
                }
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no discrete GPUs were found
        // (RPC servers do not count, otherwise the local iGPU would be dropped on iGPU+RPC setups)
        if (gpus.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                return false;
            }
            llama_device main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (const auto & dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev.dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev.dev), ggml_backend_dev_description(dev.dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    return true;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.use_mmap, params.use_direct_io,
            params.check_tensors, params.no_alloc, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        bool ok = llama_prepare_model_devices(params, model_ptr.get());
        if (!ok) {
            return {-1, nullptr};
        }

        auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
        if (model == nullptr) {
            GGML_ABORT("fatal error: model does not implement llama_model_base");
        }

        // loading time will be recalculated after the first eval, so
        // we take page faults deferred by mmap() into consideration
        model->t_load_us = 0;
        time_meas tm(model->t_load_us);

        model->t_start_us = tm.t_start_us;

        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return {0, model_ptr.release()};
        }

        if (!model->load_tensors(ml)) {
            return {-2, nullptr};
        }

        return {0, model_ptr.release()};
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return {-1, nullptr};
    }
}

static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params) {
    {
        int n_sources_defined = 0;
        if (metadata != nullptr) {
            n_sources_defined++;
        }
        if (!path_model.empty()) {
            n_sources_defined++;
        }
        if (file != nullptr) {
            n_sources_defined++;
        }
        if (n_sources_defined != 1) {
            LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
            return nullptr;
        }
    }
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    const auto [status, model] = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, file, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        if (model) {
            llama_model_free(model);
        }
        return nullptr;
    }

    return model;
}

struct llama_model * llama_model_init_from_user(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        struct llama_model_params params) {
    GGML_ASSERT(metadata != nullptr);
    std::string path_model;
    std::vector<std::string> splits = {};
    params.use_mmap = false;
    params.use_extra_bufts = false;
    return llama_model_load_from_file_impl(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, /*file*/ nullptr, params);
}
// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    splits.reserve(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, splits.front(), splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_file_ptr(FILE * file, struct llama_model_params params) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }
    std::string path_model;
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, file, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {

    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";

    const int written = snprintf(
        split_path,
        maxlen,
        SPLIT_PATH_FORMAT,
        path_prefix,
        split_no + 1,
        split_count
    );

    if (written < 0 || (size_t) written >= maxlen) {
        return 0;
    }

    return (int32_t) written;
}

int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {

    const std::string str_split_path(split_path);

    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);

    const std::string str_postfix(postfix);
    if (str_split_path.size() <= str_postfix.size()) {
        return 0;
    }

    const size_t size_prefix = str_split_path.size() - str_postfix.size();

    if (str_split_path.compare(size_prefix, std::string::npos, str_postfix) == 0) {
        const size_t copy_len = std::min(size_prefix + 1, maxlen);
        snprintf(split_prefix, copy_len, "%s", split_path);

        return (int32_t) size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

uint32_t obit_llama_abi_version(void) {
    return OBIT_LLAMA_ABI_VERSION;
}

const char * obit_llama_build_info(void) {
    return "obit-llama abi=1 stage_abi=1 stage_flags=0 boundary_info=1 single_stage=1";
}

uint32_t obit_llama_stage_abi_version(void) {
    return OBIT_LLAMA_STAGE_ABI_VERSION;
}

uint64_t obit_llama_stage_capability_flags(void) {
    return OBIT_LLAMA_STAGE_CAPABILITY_NONE;
}

const char * obit_llama_stage_unsupported_reason(void) {
    return "obit libllama stage execution hooks support only total_stages=1 covering the full layer range with emit_logits=true";
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

    if (!is_full_single_stage) {
        obit_llama_stage_set_error(obit_llama_stage_unsupported_reason());
        return nullptr;
    }

    llama_context * ctx = llama_init_from_model(model, context_params);
    if (ctx == nullptr) {
        obit_llama_stage_set_error(
                "obit libllama stage init failed to create llama_context for single-stage runtime");
        return nullptr;
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
