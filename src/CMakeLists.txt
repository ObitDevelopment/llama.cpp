llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            # Obit fork: stage-aware ABI header + translation unit. The whole
            # fork ABI lives in these two files so upstream merges against the
            # neighboring core files don't conflict with our additions -- which
            # is also why they are listed here rather than appended to
            # upstream's LLAMA_CORE_SOURCES.
            ../include/llama-obit-stage.h
            llama-obit-stage.cpp
            # Obit fork: MTP speculative shim header. The implementation is
            # added by the LLAMA_OBIT_SPEC block below, which also pulls in the
            # `common` sources it needs.
            ../include/llama-obit-spec.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# Obit fork: the stage ABI translation unit is a core source too, so it is
# excluded from the unity build for the same reason upstream excludes the rest.
set_source_files_properties(llama-obit-stage.cpp ../include/llama-obit-stage.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# Obit fork: the MTP speculative shim wraps common_speculative. libllama does
# not link the `common` library today, so with LLAMA_OBIT_SPEC=ON we compile
# the shim plus the smallest closure of `common` sources it needs straight into
# libllama. The common translation units are built with hidden visibility so
# the only new dynamic exports are obit_spec_mtp_* / obit_llama_spec_abi_version.
if (LLAMA_OBIT_SPEC)
    # common.cpp reads llama_build_number()/_commit()/etc from the generated
    # build-info TU. common/CMakeLists generates it only when the common library
    # is built, but the Obit recipe sets LLAMA_BUILD_COMMON=OFF, so generate our
    # own copy here.
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/build-info.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/obit-spec-build-info.cpp @ONLY)

    set(OBIT_SPEC_SOURCES
        llama-obit-spec.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/obit-spec-build-info.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/speculative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/log.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/sampling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/ngram-cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/ngram-map.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/ngram-mod.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/fit.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/unicode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/reasoning-budget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/trie.cpp
    )

    target_sources(llama PRIVATE ${OBIT_SPEC_SOURCES} ../include/llama-obit-spec.h)
    target_compile_definitions(llama PRIVATE LLAMA_OBIT_SPEC=1)
    target_include_directories(llama PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
    # common/json.cpp includes <nlohmann/json.hpp> from the vendored copy under
    # vendor/, the same header-only interface target common/ itself links. A
    # host with a system nlohmann header hides a missing include path, a CI box
    # without one does not (ObitDevelopment/cc #2751, run 36831427261).
    target_link_libraries(llama PRIVATE vendor::nlohmann)

    set_source_files_properties(${OBIT_SPEC_SOURCES} PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION ON)

    if (NOT MSVC)
        set_source_files_properties(
            ${CMAKE_CURRENT_BINARY_DIR}/obit-spec-build-info.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/speculative.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/common.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/log.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/sampling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/ngram-cache.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/ngram-map.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/ngram-mod.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/fit.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/unicode.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/reasoning-budget.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/trie.cpp
            PROPERTIES COMPILE_OPTIONS "-fvisibility=hidden")
    endif()
endif()

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
